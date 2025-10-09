"""SEAL-style adaptation for GPT-2 using PEFT LoRA.

Runs short inner-loop updates (budgeted by ``inner_steps``) on a 4-bit GPT-2
Small with LoRA adapters and logs metrics (Δ val loss, forgetting, tokens).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _bnb() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _tokenize(examples, tokenizer, block_size: int, text_key: str):
    return tokenizer(examples[text_key], truncation=True, max_length=block_size)


@torch.no_grad()
def _eval_loss(model, dataloader, device) -> float:
    model.eval().to(device)
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        valid = (labels != -100).sum().item()
        total_loss += loss.item() * valid
        total_tokens += valid
    return float("inf") if total_tokens == 0 else total_loss / total_tokens


def _cycle(dl):
    while True:
        for b in dl:
            yield b


@dataclass
class AdaptRecord:
    identifier: str
    inner_steps: int
    lora_rank: int
    delta_val_loss: float
    val_before: float
    val_after: float
    forgetting: float
    tokens_processed: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="SEAL (HF GPT-2 + LoRA)")
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--baseline_adapter", type=str, default=None)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--inner_steps", type=int, nargs="+", required=True)
    p.add_argument("--lora_rank", type=int, nargs="+", required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--data_format", type=str, choices=["text", "jsonl"], default="text")
    p.add_argument("--text_key", type=str, default="text")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args(list(argv) if argv is not None else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    # Data
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.data_format == "jsonl":
        ds = load_dataset(
            "json", data_files={"train": args.train_file, "val": args.val_file}
        )
        remove_cols = [args.text_key]
        text_key = args.text_key
    else:
        ds = load_dataset(
            "text", data_files={"train": args.train_file, "val": args.val_file}
        )
        remove_cols = ["text"]
        text_key = "text"
    ds_tok = ds.map(
        lambda ex: _tokenize(ex, tok, args.block_size, text_key),
        batched=True,
        remove_columns=remove_cols,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    train_dl = DataLoader(
        ds_tok["train"], batch_size=2, shuffle=True, collate_fn=collator
    )
    val_dl = DataLoader(ds_tok["val"], batch_size=2, shuffle=False, collate_fn=collator)

    results: list[AdaptRecord] = []
    stream = open(args.results, "a", encoding="utf-8")

    try:
        for steps, rank in product(args.inner_steps, args.lora_rank):
            # Fresh base + adapter per setting
            base = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=_bnb(),
                device_map="auto",
            )
            base.gradient_checkpointing_enable()
            base = prepare_model_for_kbit_training(base)

            # Fresh adapter per budget
            from peft import PeftModel

            model_lora = None
            if args.baseline_adapter:
                try:
                    model_lora = PeftModel.from_pretrained(base, args.baseline_adapter)
                except Exception:
                    model_lora = None
            if model_lora is None:
                lcfg = LoraConfig(
                    r=rank,
                    lora_alpha=max(16, rank * 2),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model_lora = get_peft_model(base, lcfg)

            # Evaluate before
            val_before = _eval_loss(model_lora, val_dl, device)
            train_before = _eval_loss(model_lora, train_dl, device)

            opt = torch.optim.AdamW(
                [p for p in model_lora.parameters() if p.requires_grad], lr=args.lr
            )
            model_lora.train().to(device)
            tokens_processed = 0
            it = _cycle(train_dl)
            for _ in range(steps):
                batch = next(it)
                inp = batch["input_ids"].to(device)
                lab = batch["labels"].to(device)
                opt.zero_grad(set_to_none=True)
                out = model_lora(input_ids=inp, labels=lab)
                out.loss.backward()
                opt.step()
                tokens_processed += (lab != -100).sum().item()

            # Evaluate after
            val_after = _eval_loss(model_lora, val_dl, device)
            train_after = _eval_loss(model_lora, train_dl, device)
            delta = val_before - val_after
            forgetting = max(0.0, train_after - train_before)

            identifier = f"inner{steps}_rank{rank}"
            ckpt_dir = args.save_dir / identifier
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_lora.save_pretrained(str(ckpt_dir))

            rec = AdaptRecord(
                identifier=identifier,
                inner_steps=steps,
                lora_rank=rank,
                delta_val_loss=float(delta),
                val_before=float(val_before),
                val_after=float(val_after),
                forgetting=float(forgetting),
                tokens_processed=int(tokens_processed),
            )
            results.append(rec)
            print(rec.to_json())
            stream.write(rec.to_json() + "\n")
            stream.flush()
    finally:
        stream.close()

    print(f"Saved adapters under {args.save_dir} and logs to {args.results}")


if __name__ == "__main__":
    main()
