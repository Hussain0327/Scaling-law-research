"""Evaluate GPT-2 Small with (optional) QLoRA adapter on text/JSONL data.

Computes token-weighted loss and perplexity over a text file using HF Trainer
utilities. Intended to run on Colab GPUs but falls back to CPU when needed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment]


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _tokenize(examples, tokenizer, block_size: int, text_key: str):
    return tokenizer(examples[text_key], truncation=True, max_length=block_size)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model.eval().to(device)
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        valid = (labels != -100).sum().item()
        total_loss += loss.item() * valid
        total_tokens += valid

    if total_tokens == 0:
        return {"loss": float("inf"), "perplexity": float("inf")}

    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "perplexity": float(math.exp(avg_loss))}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 QLoRA")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["text", "jsonl"],
        default="text",
        help="Input format for eval file",
    )
    parser.add_argument(
        "--text_key", type=str, default="text", help="JSONL text field name"
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.data_format == "jsonl":
        ds = load_dataset("json", data_files={"eval": args.eval_file})
        remove_cols = [args.text_key]
        text_key = args.text_key
    else:
        ds = load_dataset("text", data_files={"eval": args.eval_file})
        remove_cols = ["text"]
        text_key = "text"

    tokenized = ds.map(
        lambda ex: _tokenize(ex, tokenizer, args.block_size, text_key),
        batched=True,
        remove_columns=remove_cols,
    )

    from torch.utils.data import DataLoader
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dl = DataLoader(
        tokenized["eval"], batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    bnb = _bnb_config()
    # Always load base from model_name; attach adapter if provided
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        device_map="auto",
    )

    if args.adapter_dir and PeftModel is not None:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    # Optionally limit batches
    if args.max_batches is not None:
        from itertools import islice
        dl_iter = islice(iter(dl), args.max_batches)
        metrics = evaluate(model, dl_iter, device)
    else:
        metrics = evaluate(model, dl, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
