"""QLoRA training script for GPT-2 Small (HF + PEFT + bitsandbytes).

Designed for Google Colab GPUs. Trains LoRA adapters on top of a 4-bit GPT-2
base and saves the adapter weights alongside a small training summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover - PEFT installed on Colab
    raise RuntimeError(
        "PEFT is required for QLoRA. Install with `pip install peft`."
    ) from exc


@dataclass
class LoRAArgs:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "c_attn",
        "c_proj",
        "q_attn",
        "k_attn",
        "v_attn",
        "fc_in",
        "fc_out",
    )


def _build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _tokenize_function(examples, tokenizer, block_size: int, text_key: str):
    out = tokenizer(examples[text_key], truncation=True, max_length=block_size)
    return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train GPT-2 Small with QLoRA")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    # Data format
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["text", "jsonl"],
        default="text",
        help="Input format: plain text or JSONL (json)",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Field name containing text when using JSONL",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Data
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)

    # Train
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    if args.data_format == "jsonl":
        ds = load_dataset("json", data_files=data_files)
        remove_cols = [args.text_key]
        text_key = args.text_key
    else:
        ds = load_dataset("text", data_files=data_files)
        remove_cols = ["text"]
        text_key = "text"

    tokenized = ds.map(
        lambda ex: _tokenize_function(ex, tokenizer, args.block_size, text_key),
        batched=True,
        remove_columns=remove_cols,
    )

    # Model (4-bit) + PEFT
    bnb_config = _build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        evaluation_strategy="epoch" if "validation" in tokenized else "no",
        save_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save PEFT adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save summary
    summary = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Saved adapter and summary to {output_dir}")


if __name__ == "__main__":
    main()
