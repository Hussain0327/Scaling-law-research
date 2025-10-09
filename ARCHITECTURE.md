# GPT-2 QLoRA Repository — Deep Dive

Goal: a compact, reproducible scaffold for running GPT‑2 Small experiments with
QLoRA (HF Transformers + PEFT + bitsandbytes) and SEAL‑style adaptation loops.

## Overview

- Model: Hugging Face `gpt2` (Small) loaded in 4‑bit via bitsandbytes.
- Fine‑tuning: LoRA adapters applied with PEFT; gradient checkpointing enabled.
- Data: Plain text files via `datasets` (no custom Datamodule required).
- Evaluation: Perplexity over a text file using LM labels.
- SEAL: Short inner‑loop edits on top of the base using LoRA; logs Δ val loss and forgetting.

## Repo Map

```
src/
├── gpt2_qlora/
│   ├── train.py              # QLoRA training entrypoint
│   └── eval.py               # Perplexity evaluation
└── seal/
    └── hf_seal.py            # SEAL-style adaptation (LoRA) on GPT‑2

scripts/
└── run_qlora_sweep.py        # Simple sweep runner

configs/
├── gpt2_qlora.yaml           # Example training config
└── seal_qlora.yaml           # Example SEAL config
```

## Data Flow

1. Load tokenizer (`AutoTokenizer`), set PAD to EOS if missing.
2. Load text files with `datasets` and tokenize to fixed `block_size`.
3. Load GPT‑2 in 4‑bit (`BitsAndBytesConfig`).
4. Prepare model for k‑bit training and apply PEFT LoRA.
5. Train with `Trainer` (cosine LR, eval per epoch).
6. Save adapter and a small JSON training summary.
7. Evaluate perplexity on text files or run SEAL adaptation loops.

## SEAL Adapter Loop

- For each (inner_steps, LoRA rank):
  - Init a fresh LoRA adapter on the base.
  - Evaluate val/train loss before editing.
  - Run `inner_steps` gradient steps on train batches.
  - Evaluate after; compute `delta_val_loss = before - after` and
    `forgetting = max(0, train_after - train_before)`.
  - Save adapter under `checkpoints/<identifier>/` and append JSONL record.

## Notes

- Training is intended for Colab GPUs; this repo is kept lightweight.
- TinyGPT code and tests have been removed.
- For scaling‑law analysis across sweeps, adapt `scripts/run_qlora_sweep.py` and
  your plotting as needed.

