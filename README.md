# GPT-2 QLoRA – Scaling Law & SEAL Scaffold

[![CI/CD Pipeline](https://github.com/Hussain0327/Ai-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/Hussain0327/Ai-Research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hussain0327/Ai-Research/blob/main/AI_Research.ipynb)

This repository is a clean, ready-to-run scaffold focused on GPT‑2 Small with
QLoRA (4‑bit quantization + LoRA) using Hugging Face Transformers, PEFT, and
bitsandbytes. It is optimized for training and evaluation on Google Colab GPUs.

TinyGPT (custom model) and its training pipeline have been intentionally removed
to keep the scope on GPT‑2 + QLoRA and SEAL-style adaptations.

## Quickstart (Colab)

1) Install

```bash
pip install -U "transformers>=4.40" "datasets>=2.15" "peft>=0.8.0" bitsandbytes accelerate
```

2) Train QLoRA adapter (demo, supports .txt or .jsonl)

```bash
python -m src.gpt2_qlora.train \
  --model_name gpt2 \
  --train_file data/sample/train.txt \
  --output_dir checkpoints/gpt2_qlora_demo \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --block_size 128 --batch_size 2 --epochs 1 --lr 1e-4
```

JSONL example (expects key "text"):

```bash
python -m src.gpt2_qlora.train \
  --model_name gpt2 \
  --train_file /path/to/text.JSONL \
  --data_format jsonl --text_key text \
  --output_dir checkpoints/gpt2_qlora_demo \
  --lora_r 8 --block_size 128 --epochs 1
```

3) Evaluate perplexity (txt or jsonl)

```bash
python -m src.gpt2_qlora.eval \
  --model_name gpt2 \
  --adapter_dir checkpoints/gpt2_qlora_demo \
  --eval_file data/sample/train.txt \
  --block_size 128 --max_batches 50
```

JSONL example:

```bash
python -m src.gpt2_qlora.eval \
  --model_name gpt2 \
  --adapter_dir checkpoints/gpt2_qlora_demo \
  --eval_file /path/to/text.JSONL \
  --data_format jsonl --text_key text \
  --block_size 128 --max_batches 50
```

4) SEAL‑style adaptation (LoRA on GPT‑2) (txt or jsonl)

```bash
python -m src.seal.hf_seal \
  --model_name gpt2 \
  --baseline_adapter checkpoints/gpt2_qlora_demo \
  --save_dir checkpoints/gpt2_seal/adapt \
  --results results/gpt2_seal/adapt.jsonl \
  --inner_steps 1 3 10 30 \
  --lora_rank 4 8 \
  --train_file data/sample/train.txt \
  --val_file data/sample/train.txt \
  --block_size 128
```

JSONL example:

```bash
python -m src.seal.hf_seal \
  --model_name gpt2 \
  --baseline_adapter checkpoints/gpt2_qlora_demo \
  --save_dir checkpoints/gpt2_seal/adapt \
  --results results/gpt2_seal/adapt.jsonl \
  --inner_steps 1 3 10 30 \
  --lora_rank 4 8 \
  --train_file /path/to/text.JSONL \
  --val_file /path/to/text.JSONL \
  --data_format jsonl --text_key text \
  --block_size 128
```

Visualize SEAL logs with `scripts/analyze_adaptation.py` (unchanged).

## Repository Map

```
src/
├── gpt2_qlora/
│   ├── train.py              # QLoRA training (HF + PEFT + bitsandbytes)
│   └── eval.py               # Perplexity evaluation
└── seal/
    └── hf_seal.py            # SEAL-style adaptation on GPT‑2 via LoRA

scripts/
└── run_qlora_sweep.py        # Simple sweep runner (e.g., LoRA rank)

configs/
├── gpt2_qlora.yaml           # Example training config
└── seal_qlora.yaml           # Example adaptation config

data/
└── sample/train.txt          # Tiny sample for smoke tests

tests/
└── test_env.py               # Lightweight environment checks
```

## Notes

- QLoRA requires `bitsandbytes` and a CUDA-enabled GPU (Colab recommended).
- Hugging Face model downloads require network access.
- TinyGPT model and training code have been removed from this repository. Data
  samples remain to allow quick smoke tests.

## License

MIT – see LICENSE.
