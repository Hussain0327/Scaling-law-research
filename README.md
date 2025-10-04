# TinyGPT – Minimal GPT + Reproducible Scaling-Law Experiments

[![CI/CD Pipeline](https://github.com/Hussain0327/Ai-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/Hussain0327/Ai-Research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

Tiny, readable GPT with a lean training stack and guardrails for scaling-law experiments. Focus is on clarity, reproducibility, and fast iteration—not state-of-the-art tricks.

## What’s inside (today)

* **Model**: Minimal GPT with causal attention, pre-norm blocks, tied embeddings.
  `src/models/tiny_gpt.py`
* **Data**: Character & subword tokenizers; TinyStories + simple text corpora; sliding-window batching.
  `src/data/tokenizers.py`, `src/data/datamodule.py`
* **Training**: AMP optional, cosine LR, **gradient accumulation**, **config-selectable optimizer**, autosafe nested dirs, checkpointing, optional wandb hooks.
  `src/train.py`
* **Eval**: Perplexity/BPC basics; **falls back to val loader** if a test set isn’t provided.
  `src/eval.py`
* **Sweeps**: Parameter sweeps with **deep-copied configs** (no cross-run state bleed).
  `scripts/run_sweep.py`
* **Export**: Exports weights + **character tokenizer artifacts** for downstream use.
  `scripts/export_model.py`
* **Tests**: Tokenizers/data, model blocks, trainer surfaces, sweeps, export.
  `tests/`

> Scope note: claims about 150+ tests, gradient accumulation from day one, large external datasets, etc., have been aligned with reality. What’s here works and is covered; the roadmap lives below.

---

## Quickstart

### 1) Setup (macOS/Linux)

```bash
git clone https://github.com/Hussain0327/Ai-Research.git
cd Ai-Research

# recommended: venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# core deps (PyTorch CPU/MPS on Apple Silicon; CUDA users can install their variant)
pip install torch torchvision torchaudio

# project dev deps (ruff, pytest, etc.)
pip install -r requirements.txt -r requirements-dev.txt  || true
```

Apple Silicon (MPS) sanity check:

```bash
python - <<'PY'
import torch; print("torch", torch.__version__, "MPS:", torch.backends.mps.is_available())
PY
```

### 2) Run tests

```bash
ruff check .
pytest -q tests/test_training.py tests/test_data.py tests/test_run_sweep.py tests/test_export.py
```

### 3) Train a tiny model (SGD + grad accumulation)

```bash
python -m src.train \
  training.optimizer=sgd training.optimizer_kwargs.momentum=0.9 \
  training.grad_accum_steps=4 \
  training.checkpoint_dir=.ckpts/nested/sgd \
  training.output_dir=./results/nested
```

### 4) Evaluate

```bash
python -m src.eval checkpoint_path=.ckpts/nested/sgd/latest.pt
```

### 5) Export (character tokenizer bundled)

```bash
python scripts/export_model.py data.tokenizer_type=char out_dir=./exports/tiny
# => exports/tiny/tokenizer/char_vocab.json
```

---

## Configuration knobs you’ll care about

```yaml
# configs/base_config.yaml (example excerpt)
training:
  lr: 3e-4
  weight_decay: 0.0
  amp: true
  max_grad_norm: 1.0
  optimizer: adamw            # ["adamw","adam","sgd"]
  optimizer_kwargs: {}        # e.g., { momentum: 0.9, nesterov: true } for SGD
  grad_accum_steps: 1         # >1 enables micro-batch accumulation
  checkpoint_dir: ./.ckpts
  output_dir: ./results
data:
  tokenizer_type: char        # ["char","subword"]
  dataset: tinystories        # or "simple_text"
  seq_len: 256
  batch_size: 32
```

Override at the CLI:

```bash
python -m src.train training.optimizer=adam training.grad_accum_steps=8 data.seq_len=512
```

---

## Reproducible sweeps

The sweep script deep-copies configs per run/seed (so seeds/names/hparams don’t leak across runs):

```bash
python scripts/run_sweep.py --config configs/ablation_optimizer.yaml
```

Outputs a per-run summary JSON with seeds, final losses, and paths. Use `scripts/analyze_scaling.py` for plotting if you’ve installed plotting deps; otherwise, treat it as optional.

---

## Repository layout

```
src/
├── data/
│   ├── tokenizers.py
│   └── datamodule.py
├── models/
│   └── tiny_gpt.py
├── eval.py
└── train.py

scripts/
├── run_sweep.py
├── export_model.py
└── analyze_scaling.py        # optional plotting helper

tests/
├── test_models.py
├── test_training.py
├── test_data.py
├── test_run_sweep.py
└── test_export.py

configs/
├── base_config.yaml
└── ablation_optimizer.yaml   # example optimizer/accum settings

Makefile                      # lint/test shortcuts
```

---

## Current capabilities vs. roadmap

### ✅ Supported now

* Minimal GPT (causal, pre-norm, tied embeddings)
* TinyStories + SimpleText data modules
* Char & subword tokenizers
* AMP, cosine LR, gradient clipping
* **Gradient accumulation**
* **Config-selectable optimizer** (adamw/adam/sgd + kwargs)
* Autosafe nested checkpoint/result dirs
* Eval fallback to val when test is absent
* **Export with char tokenizer artifacts**
* Tests covering tokenizers/data, model blocks, trainer surfaces, sweeps, export

### 🚧 Roadmap / nice-to-haves

* More datasets (OpenWebText, The Pile subsets)
* Additional optimizers (Adafactor, Lion) via optional deps
* Richer eval metrics & text generation demos
* Extended tokenizer save/load for subword/BPE vocab artifacts
* Determinism toggles for cudnn/mps across platforms

---

## Notes on scaling-law results

This repo is an **experiment harness**. Any numeric exponents/curves will depend on your datasets, configs, and compute regime. Use `scripts/analyze_scaling.py` to visualize outcomes, and document precise configs/seeds when you publish results.

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'torch'`**
  Activate your venv or install PyTorch (`pip install torch torchvision torchaudio`).
* **macOS spawn/pickle errors in DataLoader**
  Test loaders use `num_workers=0` by default to avoid pickling lambdas/closures.
* **Apple Silicon**
  MPS is supported by upstream PyTorch; availability check printed at startup.

---

## License

MIT — see `LICENSE`.

---

### Acknowledgments

Inspired by the scaling-law literature (OpenAI, DeepMind, Anthropic) and by many elegant open-source transformer repos. This code aims to be educational first, reproducible second, fancy third.
