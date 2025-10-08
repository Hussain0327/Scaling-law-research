# TinyGPT Repository — Deep Dive & Ops Playbook

> Goal: give you a complete, at‑a‑glance understanding of the repo you pasted ("Ai-Research" / TinyGPT), how its pieces fit, how to run/extend it, and where to improve. This is based on the file list and high‑level descriptions you provided.

---

## 1) TL;DR Overview

* **Model**: Decoder‑only Transformer (TinyGPT) with tied embeddings, attention + MLP blocks, and generation helpers.
* **Data**: TinyStories (HF) or local corpora via a **DataModule**; character & subword tokenizers available.
* **Training**: Single‑GPU/accelerator trainer with AMP/mixed precision, warmup + LR scheduler, grad accumulation, checkpointing, experiment wiring.
* **Evaluation**: Checkpoint scoring, sampling, basic power‑law fits + scaling diagnostics; exports plots/CSVs.
* **Automation**: YAML configs for baseline/scaling/ablations; scripts for sweeps and analysis.
* **Tests**: Mocked, light‑weight unit tests for models, data, trainer behaviors, export and sweep glue.

---

## 2) Repo Map (by purpose)

**Top‑level / docs / tooling**

* `README.md` — setup, quickstart, experiment recipes & scaling‑law overview.
* `LICENSE`, `CONTRIBUTING.md` — legal and contribution guidelines.
* `Makefile` — common tasks (lint/test/format/run).
* `TEST_COVERAGE_REPORT.md` — intended test coverage notes.
* `pyproject.toml` — packaging + dependencies (editable install supported: `-e .`).

**Source code (`src/`)**

* `train.py` — CLI/entrypoint for training; loads config, builds model+datamodule, trainer orchestration.
* `eval.py` — evaluation CLI with `ModelEvaluator`: load checkpoints, compute metrics, fit scaling curves, emit plots.
* `config.py` — YAML/JSON config IO, default merging, validation.
* `logging.py` — logging setup/util helpers.
* `data/datamodule.py` — dataset plumbing (TinyStories or local), DataLoader creation, splits, collation.
* `data/tokenizers.py` — character and subword tokenizers; encode/decode; `collate_fn`.
* `models/tiny_gpt.py` — attention/MLP blocks; **TinyGPT** implementation; parameter counting; generate API.
* `export_model.py` — (likely) script to save/export model weights/formats.
* `run_sweep.py` — sweep automation (vary configs; submit runs; aggregate).
* `analyze_scaling.py` — offline analysis/plotting of scaling laws from checkpoints/results.

**Configs & examples (`configs/`, `data/sample/`)**

* `base_config.yaml` — default training config (model/data/opt/schedule/logging/checkpoint).
* `scaling_*.yaml` — width/depth/context/data scaling sweeps.
* `ablation_*.yaml` — optimizer/tokenizer ablations; knobs for comparative runs.
* `sample_custom.json`, `sample_custom_baseline.json` — example configs.

**Results & artifacts**

* `results/` — CSV summaries, plots, per‑experiment folders (e.g., `evaluation_results.csv`).
* `checkpoints/` — saved checkpoints for sample runs (`sample_custom/`, etc.).

**Tests (`tests/`)**

* `test_models.py`, `test_data.py`, `test_training.py`, `test_run_sweep.py`, `test_export.py` — mocked unit tests targeting core behaviors and CLI wiring.

---

## 3) Data Flow: end‑to‑end

1. **Config load** → `config.py` merges YAML/JSON + CLI overrides.
2. **Tokenizer** → `tokenizers.py` builds char/BPE tokenizer; exposes `encode`, `decode`, and `collate_fn`.
3. **DataModule** → `datamodule.py` resolves dataset (TinyStories from HF or local files), splits, DataLoader with `collate_fn`.
4. **Model** → `models/tiny_gpt.py` constructs TinyGPT per config (n_layers, d_model, n_heads, context length, dropout…); ties input/output embeddings; creates generation helpers; computes param counts.
5. **Trainer** → `train.py` sets seeds, device/precision (AMP), builds optimizer + scheduler (warmup + cosine/step, etc.), supports grad accumulation + clipping; logs; saves checkpoints.
6. **Eval** → `eval.py` loads checkpoints, computes validation loss/perplexity, optionally samples, fits power laws, saves plots + CSV.

---

## 4) Core Config Surface (typical keys)

> Names may vary; align with your `base_config.yaml`.

* **model**: `vocab_size`, `d_model`, `n_layers`, `n_heads`, `mlp_ratio` or `d_ff`, `dropout`, `bias`, `tie_embeddings`, `max_seq_len`.
* **data**: `dataset` (tinystories/local), `data_dir`, `train_frac/val_frac`, `num_workers`, `batch_size`, `tokenizer` (char/bpe), `block_size`.
* **optim**: `optimizer` (adamw), `lr`, `betas`, `weight_decay`, `eps`, `grad_clip`.
* **schedule**: `epochs` or `max_steps`, `warmup_steps`, `lr_schedule` (cosine/linear), `min_lr`.
* **runtime**: `seed`, `device`/backend, `amp`/`precision`, `grad_accum_steps`, `log_interval`.
* **checkpointing**: `save_dir`, `save_every`, `keep_last_k`, `best_by` (e.g., val_loss).
* **eval**: `eval_every`, `eval_subset_tokens` or examples, `sample_n`, `sample_temperature`.

---

## 5) Training Loop Features

* Mixed precision (AMP) with autocast + GradScaler (or backend equivalent).
* Warmup then scheduled LR; optimizer = AdamW by default.
* Gradient accumulation for large effective batch size; optional grad clipping.
* Determinism knobs (seeding, cudnn/mps flags when applicable).
* Checkpointing: periodic + best‑by‑metric; resumable.

**Quickstart (typical)**

```bash
python -m src.train \
  --config configs/base_config.yaml \
  --save_dir checkpoints/baseline \
  --seed 42
```

---

## 6) Evaluation & Scaling

* **Metrics**: validation loss, perplexity; optional qualitative samples.
* **Scaling**: fits simple power laws (loss vs parameters/data/context) using results across checkpoints/sweeps; plots diagnostics.
* **CLI (example)**

```bash
python -m src.eval \
  --config configs/base_config.yaml \
  --checkpoint_dir checkpoints/baseline \
  --output_dir results/baseline_eval
```

Artifacts: CSV with per‑checkpoint metrics, plots (loss curves, scaling fits), and optional generations.

---

## 7) Tokenizers & DataModule

* **Character tokenizer**: minimal, robust for TinyStories; small vocab.
* **Subword/BPE**: better efficiency; ensure `vocab_size` syncs with model.
* **Collate**: pads/truncates to `block_size`/`max_seq_len`; builds input/target shifts for next‑token prediction.
* **DataModule**:

  * TinyStories via HF (if available), otherwise local text under `data/`.
  * Split strategy (train/val fractions); shuffling; num workers; persistent workers for speed.

---

## 8) Tests at a Glance

These provide fast regression coverage without heavy downloads:

* **Model**: shape/forward pass, parameter counting, generation shape.
* **Data**: tokenizer encode/decode roundtrip; small DataLoader yields; collate correctness.
* **Training**: one or few steps with mocked dataset; scheduler step; checkpoint write/read.
* **Sweeps/Export**: config expansion, artifact paths, export format sanity.

> Tip: run `pytest -q` and enforce on PRs (CI). Consider adding `--durations=10` to find slow tests.

---

## 9) Repro Recipes

**Baseline**

1. Train: see Quickstart above.
2. Eval: see Evaluation CLI.
3. Inspect: `results/evaluation_results.csv` and plots under `results/*`.

**Scaling (examples)**

* Width: vary `d_model` across a log grid; hold depth/context constant.
* Depth: vary `n_layers`.
* Context: vary `max_seq_len`.
* Data: vary tokens seen (subset dataset, or `max_steps`).

Use provided `scaling_*.yaml` for convenience; run via `run_sweep.py` or your launcher.

---

## 10) Gaps & Recommendations

1. **Config validation**: strengthen type/range checks (e.g., ensure `d_model % n_heads == 1*0?` → should be divisible by `n_heads`). Fail fast.
2. **Repro & determinism**: surface flags for deterministic ops; record git SHA, Python/torch versions in logs.
3. **Logging**: add structured JSON logging (loss, lr, tokens, time/step) for easy analysis; keep text logger for humans.
4. **Metrics**: include tokens‑per‑second, examples/sec, step time, and memory footprint in logs.
5. **Checkpoint policy**: save `last`, `best`, and periodic (e.g., every N steps); compress older ones.
6. **Eval set**: pin a canonical TinyStories validation subset to stabilize scaling fits; document sample size.
7. **Tokenizer drift**: persist tokenizer state alongside checkpoints; validate `vocab_size` vs model head.
8. **AMP safety**: blacklist unstable ops in half precision; add autocast unit test.
9. **Scheduler**: expose cosine/linear; log LR each step; test warmup boundary.
10. **Scaling fits**: report confidence intervals and R²; save fit parameters in YAML/JSON.
11. **Plotting**: ensure non‑interactive backends in headless runs; save SVG + PNG.
12. **Packaging**: publish minimal wheels; pin dependency ranges; provide `requirements.txt` for quick use.
13. **CLI UX**: `--override key=value` parser to patch YAML without creating files; echo the merged config to `run_config.yaml` in the save dir.
14. **Distributed**: if needed later, abstract trainer to support DDP/FSDP; for laptops, provide MPS guards.
15. **Docs**: add `ARCHITECTURE.md` (this doc) and `DATA.md` with dataset provenance & token counts.

---

## 11) Extending the Repo

**New dataset**

* Implement a loader under `data/`; add switch in `datamodule.py` and update config with `dataset: new_name`, plus any preprocessing.

**New tokenizer**

* Add class to `tokenizers.py` with `train()`, `encode()`, `decode()`, and integrate into `collate_fn`.

**Model variants**

* Add `models/<name>.py`; make a small factory in `train.py` keyed by `config.model.name`.

**Schedulers/optimizers**

* Drop in a builder function (e.g., `build_optimizer`, `build_scheduler`) parameterized by config; write a tiny unit test.

---

## 12) Common Pitfalls & Fixes

* **Shape mismatch**: ensure `d_model % n_heads == 0`; context length = tokenizer block size.
* **Vocab mismatch**: if you switch tokenizers, rebuild model or adjust output head size; load tokenizer from checkpoint.
* **OOM**: reduce `batch_size` or increase `grad_accum_steps`; lower `d_model`/`n_layers`; turn off activations checkpointing unless implemented.
* **Divergence**: lower LR; increase warmup; check data quality and tokenization.
* **Slow data loader**: bump `num_workers`, enable persistent workers; pre‑tokenize dataset.

---

## 13) Minimal "Sanity" Config (example)

```yaml
model:
  name: tiny_gpt
  d_model: 256
  n_layers: 4
  n_heads: 4
  max_seq_len: 256
  dropout: 0.1
  tie_embeddings: true

data:
  dataset: tinystories
  tokenizer: char
  batch_size: 64
  block_size: 256
  num_workers: 4

optim:
  optimizer: adamw
  lr: 3e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0

schedule:
  max_steps: 2000
  warmup_steps: 200
  lr_schedule: cosine
  min_lr: 3e-5

runtime:
  seed: 42
  amp: true

checkpoint:
  save_dir: checkpoints/sanity
  save_every: 500
  keep_last_k: 3

eval:
  eval_every: 200
  eval_subset: 2048
  sample_n: 2
  sample_temperature: 0.8
```

---

## 14) SEAL + MLX (Integration sketch)

You pasted an orchestration that migrates to GPT‑2 via **MLX** on macOS and runs **SEAL** self‑adaptation + scaling. That can live as an optional module:

* Add `src/seal/{policy.py,edit_lang.py,executor.py,reward.py,rest_trainer.py}` (inner‑loop LoRA or last‑layer updates on MPS; outer loop ReST, keep top‑1; reward = Δval_loss).
* Config stub (your `configs/ablation_seal.yaml`) toggles `seal.enabled`, sets inner steps grid and LoRA rank, and logs to `results/seal_gpt2/adapt.jsonl`.
* Provide `scripts/analyze_adaptation.py` to plot Δloss vs update budget and a pareto curve (Δloss vs forgetting).

> This remains optional; baseline TinyGPT training/eval runs independent of MLX/SEAL.

---

## 15) Quality Bar / Definition of Done

* Reproducible baseline: run completes and writes `evaluation_results.csv` with val loss for multiple checkpoints.
* Plots: baseline scaling curves render without errors.
* Unit tests: `pytest -q` passes locally; CI configured.
* Artifacts: checkpoint folder includes tokenizer state + `run_config.yaml`.
* Docs: README updated with commands and expected artifacts; this playbook checked in as `ARCHITECTURE.md`.

---

## 16) Handy Commands

```bash
# Training
python -m src.train --config configs/base_config.yaml --save_dir checkpoints/baseline --seed 42

# Evaluation
python -m src.eval --config configs/base_config.yaml \
  --checkpoint_dir checkpoints/baseline \
  --output_dir results/baseline_eval

# Run tests
pytest -q

# Analyze scaling (example)
python -m src.analyze_scaling --checkpoint_dirs checkpoints/baseline \
  --output_dir results/baseline_plots
```

---

## 17) Next Steps (pick any)

* ✅ Re‑run baseline on your machine; confirm `evaluation_results.csv` and plots regenerate.
* 🔧 Add JSON logging and a `run_config.yaml` dumper per run dir.
* 🧪 Expand tests to cover warmup boundary conditions and autocast.
* 📈 Run a small scaling sweep (depth or width) and compare fits.
* 🧪 (Optional) Prototype `src/seal/*` with a toy LoRA edit and log Δval_loss.

---

*This document is meant to live in the repo (e.g., `ARCHITECTURE.md`). Ping me with a snippet of any config or code you want me to annotate line‑by‑line.*
