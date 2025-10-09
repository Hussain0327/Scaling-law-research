# Contributing to GPT‑2 QLoRA Scaffold

Thanks for your interest in contributing! This repo provides a minimal, reproducible
setup for GPT‑2 Small QLoRA and SEAL‑style adaptation using Hugging Face Transformers,
PEFT, and bitsandbytes.

## Getting Started

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (for training on Colab or your workstation)

Setup:
```bash
pip install -e ".[dev]"
```

Run smoke test:
```bash
make smoke-test
```

## Code Style

- PEP8, 88‑char line length (Black + isort configured)
- Type hints where practical
- Clear docstrings for functions/CLIs

## Tests

CI tests are intentionally lightweight (no large downloads). If you add new scripts,
prefer small unit tests that don’t require GPUs or datasets.

## Contributions

Good improvements include:
- Data handling: more formats (sharded JSONL/globs), streaming
- Training options: more training arguments, schedulers
- SEAL: richer metrics, plots, and summaries
- Colab notebooks: end‑to‑end examples

## Submitting a PR

1. Create a feature branch
2. Keep changes small and focused
3. Update README/ARCHITECTURE if the UX or structure changes
4. Run `make test` before submission

Thanks! All contributions are licensed under MIT, consistent with the repo.
