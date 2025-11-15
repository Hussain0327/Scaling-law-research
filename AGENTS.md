# Repository Guidelines

## Project Structure & Module Organization
`data/` stores generated addition corpora, tokenizer caches, and any attribution maps; keep large binaries out of Git by uploading them to your preferred storage and referencing download scripts. Place exploratory notebooks under `notebooks/` with a numbered prefix (e.g., `01_gpt2_probe.ipynb`). Core libraries live in `src/`, typically split into `datasets/`, `models/`, `training/`, and `analysis/` modules so that `python -m src.training.run_addition` works cleanly. Finished figures, tables, and write ups belong in `reports/` to keep them versioned separately from notebooks.

## Build, Test, and Development Commands
Use `python -m venv .venv && source .venv/bin/activate` to isolate dependencies, then `pip install -r requirements.txt` for the pinned toolchain. Run quick experiments with `python -m src.training.run_addition --dataset data/addition_pairs.json` (adjust the module name to match your script). Execute notebooks via `jupyter lab notebooks` so kernels resolve project-relative imports. Continuous testing should run through `pytest src/tests -vv` before every push.

## Coding Style & Naming Conventions
Default to 4-space indentation, `black` formatting, and type hints everywhere new public functions are introduced. Modules and functions use `snake_case`, classes use `PascalCase`, and configuration files follow the `addition_<scenario>.yaml` pattern. Keep notebooks tidy by clearing extraneous output cells before committing and prefer Markdown headers plus short bullet explanations inside them.

## Testing Guidelines
Add tests near the code they exercise (e.g., `src/tests/test_datasets.py`). Name every test `test_<behavior>` and write fixtures that load minimal JSON or tensor snippets from `data/fixtures/`. Stick with `pytest` parametrization to cover multiple number ranges, and target at least 90% branch coverage for any new training or interpretability routine.

## Commit & Pull Request Guidelines
Project history favors concise, imperative commit subjects such as “Fix Black code formatting” or “Update GPT-2 QLoRA sweep.” Follow that format, include an optional body explaining decisions, and reference issue numbers when available. Pull requests need a short change summary, reproduction commands, updated screenshots from `reports/` if visuals changed, and explicit notes about backward compatibility or data migrations.

## Security & Configuration Tips
Keep API keys and dataset licenses in a local `.env` that gets sourced inside notebooks or scripts; never commit secrets. If you must log private runs, rely on environment variables like `WANDB_API_KEY`. Document any new configuration knobs in `README.md` and provide safe defaults so agents can reproduce results without touching sensitive credentials.
