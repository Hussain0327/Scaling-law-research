"""
Command-line entry point for running SEAL adaptation sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Iterable, List

from .executor import AdaptationExecutor, AdaptationResult, save_results_jsonl
from .rest_trainer import Candidate, RESTCoordinator


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEAL self-adaptation sweeps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base_config.yaml"),
        help="Path to the base TinyGPT configuration file.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to the baseline checkpoint directory (defaults to save_dir/../baseline).",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where adapted checkpoints will be written.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to the JSONL file where adaptation results will be appended.",
    )
    parser.add_argument(
        "--inner_steps",
        type=int,
        nargs="+",
        required=True,
        help="List of inner-loop gradient steps to evaluate.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        nargs="+",
        required=True,
        help="List of LoRA ranks to evaluate.",
    )
    parser.add_argument(
        "--eval_subset",
        type=int,
        default=200,
        help="Maximum number of batches used for evaluation metrics.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate used for the inner-loop LoRA updates.",
    )
    return parser.parse_args(argv)


def _resolve_baseline_path(baseline: Path | None, save_dir: Path) -> Path:
    if baseline is not None:
        return baseline

    candidate = save_dir.parent / "baseline"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Unable to locate baseline checkpoint directory. Please provide "
        "--baseline explicitly."
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])

    baseline_path = _resolve_baseline_path(args.baseline, args.save_dir)
    executor = AdaptationExecutor(
        config_path=args.config,
        baseline_checkpoint=baseline_path,
        save_root=args.save_dir,
        eval_subset=args.eval_subset,
        learning_rate=args.learning_rate,
    )

    rest = RESTCoordinator()
    all_results: List[AdaptationResult] = []

    for inner_steps, lora_rank in product(args.inner_steps, args.lora_rank):
        print(f"[SEAL] Adapting with inner_steps={inner_steps}, lora_rank={lora_rank}...")
        result = executor.run_budget(inner_steps=inner_steps, lora_rank=lora_rank)
        all_results.append(result)

        candidate = Candidate(
            reward=result.metrics.delta_val_loss,
            identifier=result.identifier,
            metrics=result.metrics,
            checkpoint_path=result.checkpoint_path,
            metadata={
                "inner_steps": inner_steps,
                "lora_rank": lora_rank,
                "tokens_processed": result.tokens_processed,
            },
        )

        rest.consider(candidate)

        summary = {
            "identifier": result.identifier,
            "delta_val_loss": result.metrics.delta_val_loss,
            "forgetting": result.metrics.forgetting,
            "tokens_processed": result.tokens_processed,
        }
        print(f"[SEAL] Result: {json.dumps(summary, indent=2)}")

    save_results_jsonl(all_results, args.results)

    best_summary = rest.summary()
    if best_summary:
        best_dir = args.save_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = rest.best.checkpoint_path if rest.best else None
        if best_ckpt and best_ckpt.exists():
            target = best_dir / best_ckpt.name
            if target.exists():
                target.unlink()
            target.write_bytes(best_ckpt.read_bytes())
        print(f"[SEAL] Top-1 candidate: {json.dumps(best_summary, indent=2)}")


if __name__ == "__main__":
    main()
