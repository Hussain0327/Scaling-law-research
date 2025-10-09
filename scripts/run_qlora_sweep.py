"""Simple sweep runner for GPT-2 QLoRA experiments.

Invokes the training script across a grid of LoRA ranks or data fractions and
records a JSON summary for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    p = argparse.ArgumentParser(description="Run QLoRA sweeps")
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--lora_r", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--data_format", type=str, choices=["text", "jsonl"], default="text")
    p.add_argument("--text_key", type=str, default="text")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    for r in args.lora_r:
        run_dir = out_root / f"gpt2_r{r}"
        cmd = [
            "python",
            "-m",
            "src.gpt2_qlora.train",
            "--model_name",
            args.model_name,
            "--train_file",
            args.train_file,
            "--output_dir",
            str(run_dir),
            "--lora_r",
            str(r),
            "--epochs",
            str(args.epochs),
            "--block_size",
            str(args.block_size),
            "--batch_size",
            str(args.batch_size),
        ]
        if args.val_file:
            cmd += ["--val_file", args.val_file]
        if args.data_format:
            cmd += ["--data_format", args.data_format]
        if args.text_key:
            cmd += ["--text_key", args.text_key]
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd)
        all_results.append({"lora_r": r, "output_dir": str(run_dir), "return_code": rc})

    summary = {
        "train_file": args.train_file,
        "val_file": args.val_file,
        "results": all_results,
    }
    (out_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Sweep completed. Summary at", out_root / "sweep_summary.json")


if __name__ == "__main__":
    main()
