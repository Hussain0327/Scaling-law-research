"""Utility script to visualise SEAL adaptation experiment results.

The script consumes the JSONL log produced by ``src.seal.run`` and generates a
set of plots that mirror the deliverables referenced in the orchestration
prompt:

* ``adaptation_laws.png`` – log/log plot of reward (Δ validation loss) versus
  the inner-loop update budget.
* ``pareto_forgetting.png`` – improvement versus forgetting scatter with
  a Pareto-style guide.
* ``adaptation_summary.csv`` – table of aggregated metrics for downstream
  analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _coerce_dataframe(records: Iterable[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "inner_steps" in df.columns:
        df["inner_steps"] = df["inner_steps"].astype(int)
    if "lora_rank" in df.columns:
        df["lora_rank"] = df["lora_rank"].astype(int)
    if "delta_val_loss" in df.columns:
        df["delta_val_loss"] = df["delta_val_loss"].astype(float)
    if "forgetting" in df.columns:
        df["forgetting"] = df["forgetting"].astype(float)
    if "tokens_processed" in df.columns:
        df["tokens_processed"] = df["tokens_processed"].astype(int)
    return df


def _plot_adaptation_law(df: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    palette = sns.color_palette("husl", df["lora_rank"].nunique())

    for (rank), subset in df.groupby("lora_rank"):
        subset = subset.sort_values("inner_steps")
        plt.plot(
            subset["inner_steps"],
            subset["delta_val_loss"],
            marker="o",
            label=f"LoRA rank {rank}",
            color=palette.pop(0),
        )

    plt.xscale("log")
    plt.xlabel("Inner steps (log scale)")
    plt.ylabel("Δ Validation Loss (positive = improvement)")
    plt.title("Adaptation law: improvement versus update budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_forgetting(df: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    scatter = sns.scatterplot(
        data=df,
        x="forgetting",
        y="delta_val_loss",
        hue="lora_rank",
        style="inner_steps",
        palette="viridis",
        s=120,
    )
    scatter.axhline(0.0, linestyle="--", color="grey", alpha=0.6)

    plt.xlabel("Forgetting (Δ train loss, ↑ = worse)")
    plt.ylabel("Δ Validation Loss (↑ = better)")
    plt.title("Improvement versus forgetting")
    plt.tight_layout()
    plt.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close()


def _write_summary(df: pd.DataFrame, destination: Path) -> None:
    summary = (
        df.groupby(["inner_steps", "lora_rank"])
        .agg(
            delta_val_loss_mean=("delta_val_loss", "mean"),
            delta_val_loss_std=("delta_val_loss", "std"),
            forgetting_mean=("forgetting", "mean"),
            tokens_processed_mean=("tokens_processed", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(destination, index=False)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse SEAL adaptation runs.")
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Path to the JSONL log produced by src.seal.run",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for plots and summary tables.",
    )

    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(args.logs)
    if not records:
        raise ValueError(f"No records found in {args.logs}")

    df = _coerce_dataframe(records)

    _plot_adaptation_law(df, args.output_dir / "adaptation_laws.png")
    _plot_forgetting(df, args.output_dir / "pareto_forgetting.png")
    _write_summary(df, args.output_dir / "adaptation_summary.csv")

    print(f"Wrote plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
