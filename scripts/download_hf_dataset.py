#!/usr/bin/env python3
"""
Download a text dataset from Hugging Face and materialize it to local files.

- Supports any HF dataset name with optional config via --name
- Writes splits under data/hf/<dataset>[/<name>]/{train,validation,test}.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Download HF text dataset to disk")
    p.add_argument("dataset", type=str, help="HF dataset id, e.g. 'wikitext'")
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional dataset config name, e.g. 'wikitext-2-raw-v1'",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated splits to export (default: train,validation)",
    )
    p.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="Field name that contains the text (default: text)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hf"),
        help="Root output folder (default: data/hf)",
    )
    args = p.parse_args(argv)

    # Load
    ds = load_dataset(args.dataset, args.name) if args.name else load_dataset(args.dataset)

    # Prepare output
    out = args.output_dir / args.dataset
    if args.name:
        out = out / args.name
    out.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in ds:
            print(f"[warn] split '{split}' not present; available: {list(ds.keys())}")
            continue
        path = out / f"{split}.txt"
        with path.open("w", encoding="utf-8") as f:
            for ex in ds[split]:
                text = ex.get(args.text_key)
                if not text:
                    # Try common alternative keys
                    text = ex.get("content") or ex.get("instruction") or ex.get("prompt")
                if not text:
                    continue
                # Ensure newline separation
                if not text.endswith("\n"):
                    text += "\n"
                f.write(text)
        print(f"✓ wrote {path}")


if __name__ == "__main__":
    main()

