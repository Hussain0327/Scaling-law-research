#!/usr/bin/env bash
set -euo pipefail

# Convert GPT-2 (HF) to MLX 4-bit and run a tiny generation + perplexity

MODEL_DIR="models/gpt2-mlx-4bit"

if [ ! -d "$MODEL_DIR" ]; then
  echo "[MLX] Converting gpt2 -> $MODEL_DIR (4-bit affine)"
  python -m mlx_lm convert --hf-path gpt2 --mlx-path "$MODEL_DIR" -q --q-bits 4 --q-mode affine
else
  echo "[MLX] Using existing $MODEL_DIR"
fi

echo "[MLX] Generation test:"
python -m mlx_lm generate --model "$MODEL_DIR" --prompt "Hello from MLX" --max-tokens 20 --verbose F

echo "[MLX] Perplexity test on TinyStories (few samples):"
python -m mlx_lm perplexity --model "$MODEL_DIR" --data-path roneneldan/TinyStories --sequence-length 64 --num-samples 64 --batch-size 8

echo "[MLX] Done."

