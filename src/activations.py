import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATA_PATH = Path("data/raw/addition_test.csv")


def parse_ab(prompt: str) -> Tuple[int, int, int]:
    """Extract the first pair of integers from a prompt and compute their sum."""
    nums = re.findall(r"-?\d+", prompt)
    if len(nums) < 2:
        raise ValueError(f"Expected at least 2 numbers in prompt, got {len(nums)}: {prompt}")
    a, b = map(int, nums[:2])
    return a, b, a + b


def load_model(device: torch.device = DEVICE) -> HookedTransformer:
    print(f"Loading GPT-2 small on device: {device}")
    model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
    model.eval()
    return model


def load_prompts(path: Path, limit: int) -> List[str]:
    """Load prompts from the CSV, ensuring the expected column exists."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        raise ValueError(f"'prompt' column missing from {path}. Columns: {list(df.columns)}")
    prompts = df["prompt"].dropna().astype(str).tolist()
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    selected = prompts[:limit]
    print(f"Loaded {len(selected)} prompts from {path}")
    return selected


def inspect_cache(cache, layer: int):
    resid_pre_name = get_act_name("resid_pre", layer)
    attn_out_name = get_act_name("attn_out", layer)
    mlp_out_name = get_act_name("mlp_out", layer)

    resid_pre = cache[resid_pre_name]
    attn_out = cache[attn_out_name]  # [batch, seq_len, d_model]
    mlp_out = cache[mlp_out_name]  # [batch, seq_len, d_model]

    print(f"\nLayer {layer} resid_pre shape:", resid_pre.shape)
    print(f"Layer {layer} attn_out shape:", attn_out.shape)
    print(f"Layer {layer} mlp_out shape:", mlp_out.shape)

    resid_pre_cpu = resid_pre.detach().cpu()
    attn_out_cpu = attn_out.detach().cpu()
    mlp_out_cpu = mlp_out.detach().cpu()

    print("\nExample resid_pre[0, -1, :5]:", resid_pre_cpu[0, -1, :5])
    print("Example attn_out[0, -1, :5]: ", attn_out_cpu[0, -1, :5])
    print("Example mlp_out[0, -1, :5]:  ", mlp_out_cpu[0, -1, :5])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect GPT-2 activations on addition prompts.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV containing a 'prompt' column (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=32,
        help="Number of prompts to sample from the CSV.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Model layer to inspect for resid/attn/MLP activations.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    model = load_model()
    prompts = load_prompts(args.data_path, args.num_prompts)
    ab_list = [parse_ab(p) for p in prompts]
    print("\nFirst 5 (a, b, sum) triples:", ab_list[:5])

    tokens = model.to_tokens(prompts, prepend_bos=True)

    first_prompt = prompts[0]
    str_tokens = model.to_str_tokens(first_prompt, prepend_bos=True)
    print("\nFirst prompt:", repr(first_prompt))
    print("String tokens:")
    for i, tok in enumerate(str_tokens):
        print(f"  {i}: {repr(tok)}")

    print("Token tensor shape:", tokens.shape)  # [batch, seq_len]

    print("Running model with cache...")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    cache_keys = list(cache.keys())
    print("Number of cache entries:", len(cache_keys))
    print("First 10 cache keys:")
    for k in cache_keys[:10]:
        print("  ", k)

    inspect_cache(cache, args.layer)


if __name__ == "__main__":
    main()
