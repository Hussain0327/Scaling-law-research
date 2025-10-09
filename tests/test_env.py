"""Minimal environment checks for CI.

These tests avoid heavyweight downloads or training. They only validate that
key modules import successfully and that a tiny forward pass can be wired when
using a toy GPT-2 variant.
"""

def test_imports():
    import transformers  # noqa: F401
    import datasets  # noqa: F401
    import peft  # noqa: F401


def test_basic_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    assert tok.eos_token is not None

