"""
Tokenization utilities for TinyGPT.
Supports both character-level and subword tokenization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


def _resolve_gpt2_tokenizer_fast():
    """Import :class:`transformers.GPT2TokenizerFast` only when required."""

    try:
        from transformers import GPT2TokenizerFast  # type: ignore import
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "SubwordTokenizer requires the optional 'transformers' package. "
            "Install it with `pip install transformers` to use Hugging Face tokenizers."
        ) from exc

    return GPT2TokenizerFast


class CharacterTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is None:
            self.char_to_idx = {}
            self.idx_to_char = {}
            self.vocab_size = 0
        else:
            self.char_to_idx = vocab
            self.idx_to_char = {idx: char for char, idx in vocab.items()}
            self.vocab_size = len(vocab)

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from a list of texts."""
        chars = set()
        for text in texts:
            chars.update(text)

        # Add special tokens
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.eos_token,
            self.bos_token,
        ]

        # Ensure special tokens are not duplicated in regular chars
        chars = chars - set(special_tokens)
        all_chars = special_tokens + sorted(list(chars))

        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(all_chars)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids."""
        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_idx[self.bos_token])

        for char in text:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx[self.unk_token]))

        if add_special_tokens:
            tokens.append(self.char_to_idx[self.eos_token])

        return tokens

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        special_token_ids = {
            self.char_to_idx[self.pad_token],
            self.char_to_idx[self.unk_token],
            self.char_to_idx[self.eos_token],
            self.char_to_idx[self.bos_token],
        }

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            chars.append(self.idx_to_char.get(token_id, self.unk_token))

        return "".join(chars)

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        vocab_data = {
            "char_to_idx": self.char_to_idx,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "eos_token": self.eos_token,
                "bos_token": self.bos_token,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharacterTokenizer":
        """Load tokenizer from file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        tokenizer = cls(vocab=vocab_data["char_to_idx"])
        special_tokens = vocab_data["special_tokens"]
        tokenizer.pad_token = special_tokens["pad_token"]
        tokenizer.unk_token = special_tokens["unk_token"]
        tokenizer.eos_token = special_tokens["eos_token"]
        tokenizer.bos_token = special_tokens["bos_token"]

        return tokenizer

    @property
    def pad_token_id(self) -> int:
        return self.char_to_idx[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.char_to_idx[self.unk_token]

    @property
    def eos_token_id(self) -> int:
        return self.char_to_idx[self.eos_token]

    @property
    def bos_token_id(self) -> int:
        return self.char_to_idx[self.bos_token]


class SubwordTokenizer:
    """Wrapper around HuggingFace tokenizer for subword tokenization."""

    def __init__(self, model_name: str = "gpt2"):
        tokenizer_cls = _resolve_gpt2_tokenizer_fast()
        self.tokenizer = tokenizer_cls.from_pretrained(model_name)

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = len(self.tokenizer)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token ids."""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=max_length is not None,
        )

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id


def create_tokenizer(
    tokenizer_type: str = "char", **kwargs
) -> Union[CharacterTokenizer, SubwordTokenizer]:
    """
    Factory function to create tokenizers.

    Args:
        tokenizer_type: Type of tokenizer ('char' or 'subword')
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "char":
        return CharacterTokenizer(**kwargs)
    elif tokenizer_type == "subword":
        return SubwordTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def collate_fn(
    batch: List[Dict[str, Any]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching sequences.

    Args:
        batch: List of samples with 'input_ids' and optionally 'labels'
        pad_token_id: Token ID to use for padding

    Returns:
        Batched and padded tensors
    """
    input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]

    # Pad sequences to same length
    max_len = max(len(seq) for seq in input_ids)
    padded_input_ids = []

    for seq in input_ids:
        padded_seq = torch.full((max_len,), pad_token_id, dtype=seq.dtype)
        padded_seq[: len(seq)] = seq
        padded_input_ids.append(padded_seq)

    result = {"input_ids": torch.stack(padded_input_ids)}

    # Handle labels if present
    if "labels" in batch[0]:
        labels = [torch.tensor(sample["labels"]) for sample in batch]
        padded_labels = []

        for seq in labels:
            padded_seq = torch.full(
                (max_len,), -100, dtype=seq.dtype
            )  # -100 is ignore index
            padded_seq[: len(seq)] = seq
            padded_labels.append(padded_seq)

        result["labels"] = torch.stack(padded_labels)

    return result
