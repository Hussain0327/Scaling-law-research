"""
Data loading and preprocessing for TinyGPT training.
Supports TinyStories and other text datasets.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Union

from torch.utils.data import DataLoader, Dataset

from .tokenizers import (
    CharacterTokenizer,
    SubwordTokenizer,
    collate_fn,
    create_tokenizer,
)

_DATASET_LOADER: Optional[Callable[..., Any]] = None


def _resolve_load_dataset() -> Callable[..., Any]:
    """Dynamically import ``datasets.load_dataset`` when needed."""

    global _DATASET_LOADER

    if _DATASET_LOADER is not None:
        return _DATASET_LOADER

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "TinyStoriesDataModule requires the optional 'datasets' package. "
            "Install it with `pip install datasets` to enable Hugging Face dataset "
            "support."
        ) from exc

    _DATASET_LOADER = hf_load_dataset
    return _DATASET_LOADER


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Proxy to ``datasets.load_dataset`` with a helpful error if it's unavailable."""

    return _resolve_load_dataset()(*args, **kwargs)


class TextDataset(Dataset):
    """Generic text dataset for language modeling."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: Union[CharacterTokenizer, SubwordTokenizer],
        max_length: int = 256,
        stride: int = 128,
        add_special_tokens: bool = True,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.add_special_tokens = add_special_tokens

        # Pre-tokenize and create sliding windows
        self.samples = self._create_samples()

    def _create_samples(self) -> List[Dict[str, Any]]:
        """Create training samples from texts using sliding window."""
        samples = []

        for text in self.texts:
            # Tokenize the full text
            token_ids = self.tokenizer.encode(
                text, add_special_tokens=self.add_special_tokens
            )

            # Create sliding windows
            if len(token_ids) < 2:  # Need at least 2 tokens for input and label
                continue

            for i in range(0, len(token_ids) - 1, self.stride):
                window = token_ids[i : i + self.max_length]
                if len(window) >= 2:  # Need at least input and label
                    samples.append(
                        {
                            "input_ids": window[:-1],  # Input is all but last token
                            "labels": window[1:],  # Labels are shifted by one
                        }
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class TinyStoriesDataModule:
    """Data module for TinyStories dataset."""

    def __init__(
        self,
        tokenizer_type: str = "char",
        max_length: int = 256,
        stride: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        data_fraction: float = 1.0,
        seed: int = 42,
    ):
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.data_fraction = data_fraction
        self.seed = seed

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Download and prepare the dataset."""
        # Load TinyStories dataset
        print("Loading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories")

        # Take subset of data if requested
        if self.data_fraction < 1.0:
            train_size = int(len(dataset["train"]) * self.data_fraction)
            val_size = int(len(dataset["validation"]) * self.data_fraction)

            dataset["train"] = dataset["train"].select(range(train_size))
            dataset["validation"] = dataset["validation"].select(range(val_size))

        # Limit number of samples if specified
        if self.max_samples:
            train_samples = min(self.max_samples, len(dataset["train"]))
            val_samples = min(self.max_samples // 10, len(dataset["validation"]))

            dataset["train"] = dataset["train"].select(range(train_samples))
            dataset["validation"] = dataset["validation"].select(range(val_samples))

        self.raw_dataset = dataset

    def setup_tokenizer(self) -> None:
        """Set up tokenizer based on the dataset."""
        if self.tokenizer is not None:
            return

        print(f"Setting up {self.tokenizer_type} tokenizer...")

        if self.tokenizer_type == "char":
            # Build character vocabulary
            texts = []
            for split in ["train", "validation"]:
                for item in self.raw_dataset[split]:
                    texts.append(item["text"])
                    if len(texts) >= 10000:  # Sample for vocab building
                        break

            self.tokenizer = create_tokenizer("char")
            self.tokenizer.build_vocab(texts)
            print(f"Character tokenizer vocab size: {self.tokenizer.vocab_size}")

        elif self.tokenizer_type == "subword":
            self.tokenizer = create_tokenizer("subword", model_name="gpt2")
            print(f"Subword tokenizer vocab size: {self.tokenizer.vocab_size}")

        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def setup_datasets(self) -> None:
        """Create PyTorch datasets."""
        if self.train_dataset is not None:
            return

        print("Creating datasets...")

        # Extract texts
        train_texts = [item["text"] for item in self.raw_dataset["train"]]
        val_texts = [item["text"] for item in self.raw_dataset["validation"]]

        # Create datasets
        self.train_dataset = TextDataset(
            train_texts, self.tokenizer, self.max_length, self.stride
        )
        self.val_dataset = TextDataset(
            val_texts, self.tokenizer, self.max_length, self.stride
        )

        # Use validation set as test set for simplicity
        self.test_dataset = self.val_dataset

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            pin_memory=True,
        )

    def get_sample_text(self, num_samples: int = 5) -> List[str]:
        """Get sample texts for inspection."""
        samples = []
        for i, item in enumerate(self.raw_dataset["train"]):
            if i >= num_samples:
                break
            samples.append(
                item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"]
            )
        return samples


class SimpleTextDataModule:
    """Simple data module for custom text files."""

    def __init__(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        tokenizer_type: str = "char",
        max_length: int = 256,
        stride: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        self.train_file = train_file
        self.val_file = val_file
        self.tokenizer_type = tokenizer_type
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def load_text_file(self, file_path: str) -> List[str]:
        """Load text from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into sentences or paragraphs
        sentences = text.split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def prepare_data(self) -> None:
        """Load and split data."""
        print(f"Loading text from {self.train_file}...")
        train_texts = self.load_text_file(self.train_file)

        if self.val_file:
            val_texts = self.load_text_file(self.val_file)
        else:
            # Split training data
            random.seed(self.seed)
            random.shuffle(train_texts)
            split_idx = int(len(train_texts) * (1 - self.val_split))
            val_texts = train_texts[split_idx:]
            train_texts = train_texts[:split_idx]

        self.train_texts = train_texts
        self.val_texts = val_texts

        print(f"Train texts: {len(train_texts)}")
        print(f"Val texts: {len(val_texts)}")

    def setup_tokenizer(self) -> None:
        """Set up tokenizer."""
        if self.tokenizer is not None:
            return

        print(f"Setting up {self.tokenizer_type} tokenizer...")

        if self.tokenizer_type == "char":
            self.tokenizer = create_tokenizer("char")
            self.tokenizer.build_vocab(self.train_texts + self.val_texts)
            print(f"Character tokenizer vocab size: {self.tokenizer.vocab_size}")
        else:
            self.tokenizer = create_tokenizer("subword")
            print(f"Subword tokenizer vocab size: {self.tokenizer.vocab_size}")

    def setup_datasets(self) -> None:
        """Create datasets."""
        if self.train_dataset is not None:
            return

        self.train_dataset = TextDataset(
            self.train_texts, self.tokenizer, self.max_length, self.stride
        )
        self.val_dataset = TextDataset(
            self.val_texts, self.tokenizer, self.max_length, self.stride
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
        )


def create_datamodule(
    dataset_name: str, **kwargs
) -> Union[TinyStoriesDataModule, SimpleTextDataModule]:
    """
    Factory function to create data modules.

    Args:
        dataset_name: Name of dataset ('tinystories' or 'custom')
        **kwargs: Additional arguments for data module

    Returns:
        Data module instance
    """
    if dataset_name == "tinystories":
        return TinyStoriesDataModule(**kwargs)
    elif dataset_name == "custom":
        return SimpleTextDataModule(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_stats(dataloader: DataLoader) -> Dict[str, Any]:
    """Compute statistics for a dataset."""
    total_tokens = 0
    total_samples = 0
    seq_lengths = []

    for batch in dataloader:
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        total_samples += batch_size
        total_tokens += batch_size * seq_len
        seq_lengths.extend([seq_len] * batch_size)

    if total_samples == 0:
        return {
            "total_samples": 0,
            "total_tokens": 0,
            "avg_tokens_per_sample": 0.0,
            "max_seq_length": 0,
            "min_seq_length": 0,
        }

    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": total_tokens / total_samples,
        "max_seq_length": max(seq_lengths),
        "min_seq_length": min(seq_lengths),
    }
