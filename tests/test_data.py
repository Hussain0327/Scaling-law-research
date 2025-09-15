"""
Unit tests for data loading and tokenization modules.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.append("src")

from data.datamodule import (
    SimpleTextDataModule,
    TextDataset,
    TinyStoriesDataModule,
    create_datamodule,
)
from data.tokenizers import (
    CharacterTokenizer,
    SubwordTokenizer,
    collate_fn,
    create_tokenizer,
)


class TestCharacterTokenizer:
    """Test character-level tokenizer."""

    def test_build_vocab(self):
        """Test vocabulary building from texts."""
        tokenizer = CharacterTokenizer()
        texts = ["hello", "world", "test"]

        tokenizer.build_vocab(texts)

        # Check that all characters are in vocabulary
        expected_chars = set("helloworld test")
        for char in expected_chars:
            assert char in tokenizer.char_to_idx

        # Check special tokens
        assert tokenizer.pad_token in tokenizer.char_to_idx
        assert tokenizer.unk_token in tokenizer.char_to_idx
        assert tokenizer.eos_token in tokenizer.char_to_idx
        assert tokenizer.bos_token in tokenizer.char_to_idx

    def test_encode_decode(self):
        """Test encoding and decoding text."""
        tokenizer = CharacterTokenizer()
        texts = ["hello world"]
        tokenizer.build_vocab(texts)

        text = "hello"
        encoded = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        assert decoded == text

    def test_encode_without_special_tokens(self):
        """Test encoding without special tokens."""
        tokenizer = CharacterTokenizer()
        texts = ["hello"]
        tokenizer.build_vocab(texts)

        text = "hello"
        encoded = tokenizer.encode(text, add_special_tokens=False)

        # Should not contain BOS/EOS tokens
        assert tokenizer.bos_token_id not in encoded
        assert tokenizer.eos_token_id not in encoded

    def test_unknown_character_handling(self):
        """Test handling of unknown characters."""
        tokenizer = CharacterTokenizer()
        texts = ["abc"]
        tokenizer.build_vocab(texts)

        # Encode text with unknown character
        text = "abcd"  # 'd' is unknown
        encoded = tokenizer.encode(text, add_special_tokens=False)

        # Last token should be UNK
        assert encoded[-1] == tokenizer.unk_token_id

    def test_save_load(self):
        """Test saving and loading tokenizer."""
        tokenizer = CharacterTokenizer()
        texts = ["hello world"]
        tokenizer.build_vocab(texts)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            tokenizer.save(temp_path)

            # Load
            loaded_tokenizer = CharacterTokenizer.load(temp_path)

            # Check that vocabularies match
            assert loaded_tokenizer.char_to_idx == tokenizer.char_to_idx
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

            # Test encoding with loaded tokenizer
            text = "hello"
            original_encoded = tokenizer.encode(text)
            loaded_encoded = loaded_tokenizer.encode(text)
            assert original_encoded == loaded_encoded

        finally:
            Path(temp_path).unlink()

    def test_special_token_properties(self):
        """Test special token ID properties."""
        tokenizer = CharacterTokenizer()
        texts = ["hello"]
        tokenizer.build_vocab(texts)

        assert tokenizer.pad_token_id == tokenizer.char_to_idx[tokenizer.pad_token]
        assert tokenizer.unk_token_id == tokenizer.char_to_idx[tokenizer.unk_token]
        assert tokenizer.eos_token_id == tokenizer.char_to_idx[tokenizer.eos_token]
        assert tokenizer.bos_token_id == tokenizer.char_to_idx[tokenizer.bos_token]


class TestSubwordTokenizer:
    """Test subword tokenizer wrapper."""

    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_initialization(self, mock_from_pretrained):
        """Test tokenizer initialization."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.__len__ = MagicMock(return_value=1000)
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = SubwordTokenizer("gpt2")

        assert tokenizer.vocab_size == 1000
        assert tokenizer.tokenizer.pad_token == tokenizer.tokenizer.eos_token

    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_encode_decode(self, mock_from_pretrained):
        """Test encoding and decoding."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "hello world"
        mock_tokenizer.__len__ = MagicMock(return_value=1000)
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = SubwordTokenizer("gpt2")

        # Test encoding
        encoded = tokenizer.encode("hello world")
        assert encoded == [1, 2, 3]

        # Test decoding
        decoded = tokenizer.decode([1, 2, 3])
        assert decoded == "hello world"

    def test_create_tokenizer_factory(self):
        """Test tokenizer factory function."""
        # Test character tokenizer
        char_tokenizer = create_tokenizer("char")
        assert isinstance(char_tokenizer, CharacterTokenizer)

        # Test invalid tokenizer type
        with pytest.raises(ValueError):
            create_tokenizer("invalid")


class TestTextDataset:
    """Test text dataset for language modeling."""

    def test_dataset_creation(self):
        """Test dataset creation with sliding windows."""
        # Create simple tokenizer
        tokenizer = CharacterTokenizer()
        texts = ["abcdefghijk"]
        tokenizer.build_vocab(texts)

        # Create dataset
        dataset = TextDataset(
            texts=["abcdef"],
            tokenizer=tokenizer,
            max_length=4,
            stride=2,
            add_special_tokens=False,
        )

        # Check that samples are created correctly
        assert len(dataset) > 0

        # Check sample format
        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert len(sample["input_ids"]) == 3  # max_length - 1
        assert len(sample["labels"]) == 3

    def test_dataset_with_special_tokens(self):
        """Test dataset with special tokens."""
        tokenizer = CharacterTokenizer()
        texts = ["hello"]
        tokenizer.build_vocab(texts)

        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=10,
            stride=5,
            add_special_tokens=True,
        )

        assert len(dataset) > 0

    def test_empty_dataset(self):
        """Test behavior with empty texts."""
        tokenizer = CharacterTokenizer()
        texts = ["hello"]
        tokenizer.build_vocab(texts)

        dataset = TextDataset(texts=[], tokenizer=tokenizer, max_length=10, stride=5)

        assert len(dataset) == 0


class TestCollateFunction:
    """Test batch collation function."""

    def test_collate_input_ids_only(self):
        """Test collation with input_ids only."""
        batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
            {"input_ids": [6, 7, 8, 9]},
        ]

        collated = collate_fn(batch, pad_token_id=0)

        # Check shape
        assert collated["input_ids"].shape == (3, 4)  # batch_size=3, max_len=4

        # Check padding
        assert collated["input_ids"][1, 2].item() == 0  # Second sequence padded
        assert collated["input_ids"][1, 3].item() == 0

    def test_collate_with_labels(self):
        """Test collation with labels."""
        batch = [
            {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
            {"input_ids": [4, 5], "labels": [5, 6]},
        ]

        collated = collate_fn(batch, pad_token_id=0)

        assert "input_ids" in collated
        assert "labels" in collated
        assert collated["input_ids"].shape == collated["labels"].shape

        # Check that labels are padded with -100
        assert collated["labels"][1, 2].item() == -100

    def test_collate_same_length_sequences(self):
        """Test collation when all sequences have same length."""
        batch = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]

        collated = collate_fn(batch)

        # No padding needed
        assert collated["input_ids"].shape == (2, 3)
        assert torch.equal(collated["input_ids"][0], torch.tensor([1, 2, 3]))


class TestTinyStoriesDataModule:
    """Test TinyStories data module."""

    @patch("datasets.load_dataset")
    def test_data_module_initialization(self, mock_load_dataset):
        """Test data module initialization."""
        datamodule = TinyStoriesDataModule(
            tokenizer_type="char", max_length=128, batch_size=16, max_samples=100
        )

        assert datamodule.tokenizer_type == "char"
        assert datamodule.max_length == 128
        assert datamodule.batch_size == 16

    @patch("datasets.load_dataset")
    def test_prepare_data_with_fraction(self, mock_load_dataset):
        """Test data preparation with data fraction."""
        # Mock dataset
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=1000)
        mock_train.select = MagicMock(return_value=mock_train)

        mock_val = MagicMock()
        mock_val.__len__ = MagicMock(return_value=100)
        mock_val.select = MagicMock(return_value=mock_val)

        mock_dataset = {"train": mock_train, "validation": mock_val}
        mock_load_dataset.return_value = mock_dataset

        datamodule = TinyStoriesDataModule(data_fraction=0.1)
        datamodule.prepare_data()

        # Check that selection was called with correct size
        mock_train.select.assert_called_once_with(range(100))  # 10% of 1000
        mock_val.select.assert_called_once_with(range(10))  # 10% of 100

    def test_setup_character_tokenizer(self):
        """Test character tokenizer setup."""
        # Mock the dataset preparation
        datamodule = TinyStoriesDataModule(tokenizer_type="char")

        # Create mock dataset
        mock_dataset = {
            "train": [{"text": "hello"}, {"text": "world"}],
            "validation": [{"text": "test"}],
        }
        datamodule.raw_dataset = mock_dataset

        datamodule.setup_tokenizer()

        assert isinstance(datamodule.tokenizer, CharacterTokenizer)
        assert datamodule.tokenizer.vocab_size > 0

    @patch("data.tokenizers.create_tokenizer")
    def test_setup_subword_tokenizer(self, mock_create_tokenizer):
        """Test subword tokenizer setup."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 50257
        mock_create_tokenizer.return_value = mock_tokenizer

        datamodule = TinyStoriesDataModule(tokenizer_type="subword")
        datamodule.setup_tokenizer()

        assert datamodule.tokenizer is mock_tokenizer
        mock_create_tokenizer.assert_called_once_with("subword", model_name="gpt2")


class TestSimpleTextDataModule:
    """Test simple text data module for custom files."""

    def test_load_text_file(self):
        """Test loading text from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1\nLine 2\n\nLine 3\n")
            temp_path = f.name

        try:
            datamodule = SimpleTextDataModule(train_file=temp_path)
            texts = datamodule.load_text_file(temp_path)

            assert len(texts) == 3  # Empty lines should be filtered
            assert texts[0] == "Line 1"
            assert texts[1] == "Line 2"
            assert texts[2] == "Line 3"

        finally:
            Path(temp_path).unlink()

    def test_data_splitting(self):
        """Test train/validation split."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join([f"Line {i}" for i in range(10)]))
            temp_path = f.name

        try:
            datamodule = SimpleTextDataModule(
                train_file=temp_path, val_split=0.2, seed=42
            )
            datamodule.prepare_data()

            assert len(datamodule.train_texts) == 8  # 80% of 10
            assert len(datamodule.val_texts) == 2  # 20% of 10

        finally:
            Path(temp_path).unlink()


class TestDataModuleFactory:
    """Test data module factory function."""

    def test_create_tinystories_datamodule(self):
        """Test creating TinyStories data module."""
        datamodule = create_datamodule("tinystories", batch_size=16)
        assert isinstance(datamodule, TinyStoriesDataModule)
        assert datamodule.batch_size == 16

    def test_create_custom_datamodule(self):
        """Test creating custom data module."""
        datamodule = create_datamodule("custom", train_file="dummy.txt")
        assert isinstance(datamodule, SimpleTextDataModule)
        assert datamodule.train_file == "dummy.txt"

    def test_invalid_dataset_name(self):
        """Test error with invalid dataset name."""
        with pytest.raises(ValueError):
            create_datamodule("invalid_dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
