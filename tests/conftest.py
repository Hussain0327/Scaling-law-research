"""
Shared test fixtures and utilities for all tests.
Provides common test data, mock objects, and helper functions.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append("src")

import pytest
import torch
import numpy as np

from models.tiny_gpt import TinyGPT
from data.tokenizers import CharacterTokenizer


# ============= Model Fixtures =============

@pytest.fixture
def tiny_model():
    """Create a small TinyGPT model for testing."""
    return TinyGPT(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=32,
        dropout=0.1
    )


@pytest.fixture
def medium_model():
    """Create a medium-sized TinyGPT model for testing."""
    return TinyGPT(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=8,
        max_seq_len=128,
        dropout=0.1
    )


@pytest.fixture
def model_configs():
    """Provide various model configurations for testing."""
    return [
        {
            "name": "tiny",
            "vocab_size": 50,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "max_seq_len": 16
        },
        {
            "name": "small",
            "vocab_size": 100,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "max_seq_len": 32
        },
        {
            "name": "medium",
            "vocab_size": 1000,
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 8,
            "max_seq_len": 128
        }
    ]


# ============= Data Fixtures =============

@pytest.fixture
def sample_texts():
    """Provide sample text data for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data.",
        "Transformers have revolutionized NLP tasks.",
        "Attention is all you need for sequence modeling.",
        "GPT models are powerful language generators.",
        "Training neural networks requires computational resources.",
        "Optimization algorithms help models converge.",
        "Evaluation metrics measure model performance."
    ]


@pytest.fixture
def character_tokenizer(sample_texts):
    """Create a character tokenizer with vocabulary built from sample texts."""
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(sample_texts)
    return tokenizer


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    def create_batch(batch_size=2, seq_len=8, vocab_size=100):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len))
        }

    batches = [create_batch() for _ in range(10)]
    dataloader = MagicMock()
    dataloader.__iter__ = MagicMock(return_value=iter(batches))
    dataloader.__len__ = MagicMock(return_value=len(batches))

    return dataloader


@pytest.fixture
def mock_dataloader_factory():
    """Factory for creating custom mock dataloaders."""
    def _create_dataloader(num_batches=10, batch_size=4, seq_len=16, vocab_size=100):
        def create_batch():
            return {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "labels": torch.randint(0, vocab_size, (batch_size, seq_len))
            }

        batches = [create_batch() for _ in range(num_batches)]
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(return_value=iter(batches))
        dataloader.__len__ = MagicMock(return_value=len(batches))

        return dataloader

    return _create_dataloader


# ============= Configuration Fixtures =============

@pytest.fixture
def base_config():
    """Provide base configuration for experiments."""
    return {
        "model": {
            "vocab_size": 100,
            "d_model": 64,
            "n_layers": 4,
            "n_heads": 8,
            "max_seq_len": 128,
            "dropout": 0.1
        },
        "training": {
            "num_epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
            "use_amp": False,
            "eval_interval": 100,
            "save_interval": 200,
            "log_interval": 50
        },
        "data": {
            "dataset_name": "tinystories",
            "batch_size": 32,
            "max_length": 128,
            "tokenizer_type": "char",
            "stride": 64
        }
    }


@pytest.fixture
def minimal_config():
    """Provide minimal configuration for quick tests."""
    return {
        "model": {
            "vocab_size": 50,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "max_seq_len": 32,
            "dropout": 0.0
        },
        "training": {
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "use_amp": False,
            "eval_interval": 10,
            "save_interval": 20,
            "log_interval": 5
        },
        "data": {
            "dataset_name": "custom",
            "batch_size": 4,
            "max_length": 32,
            "tokenizer_type": "char"
        }
    }


# ============= File System Fixtures =============

@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_text_file(temp_dir, sample_texts):
    """Create a temporary text file with sample data."""
    file_path = temp_dir / "sample_data.txt"
    file_path.write_text("\n".join(sample_texts))
    return file_path


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """Create a temporary directory with mock checkpoints."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir()

    # Create mock checkpoints
    for i in range(3):
        checkpoint = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "config": {
                "model": {
                    "vocab_size": 100,
                    "d_model": 32,
                    "n_layers": 2,
                    "n_heads": 4
                }
            },
            "step": (i + 1) * 100,
            "epoch": i + 1,
            "val_loss": 3.0 - i * 0.5
        }
        checkpoint_path = checkpoint_dir / f"checkpoint_{i}.pt"
        torch.save(checkpoint, checkpoint_path)

    return checkpoint_dir


# ============= Testing Utilities =============

@pytest.fixture
def assert_tensor_close():
    """Utility function to assert tensors are close."""
    def _assert_close(tensor1, tensor2, rtol=1e-5, atol=1e-8):
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
            f"Tensors not close: max diff = {(tensor1 - tensor2).abs().max()}"
    return _assert_close


@pytest.fixture
def assert_gradients_valid():
    """Utility function to check if gradients are valid (finite and non-zero)."""
    def _check_gradients(model):
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient in {name}"
        assert has_gradients, "No gradients found in model"
    return _check_gradients


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _set_seed


# ============= Performance Testing Fixtures =============

@pytest.fixture
def benchmark_timer():
    """Timer utility for benchmarking."""
    import time

    class Timer:
        def __init__(self):
            self.times = []

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.times.append(self.end - self.start)

        @property
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0

        @property
        def total(self):
            return sum(self.times)

    return Timer


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import gc

    class MemoryTracker:
        def __init__(self):
            self.measurements = []

        def start(self):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self.start_memory = torch.cuda.memory_allocated()
            else:
                self.start_memory = 0

        def stop(self):
            if torch.cuda.is_available():
                self.end_memory = torch.cuda.memory_allocated()
                self.peak_memory = torch.cuda.max_memory_allocated()
                self.measurements.append({
                    "used": self.end_memory - self.start_memory,
                    "peak": self.peak_memory
                })
            else:
                self.measurements.append({"used": 0, "peak": 0})

        def get_peak_mb(self):
            if self.measurements:
                return max(m["peak"] for m in self.measurements) / (1024**2)
            return 0

    return MemoryTracker


# ============= Mock Objects Fixtures =============

@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases for testing."""
    wandb = MagicMock()
    wandb.init = MagicMock(return_value=MagicMock())
    wandb.log = MagicMock()
    wandb.watch = MagicMock()
    wandb.finish = MagicMock()
    return wandb


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(side_effect=lambda idx: {
        "text": f"Sample text {idx}",
        "input_ids": torch.randint(0, 100, (32,)),
        "labels": torch.randint(0, 100, (32,))
    })
    return dataset


# ============= Markers and Skips =============

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


@pytest.fixture(autouse=True)
def skip_gpu_if_not_available(request):
    """Skip GPU tests if CUDA is not available."""
    if request.node.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")


# ============= Cleanup Fixtures =============

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Clear any cached data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# ============= Test Data Generators =============

@pytest.fixture
def generate_random_text():
    """Generate random text for testing."""
    def _generate(length=100, vocab_size=26):
        chars = [chr(97 + i) for i in range(min(vocab_size, 26))]
        return ''.join(np.random.choice(chars, length))
    return _generate


@pytest.fixture
def generate_batch():
    """Generate random batch for model testing."""
    def _generate(batch_size=4, seq_len=32, vocab_size=100):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len)
        }
    return _generate