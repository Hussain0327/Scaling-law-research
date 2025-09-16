"""
TinyGPT: A minimal GPT implementation for scaling law research.
"""

__version__ = "0.1.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

# Import main components for easy access
from .models.tiny_gpt import TinyGPT, create_tiny_gpt
from .data.datamodule import (
    TinyStoriesDataModule,
    SimpleTextDataModule,
    create_datamodule,
)
from .data.tokenizers import CharacterTokenizer, SubwordTokenizer, create_tokenizer

__all__ = [
    "TinyGPT",
    "create_tiny_gpt",
    "TinyStoriesDataModule",
    "SimpleTextDataModule",
    "create_datamodule",
    "CharacterTokenizer",
    "SubwordTokenizer",
    "create_tokenizer",
]
