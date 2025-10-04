"""Tests for export utilities."""

import importlib.util
from pathlib import Path

import torch

from models.tiny_gpt import TinyGPT


def _load_export_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts" / "export_model.py"
    spec = importlib.util.spec_from_file_location("export_model", scripts_dir)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_export_model_saves_character_tokenizer(tmp_path):
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text("hello world\nsecond line\n")
    val_file.write_text("validation sample\n")

    model_config = {
        "vocab_size": 32,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "max_seq_len": 8,
        "dropout": 0.0,
    }

    model = TinyGPT(**model_config)

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model": model_config,
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "max_grad_norm": 1.0,
                "use_amp": False,
                "eval_interval": 1,
                "save_interval": 1,
                "log_interval": 1,
            },
            "data": {
                "dataset_name": "custom",
                "train_file": str(train_file),
                "val_file": str(val_file),
                "tokenizer_type": "char",
                "max_length": 8,
                "stride": 4,
                "batch_size": 2,
                "num_workers": 0,
                "val_split": 0.5,
                "seed": 42,
            },
        },
        "best_val_loss": 1.0,
        "step": 1,
    }
    torch.save(checkpoint, checkpoint_path)

    output_dir = tmp_path / "exported"
    export_module = _load_export_module()
    export_module.export_model(str(checkpoint_path), str(output_dir))

    tokenizer_path = output_dir / "tokenizer.json"
    assert tokenizer_path.exists()
    assert tokenizer_path.stat().st_size > 0
