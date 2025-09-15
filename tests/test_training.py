"""
Unit tests for training functionality.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

sys.path.append("src")

from models.tiny_gpt import TinyGPT
from train import Trainer, load_config, setup_experiment


class TestTrainer:
    """Test trainer functionality."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "model": {
                "vocab_size": 100,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 16,
                "dropout": 0.1,
            },
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "warmup_steps": 5,
                "max_grad_norm": 1.0,
                "use_amp": False,
                "eval_interval": 10,
                "save_interval": 20,
                "log_interval": 5,
            },
        }

    @pytest.fixture
    def mock_dataloaders(self):
        """Create mock dataloaders for testing."""

        def create_mock_batch():
            return {
                "input_ids": torch.randint(0, 100, (2, 8)),
                "labels": torch.randint(0, 100, (2, 8)),
            }

        train_loader = MagicMock()
        train_loader.__iter__ = MagicMock(
            return_value=iter([create_mock_batch() for _ in range(3)])
        )
        train_loader.__len__ = MagicMock(return_value=3)

        val_loader = MagicMock()
        val_loader.__iter__ = MagicMock(
            return_value=iter([create_mock_batch() for _ in range(2)])
        )
        val_loader.__len__ = MagicMock(return_value=2)

        return train_loader, val_loader

    def test_trainer_initialization(self, minimal_config, mock_dataloaders):
        """Test trainer initialization."""
        model = TinyGPT(**minimal_config["model"])
        train_loader, val_loader = mock_dataloaders

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=minimal_config,
            use_wandb=False,
        )

        assert trainer.model is model
        assert trainer.num_epochs == 1
        assert trainer.learning_rate == 1e-3
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_warmup_lr_calculation(self, minimal_config, mock_dataloaders):
        """Test learning rate warmup calculation."""
        model = TinyGPT(**minimal_config["model"])
        train_loader, val_loader = mock_dataloaders

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=minimal_config,
            use_wandb=False,
        )

        # Test warmup schedule
        assert trainer.warmup_lr(0) == 0.0
        assert trainer.warmup_lr(trainer.warmup_steps // 2) == trainer.learning_rate / 2
        assert trainer.warmup_lr(trainer.warmup_steps) == trainer.learning_rate

    def test_train_step(self, minimal_config, mock_dataloaders):
        """Test single training step."""
        model = TinyGPT(**minimal_config["model"])
        train_loader, val_loader = mock_dataloaders

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=minimal_config,
            use_wandb=False,
        )

        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8)),
        }

        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        loss = trainer.train_step(batch)

        # Check that loss is computed
        assert isinstance(loss, float)
        assert loss > 0

        # Check that parameters were updated
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.equal(
                    param, initial_params[name]
                ), f"Parameter {name} was not updated"

    @patch("torch.no_grad")
    def test_evaluate(self, mock_no_grad, minimal_config, mock_dataloaders):
        """Test evaluation function."""
        model = TinyGPT(**minimal_config["model"])
        train_loader, val_loader = mock_dataloaders

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=minimal_config,
            use_wandb=False,
        )

        # Mock torch.no_grad context manager
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()

        metrics = trainer.evaluate()

        assert "val_loss" in metrics
        assert "val_perplexity" in metrics
        assert isinstance(metrics["val_loss"], float)
        assert isinstance(metrics["val_perplexity"], float)

    def test_save_load_checkpoint(self, minimal_config, mock_dataloaders):
        """Test checkpoint saving and loading."""
        model = TinyGPT(**minimal_config["model"])
        train_loader, val_loader = mock_dataloaders

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=minimal_config,
                save_dir=temp_dir,
                use_wandb=False,
            )

            # Save checkpoint
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            trainer.step = 100
            trainer.epoch = 5
            trainer.best_val_loss = 2.5
            trainer.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

            # Load checkpoint
            loaded_checkpoint = trainer.load_checkpoint(str(checkpoint_path))

            assert trainer.step == 100
            assert trainer.epoch == 5
            assert trainer.best_val_loss == 2.5
            assert "model_state_dict" in loaded_checkpoint
            assert "optimizer_state_dict" in loaded_checkpoint

    @patch("wandb.init")
    @patch("wandb.watch")
    def test_wandb_integration(
        self, mock_watch, mock_init, minimal_config, mock_dataloaders
    ):
        """Test Weights & Biases integration."""
        model = TinyGPT(**minimal_config["model"])
        train_loader, val_loader = mock_dataloaders

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=minimal_config,
            use_wandb=True,
        )

        mock_init.assert_called_once()
        mock_watch.assert_called_once_with(model, log="all", log_freq=1000)


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_config(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "model": {"vocab_size": 1000, "d_model": 64},
            "training": {"num_epochs": 5, "learning_rate": 1e-4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == config_data
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_config(self):
        """Test loading invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestExperimentSetup:
    """Test experiment setup functionality."""

    @patch("data.datamodule.create_datamodule")
    def test_setup_experiment(self, mock_create_datamodule):
        """Test experiment setup with mocked data."""
        config = {
            "model": {
                "vocab_size": 100,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 16,
                "dropout": 0.1,
            },
            "data": {"dataset_name": "tinystories", "batch_size": 16},
        }

        # Mock data module
        mock_datamodule = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 100
        mock_datamodule.tokenizer = mock_tokenizer

        # Mock dataloaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_datamodule.train_dataloader.return_value = mock_train_loader
        mock_datamodule.val_dataloader.return_value = mock_val_loader

        mock_create_datamodule.return_value = mock_datamodule

        model, train_loader, val_loader = setup_experiment(config)

        assert isinstance(model, TinyGPT)
        assert model.vocab_size == 100
        assert train_loader is mock_train_loader
        assert val_loader is mock_val_loader

        # Verify data module methods were called
        mock_datamodule.prepare_data.assert_called_once()
        mock_datamodule.setup_tokenizer.assert_called_once()
        mock_datamodule.setup_datasets.assert_called_once()

    @patch("data.datamodule.create_datamodule")
    def test_setup_experiment_vocab_size_mismatch(self, mock_create_datamodule):
        """Test experiment setup when vocab sizes don't match."""
        config = {
            "model": {
                "vocab_size": 100,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 16,
                "dropout": 0.1,
            },
            "data": {"dataset_name": "tinystories", "batch_size": 16},
        }

        # Mock data module with different vocab size
        mock_datamodule = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 200  # Different from config
        mock_datamodule.tokenizer = mock_tokenizer

        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_datamodule.train_dataloader.return_value = mock_train_loader
        mock_datamodule.val_dataloader.return_value = mock_val_loader

        mock_create_datamodule.return_value = mock_datamodule

        model, train_loader, val_loader = setup_experiment(config)

        # Model should be recreated with correct vocab size
        assert model.vocab_size == 200


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_minimal_training_run(self):
        """Test a minimal training run without external dependencies."""
        # Create minimal config
        config = {
            "model": {
                "vocab_size": 50,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 8,
                "dropout": 0.0,
            },
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "warmup_steps": 2,
                "max_grad_norm": 1.0,
                "use_amp": False,
                "eval_interval": 2,
                "save_interval": 10,
                "log_interval": 1,
            },
        }

        # Create model
        model = TinyGPT(**config["model"])

        # Create simple synthetic data
        def create_batch():
            return {
                "input_ids": torch.randint(0, 50, (2, 4)),
                "labels": torch.randint(0, 50, (2, 4)),
            }

        train_data = [create_batch() for _ in range(3)]
        val_data = [create_batch() for _ in range(2)]

        # Mock dataloaders
        train_loader = MagicMock()
        train_loader.__iter__ = MagicMock(return_value=iter(train_data))
        train_loader.__len__ = MagicMock(return_value=len(train_data))

        val_loader = MagicMock()
        val_loader.__iter__ = MagicMock(return_value=iter(val_data))
        val_loader.__len__ = MagicMock(return_value=len(val_data))

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=config,
                save_dir=temp_dir,
                use_wandb=False,
            )

            # Run training
            results = trainer.train()

            # Check that training completed
            assert "final_val_loss" in results
            assert "model_parameters" in results
            assert results["model_parameters"] == model.count_parameters()

            # Check that some training steps were taken
            assert trainer.step > 0

            # Check that checkpoints were created
            checkpoint_files = list(Path(temp_dir).glob("*.pt"))
            assert len(checkpoint_files) > 0

    def test_gradient_accumulation_equivalent(self):
        """Test that training with different batch sizes gives similar results."""
        torch.manual_seed(42)

        config = {
            "model": {
                "vocab_size": 30,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 2,
                "max_seq_len": 4,
                "dropout": 0.0,
            },
            "training": {
                "num_epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "max_grad_norm": 1.0,
                "use_amp": False,
                "eval_interval": 100,
                "save_interval": 100,
                "log_interval": 100,
            },
        }

        # Fixed sequence for reproducibility
        fixed_sequence = torch.tensor([[1, 2, 3, 4]])

        def create_fixed_batch(batch_size):
            return {
                "input_ids": fixed_sequence.repeat(batch_size, 1),
                "labels": fixed_sequence.repeat(batch_size, 1),
            }

        # Test with batch size 1
        model1 = TinyGPT(**config["model"])
        data1 = [create_fixed_batch(1)]
        train_loader1 = MagicMock()
        train_loader1.__iter__ = MagicMock(return_value=iter(data1))
        train_loader1.__len__ = MagicMock(return_value=1)

        val_loader1 = MagicMock()
        val_loader1.__iter__ = MagicMock(return_value=iter(data1))
        val_loader1.__len__ = MagicMock(return_value=1)

        trainer1 = Trainer(
            model=model1,
            train_dataloader=train_loader1,
            val_dataloader=val_loader1,
            config=config,
            use_wandb=False,
        )

        # Perform one training step
        loss1 = trainer1.train_step(create_fixed_batch(1))

        # Test with batch size 2 (should be approximately same loss per sample)
        torch.manual_seed(42)  # Reset seed
        model2 = TinyGPT(**config["model"])
        trainer2 = Trainer(
            model=model2,
            train_dataloader=train_loader1,
            val_dataloader=val_loader1,
            config=config,
            use_wandb=False,
        )

        loss2 = trainer2.train_step(create_fixed_batch(2))

        # Losses should be similar (within reasonable tolerance due to batching effects)
        assert abs(loss1 - loss2) < 0.5, f"Losses too different: {loss1} vs {loss2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
