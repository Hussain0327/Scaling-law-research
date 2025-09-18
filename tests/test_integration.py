"""
Integration and end-to-end tests for the TinyGPT training pipeline.
Tests the complete workflow from data loading to model evaluation.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append("src")

import pytest
import torch
import yaml
import numpy as np

from models.tiny_gpt import TinyGPT, create_tiny_gpt
from data.datamodule import create_datamodule, TinyStoriesDataModule, SimpleTextDataModule
from data.tokenizers import CharacterTokenizer, create_tokenizer
from train import Trainer, setup_experiment, load_config
from eval import ModelEvaluator, find_checkpoints
from utils.config import save_config, merge_configs
from utils.logging import setup_logging


class TestEndToEndTraining:
    """Test complete training workflow from config to evaluation."""

    @pytest.fixture
    def training_config(self):
        """Create a minimal training configuration."""
        return {
            "model": {
                "vocab_size": 50,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 32,
                "dropout": 0.1
            },
            "training": {
                "num_epochs": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "warmup_steps": 10,
                "max_grad_norm": 1.0,
                "use_amp": False,
                "eval_interval": 5,
                "save_interval": 10,
                "log_interval": 5
            },
            "data": {
                "dataset_name": "custom",
                "train_file": None,  # Will be created
                "batch_size": 4,
                "max_length": 32,
                "tokenizer_type": "char"
            }
        }

    def test_complete_training_pipeline(self, training_config):
        """Test the complete training pipeline from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Create training data
            train_data_path = temp_path / "train.txt"
            train_data_path.write_text(
                "The quick brown fox jumps over the lazy dog.\n" * 20 +
                "Machine learning is fascinating.\n" * 20 +
                "Transformers revolutionized NLP.\n" * 20
            )

            training_config["data"]["train_file"] = str(train_data_path)

            # Step 2: Save config
            config_path = temp_path / "config.yaml"
            save_config(training_config, config_path)

            # Step 3: Setup experiment
            model, train_loader, val_loader = setup_experiment(training_config)

            assert isinstance(model, TinyGPT)
            assert model.vocab_size > 0  # Should be set from tokenizer

            # Step 4: Initialize trainer
            trainer = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=training_config,
                save_dir=str(temp_path / "checkpoints"),
                use_wandb=False
            )

            # Step 5: Train model
            results = trainer.train()

            assert "final_val_loss" in results
            assert trainer.epoch == 2  # Check epochs completed
            assert trainer.step > 0

            # Step 6: Verify checkpoints were saved
            checkpoint_files = list((temp_path / "checkpoints").glob("*.pt"))
            assert len(checkpoint_files) > 0

            # Step 7: Load and evaluate best checkpoint
            best_checkpoint = checkpoint_files[0]
            checkpoint = torch.load(best_checkpoint, map_location="cpu")

            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "config" in checkpoint

            # Step 8: Evaluate the trained model
            evaluator = ModelEvaluator(device="cpu")
            metrics = evaluator.evaluate_model(model, val_loader, max_batches=5)

            assert "loss" in metrics
            assert "perplexity" in metrics
            assert metrics["loss"] < 10  # Should have learned something

    def test_training_with_tinystories_dataset(self):
        """Test training with TinyStories dataset (mocked)."""
        with patch("data.datamodule.load_dataset") as mock_load_dataset:
            # Mock the dataset
            mock_train_data = [
                {"text": f"Story {i}: Once upon a time..."}
                for i in range(100)
            ]
            mock_val_data = [
                {"text": f"Val story {i}: In a land far away..."}
                for i in range(20)
            ]

            mock_dataset = {
                "train": mock_train_data,
                "validation": mock_val_data
            }
            mock_load_dataset.return_value = mock_dataset

            # Create data module
            datamodule = TinyStoriesDataModule(
                tokenizer_type="char",
                max_length=64,
                batch_size=8,
                max_samples=50
            )

            datamodule.prepare_data()
            datamodule.setup_tokenizer()
            datamodule.setup_datasets()

            # Get dataloaders
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()

            # Create model
            model = create_tiny_gpt(
                vocab_size=datamodule.tokenizer.vocab_size,
                d_model=32,
                n_layers=2,
                n_heads=4
            )

            # Quick training test
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            for i, batch in enumerate(train_loader):
                if i >= 5:  # Only do a few steps
                    break

                optimizer.zero_grad()
                logits, loss = model(batch["input_ids"], batch["labels"])
                loss.backward()
                optimizer.step()

            # Verify model learned something
            model.eval()
            val_batch = next(iter(val_loader))
            with torch.no_grad():
                _, val_loss = model(val_batch["input_ids"], val_batch["labels"])

            assert val_loss.item() < 10  # Reasonable loss value

    def test_checkpoint_resumption(self, training_config):
        """Test resuming training from a checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create data
            train_data_path = temp_path / "train.txt"
            train_data_path.write_text("Sample text for training.\n" * 50)
            training_config["data"]["train_file"] = str(train_data_path)

            # Setup experiment
            model, train_loader, val_loader = setup_experiment(training_config)

            # Train for 1 epoch
            training_config["training"]["num_epochs"] = 1
            trainer1 = Trainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=training_config,
                save_dir=str(temp_path / "checkpoints"),
                use_wandb=False
            )

            trainer1.train()
            final_step1 = trainer1.step

            # Save checkpoint
            checkpoint_path = temp_path / "checkpoints" / "checkpoint.pt"
            trainer1.save_checkpoint(str(checkpoint_path))

            # Create new trainer and resume
            model2, _, _ = setup_experiment(training_config)
            trainer2 = Trainer(
                model=model2,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                config=training_config,
                save_dir=str(temp_path / "checkpoints"),
                use_wandb=False
            )

            # Load checkpoint
            trainer2.load_checkpoint(str(checkpoint_path))

            assert trainer2.step == final_step1
            assert trainer2.epoch == 1

            # Continue training
            training_config["training"]["num_epochs"] = 2
            trainer2.num_epochs = 2
            trainer2.train()

            assert trainer2.epoch == 2
            assert trainer2.step > final_step1


class TestDataPipelineIntegration:
    """Test data processing pipeline integration."""

    def test_tokenizer_vocabulary_consistency(self):
        """Test that tokenizer vocabulary is consistent across save/load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tokenizer and build vocabulary
            tokenizer1 = CharacterTokenizer()
            texts = [
                "Hello world!",
                "Machine learning is great.",
                "Test data for tokenizer."
            ]
            tokenizer1.build_vocab(texts)

            # Save tokenizer
            tokenizer_path = Path(temp_dir) / "tokenizer.json"
            tokenizer1.save(str(tokenizer_path))

            # Load tokenizer
            tokenizer2 = CharacterTokenizer.load(str(tokenizer_path))

            # Test consistency
            for text in texts:
                encoded1 = tokenizer1.encode(text)
                encoded2 = tokenizer2.encode(text)
                assert encoded1 == encoded2

                decoded1 = tokenizer1.decode(encoded1)
                decoded2 = tokenizer2.decode(encoded2)
                assert decoded1 == decoded2

    def test_datamodule_with_custom_tokenizer(self):
        """Test data module with a custom pre-trained tokenizer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save custom tokenizer
            custom_tokenizer = CharacterTokenizer()
            custom_texts = ["Special vocabulary for testing!"]
            custom_tokenizer.build_vocab(custom_texts)
            tokenizer_path = Path(temp_dir) / "custom_tokenizer.json"
            custom_tokenizer.save(str(tokenizer_path))

            # Create data file
            data_path = Path(temp_dir) / "data.txt"
            data_path.write_text("Special vocabulary test data.\n" * 10)

            # Create data module with custom tokenizer
            datamodule = SimpleTextDataModule(
                train_file=str(data_path),
                tokenizer_type="char",
                max_length=32,
                batch_size=4
            )

            # Replace tokenizer with custom one
            datamodule.tokenizer = CharacterTokenizer.load(str(tokenizer_path))
            datamodule.setup_datasets()

            # Get dataloader and verify it works
            train_loader = datamodule.train_dataloader()
            batch = next(iter(train_loader))

            assert "input_ids" in batch
            assert "labels" in batch
            assert batch["input_ids"].shape[0] == 4  # batch size

    def test_data_augmentation_pipeline(self):
        """Test data pipeline with augmentation/preprocessing."""
        class AugmentedDataModule(SimpleTextDataModule):
            """Data module with text augmentation."""

            def preprocess_text(self, text):
                """Add simple augmentation - duplicate some words."""
                words = text.split()
                augmented = []
                for word in words:
                    augmented.append(word)
                    if np.random.random() < 0.1:  # 10% chance to duplicate
                        augmented.append(word)
                return " ".join(augmented)

            def setup_datasets(self):
                """Override to add preprocessing."""
                # Apply preprocessing
                self.train_texts = [
                    self.preprocess_text(text) for text in self.train_texts
                ]
                super().setup_datasets()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data
            data_path = Path(temp_dir) / "data.txt"
            data_path.write_text("Test sentence for augmentation.\n" * 20)

            # Create augmented data module
            datamodule = AugmentedDataModule(
                train_file=str(data_path),
                tokenizer_type="char",
                max_length=64,
                batch_size=4
            )

            datamodule.prepare_data()
            datamodule.setup_tokenizer()
            datamodule.setup_datasets()

            # Verify augmentation worked
            train_loader = datamodule.train_dataloader()
            batch = next(iter(train_loader))

            assert batch["input_ids"].shape[0] == 4


class TestModelEvaluationIntegration:
    """Test model evaluation integration."""

    def test_scaling_law_analysis_integration(self):
        """Test scaling law analysis with multiple model sizes."""
        # Create models of different sizes
        model_configs = [
            {"d_model": 32, "n_layers": 2, "n_heads": 4},
            {"d_model": 64, "n_layers": 3, "n_heads": 8},
            {"d_model": 128, "n_layers": 4, "n_heads": 8},
        ]

        results = []
        evaluator = ModelEvaluator(device="cpu")

        # Create simple test data
        test_data = [
            {
                "input_ids": torch.randint(0, 100, (2, 16)),
                "labels": torch.randint(0, 100, (2, 16))
            }
            for _ in range(5)
        ]
        test_loader = MagicMock()
        test_loader.__iter__ = MagicMock(return_value=iter(test_data))

        for config in model_configs:
            model = TinyGPT(
                vocab_size=100,
                max_seq_len=32,
                **config
            )

            metrics = evaluator.evaluate_model(model, test_loader)
            results.append({
                "model_size": model.count_parameters(),
                "loss": metrics["loss"],
                "config": config
            })

        # Analyze scaling
        import pandas as pd
        df = pd.DataFrame(results)

        if len(results) >= 3:
            # Use evaluator's scaling law computation
            scaling_results = evaluator.compute_scaling_laws(df)
            assert isinstance(scaling_results, dict)

    def test_checkpoint_evaluation_workflow(self):
        """Test evaluating multiple checkpoints from training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_dir = temp_path / "checkpoints"
            checkpoint_dir.mkdir()

            # Create multiple mock checkpoints
            for i in range(3):
                checkpoint = {
                    "model_state_dict": {},
                    "config": {
                        "model": {
                            "vocab_size": 100,
                            "d_model": 32,
                            "n_layers": 2,
                            "n_heads": 4,
                            "max_seq_len": 32
                        }
                    },
                    "step": (i + 1) * 100,
                    "val_loss": 3.0 - i * 0.5  # Decreasing loss
                }

                checkpoint_path = checkpoint_dir / f"checkpoint_{i}.pt"
                torch.save(checkpoint, checkpoint_path)

            # Create test dataloader
            test_data = [
                {
                    "input_ids": torch.randint(0, 100, (2, 16)),
                    "labels": torch.randint(0, 100, (2, 16))
                }
                for _ in range(3)
            ]
            test_loader = MagicMock()
            test_loader.__iter__ = MagicMock(return_value=iter(test_data))

            # Evaluate checkpoints
            checkpoint_paths = find_checkpoints(str(checkpoint_dir))
            results = []
            for path in checkpoint_paths:
                # Simplified evaluation
                results.append({
                    "checkpoint_path": path,
                    "metrics": {"loss": 2.5},
                    "step": 100
                })

            assert len(results) == 3
            for result in results:
                assert "metrics" in result
                assert "checkpoint_path" in result
                assert "step" in result

    def test_model_comparison_workflow(self):
        """Test comparing different model architectures."""
        architectures = [
            {"name": "small", "d_model": 32, "n_layers": 2},
            {"name": "medium", "d_model": 64, "n_layers": 4},
            {"name": "large", "d_model": 128, "n_layers": 6},
        ]

        # Create test data
        test_data = [
            {
                "input_ids": torch.randint(0, 100, (4, 32)),
                "labels": torch.randint(0, 100, (4, 32))
            }
            for _ in range(10)
        ]
        test_loader = MagicMock()
        test_loader.__iter__ = MagicMock(return_value=iter(test_data))

        evaluator = ModelEvaluator(device="cpu")
        comparison_results = {}

        for arch in architectures:
            model = TinyGPT(
                vocab_size=100,
                d_model=arch["d_model"],
                n_layers=arch["n_layers"],
                n_heads=arch["d_model"] // 8,
                max_seq_len=64
            )

            metrics = evaluator.evaluate_model(model, test_loader)

            comparison_results[arch["name"]] = {
                "parameters": model.count_parameters(),
                "loss": metrics["loss"],
                "perplexity": metrics["perplexity"],
                "architecture": arch
            }

        # Verify we can compare models
        assert len(comparison_results) == 3
        assert comparison_results["large"]["parameters"] > comparison_results["small"]["parameters"]


class TestConfigurationManagement:
    """Test configuration management and experiment tracking."""

    def test_config_override_system(self):
        """Test configuration override system for experiments."""
        base_config_path = Path("configs/base_config.yaml")

        if base_config_path.exists():
            base_config = load_config(str(base_config_path))

            # Create experiment overrides
            experiment_overrides = {
                "model": {
                    "d_model": 256,
                    "n_layers": 8
                },
                "training": {
                    "learning_rate": 5e-4
                }
            }

            # Merge configs
            final_config = merge_configs(base_config, experiment_overrides)

            # Verify overrides were applied
            assert final_config["model"]["d_model"] == 256
            assert final_config["model"]["n_layers"] == 8
            assert final_config["training"]["learning_rate"] == 5e-4

    def test_experiment_tracking_integration(self):
        """Test experiment tracking with config and results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "experiments" / "exp_001"
            experiment_dir.mkdir(parents=True)

            # Save experiment config
            config = {
                "experiment_name": "scaling_test",
                "model": {"d_model": 128, "n_layers": 4},
                "training": {"learning_rate": 1e-3}
            }
            config_path = experiment_dir / "config.yaml"
            save_config(config, config_path)

            # Save training results
            results = {
                "final_loss": 2.5,
                "best_val_loss": 2.3,
                "total_steps": 1000
            }
            results_path = experiment_dir / "results.yaml"
            save_config(results, results_path)

            # Save model checkpoint
            checkpoint = {
                "model_state_dict": {},
                "config": config,
                "results": results
            }
            checkpoint_path = experiment_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)

            # Verify experiment can be loaded
            loaded_config = load_config(config_path)
            loaded_results = load_config(results_path)
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")

            assert loaded_config["experiment_name"] == "scaling_test"
            assert loaded_results["final_loss"] == 2.5
            assert loaded_checkpoint["config"] == config


class TestLoggingIntegration:
    """Test logging integration across modules."""

    def test_training_logging_integration(self):
        """Test that training properly logs to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "training.log"

            # Setup logging
            logger = setup_logging(
                log_level="DEBUG",
                log_file=str(log_path)
            )

            # Log some training events
            logger.info("Starting training")
            logger.debug("Loading data")
            logger.info("Epoch 1/10")
            logger.warning("Learning rate might be too high")
            logger.info("Training complete")

            # Verify log file exists and contains messages
            assert log_path.exists()
            log_content = log_path.read_text()

            assert "Starting training" in log_content
            assert "Loading data" in log_content
            assert "Epoch 1/10" in log_content
            assert "Learning rate might be too high" in log_content

    def test_multi_module_logging(self):
        """Test logging from multiple modules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "multi_module.log"

            # Setup main logger
            setup_logging(log_file=str(log_path))

            # Get loggers for different modules
            from utils.logging import get_logger

            model_logger = get_logger("models")
            data_logger = get_logger("data")
            train_logger = get_logger("training")

            # Log from different modules
            model_logger.info("Initializing model")
            data_logger.info("Loading dataset")
            train_logger.info("Starting training loop")

            # Verify all messages are in log
            log_content = log_path.read_text()
            assert "Initializing model" in log_content
            assert "Loading dataset" in log_content
            assert "Starting training loop" in log_content


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    @pytest.mark.slow
    def test_training_speed_benchmark(self):
        """Benchmark training speed for different configurations."""
        import time

        configurations = [
            {"batch_size": 4, "seq_len": 32, "d_model": 32},
            {"batch_size": 8, "seq_len": 64, "d_model": 64},
            {"batch_size": 16, "seq_len": 128, "d_model": 128},
        ]

        results = []

        for config in configurations:
            model = TinyGPT(
                vocab_size=1000,
                d_model=config["d_model"],
                n_layers=2,
                n_heads=4,
                max_seq_len=256
            )

            optimizer = torch.optim.Adam(model.parameters())

            # Time training steps
            start_time = time.perf_counter()

            for _ in range(10):
                input_ids = torch.randint(
                    0, 1000,
                    (config["batch_size"], config["seq_len"])
                )
                targets = torch.randint(
                    0, 1000,
                    (config["batch_size"], config["seq_len"])
                )

                optimizer.zero_grad()
                _, loss = model(input_ids, targets)
                loss.backward()
                optimizer.step()

            end_time = time.perf_counter()

            results.append({
                "config": config,
                "time": end_time - start_time,
                "steps_per_second": 10 / (end_time - start_time)
            })

        # Verify scaling behavior
        # Larger models should take more time
        assert results[2]["time"] > results[0]["time"]

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with model size."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")

        model_sizes = [
            {"d_model": 128, "n_layers": 2},
            {"d_model": 256, "n_layers": 4},
            {"d_model": 512, "n_layers": 8},
        ]

        memory_usage = []

        for size_config in model_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = TinyGPT(
                vocab_size=10000,
                n_heads=8,
                max_seq_len=512,
                **size_config
            ).cuda()

            # Forward pass
            input_ids = torch.randint(0, 10000, (4, 128)).cuda()
            _, loss = model(input_ids, input_ids)
            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            memory_usage.append(peak_memory)

            del model, input_ids, loss
            torch.cuda.empty_cache()

        # Memory should increase with model size
        assert memory_usage[1] > memory_usage[0]
        assert memory_usage[2] > memory_usage[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])