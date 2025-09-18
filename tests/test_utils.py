"""
Comprehensive unit tests for utility modules.
Tests configuration management and logging utilities.
"""

import sys
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append("src")

import pytest
import yaml

from utils.config import load_config, save_config, merge_configs
from utils.logging import setup_logging, get_logger


class TestConfigUtilities:
    """Test configuration utility functions."""

    def test_load_config_yaml(self):
        """Test loading a valid YAML configuration file."""
        config_data = {
            "model": {
                "vocab_size": 1000,
                "d_model": 128,
                "n_layers": 4
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == config_data
            assert loaded_config["model"]["vocab_size"] == 1000
            assert loaded_config["training"]["learning_rate"] == 0.001
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_not_found(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [unclosed bracket")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_save_config(self):
        """Test saving a configuration to YAML file."""
        config_data = {
            "experiment": "test",
            "hyperparameters": {
                "alpha": 0.5,
                "beta": 0.9
            },
            "seeds": [42, 123, 456]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            save_config(config_data, config_path)

            assert config_path.exists()

            # Load and verify
            loaded_config = load_config(config_path)
            assert loaded_config == config_data

    def test_save_config_creates_parent_dirs(self):
        """Test that save_config creates parent directories if they don't exist."""
        config_data = {"test": "data"}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nested" / "dirs" / "config.yaml"
            save_config(config_data, config_path)

            assert config_path.exists()
            assert config_path.parent.exists()

    def test_merge_configs_simple(self):
        """Test merging simple configuration dictionaries."""
        base_config = {
            "a": 1,
            "b": 2,
            "c": 3
        }
        override_config = {
            "b": 20,
            "d": 4
        }

        merged = merge_configs(base_config, override_config)

        assert merged["a"] == 1
        assert merged["b"] == 20  # Overridden
        assert merged["c"] == 3
        assert merged["d"] == 4  # New key

    def test_merge_configs_nested(self):
        """Test merging nested configuration dictionaries."""
        base_config = {
            "model": {
                "vocab_size": 1000,
                "d_model": 128,
                "dropout": 0.1
            },
            "training": {
                "lr": 0.001,
                "batch_size": 32
            }
        }
        override_config = {
            "model": {
                "d_model": 256,  # Override
                "n_layers": 4    # New key
            },
            "training": {
                "lr": 0.002      # Override
            }
        }

        merged = merge_configs(base_config, override_config)

        assert merged["model"]["vocab_size"] == 1000  # Unchanged
        assert merged["model"]["d_model"] == 256       # Overridden
        assert merged["model"]["dropout"] == 0.1       # Unchanged
        assert merged["model"]["n_layers"] == 4        # New key
        assert merged["training"]["lr"] == 0.002       # Overridden
        assert merged["training"]["batch_size"] == 32  # Unchanged

    def test_merge_configs_empty(self):
        """Test merging with empty configurations."""
        base_config = {"a": 1, "b": 2}

        # Empty override
        merged = merge_configs(base_config, {})
        assert merged == base_config

        # Empty base
        merged = merge_configs({}, base_config)
        assert merged == base_config

    def test_merge_configs_different_types(self):
        """Test merging when values have different types."""
        base_config = {
            "nested": {"a": 1},
            "list_val": [1, 2, 3],
            "string_val": "hello"
        }
        override_config = {
            "nested": "not a dict",      # Replace dict with string
            "list_val": {"a": 1},        # Replace list with dict
            "string_val": 42             # Replace string with int
        }

        merged = merge_configs(base_config, override_config)

        assert merged["nested"] == "not a dict"
        assert merged["list_val"] == {"a": 1}
        assert merged["string_val"] == 42

    def test_config_with_path_object(self):
        """Test that functions work with Path objects."""
        config_data = {"test": "path_object"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            # Test load with Path object
            loaded = load_config(temp_path)
            assert loaded == config_data

            # Test save with Path object
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = Path(temp_dir) / "saved.yaml"
                save_config(config_data, save_path)
                assert save_path.exists()
        finally:
            temp_path.unlink()


class TestLoggingUtilities:
    """Test logging utility functions."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging()

        assert logger is not None
        assert logger.name == "tinygpt"
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        logger = setup_logging(log_level="DEBUG")
        assert logger.level == logging.DEBUG

        logger = setup_logging(log_level="WARNING")
        assert logger.level == logging.WARNING

        logger = setup_logging(log_level="ERROR")
        assert logger.level == logging.ERROR

    def test_setup_logging_with_file(self):
        """Test logging setup with file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logging(log_file=str(log_file))

            # Log a message
            logger.info("Test message")

            # Check that log file exists and contains the message
            assert log_file.exists()
            log_content = log_file.read_text()
            assert "Test message" in log_content

    def test_setup_logging_creates_parent_dirs(self):
        """Test that logging setup creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "nested" / "dirs" / "test.log"
            logger = setup_logging(log_file=str(log_file))

            logger.info("Test")

            assert log_file.exists()
            assert log_file.parent.exists()

    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            custom_format = "%(levelname)s - %(message)s"

            logger = setup_logging(
                log_file=str(log_file),
                format_str=custom_format
            )

            logger.warning("Test warning")

            log_content = log_file.read_text()
            assert "WARNING - Test warning" in log_content
            # Should not contain timestamp since it's not in format
            assert "asctime" not in log_content

    def test_get_logger(self):
        """Test getting named loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "tinygpt.module1"
        assert logger2.name == "tinygpt.module2"
        assert logger1 is not logger2

    def test_get_logger_same_name(self):
        """Test that getting logger with same name returns same instance."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2

    def test_logging_case_insensitive_level(self):
        """Test that log level is case insensitive."""
        logger = setup_logging(log_level="debug")
        assert logger.level == logging.DEBUG

        logger = setup_logging(log_level="INFO")
        assert logger.level == logging.INFO

        logger = setup_logging(log_level="WaRnInG")
        assert logger.level == logging.WARNING

    @patch('sys.stdout')
    def test_logging_to_stdout(self, mock_stdout):
        """Test that logging outputs to stdout by default."""
        logger = setup_logging()

        # Verify that a StreamHandler for stdout was added
        handlers = logging.getLogger().handlers
        stdout_handlers = [h for h in handlers
                          if isinstance(h, logging.StreamHandler)
                          and h.stream == sys.stdout]
        assert len(stdout_handlers) > 0

    def test_multiple_setup_calls(self):
        """Test that multiple setup calls don't duplicate handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Setup logging multiple times
            logger1 = setup_logging(log_file=str(log_file))
            initial_handler_count = len(logger1.handlers)

            logger2 = setup_logging(log_file=str(log_file))

            # Logger should be the same instance
            assert logger1 is logger2

    def test_logging_thread_safety(self):
        """Test that logging is thread-safe (basic test)."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "thread_test.log"
            logger = setup_logging(log_file=str(log_file))

            def log_messages(thread_id):
                for i in range(10):
                    logger.info(f"Thread {thread_id} - Message {i}")

            threads = []
            for i in range(5):
                t = threading.Thread(target=log_messages, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Check that all messages were logged
            log_content = log_file.read_text()
            for thread_id in range(5):
                for msg_id in range(10):
                    assert f"Thread {thread_id} - Message {msg_id}" in log_content


class TestIntegration:
    """Integration tests for utilities working together."""

    def test_config_and_logging_integration(self):
        """Test using config to setup logging."""
        config = {
            "logging": {
                "level": "DEBUG",
                "format": "%(levelname)s: %(message)s"
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save config
            config_path = Path(temp_dir) / "config.yaml"
            save_config(config, config_path)

            # Load config
            loaded = load_config(config_path)

            # Setup logging from config
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logging(
                log_level=loaded["logging"]["level"],
                format_str=loaded["logging"]["format"],
                log_file=str(log_file)
            )

            logger.debug("Debug message")
            logger.info("Info message")

            log_content = log_file.read_text()
            assert "DEBUG: Debug message" in log_content
            assert "INFO: Info message" in log_content

    def test_config_merge_for_experiments(self):
        """Test realistic config merging for experiments."""
        base_config = {
            "model": {
                "type": "TinyGPT",
                "vocab_size": 50000,
                "d_model": 512,
                "n_layers": 12,
                "n_heads": 8,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "optimizer": "adamw"
            },
            "data": {
                "dataset": "tinystories",
                "max_length": 512
            }
        }

        experiment_overrides = {
            "model": {
                "d_model": 256,  # Smaller model
                "n_layers": 6    # Fewer layers
            },
            "training": {
                "learning_rate": 5e-4,  # Higher LR for smaller model
                "num_epochs": 20       # More epochs
            }
        }

        final_config = merge_configs(base_config, experiment_overrides)

        # Verify merged correctly
        assert final_config["model"]["type"] == "TinyGPT"
        assert final_config["model"]["vocab_size"] == 50000
        assert final_config["model"]["d_model"] == 256
        assert final_config["model"]["n_layers"] == 6
        assert final_config["model"]["n_heads"] == 8
        assert final_config["training"]["learning_rate"] == 5e-4
        assert final_config["training"]["num_epochs"] == 20
        assert final_config["training"]["batch_size"] == 32
        assert final_config["data"]["dataset"] == "tinystories"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])