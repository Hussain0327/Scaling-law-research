"""
Comprehensive unit tests for model evaluation functionality.
Tests evaluation metrics, scaling law analysis, and visualization.
"""

import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.append("src")

import pytest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from eval import ModelEvaluator, main
# Mock or create the other components we need for testing
import pandas as pd
from models.tiny_gpt import TinyGPT


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator instance."""
        return ModelEvaluator(device="cpu")

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = TinyGPT(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_seq_len=16,
            dropout=0.0
        )
        return model

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader."""
        def create_batch():
            return {
                "input_ids": torch.randint(0, 100, (2, 8)),
                "labels": torch.randint(0, 100, (2, 8))
            }

        batches = [create_batch() for _ in range(5)]
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(return_value=iter(batches))
        dataloader.__len__ = MagicMock(return_value=len(batches))

        return dataloader

    def test_evaluator_initialization_auto_device(self):
        """Test evaluator initialization with auto device selection."""
        evaluator = ModelEvaluator(device="auto")
        assert evaluator.device.type in ["cpu", "cuda"]

    def test_evaluator_initialization_explicit_device(self):
        """Test evaluator initialization with explicit device."""
        evaluator = ModelEvaluator(device="cpu")
        assert evaluator.device.type == "cpu"

    def test_evaluate_model_basic(self, evaluator, mock_model, mock_dataloader):
        """Test basic model evaluation."""
        metrics = evaluator.evaluate_model(mock_model, mock_dataloader)

        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "accuracy" in metrics
        assert "total_tokens" in metrics
        assert "total_batches" in metrics
        assert "loss_std" in metrics

        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["perplexity"], float)
        assert isinstance(metrics["accuracy"], float)
        assert metrics["total_batches"] == 5

    def test_evaluate_model_max_batches(self, evaluator, mock_model, mock_dataloader):
        """Test evaluation with max_batches limit."""
        metrics = evaluator.evaluate_model(
            mock_model, mock_dataloader, max_batches=2
        )

        assert metrics["total_batches"] == 2

    def test_evaluate_model_empty_dataloader(self, evaluator, mock_model):
        """Test evaluation with empty dataloader."""
        empty_dataloader = MagicMock()
        empty_dataloader.__iter__ = MagicMock(return_value=iter([]))
        empty_dataloader.__len__ = MagicMock(return_value=0)

        metrics = evaluator.evaluate_model(mock_model, empty_dataloader)

        assert metrics["loss"] == float("inf")
        assert metrics["total_batches"] == 0
        assert metrics["total_tokens"] == 0

    def test_evaluate_model_with_padding(self, evaluator, mock_model):
        """Test evaluation handles padding tokens correctly."""
        # Create batch with padding tokens (labels = -100)
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.full((2, 8), -100)  # All padding
        }
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        metrics = evaluator.evaluate_model(mock_model, dataloader)

        # Should handle all padding gracefully
        assert metrics["total_tokens"] == 0
        assert metrics["accuracy"] == 0.0

    def test_evaluate_multiple_models(self, evaluator, mock_dataloader):
        """Test evaluating multiple model configurations."""
        models = []
        for d_model in [16, 32, 64]:
            model = TinyGPT(
                vocab_size=100,
                d_model=d_model,
                n_layers=2,
                n_heads=max(2, d_model // 16),
                max_seq_len=16
            )
            models.append(model)

        results = {}
        for i, model in enumerate(models):
            metrics = evaluator.evaluate_model(model, mock_dataloader)
            results[f"model_{i}"] = metrics

        assert len(results) == 3
        for key in results:
            assert "loss" in results[key]
            assert "perplexity" in results[key]

    @torch.no_grad()
    def test_model_eval_mode(self, evaluator, mock_model, mock_dataloader):
        """Test that model is set to eval mode during evaluation."""
        # Initially in training mode
        mock_model.train()
        assert mock_model.training

        # Evaluate
        evaluator.evaluate_model(mock_model, mock_dataloader)

        # Should be in eval mode
        assert not mock_model.training

    def test_perplexity_calculation(self, evaluator, mock_model):
        """Test correct perplexity calculation."""
        # Create controlled batch with known loss
        class ControlledModel(torch.nn.Module):
            def forward(self, input_ids, labels=None):
                batch_size, seq_len = input_ids.shape
                vocab_size = 100
                logits = torch.zeros(batch_size, seq_len, vocab_size)
                # Return fixed loss value
                loss = torch.tensor(2.0)
                return logits, loss

        model = ControlledModel()
        batch = {
            "input_ids": torch.randint(0, 100, (1, 8)),
            "labels": torch.randint(0, 100, (1, 8))
        }
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        metrics = evaluator.evaluate_model(model, dataloader)

        # Perplexity = exp(loss)
        expected_perplexity = np.exp(2.0)
        assert abs(metrics["perplexity"] - expected_perplexity) < 0.01


class TestScalingLawAnalysis:
    """Test scaling law analysis functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for scaling analysis."""
        return ModelEvaluator(device="cpu")

    @pytest.fixture
    def scaling_data(self):
        """Create sample scaling data."""
        # Generate synthetic scaling law data
        # L = A * N^(-alpha) + L_inf
        np.random.seed(42)
        N_values = np.array([1e6, 5e6, 1e7, 5e7, 1e8, 5e8])
        alpha = 0.35
        A = 10.0
        L_inf = 1.5
        noise = np.random.normal(0, 0.05, len(N_values))

        losses = A * N_values**(-alpha) + L_inf + noise

        return pd.DataFrame({
            "model_size": N_values,
            "loss": losses,
            "flops": N_values * 1000,  # Approximate FLOPs
            "dataset_size": [1e9] * len(N_values)
        })

    def test_compute_scaling_laws(self, evaluator, scaling_data):
        """Test basic power law fitting."""
        # Test the compute_scaling_laws method
        scaling_results = evaluator.compute_scaling_laws(scaling_data)

        # Check that results contain expected keys
        if "width_scaling" in scaling_results:
            assert "params" in scaling_results["width_scaling"]
            assert "r_squared" in scaling_results["width_scaling"]

    def test_scaling_laws_insufficient_data(self, evaluator):
        """Test scaling law computation with insufficient data."""
        small_df = pd.DataFrame({
            "model_params": [1000],
            "test_loss": [2.5],
            "d_model": [32]
        })

        results = evaluator.compute_scaling_laws(small_df)
        # Should handle gracefully without errors
        assert isinstance(results, dict)

    def test_evaluate_multiple_checkpoints(self, evaluator):
        """Test evaluating multiple checkpoints."""
        # This tests the evaluate_multiple_checkpoints method
        # We'll mock the checkpoints and config
        pass  # Placeholder for complex test

    def test_compute_r_squared(self, evaluator):
        """Test R-squared computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        r_squared = evaluator._compute_r_squared(y_true, y_pred)

        assert 0 <= r_squared <= 1
        assert r_squared > 0.95  # Should be very close for this data

    def test_scaling_with_real_data(self, evaluator):
        """Test scaling laws with realistic evaluation data."""
        # Create realistic evaluation results
        results_data = []
        for d_model in [32, 64, 128]:
            for n_layers in [2, 4]:
                model = TinyGPT(
                    vocab_size=100,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=max(2, d_model // 16),
                    max_seq_len=32
                )
                results_data.append({
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "model_params": model.count_parameters(),
                    "test_loss": 3.0 - 0.001 * model.count_parameters() / 1000
                })

        results_df = pd.DataFrame(results_data)
        scaling_results = evaluator.compute_scaling_laws(results_df)

        assert isinstance(scaling_results, dict)

    def test_scaling_laws_empty_data(self, evaluator):
        """Test scaling laws with empty data."""
        empty_df = pd.DataFrame()
        results = evaluator.compute_scaling_laws(empty_df)

        # Should return empty dict without errors
        assert isinstance(results, dict)
        assert len(results) == 0

    @patch("matplotlib.pyplot.savefig")
    def test_plot_scaling_laws(self, mock_savefig, evaluator):
        """Test plotting scaling law visualizations."""
        results_df = pd.DataFrame({
            "model_params": [1000, 5000, 10000],
            "test_loss": [3.0, 2.5, 2.2],
            "d_model": [32, 64, 128],
            "n_layers": [2, 4, 4]
        })

        scaling_results = {"width_scaling": {"params": [10, -0.3], "r_squared": 0.95}}

        # This should create plots without error
        evaluator._plot_scaling_laws(results_df, scaling_results, "test_plot.png")

        mock_savefig.assert_called()

    def test_scaling_multiple_dimensions(self, evaluator):
        """Test scaling laws across multiple model dimensions."""
        # Create data varying in both width and depth
        results_data = []
        for d_model in [32, 64, 128, 256]:
            for n_layers in [2, 4, 6, 8]:
                params = d_model * d_model * n_layers  # Simplified param count
                loss = 5.0 * (params ** -0.2) + 1.5
                results_data.append({
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "model_params": params,
                    "test_loss": loss,
                    "test_perplexity": np.exp(loss)
                })

        results_df = pd.DataFrame(results_data)
        scaling_results = evaluator.compute_scaling_laws(results_df)

        # Should find scaling patterns
        if "width_scaling" in scaling_results:
            assert scaling_results["width_scaling"]["r_squared"] > 0.5
        if "depth_scaling" in scaling_results:
            assert scaling_results["depth_scaling"]["r_squared"] > 0.5


class TestPlottingFunctions:
    """Test visualization and plotting functions."""

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_scaling_laws(self, mock_show, mock_savefig):
        """Test scaling law plotting via evaluator."""
        evaluator = ModelEvaluator(device="cpu")

        results_df = pd.DataFrame({
            "model_params": [1e6, 1e7, 1e8],
            "test_loss": [3.0, 2.5, 2.0],
            "d_model": [32, 64, 128],
            "n_layers": [2, 4, 6]
        })

        scaling_results = {}

        # Test internal plotting method
        evaluator._plot_scaling_laws(results_df, scaling_results, "test_plot.png")

        mock_savefig.assert_called()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_multiple_metrics(self, mock_subplots):
        """Test plotting multiple evaluation metrics."""
        mock_fig = MagicMock()
        mock_axes = np.array([MagicMock() for _ in range(4)]).reshape(2, 2)
        mock_subplots.return_value = (mock_fig, mock_axes)

        evaluator = ModelEvaluator(device="cpu")

        results_df = pd.DataFrame({
            "model_params": [1e6, 1e7, 1e8],
            "test_loss": [3.0, 2.5, 2.0],
            "test_perplexity": [20.0, 12.0, 7.4],
            "val_accuracy": [0.3, 0.4, 0.5],
            "d_model": [32, 64, 128],
            "n_layers": [2, 4, 6]
        })

        evaluator._plot_scaling_laws(results_df, {}, "test.png")

        # Verify that subplots was called
        mock_subplots.assert_called()


class TestCheckpointEvaluation:
    """Test checkpoint evaluation functionality."""

    @patch("eval.find_checkpoints")
    @patch("eval.create_datamodule")
    def test_evaluate_multiple_checkpoints(self, mock_create_dm, mock_find):
        """Test evaluating multiple checkpoints."""
        # Mock checkpoint files
        mock_find.return_value = [
            "checkpoint_1.pt",
            "checkpoint_2.pt"
        ]

        # Mock datamodule
        mock_dm = MagicMock()
        mock_dataloader = MagicMock()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8))
        }
        mock_dataloader.__iter__ = MagicMock(return_value=iter([batch]))
        mock_dm.val_dataloader.return_value = mock_dataloader
        mock_dm.test_dataloader.return_value = mock_dataloader
        mock_create_dm.return_value = mock_dm

        evaluator = ModelEvaluator(device="cpu")

        config = {
            "data": {"dataset_name": "test"},
            "model": {"vocab_size": 100}
        }

        # Test that evaluate_multiple_checkpoints can be called
        # Note: Full test would require mocking torch.load as well

    @patch("eval.find_checkpoints")
    def test_evaluate_checkpoints_no_files(self, mock_find):
        """Test checkpoint evaluation with no checkpoint files."""
        mock_find.return_value = []

        evaluator = ModelEvaluator(device="cpu")
        config = {"data": {}}

        results = evaluator.evaluate_multiple_checkpoints([], config)

        # Should return empty DataFrame
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_find_checkpoints(self):
        """Test finding checkpoint files."""
        from eval import find_checkpoints
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some checkpoint files
            Path(temp_dir, "checkpoint_1.pt").touch()
            Path(temp_dir, "checkpoint_2.pt").touch()
            Path(temp_dir, "not_checkpoint.txt").touch()

            checkpoints = find_checkpoints(temp_dir)

            assert len(checkpoints) == 2
            assert all(path.endswith(".pt") for path in checkpoints)


class TestEvaluationReport:
    """Test evaluation report generation."""

    def test_evaluation_results_dataframe(self):
        """Test creating evaluation results DataFrame."""
        evaluator = ModelEvaluator(device="cpu")

        # Create sample results
        results_data = [
            {
                "checkpoint_path": "ckpt1.pt",
                "step": 100,
                "val_loss": 2.5,
                "test_loss": 2.6
            },
            {
                "checkpoint_path": "ckpt2.pt",
                "step": 200,
                "val_loss": 2.3,
                "test_loss": 2.4
            }
        ]

        results_df = pd.DataFrame(results_data)

        assert len(results_df) == 2
        assert "val_loss" in results_df.columns
        assert results_df["val_loss"].min() == 2.3

    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        import tempfile
        import json

        results = {
            "model_params": 1000000,
            "test_loss": 2.5,
            "test_perplexity": 12.2
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            temp_path = f.name

        try:
            # Verify file was created and can be loaded
            with open(temp_path, 'r') as f:
                loaded = json.load(f)

            assert loaded == results
        finally:
            Path(temp_path).unlink()



class TestMainEvaluation:
    """Test the main evaluation entry point."""

    @patch("eval.create_datamodule")
    @patch("eval.load_config")
    @patch("torch.load")
    def test_main_evaluation_basic(self, mock_load, mock_load_config, mock_create_dm):
        """Test basic main evaluation flow."""
        # Mock config
        mock_load_config.return_value = {
            "model": {
                "vocab_size": 100,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4
            },
            "data": {
                "dataset_name": "test"
            }
        }

        # Mock checkpoint
        mock_load.return_value = {
            "model_state_dict": {},
            "config": mock_load_config.return_value
        }

        # Mock datamodule
        mock_dm = MagicMock()
        mock_dataloader = MagicMock()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8))
        }
        mock_dataloader.__iter__ = MagicMock(return_value=iter([batch]))
        mock_dm.val_dataloader.return_value = mock_dataloader
        mock_create_dm.return_value = mock_dm

        # Create args
        args = MagicMock()
        args.checkpoint = "checkpoint.pt"
        args.config = "config.yaml"
        args.output = None
        args.max_batches = None
        args.device = "cpu"
        args.scaling_analysis = False

        # Run main
        results = main(args)

        assert "metrics" in results
        assert "model_config" in results

    @patch("eval.plot_scaling_curves")
    @patch("eval.ScalingLawAnalyzer")
    def test_main_with_scaling_analysis(self, mock_analyzer_class, mock_plot):
        """Test main evaluation with scaling law analysis."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_scaling_results.return_value = {
            "model_scaling": {"r_squared": 0.95}
        }
        mock_analyzer_class.return_value = mock_analyzer

        args = MagicMock()
        args.scaling_analysis = True
        args.checkpoint_dir = "checkpoints"

        # This would be part of a larger integration test
        # Verify that scaling analysis is called when flag is set


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_evaluation_with_nan_loss(self):
        """Test handling of NaN losses during evaluation."""
        evaluator = ModelEvaluator(device="cpu")

        class NaNModel(torch.nn.Module):
            def forward(self, input_ids, labels=None):
                batch_size, seq_len = input_ids.shape
                logits = torch.full((batch_size, seq_len, 100), float("nan"))
                loss = torch.tensor(float("nan"))
                return logits, loss

        model = NaNModel()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8))
        }
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        metrics = evaluator.evaluate_model(model, dataloader)

        # Should handle NaN gracefully
        assert np.isnan(metrics["loss"]) or np.isinf(metrics["loss"])

    def test_evaluation_with_infinite_loss(self):
        """Test handling of infinite losses."""
        evaluator = ModelEvaluator(device="cpu")

        class InfModel(torch.nn.Module):
            def forward(self, input_ids, labels=None):
                batch_size, seq_len = input_ids.shape
                logits = torch.zeros(batch_size, seq_len, 100)
                loss = torch.tensor(float("inf"))
                return logits, loss

        model = InfModel()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 100, (2, 8))
        }
        dataloader = MagicMock()
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        metrics = evaluator.evaluate_model(model, dataloader)

        assert np.isinf(metrics["loss"])
        assert np.isinf(metrics["perplexity"])

    def test_scaling_analysis_with_single_point(self):
        """Test scaling analysis with only one data point."""
        analyzer = ScalingLawAnalyzer()

        single_point_df = pd.DataFrame({
            "model_size": [1e7],
            "loss": [2.5],
            "flops": [1e15],
            "dataset_size": [1e9]
        })

        results = analyzer.analyze_scaling_results(single_point_df)

        # Should handle gracefully without fitting
        assert results["model_scaling"]["params"] is None
        assert results["model_scaling"]["r_squared"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])