"""
Evaluation script for TinyGPT models.
Includes metrics computation, scaling law analysis, and curve plotting.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import curve_fit
from tqdm import tqdm

from data.datamodule import create_datamodule
from models.tiny_gpt import TinyGPT


class ModelEvaluator:
    """Comprehensive model evaluation with scaling law analysis."""

    def __init__(self, device: str = "auto"):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def evaluate_model(
        self, model: TinyGPT, dataloader, max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model on a dataset.

        Args:
            model: The model to evaluate
            dataloader: DataLoader for evaluation
            max_batches: Maximum number of batches to evaluate (for faster evaluation)

        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        model.to(self.device)

        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        batch_losses = []

        batches_processed = 0
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if max_batches and batches_processed >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, loss = model(input_ids, labels)

            # Accumulate loss (weighted by number of valid tokens)
            valid_mask = labels != -100
            valid_tokens = valid_mask.sum().item()

            if valid_tokens > 0:
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
                batch_losses.append(loss.item())

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct = ((predictions == labels) & valid_mask).sum().item()
                correct_predictions += correct
                total_predictions += valid_tokens

            batches_processed += 1

        # Compute metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = np.exp(avg_loss)
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )
        loss_std = np.std(batch_losses) if batch_losses else 0.0

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "total_tokens": total_tokens,
            "total_batches": batches_processed,
            "loss_std": loss_std,
        }

    def evaluate_multiple_checkpoints(
        self,
        checkpoint_paths: List[str],
        config: Dict[str, Any],
        max_batches: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate multiple model checkpoints.

        Args:
            checkpoint_paths: List of paths to model checkpoints
            config: Configuration dictionary for data setup
            max_batches: Maximum number of batches per evaluation

        Returns:
            DataFrame with evaluation results
        """
        # Set up data
        data_module = create_datamodule(**config["data"])
        data_module.prepare_data()
        data_module.setup_tokenizer()
        data_module.setup_datasets()

        val_loader = data_module.val_dataloader()
        test_loader = None

        if hasattr(data_module, "test_dataloader"):
            try:
                test_candidate = data_module.test_dataloader()
            except TypeError:
                test_candidate = None
            test_loader = test_candidate

        if test_loader is None:
            test_loader = data_module.val_dataloader()

        results = []

        for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model_config = checkpoint["config"]["model"]

                # Create and load model
                model = TinyGPT(**model_config)
                model.load_state_dict(checkpoint["model_state_dict"])

                # Evaluate on validation and test sets
                val_metrics = self.evaluate_model(model, val_loader, max_batches)
                test_metrics = self.evaluate_model(model, test_loader, max_batches)

                # Collect results
                result = {
                    "checkpoint_path": checkpoint_path,
                    "step": checkpoint.get("step", 0),
                    "epoch": checkpoint.get("epoch", 0),
                    "model_params": model.count_parameters(),
                    "d_model": model_config["d_model"],
                    "n_layers": model_config["n_layers"],
                    "n_heads": model_config["n_heads"],
                    "val_loss": val_metrics["loss"],
                    "val_perplexity": val_metrics["perplexity"],
                    "val_accuracy": val_metrics["accuracy"],
                    "test_loss": test_metrics["loss"],
                    "test_perplexity": test_metrics["perplexity"],
                    "test_accuracy": test_metrics["accuracy"],
                }

                results.append(result)
                print(
                    f"Evaluated {checkpoint_path}: val_loss={val_metrics['loss']:.4f}, "
                    f"test_loss={test_metrics['loss']:.4f}"
                )

            except Exception as e:
                print(f"Error evaluating {checkpoint_path}: {e}")
                continue

        return pd.DataFrame(results)

    def compute_scaling_laws(
        self, results_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute scaling law relationships from evaluation results.

        Args:
            results_df: DataFrame with evaluation results
            save_path: Optional path to save scaling law plots

        Returns:
            Dictionary with scaling law parameters and plots
        """

        def power_law(x, a, b):
            """Power law function: y = a * x^b"""
            return a * np.power(x, b)

        scaling_results = {}

        # Group by different scaling dimensions
        if "d_model" in results_df.columns and len(results_df["d_model"].unique()) > 1:
            # Scaling with model width
            width_data = (
                results_df.groupby("d_model")
                .agg(
                    {
                        "test_loss": "mean",
                        "test_perplexity": "mean",
                        "model_params": "first",
                    }
                )
                .reset_index()
            )

            try:
                # Fit power law: loss = a * params^b
                popt, pcov = curve_fit(
                    power_law,
                    width_data["model_params"],
                    width_data["test_loss"],
                    p0=[1.0, -0.1],
                )
                scaling_results["width_scaling"] = {
                    "params": popt.tolist(),
                    "r_squared": self._compute_r_squared(
                        width_data["test_loss"],
                        power_law(width_data["model_params"], *popt),
                    ),
                }
            except Exception as e:
                print(f"Error fitting width scaling law: {e}")

        # Scaling with number of layers
        if (
            "n_layers" in results_df.columns
            and len(results_df["n_layers"].unique()) > 1
        ):
            layer_data = (
                results_df.groupby("n_layers")
                .agg(
                    {
                        "test_loss": "mean",
                        "test_perplexity": "mean",
                        "model_params": "first",
                    }
                )
                .reset_index()
            )

            try:
                popt, pcov = curve_fit(
                    power_law,
                    layer_data["model_params"],
                    layer_data["test_loss"],
                    p0=[1.0, -0.1],
                )
                scaling_results["depth_scaling"] = {
                    "params": popt.tolist(),
                    "r_squared": self._compute_r_squared(
                        layer_data["test_loss"],
                        power_law(layer_data["model_params"], *popt),
                    ),
                }
            except Exception as e:
                print(f"Error fitting depth scaling law: {e}")

        # Create plots if save path provided
        if save_path:
            self._plot_scaling_laws(results_df, scaling_results, save_path)

        return scaling_results

    def _compute_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared coefficient."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def _plot_scaling_laws(
        self, results_df: pd.DataFrame, scaling_results: Dict[str, Any], save_path: str
    ):
        """Create and save scaling law plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("TinyGPT Scaling Laws", fontsize=16)

        # Plot 1: Loss vs Model Parameters
        if not results_df.empty:
            ax = axes[0, 0]
            scatter = ax.scatter(
                results_df["model_params"],
                results_df["test_loss"],
                c=results_df["d_model"],
                cmap="viridis",
                alpha=0.7,
            )
            ax.set_xlabel("Model Parameters")
            ax.set_ylabel("Test Loss")
            ax.set_title("Test Loss vs Model Parameters")
            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.colorbar(scatter, ax=ax, label="d_model")

            # Add power law fit if available
            if "width_scaling" in scaling_results:
                params = scaling_results["width_scaling"]["params"]
                x_fit = np.logspace(
                    np.log10(results_df["model_params"].min()),
                    np.log10(results_df["model_params"].max()),
                    100,
                )
                y_fit = params[0] * np.power(x_fit, params[1])
                ax.plot(
                    x_fit,
                    y_fit,
                    "r--",
                    alpha=0.8,
                    label=f"Power law: {params[0]:.3f} * N^{params[1]:.3f}",
                )
                ax.legend()

        # Plot 2: Perplexity vs Model Parameters
        ax = axes[0, 1]
        if not results_df.empty:
            ax.scatter(
                results_df["model_params"],
                results_df["test_perplexity"],
                c=results_df["n_layers"],
                cmap="plasma",
                alpha=0.7,
            )
            ax.set_xlabel("Model Parameters")
            ax.set_ylabel("Test Perplexity")
            ax.set_title("Test Perplexity vs Model Parameters")
            ax.set_xscale("log")

        # Plot 3: Loss vs Width (d_model)
        ax = axes[1, 0]
        if "d_model" in results_df.columns:
            width_means = (
                results_df.groupby("d_model")["test_loss"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                width_means["d_model"],
                width_means["mean"],
                yerr=width_means["std"],
                marker="o",
                capsize=5,
            )
            ax.set_xlabel("Model Width (d_model)")
            ax.set_ylabel("Test Loss")
            ax.set_title("Test Loss vs Model Width")

        # Plot 4: Loss vs Depth (n_layers)
        ax = axes[1, 1]
        if "n_layers" in results_df.columns:
            depth_means = (
                results_df.groupby("n_layers")["test_loss"]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                depth_means["n_layers"],
                depth_means["mean"],
                yerr=depth_means["std"],
                marker="s",
                capsize=5,
            )
            ax.set_xlabel("Model Depth (n_layers)")
            ax.set_ylabel("Test Loss")
            ax.set_title("Test Loss vs Model Depth")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Scaling law plots saved to {save_path}")

    def generate_text_samples(
        self,
        model: TinyGPT,
        tokenizer,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        num_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate text samples from the model."""
        model.eval()
        model.to(self.device)

        samples = []

        for prompt in prompts:
            prompt_samples = []

            for _ in range(num_samples):
                # Encode prompt
                input_ids = torch.tensor(
                    tokenizer.encode(prompt, add_special_tokens=False),
                    device=self.device,
                ).unsqueeze(0)

                # Generate
                with torch.no_grad():
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                    )

                # Decode
                generated_text = tokenizer.decode(
                    generated[0], skip_special_tokens=True
                )
                prompt_samples.append(generated_text)

            samples.append({"prompt": prompt, "samples": prompt_samples})

        return samples


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def find_checkpoints(checkpoint_dir: str, pattern: str = "*.pt") -> List[str]:
    """Find all checkpoint files in a directory."""
    checkpoint_path = Path(checkpoint_dir)
    return sorted([str(p) for p in checkpoint_path.glob(pattern)])


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyGPT models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--max_batches", type=int, default=None, help="Maximum batches for evaluation"
    )
    parser.add_argument(
        "--generate_samples", action="store_true", help="Generate text samples"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Find checkpoints
    checkpoint_paths = find_checkpoints(args.checkpoint_dir)
    if not checkpoint_paths:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints")

    # Evaluate models
    results_df = evaluator.evaluate_multiple_checkpoints(
        checkpoint_paths, config, args.max_batches
    )

    if results_df.empty:
        print("No successful evaluations")
        return

    # Save results
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")

    # Compute scaling laws
    scaling_results = evaluator.compute_scaling_laws(
        results_df, save_path=str(output_dir / "scaling_laws.png")
    )

    # Save scaling law results
    scaling_path = output_dir / "scaling_laws.json"
    with open(scaling_path, "w") as f:
        json.dump(scaling_results, f, indent=2)
    print(f"Scaling law results saved to {scaling_path}")

    # Generate text samples if requested
    if args.generate_samples:
        print("Generating text samples...")

        # Load best model
        best_model_path = results_df.loc[
            results_df["test_loss"].idxmin(), "checkpoint_path"
        ]
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model = TinyGPT(**checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model_state_dict"])

        # Set up tokenizer
        data_module = create_datamodule(**config["data"])
        data_module.prepare_data()
        data_module.setup_tokenizer()

        # Sample prompts
        prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight",
            "On a sunny day",
        ]

        samples = evaluator.generate_text_samples(model, data_module.tokenizer, prompts)

        # Save samples
        samples_path = output_dir / "text_samples.json"
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Text samples saved to {samples_path}")

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
