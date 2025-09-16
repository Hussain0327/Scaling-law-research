"""
Training script for TinyGPT with scaling law experiments.
Supports various model sizes and configurations for research.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import datamodule as data_datamodule
from models.tiny_gpt import TinyGPT


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)


class Trainer:
    """Training class for TinyGPT models."""

    def __init__(
        self,
        model: TinyGPT,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any],
        save_dir: str = "checkpoints",
        use_wandb: bool = True,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb

        # Training configuration
        self.num_epochs = config["training"]["num_epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.warmup_steps = config["training"]["warmup_steps"]
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)
        self.eval_interval = config["training"].get("eval_interval", 500)
        self.save_interval = config["training"].get("save_interval", 1000)
        self.log_interval = config["training"].get("log_interval", 100)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * self.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.1
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        # Mixed precision setup
        self.use_amp = (
            config["training"].get("use_amp", True) and torch.cuda.is_available()
        )
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            wandb.init(
                project="tiny-lm-scaling",
                config=self.config,
                name=(
                    f"tinygpt_{self.config['model']['d_model']}d"
                    f"_{self.config['model']['n_layers']}l"
                ),
            )
            wandb.watch(self.model, log="all", log_freq=1000)
        except ImportError:
            print("Warning: wandb not available. Logging disabled.")
            self.use_wandb = False

    def warmup_lr(self, step: int) -> float:
        """Compute learning rate with warmup."""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        return self.learning_rate

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits, loss = self.model(input_ids, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, loss = self.model(input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Update learning rate with warmup
        if self.step < self.warmup_steps:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.warmup_lr(self.step)
        else:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, loss = self.model(input_ids, labels)
            else:
                logits, loss = self.model(input_ids, labels)

            # Count non-padded tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        if total_tokens == 0:
            return {"val_loss": float("inf"), "val_perplexity": float("inf")}

        avg_loss = total_loss / total_tokens
        avg_perplexity = float(np.exp(avg_loss))

        return {"val_loss": avg_loss, "val_perplexity": avg_perplexity}

    def save_checkpoint(self, filepath: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if filepath is None:
            filepath = self.save_dir / f"checkpoint_step_{self.step}.pt"

        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Checkpoint loaded from {filepath}")
        return checkpoint

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")

        start_time = time.time()
        running_loss = 0.0

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training loop
            for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="Training")
            ):
                loss = self.train_step(batch)
                running_loss += loss
                self.step += 1

                # Logging
                if self.step % self.log_interval == 0:
                    avg_loss = running_loss / self.log_interval
                    self.train_losses.append(avg_loss)

                    lr = self.optimizer.param_groups[0]["lr"]
                    tokens_per_sec = (
                        (self.log_interval * batch["input_ids"].numel())
                        / (time.time() - start_time)
                        if self.step > self.log_interval
                        else 0
                    )

                    log_data = {
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "tokens_per_sec": tokens_per_sec,
                        "step": self.step,
                        "epoch": epoch,
                    }

                    if self.use_wandb:
                        try:
                            import wandb

                            wandb.log(log_data)
                        except ImportError:
                            pass

                    print(
                        f"Step {self.step}: loss={avg_loss:.4f}, lr={lr:.2e}, "
                        f"tokens/s={tokens_per_sec:.1f}"
                    )
                    running_loss = 0.0
                    start_time = time.time()

                # Evaluation
                if self.step % self.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.val_losses.append(eval_metrics["val_loss"])

                    print(
                        f"Validation - Loss: {eval_metrics['val_loss']:.4f}, "
                        f"Perplexity: {eval_metrics['val_perplexity']:.2f}"
                    )

                    if self.use_wandb:
                        try:
                            import wandb

                            wandb.log(eval_metrics)
                        except ImportError:
                            pass

                    # Save best model
                    if eval_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["val_loss"]
                        self.save_checkpoint(is_best=True)

                # Save checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()

        # Final evaluation
        final_metrics = self.evaluate()
        print(
            f"\nFinal validation - Loss: {final_metrics['val_loss']:.4f}, "
            f"Perplexity: {final_metrics['val_perplexity']:.2f}"
        )

        # Save final model
        self.save_checkpoint(self.save_dir / "final_model.pt")

        return {
            "final_val_loss": final_metrics["val_loss"],
            "final_val_perplexity": final_metrics["val_perplexity"],
            "best_val_loss": self.best_val_loss,
            "total_steps": self.step,
            "model_parameters": self.model.count_parameters(),
        }


def setup_experiment(config: Dict[str, Any]) -> Tuple[TinyGPT, DataLoader, DataLoader]:
    """Set up model and data for experiment."""
    # Create model
    model = TinyGPT(**config["model"])
    print(f"Created model with {model.count_parameters():,} parameters")

    # Create data module
    data_module = data_datamodule.create_datamodule(**config["data"])
    data_module.prepare_data()
    data_module.setup_tokenizer()
    data_module.setup_datasets()

    # Update vocab size in model if using character tokenizer
    if hasattr(data_module.tokenizer, "vocab_size"):
        if model.vocab_size != data_module.tokenizer.vocab_size:
            print(
                f"Updating model vocab size from {model.vocab_size} to "
                f"{data_module.tokenizer.vocab_size}"
            )
            # Recreate model with correct vocab size
            config["model"]["vocab_size"] = data_module.tokenizer.vocab_size
            model = TinyGPT(**config["model"])

    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Print dataset statistics
    train_stats = data_datamodule.get_dataset_stats(train_loader)
    print(f"Training set: {train_stats}")

    return model, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train TinyGPT model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set up experiment
    model, train_loader, val_loader = setup_experiment(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        save_dir=args.save_dir,
        use_wandb=not args.no_wandb,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train model
    results = trainer.train()

    # Save results
    results_path = Path(args.save_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Training completed! Results saved to {results_path}")

    if not args.no_wandb:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
