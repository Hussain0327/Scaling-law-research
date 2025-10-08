"""
Execution helpers for SEAL self-adaptation experiments.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import json
import yaml

import torch
from torch.utils.data import DataLoader

from ..data import datamodule as data_datamodule
from ..models.tiny_gpt import TinyGPT
from .edit_lang import LoRAAdapter, LoRAConfig
from .reward import RewardMetrics, build_reward_metrics


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cycle(dataloader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield batches indefinitely."""

    while True:
        for batch in dataloader:
            yield batch


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_checkpoint(ckpt_dir: Path) -> Dict:
    """
    Load a TinyGPT checkpoint, preferring ``best_model.pt`` followed by
    ``final_model.pt`` and lastly any ``checkpoint_step_*.pt`` file.
    """

    if ckpt_dir.is_file():
        return torch.load(ckpt_dir, map_location="cpu")

    for candidate in ("best_model.pt", "final_model.pt"):
        ckpt_path = ckpt_dir / candidate
        if ckpt_path.exists():
            return torch.load(ckpt_path, map_location="cpu")

    checkpoints = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No TinyGPT checkpoints found in {ckpt_dir}")

    return torch.load(checkpoints[-1], map_location="cpu")


def _evaluate_model(
    model: TinyGPT,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> Tuple[float, float]:
    """
    Return token-weighted loss and perplexity over ``dataloader``.
    """

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels)

            valid_tokens = (labels != -100).sum().item()
            if valid_tokens <= 0:
                continue

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            batches += 1

            if max_batches is not None and batches >= max_batches:
                break

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    return avg_loss, perplexity


def _compute_tokens(batch: Dict[str, torch.Tensor]) -> int:
    labels = batch["labels"]
    valid = (labels != -100).sum().item()
    return max(valid, 0)


@dataclass
class AdaptationResult:
    identifier: str
    inner_steps: int
    lora_rank: int
    metrics: RewardMetrics
    checkpoint_path: Path
    tokens_processed: int
    wall_clock_steps: int

    def to_json(self) -> str:
        payload = {
            "identifier": self.identifier,
            "inner_steps": self.inner_steps,
            "lora_rank": self.lora_rank,
            "checkpoint_path": str(self.checkpoint_path),
            "tokens_processed": self.tokens_processed,
            "wall_clock_steps": self.wall_clock_steps,
        }
        payload.update(self.metrics.to_dict())
        return json.dumps(payload)


class AdaptationExecutor:
    """
    Run LoRA-based self-adaptation experiments on TinyGPT.
    """

    def __init__(
        self,
        config_path: Path,
        baseline_checkpoint: Path,
        save_root: Path,
        device: Optional[torch.device] = None,
        eval_subset: Optional[int] = 200,
        learning_rate: float = 5e-4,
    ) -> None:
        self.config = self._load_training_config(config_path)
        self.baseline_checkpoint = baseline_checkpoint
        self.save_root = save_root
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.device = device or _default_device()
        self.eval_subset = eval_subset
        self.learning_rate = learning_rate

        data_cfg = dict(self.config["data"])
        data_cfg.setdefault("dataset_name", "tinystories")
        self.data_module = data_datamodule.create_datamodule(**data_cfg)
        self.data_module.prepare_data()
        self.data_module.setup_tokenizer()
        self.data_module.setup_datasets()

        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()

        checkpoint = _load_checkpoint(baseline_checkpoint)
        self._baseline_state = checkpoint["model_state_dict"]

    def run_budget(self, inner_steps: int, lora_rank: int) -> AdaptationResult:
        model = TinyGPT(**self.config["model"])
        model.load_state_dict(self._baseline_state, strict=True)
        model.to(self.device)

        lora_cfg = LoRAConfig(rank=lora_rank)
        adapter = LoRAAdapter(model, lora_cfg)

        # Pre-edit evaluation
        val_loss_before, _ = _evaluate_model(
            model, self.val_loader, self.device, self.eval_subset
        )
        train_loss_before, _ = _evaluate_model(
            model, self.train_loader, self.device, self.eval_subset
        )

        params = list(adapter.parameters)
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        model.train()
        tokens_processed = 0

        data_iter = _cycle(self.train_loader)

        for step in range(inner_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            tokens_processed += _compute_tokens(batch)

        # Post-edit evaluation
        val_loss_after, _ = _evaluate_model(
            model, self.val_loader, self.device, self.eval_subset
        )
        train_loss_after, _ = _evaluate_model(
            model, self.train_loader, self.device, self.eval_subset
        )

        metrics = build_reward_metrics(
            val_before=val_loss_before,
            val_after=val_loss_after,
            train_before=train_loss_before,
            train_after=train_loss_after,
        )

        identifier = f"inner{inner_steps}_rank{lora_rank}"
        checkpoint_dir = self.save_root / identifier
        _ensure_dir(checkpoint_dir)
        ckpt_path = checkpoint_dir / "seal_adapter.pt"
        torch.save(
            {
                "config": self.config,
                "inner_steps": inner_steps,
                "lora_rank": lora_rank,
                "tokens_processed": tokens_processed,
                "model_state_dict": model.state_dict(),
            },
            ckpt_path,
        )

        return AdaptationResult(
            identifier=identifier,
            inner_steps=inner_steps,
            lora_rank=lora_rank,
            metrics=metrics,
            checkpoint_path=ckpt_path,
            tokens_processed=tokens_processed,
            wall_clock_steps=inner_steps,
        )

    @staticmethod
    def _load_training_config(path: Path) -> Dict:
        """
        Load a training configuration from YAML or JSON without depending on
        :mod:`src.train` (which in turn expects specific import semantics).
        """

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(text)
        return json.loads(text)


def save_results_jsonl(results: Iterable[AdaptationResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.to_json() + "\n")
