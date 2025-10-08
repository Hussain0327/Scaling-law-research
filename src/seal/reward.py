"""
Reward helpers for SEAL adaptation loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardMetrics:
    """Convenience container for reporting reward-related figures."""

    delta_val_loss: float
    val_loss_before: float
    val_loss_after: float
    train_loss_before: float
    train_loss_after: float
    forgetting: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "delta_val_loss": self.delta_val_loss,
            "val_loss_before": self.val_loss_before,
            "val_loss_after": self.val_loss_after,
            "train_loss_before": self.train_loss_before,
            "train_loss_after": self.train_loss_after,
            "forgetting": self.forgetting,
        }


def compute_delta_val_loss(before: float, after: float) -> float:
    """Positive numbers correspond to an improvement (lower loss after editing)."""

    return before - after


def compute_forgetting(train_before: float, train_after: float) -> float:
    """
    Basic forgetting metric used by the orchestration prompt.

    We measure the increase in training loss after a self-edit.  Positive values
    imply the model has forgotten some of the original training distribution.
    """

    return max(0.0, train_after - train_before)


def build_reward_metrics(
    val_before: float,
    val_after: float,
    train_before: float,
    train_after: float,
) -> RewardMetrics:
    """Bundle the different reward components into a single object."""

    delta = compute_delta_val_loss(val_before, val_after)
    forgetting = compute_forgetting(train_before, train_after)
    return RewardMetrics(
        delta_val_loss=delta,
        val_loss_before=val_before,
        val_loss_after=val_after,
        train_loss_before=train_before,
        train_loss_after=train_after,
        forgetting=forgetting,
    )
