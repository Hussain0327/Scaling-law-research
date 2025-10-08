"""
Minimal ReST (Reinforcement via Self-Training) coordinator.

The canonical SEAL setup performs a best-of-N selection over multiple edited
models.  We implement a lightweight ``RESTCoordinator`` that tracks the highest
reward candidate and exposes helpers for saving the champion's checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .reward import RewardMetrics


@dataclass(order=True)
class Candidate:
    reward: float
    identifier: str
    metrics: RewardMetrics = field(compare=False)
    checkpoint_path: Optional[Path] = field(default=None, compare=False)
    metadata: Dict[str, int | float | str] = field(default_factory=dict, compare=False)


class RESTCoordinator:
    """
    Track the best-performing adaptation (top-1) while streaming results to disk.
    """

    def __init__(self) -> None:
        self._best: Optional[Candidate] = None

    @property
    def best(self) -> Optional[Candidate]:
        return self._best

    def consider(self, candidate: Candidate) -> None:
        """
        Compare ``candidate`` against the current best entry and update whenever
        a larger reward is observed.
        """

        if self._best is None or candidate.reward > self._best.reward:
            self._best = candidate

    def summary(self) -> Dict[str, float | str]:
        """Short-form report suitable for logging."""

        if self._best is None:
            return {}

        payload: Dict[str, float | str] = {
            "identifier": self._best.identifier,
            "reward": self._best.reward,
            "delta_val_loss": self._best.metrics.delta_val_loss,
            "forgetting": self._best.metrics.forgetting,
        }
        payload.update(self._best.metadata)
        return payload
