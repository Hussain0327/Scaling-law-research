"""
Parameter-efficient fine-tuning utilities (LoRA) for TinyGPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


def _lora_forward(module: nn.Linear, input_: torch.Tensor) -> torch.Tensor:
    """Replacement forward pass that augments the frozen weight with a LoRA term."""

    result = module._lora_original_forward(input_)  # type: ignore[attr-defined]
    lora_a = module.lora_A  # type: ignore[attr-defined]
    lora_b = module.lora_B  # type: ignore[attr-defined]
    scaling = module.lora_scaling  # type: ignore[attr-defined]

    update = lora_b(lora_a(input_))
    return result + scaling * update


def _inject_lora(module: nn.Linear, rank: int, alpha: float) -> None:
    """Attach LoRA adaptation parameters to ``module`` in-place."""

    if hasattr(module, "lora_A"):
        return

    in_features = module.in_features
    out_features = module.out_features

    module.weight.requires_grad_(False)
    if module.bias is not None:
        module.bias.requires_grad_(False)

    lora_a = nn.Linear(in_features, rank, bias=False)
    lora_b = nn.Linear(rank, out_features, bias=False)

    nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
    nn.init.zeros_(lora_b.weight)

    module.register_module("lora_A", lora_a)
    module.register_module("lora_B", lora_b)
    module.lora_scaling = alpha / max(1, rank)  # type: ignore[attr-defined]
    module._lora_original_forward = module.forward  # type: ignore[attr-defined]
    module.forward = lambda input_, module=module: _lora_forward(module, input_)  # type: ignore[method-assign]


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection."""

    rank: int = 8
    alpha: float = 16.0
    target_modules: Sequence[str] = (
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
        "lm_head",
    )


class LoRAAdapter:
    """
    Manage LoRA layers injected into a ``TinyGPT`` model.
    """

    def __init__(self, model: nn.Module, config: LoRAConfig) -> None:
        self.model = model
        self.config = config
        self._inject()

    @property
    def parameters(self) -> Iterable[nn.Parameter]:
        """
        Return an iterator over trainable LoRA parameters.
        """

        for module in self.model.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "lora_A"):
                yield from module.lora_A.parameters()  # type: ignore[attr-defined]
                yield from module.lora_B.parameters()  # type: ignore[attr-defined]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters)

    def reset(self) -> None:
        """Zero all LoRA parameters."""

        for module in self.model.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "lora_A"):
                nn.init.zeros_(module.lora_B.weight)  # type: ignore[attr-defined]

    def _inject(self) -> None:
        target_suffixes = tuple(self.config.target_modules)

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not name.endswith(target_suffixes):
                continue

            _inject_lora(module, rank=self.config.rank, alpha=self.config.alpha)


def collect_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Convenience helper mirroring :meth:`LoRAAdapter.parameters`."""

    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora_A"):
            params.extend(module.lora_A.parameters())  # type: ignore[attr-defined]
            params.extend(module.lora_B.parameters())  # type: ignore[attr-defined]
    return params
