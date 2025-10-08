"""
SEAL (Self-Editing with Assisted Learning) utilities.

This package provides a lightweight, repo-specific take on the SEAL workflow
described in the orchestration prompt.  The goal is to keep the implementation
simple enough to run on a local Apple Silicon machine while still providing the
hooks required by the automation scripts we add in this change-set.
"""

from .executor import AdaptationExecutor, AdaptationResult
from .rest_trainer import RESTCoordinator

__all__ = [
    "AdaptationExecutor",
    "AdaptationResult",
    "RESTCoordinator",
]
