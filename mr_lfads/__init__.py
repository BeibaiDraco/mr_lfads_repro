"""Faithful PyTorch reimplementation of MR-LFADS.

This package is a paper-faithful reimplementation of
"Accurate Identification of Communication Between Multiple Interacting Neural Populations"
(Belle Liu, Jacob Sacks, Matthew D. Golub, ICML 2025).

It is built from the public LFADS-Torch codebase the user supplied, but rewritten as a
standalone multi-region package so the MR-LFADS architecture and training schedule are
explicit and easy to modify.
"""

from .model import MRLFADS
from .train import TrainConfig, train_mr_lfads

__all__ = ["MRLFADS", "TrainConfig", "train_mr_lfads"]
