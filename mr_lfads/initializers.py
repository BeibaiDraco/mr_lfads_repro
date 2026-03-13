"""Initializers adapted from the public lfads-torch repository provided by the user."""

from __future__ import annotations

import torch
from torch import nn


def init_variance_scaling_(weight: torch.Tensor, scale_dim: int) -> None:
    scale = torch.tensor(float(scale_dim), dtype=weight.dtype, device=weight.device)
    nn.init.normal_(weight, std=1.0 / torch.sqrt(scale))


def init_linear_(linear: nn.Linear) -> None:
    init_variance_scaling_(linear.weight, linear.in_features)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_gru_cell_(cell: nn.GRUCell, scale_dim: int | None = None) -> None:
    if scale_dim is None:
        ih_scale = cell.input_size
        hh_scale = cell.hidden_size
    else:
        ih_scale = scale_dim
        hh_scale = scale_dim
    init_variance_scaling_(cell.weight_ih, ih_scale)
    init_variance_scaling_(cell.weight_hh, hh_scale)
    nn.init.ones_(cell.bias_ih)
    cell.bias_ih.data[-cell.hidden_size :] = 0.0
    nn.init.zeros_(cell.bias_hh)
