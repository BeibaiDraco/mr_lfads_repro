"""Recurrent building blocks adapted from the public lfads-torch repository."""

from __future__ import annotations

import torch
from torch import nn

from .initializers import init_gru_cell_


class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ) -> None:
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        scale_dim = input_size + hidden_size if is_encoder else None
        init_gru_cell_(self, scale_dim=scale_dim)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1.0 - z) * n
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden


class ClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ClippedGRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            clip_value=clip_value,
            is_encoder=True,
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = h_0
        input_t = torch.transpose(input, 0, 1)
        output = []
        for step in input_t:
            hidden = self.cell(step, hidden)
            output.append(hidden)
        return torch.stack(output, dim=1), hidden


class BidirectionalClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ) -> None:
        super().__init__()
        self.fwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0_fwd, h0_bwd = h_0
        output_fwd, hn_fwd = self.fwd_gru(input, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(torch.flip(input, dims=[1]), h0_bwd)
        output_bwd = torch.flip(output_bwd, dims=[1])
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd], dim=0)
        return output, h_n
