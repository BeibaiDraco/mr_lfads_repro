"""Losses and KL utilities for MR-LFADS."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def diag_gaussian_kl(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> torch.Tensor:
    """Elementwise KL[q || p] for diagonal Gaussians.

    Returns a tensor reduced only over the latent dimension.
    """
    logvar = torch.clamp(logvar, min=-20.0, max=80.0)
    var = torch.exp(logvar)
    prior_logvar = math.log(prior_var)
    kl = 0.5 * (
        (var + (mean - prior_mean) ** 2) / prior_var - 1.0 - logvar + prior_logvar
    )
    return kl.sum(dim=-1)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return F.gaussian_nll_loss(mean, target, var=torch.exp(logvar), reduction="none")


def poisson_nll(target: torch.Tensor, lograte: torch.Tensor) -> torch.Tensor:
    lograte = torch.clamp(lograte, min=-20.0, max=20.0)
    return F.poisson_nll_loss(lograte, target, full=True, reduction="none", log_input=True)


def mean_over_collection(values: Iterable[torch.Tensor]) -> torch.Tensor:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty collection.")
    return torch.stack([v.mean() for v in values]).mean()


def safe_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true_np = y_true.detach().cpu().reshape(-1, y_true.shape[-1]).numpy()
    y_pred_np = y_pred.detach().cpu().reshape(-1, y_pred.shape[-1]).numpy()
    ss_res = ((y_true_np - y_pred_np) ** 2).sum(axis=0)
    ss_tot = ((y_true_np - y_true_np.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    valid = ss_tot > 1e-12
    if not valid.any():
        return float("nan")
    r2 = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(r2.mean())
