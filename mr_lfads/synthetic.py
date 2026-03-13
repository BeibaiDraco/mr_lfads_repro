"""Synthetic data generators for MR-LFADS demos.

The default generator reproduces the spirit of Experiment 1 from the MR-LFADS paper:
a ring of GRU modules, each receiving a private stimulus and delayed communication from
an upstream region. The data-generating network is itself trained so the hidden states
serve as realistic synthetic neural activity, matching the paper's procedure where
synthetic datasets are produced by task-trained recurrent modules.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .initializers import init_linear_
from .recurrent import ClippedGRUCell


@dataclass
class MemoryRingConfig:
    stim_dims: list[int]
    hidden_size: int = 64
    total_steps: int = 200
    tau: int = 10
    noise_std: float = 0.1
    delay: int = 2
    history_window: int = 5
    train_epochs: int = 300
    train_batch_size: int = 2048
    train_batches_per_epoch: int = 1
    train_lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.stim_dims) < 2:
            raise ValueError("Need at least two regions for a communication dataset.")
        if self.tau <= 0 or self.tau >= self.total_steps:
            raise ValueError("tau must satisfy 0 < tau < total_steps.")
        if self.delay < 1:
            raise ValueError("delay must be >= 1.")
        if self.history_window < 1:
            raise ValueError("history_window must be >= 1.")


class _MemoryRegion(nn.Module):
    def __init__(self, stim_dim: int, incoming_dim: int, hidden_size: int, history_window: int) -> None:
        super().__init__()
        self.stim_dim = stim_dim
        self.incoming_dim = incoming_dim
        self.hidden_size = hidden_size
        self.history_window = history_window
        self.cell = ClippedGRUCell(stim_dim + incoming_dim, hidden_size, clip_value=5.0)
        self.msg_head = nn.Linear(hidden_size, stim_dim)
        self.readout_head = nn.Linear(hidden_size, history_window * (stim_dim + incoming_dim))
        init_linear_(self.msg_head)
        init_linear_(self.readout_head)


class MemoryRingDGN(nn.Module):
    def __init__(self, config: MemoryRingConfig) -> None:
        super().__init__()
        self.config = config
        n_regions = len(config.stim_dims)
        self.n_regions = n_regions
        self.upstream = {i: (i - 1) % n_regions for i in range(n_regions)}
        self.regions = nn.ModuleList(
            [
                _MemoryRegion(
                    stim_dim=config.stim_dims[i],
                    incoming_dim=config.stim_dims[self.upstream[i]],
                    hidden_size=config.hidden_size,
                    history_window=config.history_window,
                )
                for i in range(n_regions)
            ]
        )

    def rollout(self, stimuli: list[torch.Tensor], add_noise: bool = True) -> dict[str, list[torch.Tensor]]:
        batch_size = stimuli[0].shape[0]
        device = stimuli[0].device
        steps = stimuli[0].shape[1]
        hidden = [torch.zeros(batch_size, self.config.hidden_size, device=device) for _ in range(self.n_regions)]
        prev_msgs = [torch.zeros(batch_size, self.config.stim_dims[i], device=device) for i in range(self.n_regions)]

        hidden_traces = [[] for _ in range(self.n_regions)]
        emitted_msgs = [[] for _ in range(self.n_regions)]
        incoming_msgs = [[] for _ in range(self.n_regions)]
        readouts = [[] for _ in range(self.n_regions)]

        for t in range(steps):
            current_msgs: list[torch.Tensor] = []
            for i, region in enumerate(self.regions):
                incoming = prev_msgs[self.upstream[i]]
                x_t = torch.cat([stimuli[i][:, t], incoming], dim=1)
                h_t = region.cell(x_t, hidden[i])
                if add_noise:
                    h_t = h_t + self.config.noise_std * torch.randn_like(h_t)
                msg_t = region.msg_head(h_t)
                readout_t = region.readout_head(h_t)
                hidden[i] = h_t
                current_msgs.append(msg_t)
                hidden_traces[i].append(h_t)
                emitted_msgs[i].append(msg_t)
                incoming_msgs[i].append(incoming)
                readouts[i].append(readout_t)
            prev_msgs = current_msgs

        return {
            "hidden": [torch.stack(v, dim=1) for v in hidden_traces],
            "messages": [torch.stack(v, dim=1) for v in emitted_msgs],
            "incoming": [torch.stack(v, dim=1) for v in incoming_msgs],
            "readouts": [torch.stack(v, dim=1) for v in readouts],
        }


def _build_history_targets(sequence: torch.Tensor, window: int) -> torch.Tensor:
    batch, steps, dim = sequence.shape
    padded = torch.zeros(batch, steps + window - 1, dim, device=sequence.device, dtype=sequence.dtype)
    padded[:, window - 1:] = sequence
    # unfold → (batch, steps, dim, window), permute to (batch, steps, window, dim)
    return padded.unfold(1, window, 1).permute(0, 1, 3, 2).reshape(batch, steps, window * dim)


def _delayed(sequence: torch.Tensor, delay: int) -> torch.Tensor:
    out = torch.zeros_like(sequence)
    if delay < sequence.shape[1]:
        out[:, delay:] = sequence[:, :-delay]
    return out


def _sample_stimuli(cfg: MemoryRingConfig, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, cfg.total_steps, dim, device=device)
        for dim in cfg.stim_dims
    ]


def train_memory_ring_dgn(
    config: MemoryRingConfig,
    device: str | torch.device = "cpu",
    early_stop_patience: int = 30,
    compile_model: bool = False,
) -> MemoryRingDGN:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = MemoryRingDGN(config).to(device)
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[DGN] using torch.compile")
        except Exception:
            pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
    model.train()

    best_loss = float("inf")
    stall_count = 0
    for epoch in range(config.train_epochs):
        epoch_loss = 0.0
        for _ in range(config.train_batches_per_epoch):
            stimuli = _sample_stimuli(config, config.train_batch_size, torch.device(device))
            rollout = model.rollout(stimuli, add_noise=True)
            loss = torch.tensor(0.0, device=device)
            for i in range(model.n_regions):
                msg_target = _delayed(stimuli[i], config.delay)
                incoming_hist = _build_history_targets(rollout["incoming"][i].detach(), config.history_window)
                stim_hist = _build_history_targets(stimuli[i], config.history_window)
                readout_target = torch.cat([stim_hist, incoming_hist], dim=-1)
                loss = loss + nn.functional.mse_loss(rollout["messages"][i], msg_target)
                loss = loss + nn.functional.mse_loss(rollout["readouts"][i], readout_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        mean_loss = epoch_loss / float(config.train_batches_per_epoch)
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            stall_count = 0
        else:
            stall_count += 1
        print(f"[DGN] epoch {epoch + 1:4d}/{config.train_epochs}  loss={mean_loss:.6f}  best={best_loss:.6f}  stall={stall_count}")
        if early_stop_patience > 0 and stall_count >= early_stop_patience:
            print(f"[DGN] early stopping at epoch {epoch + 1} (loss={mean_loss:.6f})")
            break
    if compile_model and hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


@torch.no_grad()
def generate_memory_ring_dataset(
    dgn: MemoryRingDGN,
    n_trials: int,
    config: MemoryRingConfig,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    dgn.eval()
    stimuli = _sample_stimuli(config, n_trials, torch.device(device))
    rollout = dgn.rollout(stimuli, add_noise=True)
    payload: dict[str, np.ndarray] = {
        "tau": np.asarray(config.tau, dtype=np.int64),
        "output_type": np.asarray("gaussian"),
    }
    for i in range(dgn.n_regions):
        payload[f"region{i}"] = rollout["hidden"][i].cpu().numpy().astype(np.float32)
        payload[f"gt_input_region{i}"] = stimuli[i][:, config.tau :].cpu().numpy().astype(np.float32)
        tgt = (i + 1) % dgn.n_regions
        payload[f"gt_messages_{i}to{tgt}"] = rollout["messages"][i][:, config.tau :].cpu().numpy().astype(np.float32)
    return payload


def save_memory_ring_dataset(
    out_path: str | Path,
    dataset: dict[str, np.ndarray],
    config: MemoryRingConfig,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    meta = asdict(config)
    if extra_meta is not None:
        meta.update(extra_meta)
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
