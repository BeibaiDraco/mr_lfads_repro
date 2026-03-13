"""Dataset utilities for MR-LFADS."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REGION_PATTERNS = [
    re.compile(r"region(\d+)"),
    re.compile(r"spikes_region(\d+)"),
    re.compile(r"data_region(\d+)"),
]


@dataclass
class LoadedDataset:
    region_data: list[np.ndarray]
    tau: int
    total_steps: int
    main_steps: int
    output_type: str
    meta: dict[str, Any]
    raw_npz: dict[str, Any]


class MRLFADSDataset(Dataset):
    def __init__(self, region_data: Sequence[np.ndarray], tau: int, indices: Sequence[int]):
        self.region_data = [torch.tensor(arr, dtype=torch.float32) for arr in region_data]
        self.tau = int(tau)
        self.indices = np.asarray(indices, dtype=np.int64)
        if not self.region_data:
            raise ValueError("No region arrays were provided.")
        total_steps = self.region_data[0].shape[1]
        for arr in self.region_data:
            if arr.ndim != 3:
                raise ValueError(f"Each region array must have shape [trials, time, neurons], got {tuple(arr.shape)}.")
            if arr.shape[1] != total_steps:
                raise ValueError("All regions must share the same time dimension.")
        if tau <= 0 or tau >= total_steps:
            raise ValueError(f"tau must satisfy 0 < tau < total_steps, got tau={tau}, total_steps={total_steps}.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        trial = int(self.indices[idx])
        history = [arr[trial, : self.tau] for arr in self.region_data]
        obs = [arr[trial, self.tau :] for arr in self.region_data]
        return {"history": history, "obs": obs, "trial_index": trial}


class FullSplitDataset(MRLFADSDataset):
    def __init__(
        self,
        region_data: Sequence[np.ndarray],
        tau: int,
        indices: Sequence[int],
        raw_npz: Mapping[str, Any],
    ):
        super().__init__(region_data, tau, indices)
        self.extra_tensors: dict[str, Any] = {}
        self.extra_arrays: dict[str, np.ndarray] = {}
        for key, value in raw_npz.items():
            if key.startswith("gt_input_region") or key.startswith("gt_messages_"):
                self.extra_arrays[key] = np.asarray(value)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        trial = item["trial_index"]
        extras: dict[str, Any] = {}
        for key, value in self.extra_arrays.items():
            extras[key] = torch.tensor(value[trial], dtype=torch.float32)
        item["extras"] = extras
        return item


def _find_region_keys(npz_keys: Iterable[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for key in npz_keys:
        for pattern in _REGION_PATTERNS:
            m = pattern.fullmatch(key)
            if m is not None:
                found.append((int(m.group(1)), key))
                break
    if not found:
        raise ValueError(
            "Could not find any region arrays. Expected keys like 'region0', 'spikes_region0', or 'data_region0'."
        )
    found.sort(key=lambda pair: pair[0])
    return [key for _, key in found]


def load_npz_dataset(path: str | Path, tau: int | None = None) -> LoadedDataset:
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    region_keys = _find_region_keys(keys)
    region_data = [np.asarray(npz[k], dtype=np.float32) for k in region_keys]
    inferred_tau = int(npz["tau"]) if "tau" in keys else None
    if tau is None and inferred_tau is None:
        raise ValueError("tau was not provided and was not found inside the dataset.")
    tau_value = int(tau if tau is not None else inferred_tau)
    total_steps = int(region_data[0].shape[1])
    main_steps = total_steps - tau_value
    if "output_type" in keys:
        output_raw = npz["output_type"]
        if isinstance(output_raw, np.ndarray):
            output_raw = output_raw.item()
        if isinstance(output_raw, bytes):
            output_raw = output_raw.decode()
        output_type = str(output_raw)
    else:
        output_type = "gaussian"
    meta = {
        "region_keys": region_keys,
        "region_dims": [int(arr.shape[-1]) for arr in region_data],
        "n_regions": len(region_data),
        "n_trials": int(region_data[0].shape[0]),
        "tau": tau_value,
        "total_steps": total_steps,
        "main_steps": main_steps,
        "output_type": output_type,
    }
    raw_npz = {k: npz[k] for k in keys}
    return LoadedDataset(
        region_data=region_data,
        tau=tau_value,
        total_steps=total_steps,
        main_steps=main_steps,
        output_type=output_type,
        meta=meta,
        raw_npz=raw_npz,
    )


def make_split_indices(
    n_trials: int,
    valid_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_trials, dtype=np.int64)
    rng.shuffle(indices)
    n_valid = max(1, int(round(n_trials * valid_fraction)))
    valid_idx = np.sort(indices[:n_valid])
    train_idx = np.sort(indices[n_valid:])
    return train_idx, valid_idx


def create_dataloaders(
    loaded: LoadedDataset,
    batch_size: int,
    valid_fraction: float = 0.15,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_idx, valid_idx = make_split_indices(loaded.meta["n_trials"], valid_fraction=valid_fraction, seed=seed)
    train_ds = FullSplitDataset(loaded.region_data, loaded.tau, train_idx, loaded.raw_npz)
    valid_ds = FullSplitDataset(loaded.region_data, loaded.tau, valid_idx, loaded.raw_npz)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, train_idx, valid_idx


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")
