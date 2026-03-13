"""Evaluation helpers for MR-LFADS."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


Edge = Tuple[int, int]


def effectome_from_messages(message_means: Mapping[Edge, np.ndarray]) -> np.ndarray:
    """Compute target-by-source effectome from inferred message means.

    Each value is the trial/time average L2 norm of the corresponding message.
    """
    edges = list(message_means.keys())
    if not edges:
        raise ValueError("message_means is empty.")
    n_regions = 1 + max(max(src, tgt) for src, tgt in edges)
    eff = np.zeros((n_regions, n_regions), dtype=np.float32)
    for (src, tgt), msg in message_means.items():
        norms = np.linalg.norm(msg, axis=-1)  # [trial, time]
        eff[tgt, src] = float(norms.mean())
    return eff


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def ground_truth_effectome(npz_data: Mapping[str, np.ndarray]) -> np.ndarray:
    gt_messages: Dict[Edge, np.ndarray] = {}
    for key, value in npz_data.items():
        if key.startswith("gt_messages_") and "to" in key:
            payload = key.replace("gt_messages_", "")
            src_s, tgt_s = payload.split("to")
            gt_messages[(int(src_s), int(tgt_s))] = value
    return effectome_from_messages(gt_messages)


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x2 = x.reshape(-1, x.shape[-1])
    y2 = y.reshape(-1, y.shape[-1])
    x_aug = np.concatenate([x2, np.ones((x2.shape[0], 1), dtype=x2.dtype)], axis=1)
    beta, *_ = np.linalg.lstsq(x_aug, y2, rcond=None)
    pred = x_aug @ beta
    ss_res = ((y2 - pred) ** 2).sum(axis=0)
    ss_tot = ((y2 - y2.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    valid = ss_tot > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(1.0 - ss_res[valid] / ss_tot[valid]))


def message_content_r2(
    inferred_messages: Mapping[Edge, np.ndarray],
    npz_data: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for (src, tgt), inferred in inferred_messages.items():
        key = f"gt_messages_{src}to{tgt}"
        if key not in npz_data:
            continue
        metrics[key] = _linear_r2(inferred, np.asarray(npz_data[key]))
    return metrics


def input_content_r2(
    inferred_inputs: Mapping[int, np.ndarray],
    npz_data: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for region, inferred in inferred_inputs.items():
        key = f"gt_input_region{region}"
        if key not in npz_data:
            continue
        metrics[key] = _linear_r2(inferred, np.asarray(npz_data[key]))
    return metrics
