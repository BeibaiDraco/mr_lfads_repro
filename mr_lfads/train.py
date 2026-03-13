"""Training loop for MR-LFADS."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .data import create_dataloaders, load_npz_dataset, save_json
from .evaluate import cosine_similarity, effectome_from_messages, ground_truth_effectome, input_content_r2, message_content_r2
from .losses import safe_r2
from .model import MRLFADS, MRLFADSConfig


@dataclass
class TrainConfig:
    # Data / architecture
    tau: int | None = None
    output_type: str | None = None
    gen_dim: int | None = None
    fac_dim: int | None = None
    g0_encoder_dim: int | None = None
    causal_encoder_dim: int | None = None
    controller_dim: int | None = None
    input_dim: int = 4
    message_dim: int = 4
    message_from: str = "rates"
    message_timing: str = "lagged"
    fixed_point_iters: int = 2
    prior_variance: float = 1.0
    posterior_var_min: float = 1.0e-4
    dropout_rate: float = 0.02
    cell_clip: float = 5.0

    # Regularization (paper defaults)
    beta_u: float = 0.1
    beta_m: float = 0.01
    beta_g0: float | None = None
    beta_u_start_epoch: int = 50
    beta_u_increase_epoch: int = 200
    beta_m_start_epoch: int = 150
    beta_m_increase_epoch: int = 100
    l2_alpha: float = 1.0e-4
    l2_start_epoch: int = 0
    l2_increase_epoch: int = 80
    loss_scale: float = 1.0

    # Optimization
    seed: int = 0
    batch_size: int = 64
    max_epochs: int = 350
    lr_init: float = 4.0e-3
    lr_min: float = 1.0e-5
    lr_decay: float = 0.95
    lr_patience: int = 6
    lr_adam_beta1: float = 0.9
    lr_adam_beta2: float = 0.999
    lr_adam_epsilon: float = 1.0e-7
    weight_decay: float = 0.0
    grad_clip: float = 200.0
    valid_fraction: float = 0.15
    num_workers: int = 0
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics if k in m and not math.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


@torch.no_grad()
def _collect_posteriors(
    model: MRLFADS,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    recon: dict[int, list[np.ndarray]] = {i: [] for i in range(model.n_regions)}
    obs: dict[int, list[np.ndarray]] = {i: [] for i in range(model.n_regions)}
    inferred_inputs: dict[int, list[np.ndarray]] = {i: [] for i in range(model.n_regions)}
    messages: dict[tuple[int, int], list[np.ndarray]] = {edge: [] for edge in model.edge_keys}
    g_states: dict[int, list[np.ndarray]] = {i: [] for i in range(model.n_regions)}
    factors: dict[int, list[np.ndarray]] = {i: [] for i in range(model.n_regions)}
    trial_indices: list[np.ndarray] = []

    model.eval()
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model.posterior_means(batch["history"], batch["obs"])
        trial_indices.append(batch["trial_index"].detach().cpu().numpy())
        for i in range(model.n_regions):
            recon[i].append(outputs["recon_main"][i].detach().cpu().numpy())
            obs[i].append(batch["obs"][i].detach().cpu().numpy())
            inferred_inputs[i].append(outputs["input_means"][i].detach().cpu().numpy())
            g_states[i].append(outputs["gen_states"][i].detach().cpu().numpy())
            factors[i].append(outputs["factors"][i].detach().cpu().numpy())
        for edge in model.edge_keys:
            messages[edge].append(outputs["message_means"][edge].detach().cpu().numpy())

    return {
        "trial_indices": np.concatenate(trial_indices, axis=0),
        "recon": {i: np.concatenate(v, axis=0) for i, v in recon.items()},
        "obs": {i: np.concatenate(v, axis=0) for i, v in obs.items()},
        "inferred_inputs": {i: np.concatenate(v, axis=0) for i, v in inferred_inputs.items()},
        "messages": {edge: np.concatenate(v, axis=0) for edge, v in messages.items()},
        "gen_states": {i: np.concatenate(v, axis=0) for i, v in g_states.items()},
        "factors": {i: np.concatenate(v, axis=0) for i, v in factors.items()},
    }


def _subset_raw_npz(raw_npz: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key, value in raw_npz.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == indices.max() + 1 or (arr.ndim >= 1 and arr.shape[0] >= indices.shape[0] and key.startswith(("region", "spikes_region", "data_region", "gt_input_region", "gt_messages_"))):
            try:
                subset[key] = arr[indices]
            except Exception:
                subset[key] = value
        else:
            subset[key] = value
    return subset


def _save_predictions(save_dir: Path, posteriors: Mapping[str, Any]) -> None:
    payload: dict[str, Any] = {"trial_indices": posteriors["trial_indices"]}
    for i, arr in posteriors["recon"].items():
        payload[f"recon_region{i}"] = arr
        payload[f"obs_region{i}"] = posteriors["obs"][i]
        payload[f"inferred_input_region{i}"] = posteriors["inferred_inputs"][i]
        payload[f"gen_state_region{i}"] = posteriors["gen_states"][i]
        payload[f"factors_region{i}"] = posteriors["factors"][i]
    for (src, tgt), arr in posteriors["messages"].items():
        payload[f"inferred_message_{src}to{tgt}"] = arr
    np.savez_compressed(save_dir / "posterior_means_valid.npz", **payload)


def train_mr_lfads(
    dataset_path: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    train_config: TrainConfig,
) -> dict[str, Any]:
    _set_seed(int(train_config.seed))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_npz_dataset(dataset_path, tau=train_config.tau)
    output_type = train_config.output_type or loaded.output_type
    if output_type not in {"gaussian", "poisson"}:
        raise ValueError(f"Unsupported output_type={output_type}.")

    model_config = MRLFADSConfig(
        region_dims=list(loaded.meta["region_dims"]),
        tau=int(loaded.tau),
        main_steps=int(loaded.main_steps),
        output_type=output_type,  # type: ignore[arg-type]
        gen_dim=train_config.gen_dim,
        fac_dim=train_config.fac_dim,
        g0_encoder_dim=train_config.g0_encoder_dim,
        causal_encoder_dim=train_config.causal_encoder_dim,
        controller_dim=train_config.controller_dim,
        input_dim=train_config.input_dim,
        message_dim=train_config.message_dim,
        message_from=train_config.message_from,  # type: ignore[arg-type]
        message_timing=train_config.message_timing,  # type: ignore[arg-type]
        fixed_point_iters=train_config.fixed_point_iters,
        prior_variance=train_config.prior_variance,
        posterior_var_min=train_config.posterior_var_min,
        dropout_rate=train_config.dropout_rate,
        cell_clip=train_config.cell_clip,
        beta_u=train_config.beta_u,
        beta_m=train_config.beta_m,
        beta_g0=train_config.beta_g0,
        beta_u_start_epoch=train_config.beta_u_start_epoch,
        beta_u_increase_epoch=train_config.beta_u_increase_epoch,
        beta_m_start_epoch=train_config.beta_m_start_epoch,
        beta_m_increase_epoch=train_config.beta_m_increase_epoch,
        l2_alpha=train_config.l2_alpha,
        l2_start_epoch=train_config.l2_start_epoch,
        l2_increase_epoch=train_config.l2_increase_epoch,
        loss_scale=train_config.loss_scale,
    )

    train_loader, valid_loader, train_idx, valid_idx = create_dataloaders(
        loaded,
        batch_size=int(train_config.batch_size),
        valid_fraction=float(train_config.valid_fraction),
        seed=int(train_config.seed),
        num_workers=int(train_config.num_workers),
    )

    device = torch.device(train_config.resolved_device())
    model = MRLFADS(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.lr_init),
        betas=(float(train_config.lr_adam_beta1), float(train_config.lr_adam_beta2)),
        eps=float(train_config.lr_adam_epsilon),
        weight_decay=float(train_config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_config.lr_decay),
        patience=int(train_config.lr_patience),
        threshold=0.0,
        min_lr=float(train_config.lr_min)
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_recon": [],
        "train_kl_g0": [],
        "train_kl_u": [],
        "train_kl_m": [],
        "valid_loss": [],
        "valid_recon": [],
        "valid_kl_g0": [],
        "valid_kl_u": [],
        "valid_kl_m": [],
        "lr": [],
    }
    best_val_recon = float("inf")
    best_ckpt = save_dir / "best_model.pt"
    last_ckpt = save_dir / "last_model.pt"

    for epoch in range(int(train_config.max_epochs)):
        model.train()
        train_metrics: list[dict[str, float]] = []
        prog = tqdm(train_loader, leave=False, desc=f"epoch {epoch + 1}/{train_config.max_epochs}")
        for batch in prog:
            batch = _to_device(batch, device)
            outputs = model(batch["history"], batch["obs"], sample_posteriors=True)
            loss, metrics = model.compute_loss(batch["obs"], outputs, epoch=epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_config.grad_clip))
            optimizer.step()
            train_metrics.append(metrics)
            prog.set_postfix(loss=f"{metrics['loss']:.3f}", recon=f"{metrics['recon']:.3f}")

        model.eval()
        valid_metrics: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = _to_device(batch, device)
                outputs = model(batch["history"], batch["obs"], sample_posteriors=False)
                _, metrics = model.compute_loss(batch["obs"], outputs, epoch=epoch)
                valid_metrics.append(metrics)

        train_mean = _mean_metrics(train_metrics)
        valid_mean = _mean_metrics(valid_metrics)
        scheduler.step(valid_mean["recon"])
        lr_value = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_mean["loss"])
        history["train_recon"].append(train_mean["recon"])
        history["train_kl_g0"].append(train_mean["kl_g0"])
        history["train_kl_u"].append(train_mean["kl_u"])
        history["train_kl_m"].append(train_mean["kl_m"])
        history["valid_loss"].append(valid_mean["loss"])
        history["valid_recon"].append(valid_mean["recon"])
        history["valid_kl_g0"].append(valid_mean["kl_g0"])
        history["valid_kl_u"].append(valid_mean["kl_u"])
        history["valid_kl_m"].append(valid_mean["kl_m"])
        history["lr"].append(lr_value)

        ckpt_payload = {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "epoch": epoch,
            "history": history,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
        }
        torch.save(ckpt_payload, last_ckpt)
        if valid_mean["recon"] < best_val_recon:
            best_val_recon = valid_mean["recon"]
            torch.save(ckpt_payload, best_ckpt)

        print(
            f"[MR-LFADS] epoch {epoch + 1:4d}/{train_config.max_epochs} "
            f"train_recon={train_mean['recon']:.6f} valid_recon={valid_mean['recon']:.6f} lr={lr_value:.3e}"
        )

    best_payload = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    posteriors = _collect_posteriors(model, valid_loader, device)
    _save_predictions(save_dir, posteriors)

    effectome = effectome_from_messages(posteriors["messages"])
    np.save(save_dir / "effectome_valid.npy", effectome)

    final_metrics: dict[str, Any] = {
        "best_valid_recon": float(best_val_recon),
        "effectome_valid": effectome.tolist(),
        "valid_recon_r2": {
            f"region{i}": safe_r2(torch.tensor(posteriors["obs"][i]), torch.tensor(posteriors["recon"][i]))
            for i in range(model.n_regions)
        },
        "valid_indices": valid_idx.tolist(),
        "train_indices": train_idx.tolist(),
    }

    raw_subset = {k: v for k, v in loaded.raw_npz.items()}
    try:
        gt_effectome = ground_truth_effectome({
            k: np.asarray(v)[valid_idx] if k.startswith("gt_messages_") else v
            for k, v in raw_subset.items()
        })
        final_metrics["gt_effectome_valid"] = gt_effectome.tolist()
        final_metrics["effectome_cosine_valid"] = cosine_similarity(effectome, gt_effectome)
    except Exception:
        pass

    msg_r2 = message_content_r2(
        posteriors["messages"],
        {k: np.asarray(v)[valid_idx] if k.startswith("gt_messages_") else v for k, v in raw_subset.items()},
    )
    if msg_r2:
        final_metrics["message_content_r2_valid"] = msg_r2
    inp_r2 = input_content_r2(
        posteriors["inferred_inputs"],
        {k: np.asarray(v)[valid_idx] if k.startswith("gt_input_region") else v for k, v in raw_subset.items()},
    )
    if inp_r2:
        final_metrics["input_content_r2_valid"] = inp_r2

    save_json(save_dir / "metrics.json", final_metrics)
    save_json(save_dir / "history.json", history)
    save_json(
        save_dir / "config_resolved.json",
        {
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "dataset_meta": loaded.meta,
        },
    )

    return {
        "model": model,
        "model_config": model_config,
        "train_config": train_config,
        "history": history,
        "metrics": final_metrics,
        "save_dir": str(save_dir),
    }
