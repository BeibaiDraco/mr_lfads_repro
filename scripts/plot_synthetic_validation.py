#!/usr/bin/env python3
"""Generate validation figures for synthetic MR-LFADS runs.

Produces 4 figures comparing inferred quantities against ground truth:
  1. Reconstruction quality (R² scatter + example neuron time courses)
  2. Message recovery (GT vs inferred message norm per active pathway)
  3. Effectome comparison (side-by-side GT vs inferred heatmaps)
  4. Hidden-input recovery (GT vs inferred input norm per region)

Usage:
    python scripts/plot_synthetic_validation.py \
        --run_dir runs/memory_ring_full \
        --data data/memory_ring_exp1.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REGION_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
GT_COLOR = "#333333"
INFERRED_MSG_COLOR = "#1f77b4"
BIN_MS = 10


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_data(run_dir: Path, data_path: Path):
    posteriors = np.load(run_dir / "posterior_means_valid.npz", allow_pickle=True)
    effectome = np.load(run_dir / "effectome_valid.npy")

    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    config = {}
    config_path = run_dir / "config_resolved.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    raw_npz = np.load(data_path, allow_pickle=True)
    valid_idx = np.asarray(posteriors["trial_indices"])

    gt_subset: dict[str, np.ndarray] = {}
    for key in raw_npz.files:
        if key.startswith(("gt_messages_", "gt_input_")):
            gt_subset[key] = np.asarray(raw_npz[key])[valid_idx]

    output_type = config.get("model_config", {}).get("output_type", "gaussian")
    if "output_type" in raw_npz:
        output_type = str(raw_npz["output_type"])

    n_regions = effectome.shape[0]
    main_steps = np.asarray(posteriors["obs_region0"]).shape[1]
    time_ms = np.arange(main_steps) * BIN_MS

    return posteriors, effectome, metrics, gt_subset, n_regions, time_ms, output_type


# ──────────────────────────────────────────────────────────────────────
# Figure 1: Reconstruction quality
# ──────────────────────────────────────────────────────────────────────

def _recon_to_rate(recon: np.ndarray, output_type: str) -> np.ndarray:
    if output_type == "poisson":
        return np.exp(np.clip(recon, -20, 20))
    return recon


def fig_reconstruction(posteriors, metrics: dict, n_regions: int,
                       time_ms: np.ndarray, save_path: Path,
                       output_type: str = "gaussian"):
    """R² scatter per region (left column) + 3 example neuron traces (right columns)."""
    fig = plt.figure(figsize=(16, 3.6 * n_regions))
    gs = fig.add_gridspec(n_regions, 4, width_ratios=[1.2, 1, 1, 1],
                          hspace=0.5, wspace=0.35)

    r2_by_region = metrics.get("valid_recon_r2", {})

    for i in range(n_regions):
        obs = np.asarray(posteriors[f"obs_region{i}"])
        recon = np.asarray(posteriors[f"recon_region{i}"])
        rate = _recon_to_rate(recon, output_type)

        avg_obs = obs.mean(axis=0)
        avg_rate = rate.mean(axis=0)
        n_neurons = obs.shape[-1]

        r2_per_neuron = np.zeros(n_neurons)
        for n in range(n_neurons):
            y, yhat = avg_obs[:, n], avg_rate[:, n]
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2_per_neuron[n] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

        region_r2 = r2_by_region.get(f"region{i}", np.nanmean(r2_per_neuron))
        color = REGION_COLORS.get(i, "C0")

        # Scatter panel
        ax_sc = fig.add_subplot(gs[i, 0])
        modulated = np.where(avg_obs.var(axis=0) > 1e-6)[0]
        x_all = avg_obs[:, modulated].flatten()
        y_all = avg_rate[:, modulated].flatten()
        ax_sc.scatter(x_all, y_all, s=1, alpha=0.15, color=color, rasterized=True)
        lim = max(x_all.max(), y_all.max()) * 1.05
        ax_sc.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        corr = np.corrcoef(x_all, y_all)[0, 1]
        ax_sc.set_title(f"R{i}   R²={region_r2:.3f}  r={corr:.3f}",
                        fontsize=11, fontweight="bold")
        ax_sc.set_xlabel("Observed (trial avg)", fontsize=9)
        ax_sc.set_ylabel("Model rate (trial avg)", fontsize=9)
        ax_sc.set_xlim(0, lim)
        ax_sc.set_ylim(0, lim)
        ax_sc.set_aspect("equal")
        ax_sc.spines["top"].set_visible(False)
        ax_sc.spines["right"].set_visible(False)

        # Example neuron panels
        sorted_neurons = np.argsort(r2_per_neuron)[::-1]
        picks = [0, len(sorted_neurons) // 4, len(sorted_neurons) // 2]
        pick_labels = ["best", "Q1", "median"]
        for col, (rank, plabel) in enumerate(zip(picks, pick_labels)):
            ax = fig.add_subplot(gs[i, col + 1])
            n_idx = sorted_neurons[rank]
            psth = avg_obs[:, n_idx]
            model = avg_rate[:, n_idx]
            r2v = r2_per_neuron[n_idx]

            ax.plot(time_ms, psth, color="black", linewidth=1.2, linestyle="--",
                    label="Observed" if i == 0 and col == 0 else None)
            ax.plot(time_ms, model, color=color, linewidth=1.8,
                    label="Model" if i == 0 and col == 0 else None)
            ax.fill_between(time_ms, 0, psth, alpha=0.07, color="black")
            ax.set_title(f"neuron {n_idx} ({plabel})  R²={r2v:.3f}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == n_regions - 1:
                ax.set_xlabel("Time (ms)", fontsize=9)
            if col == 0:
                ax.set_ylabel("Activity", fontsize=9)
            if i == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Figure 1: Reconstruction Quality — trial-averaged observed vs model rate",
                 fontsize=14, y=1.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")  # gridspec handles layout
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Message recovery (GT vs inferred)
# ──────────────────────────────────────────────────────────────────────

def _find_active_pathways(gt_subset: dict, n_regions: int):
    """Return list of (src, tgt) for pathways that have ground-truth messages."""
    edges = []
    for src in range(n_regions):
        for tgt in range(n_regions):
            if src != tgt and f"gt_messages_{src}to{tgt}" in gt_subset:
                edges.append((src, tgt))
    return edges


def fig_message_recovery(posteriors, gt_subset: dict, metrics: dict,
                         n_regions: int, time_ms: np.ndarray, save_path: Path):
    """GT vs inferred trial-averaged message norm for each active pathway."""
    edges = _find_active_pathways(gt_subset, n_regions)
    n_edges = len(edges)
    msg_r2 = metrics.get("message_content_r2_valid", {})

    fig, axes = plt.subplots(1, n_edges, figsize=(5.5 * n_edges, 4), sharey=True)
    if n_edges == 1:
        axes = [axes]

    for idx, (src, tgt) in enumerate(edges):
        ax = axes[idx]
        gt_key = f"gt_messages_{src}to{tgt}"
        inf_key = f"inferred_message_{src}to{tgt}"

        gt = np.asarray(gt_subset[gt_key])
        inferred = np.asarray(posteriors[inf_key])

        gt_norms = np.linalg.norm(gt, axis=-1)
        inf_norms = np.linalg.norm(inferred, axis=-1)

        gt_mean, gt_std = gt_norms.mean(0), gt_norms.std(0)
        inf_mean, inf_std = inf_norms.mean(0), inf_norms.std(0)

        ax.plot(time_ms, gt_mean, color=GT_COLOR, linewidth=2.0, label="Ground truth")
        ax.fill_between(time_ms, gt_mean - gt_std, gt_mean + gt_std,
                        alpha=0.15, color=GT_COLOR)
        ax.plot(time_ms, inf_mean, color=INFERRED_MSG_COLOR, linewidth=2.0,
                linestyle="--", label="Inferred")
        ax.fill_between(time_ms, inf_mean - inf_std, inf_mean + inf_std,
                        alpha=0.15, color=INFERRED_MSG_COLOR)

        r2_val = msg_r2.get(gt_key, None)
        title = f"R{src} → R{tgt}"
        if r2_val is not None:
            title += f"   R²={r2_val:.3f}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time (ms)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Message norm (trial avg ± 1 SD)", fontsize=10)
            ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Figure 2: Message Recovery — ground truth vs inferred",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Effectome comparison
# ──────────────────────────────────────────────────────────────────────

def fig_effectome_comparison(effectome: np.ndarray, metrics: dict,
                             n_regions: int, save_path: Path):
    """Side-by-side GT vs inferred effectome heatmaps."""
    gt_eff_list = metrics.get("gt_effectome_valid", None)
    if gt_eff_list is None:
        print("Skipping effectome comparison: no gt_effectome_valid in metrics.")
        return
    gt_eff = np.array(gt_eff_list)
    cosine = metrics.get("effectome_cosine_valid", None)

    labels = [f"R{i}" for i in range(n_regions)]
    mask = np.eye(n_regions, dtype=bool)

    gt_display = gt_eff.copy()
    gt_display[mask] = np.nan
    inf_display = effectome.copy()
    inf_display[mask] = np.nan
    vmax = max(np.nanmax(np.abs(gt_display)), np.nanmax(np.abs(inf_display)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 0.06]})
    ax_gt, ax_inf, ax_cb = axes

    for ax, data, title in [(ax_gt, gt_eff, "Ground Truth"),
                             (ax_inf, effectome, "Inferred")]:
        display = data.copy()
        display[mask] = np.nan
        im = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticks(range(n_regions))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Source region", fontsize=11)
        ax.set_ylabel("Target region", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        for r in range(n_regions):
            for c in range(n_regions):
                if r == c:
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=12, color="gray")
                else:
                    val = data[r, c]
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if val > 0.5 * vmax else "black")

    fig.colorbar(im, cax=ax_cb, label="Communication strength")

    suptitle = "Figure 3: Effectome Recovery"
    if cosine is not None:
        suptitle += f"  (cosine similarity = {cosine:.3f})"
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Figure 4: Hidden-input recovery (GT vs inferred)
# ──────────────────────────────────────────────────────────────────────

def fig_input_recovery(posteriors, gt_subset: dict, metrics: dict,
                       n_regions: int, time_ms: np.ndarray, save_path: Path):
    """GT vs inferred trial-averaged hidden-input norm for each region."""
    inp_r2 = metrics.get("input_content_r2_valid", {})

    fig, axes = plt.subplots(1, n_regions, figsize=(5.5 * n_regions, 4), sharey=True)
    if n_regions == 1:
        axes = [axes]

    for r in range(n_regions):
        ax = axes[r]
        gt_key = f"gt_input_region{r}"
        inf_key = f"inferred_input_region{r}"

        if gt_key not in gt_subset:
            continue

        gt = np.asarray(gt_subset[gt_key])
        inferred = np.asarray(posteriors[inf_key])

        gt_norms = np.linalg.norm(gt, axis=-1)
        inf_norms = np.linalg.norm(inferred, axis=-1)

        gt_mean, gt_std = gt_norms.mean(0), gt_norms.std(0)
        inf_mean, inf_std = inf_norms.mean(0), inf_norms.std(0)

        color = REGION_COLORS.get(r, "C0")

        ax.plot(time_ms, gt_mean, color=GT_COLOR, linewidth=2.0, label="Ground truth")
        ax.fill_between(time_ms, gt_mean - gt_std, gt_mean + gt_std,
                        alpha=0.15, color=GT_COLOR)
        ax.plot(time_ms, inf_mean, color=color, linewidth=2.0,
                linestyle="--", label="Inferred")
        ax.fill_between(time_ms, inf_mean - inf_std, inf_mean + inf_std,
                        alpha=0.15, color=color)

        r2_val = inp_r2.get(gt_key, None)
        title = f"R{r}"
        if r2_val is not None:
            title += f"   R²={r2_val:.3f}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time (ms)", fontsize=10)
        if r == 0:
            ax.set_ylabel("Input norm (trial avg ± 1 SD)", fontsize=10)
            ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Figure 4: Hidden-Input Recovery — ground truth vs inferred",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic validation figures for MR-LFADS")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to run directory (e.g. runs/memory_ring_full)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to synthetic .npz dataset")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: run_dir/figures_synthetic)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    posteriors, effectome, metrics, gt_subset, n_regions, time_ms, output_type = \
        load_data(run_dir, data_path)

    fig_reconstruction(posteriors, metrics, n_regions, time_ms,
                       out_dir / "1_reconstruction.png",
                       output_type=output_type)

    fig_message_recovery(posteriors, gt_subset, metrics, n_regions, time_ms,
                         out_dir / "2_message_recovery.png")

    fig_effectome_comparison(effectome, metrics, n_regions,
                             out_dir / "3_effectome_comparison.png")

    fig_input_recovery(posteriors, gt_subset, metrics, n_regions, time_ms,
                       out_dir / "4_input_recovery.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
