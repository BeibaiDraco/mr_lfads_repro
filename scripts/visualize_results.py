#!/usr/bin/env python3
"""Visualize MR-LFADS results: effectome, messages, reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

REGION_NAMES = {0: "FEF", 1: "LIP", 2: "SC"}
REGION_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
BIN_MS = 10


def load_results(run_dir: Path, data_path: Path | None = None):
    posteriors = np.load(run_dir / "posterior_means_valid.npz", allow_pickle=True)
    effectome = np.load(run_dir / "effectome_valid.npy")
    with open(run_dir / "config_resolved.json") as f:
        config = json.load(f)

    raw_npz = None
    if data_path and data_path.exists():
        raw_npz = np.load(data_path, allow_pickle=True)
    tau_bins = int(raw_npz["tau"]) if raw_npz is not None and "tau" in raw_npz else 10

    # Build time axes from actual time_bins_s if available
    if raw_npz is not None and "time_bins_s" in raw_npz:
        time_bins_ms = np.asarray(raw_npz["time_bins_s"]) * 1000.0
        time_hist_ms = time_bins_ms[:tau_bins]
        time_main_ms = time_bins_ms[tau_bins:]
    else:
        time_hist_ms = np.arange(-tau_bins, 0) * BIN_MS
        time_main_ms = np.arange(posteriors["obs_region0"].shape[1]) * BIN_MS

    return posteriors, effectome, config, tau_bins, raw_npz, time_hist_ms, time_main_ms


def fig_effectome(effectome: np.ndarray, n_regions: int, save_path: Path):
    """Figure 1: Effectome heatmap."""
    labels = [REGION_NAMES.get(i, f"R{i}") for i in range(n_regions)]

    fig, ax = plt.subplots(figsize=(5, 4))
    mask = np.eye(n_regions, dtype=bool)
    display = effectome.copy()
    display[mask] = np.nan

    vmax = np.nanmax(np.abs(display))
    im = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Source region", fontsize=12)
    ax.set_ylabel("Target region", fontsize=12)
    ax.set_title("Inferred Effectome\n(avg message L2 norm)", fontsize=13)

    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="gray")
            else:
                ax.text(j, i, f"{effectome[i, j]:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if effectome[i, j] > 0.6 * vmax else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Communication strength", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_message_timecourses(posteriors, n_regions: int, time_main_ms: np.ndarray,
                            save_path: Path):
    """Figure 2: Per-pathway message norm over time, excluding initialization transient."""
    edges = [(s, t) for s in range(n_regions) for t in range(n_regions) if s != t]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)

    # Skip bins before -100ms to remove initialization transient
    skip_until_ms = -100.0
    start_idx = int(np.searchsorted(time_main_ms, skip_until_ms))

    for idx, (src, tgt) in enumerate(edges):
        ax = axes[idx // 3, idx % 3]
        key = f"inferred_message_{src}to{tgt}"
        msg = np.asarray(posteriors[key])[:, start_idx:, :]
        norms = np.linalg.norm(msg, axis=-1)
        mean_norm = norms.mean(axis=0)
        std_norm = norms.std(axis=0)
        t = time_main_ms[start_idx:]

        ax.plot(t, mean_norm, color="C0", linewidth=1.5)
        ax.fill_between(t, mean_norm - std_norm, mean_norm + std_norm,
                         alpha=0.2, color="C0")
        src_name = REGION_NAMES.get(src, f"R{src}")
        tgt_name = REGION_NAMES.get(tgt, f"R{tgt}")
        ax.set_title(f"{src_name} → {tgt_name}", fontsize=12, fontweight="bold")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // 3 == 1:
            ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Message norm", fontsize=10)

    fig.suptitle("Inter-region message strength over time\n(trial-averaged ± 1 SD)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_recon_quality(posteriors, n_regions: int, time_main_ms: np.ndarray,
                      save_path: Path):
    """Figure 3: Reconstruction quality — trial-averaged PSTH vs model rate.

    Left: scatter of model rate vs observed PSTH (pooled across neurons & time).
    Right panels: top-3 neuron PSTHs per region with overlaid model rate.
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(n_regions, 4, width_ratios=[1.2, 1, 1, 1], hspace=0.45, wspace=0.35)

    all_r2_trial_avg = []

    for i in range(n_regions):
        obs = np.asarray(posteriors[f"obs_region{i}"])
        recon = np.asarray(posteriors[f"recon_region{i}"])
        rate = np.exp(np.clip(recon, -20, 20))

        avg_obs = obs.mean(axis=0)
        avg_rate = rate.mean(axis=0)
        n_neurons = obs.shape[-1]
        time_ms = time_main_ms

        r2_per_neuron = np.zeros(n_neurons)
        for n in range(n_neurons):
            y, yhat = avg_obs[:, n], avg_rate[:, n]
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2_per_neuron[n] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        all_r2_trial_avg.append(r2_per_neuron)

        # Left panel: scatter
        ax_scatter = fig.add_subplot(gs[i, 0])
        modulated = np.where(avg_obs.var(axis=0) > 1e-6)[0]
        x_all = avg_obs[:, modulated].flatten()
        y_all = avg_rate[:, modulated].flatten()
        ax_scatter.scatter(x_all, y_all, s=1, alpha=0.15, color=REGION_COLORS[i], rasterized=True)
        lim = max(x_all.max(), y_all.max()) * 1.05
        ax_scatter.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        corr = np.corrcoef(x_all, y_all)[0, 1]
        region_name = REGION_NAMES.get(i, f"R{i}")
        ax_scatter.set_title(f"{region_name}  r={corr:.3f}", fontsize=11, fontweight="bold")
        ax_scatter.set_xlabel("Observed PSTH", fontsize=9)
        ax_scatter.set_ylabel("Model rate", fontsize=9)
        ax_scatter.set_xlim(0, lim)
        ax_scatter.set_ylim(0, lim)
        ax_scatter.set_aspect("equal")
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)

        # Right 3 panels: top-3 neuron PSTHs
        sorted_neurons = np.argsort(r2_per_neuron)[::-1]
        for rank in range(3):
            ax = fig.add_subplot(gs[i, rank + 1])
            n_idx = sorted_neurons[rank]
            psth = avg_obs[:, n_idx]
            model = avg_rate[:, n_idx]
            r2_val = r2_per_neuron[n_idx]

            ax.plot(time_ms, psth, color="black", linewidth=1.5, label="PSTH")
            ax.plot(time_ms, model, color=REGION_COLORS[i], linewidth=2.0, label="Model rate")
            ax.fill_between(time_ms, 0, psth, alpha=0.1, color="black")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_title(f"neuron {n_idx}  R²={r2_val:.3f}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == n_regions - 1:
                ax.set_xlabel("Time from onset (ms)", fontsize=9)
            if rank == 0:
                ax.set_ylabel("Spikes/bin", fontsize=9)
            if i == 0 and rank == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Reconstruction quality: trial-averaged PSTH vs model rate", fontsize=14, y=1.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return all_r2_trial_avg


def fig_example_neurons(posteriors, all_r2: list, n_regions: int,
                        time_hist_ms: np.ndarray, time_main_ms: np.ndarray,
                        raw_npz, tau_bins: int, save_path: Path):
    """Figure 4: 6 example neurons with history period shown as negative time."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    neuron_picks = []
    for i in range(n_regions):
        r2 = all_r2[i]
        valid_idx = np.where(np.isfinite(r2))[0]
        sorted_by_r2 = valid_idx[np.argsort(r2[valid_idx])[::-1]]
        best = sorted_by_r2[0]
        mid = sorted_by_r2[len(sorted_by_r2) // 4]
        neuron_picks.append((i, best, "best"))
        neuron_picks.append((i, mid, "mid"))

    valid_trial_idx = np.asarray(posteriors["trial_indices"])

    for panel, (region, neuron, label) in enumerate(neuron_picks):
        ax = axes[panel // 3, panel % 3]
        obs = np.asarray(posteriors[f"obs_region{region}"])
        recon = np.asarray(posteriors[f"recon_region{region}"])
        rate = np.exp(np.clip(recon, -20, 20))

        raw_key = f"region{region}"
        has_history = raw_npz is not None and raw_key in raw_npz
        if has_history:
            raw_all = np.asarray(raw_npz[raw_key])
            raw_valid = raw_all[valid_trial_idx]
            history_obs = raw_valid[:, :tau_bins, :]

        avg_obs_main = obs[:, :, neuron].mean(axis=0)
        avg_rate = rate[:, :, neuron].mean(axis=0)

        if has_history:
            avg_hist = history_obs[:, :, neuron].mean(axis=0)
            ax.axvspan(time_hist_ms[0], time_hist_ms[-1] + BIN_MS, alpha=0.06, color="gray")
            ax.plot(time_hist_ms, avg_hist, color="black", linewidth=1.2, linestyle=":")

        ax.plot(time_main_ms, avg_obs_main, color="black", linewidth=1.5, linestyle="--", label="Observed PSTH")
        ax.plot(time_main_ms, avg_rate, color="C0", linewidth=2.0, label="Model rate")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

        region_name = REGION_NAMES.get(region, f"R{region}")
        r2_val = all_r2[region][neuron]
        ax.set_title(f"{region_name} neuron {neuron} ({label}, R²={r2_val:.3f})",
                     fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel // 3 == 1:
            ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if panel % 3 == 0:
            ax.set_ylabel("Spikes / bin", fontsize=10)
        if panel == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Example neuron reconstructions: observed PSTH vs inferred firing rate",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Shared helpers for decoder figures (2b–2e)
#
# Decoders are trained by POOLING post-onset time bins as individual
# samples.  CV is trial-grouped 5-fold: for each held-out fold the
# decoder's raw output is AVERAGED across the trial's post-onset bins,
# then evaluated — giving one robust prediction per trial with no
# bin-selection bias.
# ──────────────────────────────────────────────────────────────────────

_DISPLAY_FROM_MS = -100.0
_N_CV_FOLDS = 5


def _all_edges(n_regions: int):
    return [(s, t) for s in range(n_regions) for t in range(n_regions) if s != t]


def _onset_and_display(time_main_ms):
    """Return (onset_idx, display_idx) — first post-onset bin and first display bin."""
    onset = int(np.searchsorted(time_main_ms, 0.0))
    display = int(np.searchsorted(time_main_ms, _DISPLAY_FROM_MS))
    return onset, display


def _pooled_category_decoder(X_3d, category, onset_idx):
    """Train category LogReg by pooling post-onset bins; 5-fold trial-level CV.

    Returns (w, cv_acc).
    """
    N, T, D = X_3d.shape
    X_post = X_3d[:, onset_idx:, :]
    T_post = T - onset_idx

    X_pool = X_post.reshape(-1, D)
    y_pool = np.repeat(category, T_post)
    sc = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(sc.fit_transform(X_pool), y_pool)
    w = clf.coef_[0] / sc.scale_

    correct = 0
    skf = StratifiedKFold(n_splits=_N_CV_FOLDS, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(np.arange(N), category):
        X_tr = X_post[train_idx].reshape(-1, D)
        y_tr = np.repeat(category[train_idx], T_post)
        sc_cv = StandardScaler()
        clf_cv = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_cv.fit(sc_cv.fit_transform(X_tr), y_tr)
        for i in test_idx:
            df = clf_cv.decision_function(sc_cv.transform(X_post[i]))
            pred = clf_cv.classes_[1] if df.mean() > 0 else clf_cv.classes_[0]
            correct += int(pred == category[i])

    return w, correct / N


def _pooled_direction_decoder(X_3d, direction_rad, onset_idx):
    """Train cos/sin Ridge by pooling post-onset bins; 5-fold trial-level CV.

    Returns (wc, ws, cv_evidence).
    """
    N, T, D = X_3d.shape
    X_post = X_3d[:, onset_idx:, :]
    T_post = T - onset_idx

    X_pool = X_post.reshape(-1, D)
    y_cos = np.repeat(np.cos(direction_rad), T_post)
    y_sin = np.repeat(np.sin(direction_rad), T_post)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_pool)
    wc = Ridge(alpha=1.0).fit(X_sc, y_cos).coef_ / sc.scale_
    ws = Ridge(alpha=1.0).fit(X_sc, y_sin).coef_ / sc.scale_

    evidences = np.zeros(N)
    kf = KFold(n_splits=_N_CV_FOLDS, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(np.arange(N)):
        X_tr = X_post[train_idx].reshape(-1, D)
        sc_cv = StandardScaler()
        X_sc_cv = sc_cv.fit_transform(X_tr)
        rc = Ridge(alpha=1.0).fit(X_sc_cv, np.repeat(np.cos(direction_rad[train_idx]), T_post))
        rs = Ridge(alpha=1.0).fit(X_sc_cv, np.repeat(np.sin(direction_rad[train_idx]), T_post))
        for i in test_idx:
            X_te = sc_cv.transform(X_post[i])
            mean_pc = rc.predict(X_te).mean()
            mean_ps = rs.predict(X_te).mean()
            evidences[i] = mean_pc * np.cos(direction_rad[i]) + mean_ps * np.sin(direction_rad[i])

    return wc, ws, evidences.mean()


def _pooled_regression(X_msg_3d, y_score_2d, onset_idx):
    """Ridge regression from message to coding score, pooled post-onset.

    X_msg_3d: (N, T, 8), y_score_2d: (N, T).  Returns w_msg (8,).
    """
    X = X_msg_3d[:, onset_idx:, :].reshape(-1, X_msg_3d.shape[-1])
    y = y_score_2d[:, onset_idx:].reshape(-1)
    sc = StandardScaler()
    return Ridge(alpha=1.0).fit(sc.fit_transform(X), y).coef_ / sc.scale_


# ──────────────────────────────────────────────────────────────────────
# 2b / 2c — Message-content figures (independent per-pathway decoders)
# ──────────────────────────────────────────────────────────────────────

def fig_category_decoder_loading(posteriors, raw_npz, n_regions, time_main_ms,
                                 save_path):
    """Fig 2b: category separation in each message (pooled per-pathway decoder)."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    edges = _all_edges(n_regions)
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    cat_pos, cat_neg = category == 1.0, category == -1.0

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    for idx, (src, tgt) in enumerate(edges):
        ax = axes[idx // 3, idx % 3]
        msg = np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])

        w, loo_acc = _pooled_category_decoder(msg, category, onset)
        loading = msg[:, display:, :] @ w
        separation = loading[cat_pos].mean(0) - loading[cat_neg].mean(0)
        sem = np.sqrt((loading[cat_pos].std(0) / np.sqrt(cat_pos.sum())) ** 2 +
                      (loading[cat_neg].std(0) / np.sqrt(cat_neg.sum())) ** 2)

        ax.plot(t, separation, color="#2ca02c", linewidth=1.5)
        ax.fill_between(t, separation - sem, separation + sem,
                        alpha=0.2, color="#2ca02c")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        src_name = REGION_NAMES.get(src, f"R{src}")
        tgt_name = REGION_NAMES.get(tgt, f"R{tgt}")
        ax.set_title(f"{src_name} \u2192 {tgt_name}   CV acc={loo_acc:.0%}",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // 3 == 1:
            ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Category separation (a.u.)", fontsize=10)

    fig.suptitle("Category information in messages\n"
                 "(cat+1 minus cat\u22121 loading, pooled per-pathway decoder, \u00b1 SEM)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_direction_decoder_loading(posteriors, raw_npz, n_regions, time_main_ms,
                                  save_path):
    """Fig 2c: direction evidence in each message (pooled per-pathway decoder)."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    direction_deg = np.asarray(raw_npz["direction"])[trial_idx]
    direction_rad = np.deg2rad(direction_deg)
    edges = _all_edges(n_regions)
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    n_trials = len(direction_rad)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    for idx, (src, tgt) in enumerate(edges):
        ax = axes[idx // 3, idx % 3]
        msg = np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])

        wc, ws, loo_ev = _pooled_direction_decoder(msg, direction_rad, onset)
        pred_cos = msg[:, display:, :] @ wc
        pred_sin = msg[:, display:, :] @ ws
        evidence = (pred_cos * np.cos(direction_rad)[:, None] +
                    pred_sin * np.sin(direction_rad)[:, None])
        m = evidence.mean(axis=0)
        sem = evidence.std(axis=0) / np.sqrt(n_trials)

        ax.plot(t, m, color="#9467bd", linewidth=1.5)
        ax.fill_between(t, m - sem, m + sem, alpha=0.2, color="#9467bd")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        src_name = REGION_NAMES.get(src, f"R{src}")
        tgt_name = REGION_NAMES.get(tgt, f"R{tgt}")
        ax.set_title(f"{src_name} \u2192 {tgt_name}   CV ev={loo_ev:.3f}",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // 3 == 1:
            ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Direction evidence (a.u.)", fontsize=10)

    fig.suptitle("Direction information in messages\n"
                 "(mean correct-direction evidence, pooled per-pathway decoder, \u00b1 SEM)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 2d / 2e — Contribution to target-region coding (factors-based)
# ──────────────────────────────────────────────────────────────────────

def fig_category_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                              save_path):
    """Fig 2d: message contribution to target-region category coding."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    edges = _all_edges(n_regions)
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    cat_pos, cat_neg = category == 1.0, category == -1.0

    fac = {}
    for r in range(n_regions):
        factors = np.asarray(posteriors[f"factors_region{r}"])
        w_fac, loo_acc = _pooled_category_decoder(factors, category, onset)
        score = np.einsum("ntf,f->nt", factors, w_fac)
        fac[r] = (score, loo_acc)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    for idx, (src, tgt) in enumerate(edges):
        ax = axes[idx // 3, idx % 3]
        score, loo_acc = fac[tgt]
        msg = np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])

        w_msg = _pooled_regression(msg, score, onset)
        contrib = msg[:, display:, :] @ w_msg
        separation = contrib[cat_pos].mean(0) - contrib[cat_neg].mean(0)
        sem = np.sqrt((contrib[cat_pos].std(0) / np.sqrt(cat_pos.sum())) ** 2 +
                      (contrib[cat_neg].std(0) / np.sqrt(cat_neg.sum())) ** 2)

        ax.plot(t, separation, color="#e377c2", linewidth=1.5)
        ax.fill_between(t, separation - sem, separation + sem,
                        alpha=0.2, color="#e377c2")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        src_name = REGION_NAMES.get(src, f"R{src}")
        tgt_name = REGION_NAMES.get(tgt, f"R{tgt}")
        ax.set_title(f"{src_name} \u2192 {tgt_name}   {tgt_name} CV acc={loo_acc:.0%}",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // 3 == 1:
            ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Category contribution (a.u.)", fontsize=10)

    fig.suptitle("Message contribution to target-region category coding\n"
                 "(cat+1 minus cat\u22121, bridged through factors decoder, \u00b1 SEM)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_direction_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                               save_path):
    """Fig 2e: message contribution to target-region direction coding."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    direction_deg = np.asarray(raw_npz["direction"])[trial_idx]
    direction_rad = np.deg2rad(direction_deg)
    edges = _all_edges(n_regions)
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    n_trials = len(direction_rad)

    fac = {}
    for r in range(n_regions):
        factors = np.asarray(posteriors[f"factors_region{r}"])
        wc_fac, ws_fac, loo_ev = _pooled_direction_decoder(factors, direction_rad, onset)
        score_cos = np.einsum("ntf,f->nt", factors, wc_fac)
        score_sin = np.einsum("ntf,f->nt", factors, ws_fac)
        fac[r] = (score_cos, score_sin, loo_ev)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    for idx, (src, tgt) in enumerate(edges):
        ax = axes[idx // 3, idx % 3]
        score_cos, score_sin, loo_ev = fac[tgt]
        msg = np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])

        wc_msg = _pooled_regression(msg, score_cos, onset)
        ws_msg = _pooled_regression(msg, score_sin, onset)
        pred_cos = msg[:, display:, :] @ wc_msg
        pred_sin = msg[:, display:, :] @ ws_msg
        evidence = (pred_cos * np.cos(direction_rad)[:, None] +
                    pred_sin * np.sin(direction_rad)[:, None])
        m = evidence.mean(axis=0)
        sem = evidence.std(axis=0) / np.sqrt(n_trials)

        ax.plot(t, m, color="#ff7f0e", linewidth=1.5)
        ax.fill_between(t, m - sem, m + sem, alpha=0.2, color="#ff7f0e")
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        src_name = REGION_NAMES.get(src, f"R{src}")
        tgt_name = REGION_NAMES.get(tgt, f"R{tgt}")
        ax.set_title(f"{src_name} \u2192 {tgt_name}   {tgt_name} CV ev={loo_ev:.3f}",
                     fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // 3 == 1:
            ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Direction contribution (a.u.)", fontsize=10)

    fig.suptitle("Message contribution to target-region direction coding\n"
                 "(mean correct-direction evidence, bridged through factors decoder, \u00b1 SEM)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 5a–5e — Hidden-input figures (one subplot per region)
# ──────────────────────────────────────────────────────────────────────

def fig_hidden_input_timecourses(posteriors, n_regions, time_main_ms, save_path):
    """Fig 5a: Hidden-input norm over time for each region."""
    skip_ms = _DISPLAY_FROM_MS
    start = int(np.searchsorted(time_main_ms, skip_ms))
    t = time_main_ms[start:]

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True, sharey=True)
    if n_regions == 1:
        axes = [axes]

    for r in range(n_regions):
        ax = axes[r]
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])[:, start:, :]
        norms = np.linalg.norm(inp, axis=-1)
        m = norms.mean(0)
        sd = norms.std(0)
        ax.plot(t, m, color=REGION_COLORS[r], linewidth=1.5)
        ax.fill_between(t, m - sd, m + sd, alpha=0.2, color=REGION_COLORS[r])
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(REGION_NAMES.get(r, f"R{r}"), fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if r == 0:
            ax.set_ylabel("Input norm", fontsize=10)

    fig.suptitle("Hidden-input strength over time\n(trial-averaged ± 1 SD)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_hidden_input_category(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 5b: Category separation decoded from hidden inputs per region."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    cat_pos, cat_neg = category == 1.0, category == -1.0

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for r in range(n_regions):
        ax = axes[r]
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])
        w, cv_acc = _pooled_category_decoder(inp, category, onset)
        loading = inp[:, display:, :] @ w
        sep = loading[cat_pos].mean(0) - loading[cat_neg].mean(0)
        sem = np.sqrt((loading[cat_pos].std(0) / np.sqrt(cat_pos.sum())) ** 2 +
                      (loading[cat_neg].std(0) / np.sqrt(cat_neg.sum())) ** 2)
        ax.plot(t, sep, color=REGION_COLORS[r], linewidth=1.5)
        ax.fill_between(t, sep - sem, sep + sem, alpha=0.2, color=REGION_COLORS[r])
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        name = REGION_NAMES.get(r, f"R{r}")
        ax.set_title(f"{name}   CV acc={cv_acc:.0%}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if r == 0:
            ax.set_ylabel("Category separation (a.u.)", fontsize=10)

    fig.suptitle("Category information in hidden inputs\n"
                 "(cat+1 minus cat−1 loading, ± SEM)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_hidden_input_direction(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 5c: Direction evidence decoded from hidden inputs per region."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    direction_deg = np.asarray(raw_npz["direction"])[trial_idx]
    direction_rad = np.deg2rad(direction_deg)
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    n_trials = len(direction_rad)

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for r in range(n_regions):
        ax = axes[r]
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])
        wc, ws, cv_ev = _pooled_direction_decoder(inp, direction_rad, onset)
        pred_cos = inp[:, display:, :] @ wc
        pred_sin = inp[:, display:, :] @ ws
        evidence = (pred_cos * np.cos(direction_rad)[:, None] +
                    pred_sin * np.sin(direction_rad)[:, None])
        m = evidence.mean(0)
        sem = evidence.std(0) / np.sqrt(n_trials)
        ax.plot(t, m, color=REGION_COLORS[r], linewidth=1.5)
        ax.fill_between(t, m - sem, m + sem, alpha=0.2, color=REGION_COLORS[r])
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        name = REGION_NAMES.get(r, f"R{r}")
        ax.set_title(f"{name}   CV ev={cv_ev:.3f}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if r == 0:
            ax.set_ylabel("Direction evidence (a.u.)", fontsize=10)

    fig.suptitle("Direction information in hidden inputs\n"
                 "(mean correct-direction evidence, ± SEM)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_hidden_input_category_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                                           save_path):
    """Fig 5d: Hidden-input contribution to region's own category coding."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    cat_pos, cat_neg = category == 1.0, category == -1.0

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for r in range(n_regions):
        ax = axes[r]
        factors = np.asarray(posteriors[f"factors_region{r}"])
        w_fac, cv_acc = _pooled_category_decoder(factors, category, onset)
        score = np.einsum("ntf,f->nt", factors, w_fac)

        inp = np.asarray(posteriors[f"inferred_input_region{r}"])
        w_inp = _pooled_regression(inp, score, onset)
        contrib = inp[:, display:, :] @ w_inp
        sep = contrib[cat_pos].mean(0) - contrib[cat_neg].mean(0)
        sem = np.sqrt((contrib[cat_pos].std(0) / np.sqrt(cat_pos.sum())) ** 2 +
                      (contrib[cat_neg].std(0) / np.sqrt(cat_neg.sum())) ** 2)
        ax.plot(t, sep, color=REGION_COLORS[r], linewidth=1.5)
        ax.fill_between(t, sep - sem, sep + sem, alpha=0.2, color=REGION_COLORS[r])
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        name = REGION_NAMES.get(r, f"R{r}")
        ax.set_title(f"{name}   factors CV acc={cv_acc:.0%}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if r == 0:
            ax.set_ylabel("Category contribution (a.u.)", fontsize=10)

    fig.suptitle("Hidden-input contribution to region's category coding\n"
                 "(cat+1 minus cat−1, bridged through factors decoder, ± SEM)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_hidden_input_direction_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                                            save_path):
    """Fig 5e: Hidden-input contribution to region's own direction coding."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    direction_deg = np.asarray(raw_npz["direction"])[trial_idx]
    direction_rad = np.deg2rad(direction_deg)
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    n_trials = len(direction_rad)

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for r in range(n_regions):
        ax = axes[r]
        factors = np.asarray(posteriors[f"factors_region{r}"])
        wc_fac, ws_fac, cv_ev = _pooled_direction_decoder(factors, direction_rad, onset)
        score_cos = np.einsum("ntf,f->nt", factors, wc_fac)
        score_sin = np.einsum("ntf,f->nt", factors, ws_fac)

        inp = np.asarray(posteriors[f"inferred_input_region{r}"])
        wc_inp = _pooled_regression(inp, score_cos, onset)
        ws_inp = _pooled_regression(inp, score_sin, onset)
        pred_cos = inp[:, display:, :] @ wc_inp
        pred_sin = inp[:, display:, :] @ ws_inp
        evidence = (pred_cos * np.cos(direction_rad)[:, None] +
                    pred_sin * np.sin(direction_rad)[:, None])
        m = evidence.mean(0)
        sem = evidence.std(0) / np.sqrt(n_trials)
        ax.plot(t, m, color=REGION_COLORS[r], linewidth=1.5)
        ax.fill_between(t, m - sem, m + sem, alpha=0.2, color=REGION_COLORS[r])
        ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
        name = REGION_NAMES.get(r, f"R{r}")
        ax.set_title(f"{name}   factors CV ev={cv_ev:.3f}", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time from onset (ms)", fontsize=10)
        if r == 0:
            ax.set_ylabel("Direction contribution (a.u.)", fontsize=10)

    fig.suptitle("Hidden-input contribution to region's direction coding\n"
                 "(mean correct-direction evidence, bridged through factors decoder, ± SEM)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# 6 — Single-trial RT-sorted analyses
# ──────────────────────────────────────────────────────────────────────

def _get_cat_weights_per_pathway(posteriors, category, onset, n_regions):
    """Return dict (src,tgt)->w for message category decoders."""
    weights = {}
    for s in range(n_regions):
        for t in range(n_regions):
            if s == t:
                continue
            msg = np.asarray(posteriors[f"inferred_message_{s}to{t}"])
            w, _ = _pooled_category_decoder(msg, category, onset)
            weights[(s, t)] = w
    return weights


def _get_hidden_cat_weights(posteriors, category, onset, n_regions):
    """Return dict r->w for hidden-input category decoders."""
    weights = {}
    for r in range(n_regions):
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])
        w, _ = _pooled_category_decoder(inp, category, onset)
        weights[r] = w
    return weights


def _rt_sorted_heatmap_grid(axes_flat, data_3d_list, weights_list, labels,
                            sort_idx, rt_sorted, cat_sorted, t, mode):
    """Shared logic for RT-sorted heatmaps (loading or norm)."""
    sign = np.where(cat_sorted > 0, 1.0, -1.0)
    for pi, (data_3d, w, lbl) in enumerate(zip(data_3d_list, weights_list, labels)):
        ax = axes_flat[pi]
        if mode == "loading":
            vals = (data_3d @ w)[sort_idx] * sign[:, None]
            cmap = "RdBu_r"
        else:
            vals = np.linalg.norm(data_3d, axis=-1)[sort_idx]
            cmap = "viridis"
        vmax = np.percentile(np.abs(vals), 95)
        kwargs = {"vmin": -vmax, "vmax": vmax} if mode == "loading" else {"vmin": 0}
        ax.pcolormesh(t, np.arange(len(sort_idx)), vals,
                      cmap=cmap, shading="auto", **kwargs)
        ax.plot(rt_sorted, np.arange(len(sort_idx)), color="black", linewidth=1.5)
        ax.axvline(0, color="green" if mode == "loading" else "red",
                   linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_ylabel("Trial (fast→slow)", fontsize=8)


def fig_rt_msg_loading(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 6a: RT-sorted message category loading heatmaps."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    rt = np.asarray(raw_npz["rt_ms"])[trial_idx]
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    edges = _all_edges(n_regions)
    msg_weights = _get_cat_weights_per_pathway(posteriors, category, onset, n_regions)
    sort_idx = np.argsort(rt)

    data_list, w_list, labels = [], [], []
    for src, tgt in edges:
        data_list.append(np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])[:, display:, :])
        w_list.append(msg_weights[(src, tgt)])
        labels.append(f"{REGION_NAMES[src]}→{REGION_NAMES[tgt]}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    _rt_sorted_heatmap_grid(axes.flatten(), data_list, w_list, labels,
                            sort_idx, rt[sort_idx], category[sort_idx], t, "loading")
    for ax in axes[1]:
        ax.set_xlabel("Time from onset (ms)", fontsize=9)
    fig.suptitle("RT-sorted message category loading\n"
                 "(sign-flipped: red=correct direction; black line=RT)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_rt_msg_norm(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 6b: RT-sorted message norm heatmaps."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    rt = np.asarray(raw_npz["rt_ms"])[trial_idx]
    category = np.asarray(raw_npz["category"])[trial_idx]
    _, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    edges = _all_edges(n_regions)
    sort_idx = np.argsort(rt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for idx, (src, tgt) in enumerate(edges):
        ax = axes.flatten()[idx]
        msg = np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])[:, display:, :]
        norms = np.linalg.norm(msg, axis=-1)[sort_idx]
        ax.pcolormesh(t, np.arange(len(sort_idx)), norms,
                      cmap="viridis", shading="auto")
        ax.plot(rt[sort_idx], np.arange(len(sort_idx)), color="white", linewidth=1.5)
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"{REGION_NAMES[src]}→{REGION_NAMES[tgt]}",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Trial (fast→slow)", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Time from onset (ms)", fontsize=9)
    fig.suptitle("RT-sorted message norm\n"
                 "(brighter=stronger message; white line=RT)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_rt_input_loading(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 6c: RT-sorted hidden-input category loading heatmaps."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    rt = np.asarray(raw_npz["rt_ms"])[trial_idx]
    onset, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    inp_weights = _get_hidden_cat_weights(posteriors, category, onset, n_regions)
    sort_idx = np.argsort(rt)
    sign = np.where(category[sort_idx] > 0, 1.0, -1.0)

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]
    for r in range(n_regions):
        ax = axes[r]
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])[:, display:, :]
        w = inp_weights[r]
        loading = (inp @ w)[sort_idx] * sign[:, None]
        vmax = np.percentile(np.abs(loading), 95)
        ax.pcolormesh(t, np.arange(len(sort_idx)), loading,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.plot(rt[sort_idx], np.arange(len(sort_idx)), color="black", linewidth=1.5)
        ax.axvline(0, color="green", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"input {REGION_NAMES.get(r, f'R{r}')}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Trial (fast→slow)", fontsize=9)
        ax.set_xlabel("Time from onset (ms)", fontsize=9)
    fig.suptitle("RT-sorted hidden-input category loading\n"
                 "(sign-flipped: red=correct direction; black line=RT)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_rt_input_norm(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 6d: RT-sorted hidden-input norm heatmaps."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    rt = np.asarray(raw_npz["rt_ms"])[trial_idx]
    _, display = _onset_and_display(time_main_ms)
    t = time_main_ms[display:]
    sort_idx = np.argsort(rt)

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4), sharex=True)
    if n_regions == 1:
        axes = [axes]
    for r in range(n_regions):
        ax = axes[r]
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])[:, display:, :]
        norms = np.linalg.norm(inp, axis=-1)[sort_idx]
        ax.pcolormesh(t, np.arange(len(sort_idx)), norms,
                      cmap="viridis", shading="auto")
        ax.plot(rt[sort_idx], np.arange(len(sort_idx)), color="white", linewidth=1.5)
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(f"input {REGION_NAMES.get(r, f'R{r}')}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Trial (fast→slow)", fontsize=9)
        ax.set_xlabel("Time from onset (ms)", fontsize=9)
    fig.suptitle("RT-sorted hidden-input norm\n"
                 "(brighter=stronger input; white line=RT)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def fig_cross_pathway_correlation(posteriors, raw_npz, n_regions, time_main_ms, save_path):
    """Fig 6e: Trial-to-trial correlation of category loading between pairs of
    pathways / hidden inputs, averaged over post-onset time bins."""
    trial_idx = np.asarray(posteriors["trial_indices"])
    category = np.asarray(raw_npz["category"])[trial_idx]
    onset, _ = _onset_and_display(time_main_ms)
    edges = _all_edges(n_regions)
    sign = np.where(category > 0, 1.0, -1.0)

    msg_weights = _get_cat_weights_per_pathway(posteriors, category, onset, n_regions)
    inp_weights = _get_hidden_cat_weights(posteriors, category, onset, n_regions)

    labels = []
    post_onset_loadings = []

    for src, tgt in edges:
        msg = np.asarray(posteriors[f"inferred_message_{src}to{tgt}"])
        w = msg_weights[(src, tgt)]
        loading = (msg[:, onset:, :] @ w) * sign[:, None]
        post_onset_loadings.append(loading.mean(axis=1))
        labels.append(f"m:{REGION_NAMES[src]}→{REGION_NAMES[tgt]}")

    for r in range(n_regions):
        inp = np.asarray(posteriors[f"inferred_input_region{r}"])
        w = inp_weights[r]
        loading = (inp[:, onset:, :] @ w) * sign[:, None]
        post_onset_loadings.append(loading.mean(axis=1))
        labels.append(f"i:{REGION_NAMES.get(r, f'R{r}')}")

    loadings = np.column_stack(post_onset_loadings)
    corr = np.corrcoef(loadings.T)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.max(np.abs(corr - np.eye(n)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr[i,j]) > 0.4 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("Pearson r", fontsize=10)
    ax.set_title("Cross-pathway trial-to-trial correlation\n"
                 "(post-onset avg category loading, sign-flipped)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MR-LFADS results")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--data", type=str, default=None, help="Path to .npz dataset (for history period)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: run_dir/figures)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = Path(args.data) if args.data else None
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    posteriors, effectome, config, tau_bins, raw_npz, time_hist_ms, time_main_ms = \
        load_results(run_dir, data_path)
    n_regions = effectome.shape[0]

    fig_effectome(effectome, n_regions, out_dir / "1_effectome.png")
    fig_message_timecourses(posteriors, n_regions, time_main_ms,
                            out_dir / "2_message_timecourses.png")
    all_r2 = fig_recon_quality(posteriors, n_regions, time_main_ms,
                               out_dir / "3_recon_r2.png")
    fig_example_neurons(posteriors, all_r2, n_regions, time_hist_ms, time_main_ms,
                        raw_npz, tau_bins, out_dir / "4_example_neurons.png")

    if raw_npz is not None and "category" in raw_npz and "direction" in raw_npz:
        fig_category_decoder_loading(posteriors, raw_npz, n_regions, time_main_ms,
                                     out_dir / "2b_category_decoder_loading.png")
        fig_direction_decoder_loading(posteriors, raw_npz, n_regions, time_main_ms,
                                      out_dir / "2c_direction_decoder_loading.png")
        fig_category_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                                  out_dir / "2d_category_contribution.png")
        fig_direction_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                                   out_dir / "2e_direction_contribution.png")

    fig_hidden_input_timecourses(posteriors, n_regions, time_main_ms,
                                 out_dir / "5a_hidden_input_timecourses.png")

    if raw_npz is not None and "category" in raw_npz and "direction" in raw_npz:
        fig_hidden_input_category(posteriors, raw_npz, n_regions, time_main_ms,
                                  out_dir / "5b_hidden_input_category.png")
        fig_hidden_input_direction(posteriors, raw_npz, n_regions, time_main_ms,
                                   out_dir / "5c_hidden_input_direction.png")
        fig_hidden_input_category_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                                               out_dir / "5d_hidden_input_cat_contribution.png")
        fig_hidden_input_direction_contribution(posteriors, raw_npz, n_regions, time_main_ms,
                                                out_dir / "5e_hidden_input_dir_contribution.png")

    if raw_npz is not None and "category" in raw_npz and "rt_ms" in raw_npz:
        fig_rt_msg_loading(posteriors, raw_npz, n_regions, time_main_ms,
                           out_dir / "6a_rt_msg_loading.png")
        fig_rt_msg_norm(posteriors, raw_npz, n_regions, time_main_ms,
                        out_dir / "6b_rt_msg_norm.png")
        fig_rt_input_loading(posteriors, raw_npz, n_regions, time_main_ms,
                             out_dir / "6c_rt_input_loading.png")
        fig_rt_input_norm(posteriors, raw_npz, n_regions, time_main_ms,
                          out_dir / "6d_rt_input_norm.png")
        fig_cross_pathway_correlation(posteriors, raw_npz, n_regions, time_main_ms,
                                      out_dir / "6e_cross_pathway_corr.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
