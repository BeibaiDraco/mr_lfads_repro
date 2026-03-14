# MR-LFADS

A paper-faithful reimplementation of **Multi-Region Latent Factor Analysis via Dynamical Systems (MR-LFADS)** from:

> Belle Liu, Jacob Sacks, Matthew D. Golub.
> [*Accurate Identification of Communication Between Multiple Interacting Neural Populations.*](https://openreview.net/pdf?id=O14GjxDAt3)
> ICML 2025.

Built on top of [`lfads-torch`](https://github.com/arsedler9/lfads-torch).

If you use this code, please cite the original paper:

```bibtex
@inproceedings{liu2025mrlfads,
  title     = {Accurate Identification of Communication Between Multiple Interacting Neural Populations},
  author    = {Belle Liu and Jacob Sacks and Matthew D. Golub},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

---

## Overview

MR-LFADS learns a generative model of multi-region neural population dynamics, jointly inferring within-region latent factors and between-region communication. This codebase reproduces the architecture, training procedure, and synthetic benchmarks described in the paper.

### What is implemented

| Component | Details |
|---|---|
| **Per-region LFADS modules** | One encoder → controller → generator per recorded region |
| **Direct `g₀` posterior** | Posterior over the generator initial state for each region |
| **Causal inferred inputs** | Unidirectional encoder + controller using `[eₜ ; fₜ₋₁]` |
| **Communication posteriors** | `q(mʲ→ⁱₜ ∣ rʲₜ)` with separate affine maps per source–target pair |
| **Structured KL bottleneck** | Independent penalties for `g₀`, `u`, and `m` with staged ramp schedule |
| **Observation models** | Gaussian (synthetic) and Poisson (spike counts) |
| **Factor dim = neuron count** | Per Appendix A.1 / Table S1 |
| **L2 regularisation** | Applied to recurrent GRU weights |

### A note on message timing

The paper's equations create a within-timestep cycle: `gₜ` depends on messages `mₜ`, which are inferred from `rₜ`, which in turn depends on `gₜ`. Because the official code is not public, this reimplementation exposes two resolutions:

| Mode | Flag | Behaviour |
|---|---|---|
| **Lagged** *(default)* | `"message_timing": "lagged"` | Messages emitted at `t` are consumed at `t + 1` |
| **Fixed-point** | `"message_timing": "fixed_point"` | A small fixed-point loop approximates instantaneous coupling |

See [Choosing fixed-point message timing](#choosing-fixed-point-message-timing) for configuration details.

---

## Installation

```bash
git clone https://github.com/BeibaiDraco/mr_lfads_repro.git
cd mr_lfads_repro
pip install -r requirements.txt
```

## Quick start

### 1. Generate the synthetic dataset

Trains a small data-generating network and samples trials. Only needs to be run once.

```bash
python scripts/generate_memory_dataset.py \
  --out data/memory_ring_exp1.npz \
  --stim_dims 2,3,4 \
  --hidden_size 64 \
  --n_trials 1024 \
  --train_epochs 300
```

### 2. Train MR-LFADS

Pick a configuration that suits your needs:

| Config | Batch size | L2 | Notes |
|---|---|---|---|
| [`memory_ring_full.json`](configs/memory_ring_full.json) | 128 | 0.014 | Main config (stronger L2) |
| [`smoke_test.json`](configs/smoke_test.json) | — | — | Tiny run for CI / sanity checks |

```bash
python scripts/train_mr_lfads.py \
  --data data/memory_ring_exp1.npz \
  --save_dir runs/memory_ring_full \
  --config_json configs/memory_ring_full.json
```

### 3. Outputs

Each run writes the following to `save_dir`:

```
runs/<experiment>/
├── best_model.pt
├── last_model.pt
├── history.json
├── metrics.json
├── config_resolved.json
├── posterior_means_valid.npz
└── effectome_valid.npy
```

### Choosing fixed-point message timing

All configs default to `"message_timing": "lagged"`. To switch to the fixed-point variant, set the field in any config JSON:

```json
{
  "message_timing": "fixed_point",
  "fixed_point_iters": 2
}
```

---

## Dataset format

The training script expects a `.npz` file with one array per region, each of shape `(trials, time, neurons)`:

```
region0, region1, ...        # or
spikes_region0, spikes_region1, ...
```

**Optional keys** (used for evaluation):

| Key | Description |
|---|---|
| `tau` | History length for `q(g₀ ∣ x₋τ:₀)` |
| `output_type` | `"gaussian"` or `"poisson"` |
| `gt_input_region{i}` | Ground-truth unobserved inputs (aligned to modelled segment) |
| `gt_messages_{src}to{tgt}` | Ground-truth messages (aligned to modelled segment) |

---

## Hyperparameters

Defaults follow the paper where values were explicitly stated:

| Parameter | Value |
|---|---|
| τ (history length) | 10 |
| Modelled segment length *T* | 190 (total trial = 200) |
| Generator dim | 2 × *n*_neurons (auto-resolved) |
| Factor dim | *n*_neurons |
| Inferred-input / message dim | 4 |
| β_u | 0.1 |
| β_m | 0.01 |
| β_g₀ | β_u |
| Input / g₀ KL ramp | epoch 50 → 250 |
| Message KL ramp | epoch 150 → 250 |
| L2 weight (default) | 1 × 10⁻⁴ |
| Max epochs | 350 |
| Initial learning rate | 0.004 |

---

## Package layout

```
mr_lfads/
├── model.py        # MR-LFADS architecture
├── train.py        # Training loop & checkpointing
├── synthetic.py    # Synthetic multi-region data generator
└── evaluate.py     # Effectome & message-content metrics

scripts/
├── generate_memory_dataset.py
└── train_mr_lfads.py

configs/
├── memory_ring_full.json
└── smoke_test.json
```

---

## Differences from the original

Because the authors have not released code, this is a **faithful reimplementation**, not a byte-for-byte reproduction. Known underspecified details include:

- Resolution of the within-timestep message cycle (see [above](#a-note-on-message-timing))
- Exact hidden dimensions for encoders and controllers
- Some optimizer and dropout settings not listed in the appendix

---

## License

This project is released under the [MIT License](LICENSE).