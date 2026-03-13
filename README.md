# MR-LFADS reproduction

This is a paper-faithful reimplementation of **MR-LFADS** from:

> Belle Liu, Jacob Sacks, Matthew D. Golub. *Accurate Identification of Communication Between Multiple Interacting Neural Populations*. ICML 2025.

It is built from the `lfads-torch` (https://github.com/arsedler9/lfads-torch).

## What is implemented

The code matches the paper's core design choices:

- **One LFADS-like module per recorded region**.
- **Direct posterior over the generator initial state** `g0` for each region.
- **Causal inferred inputs** `u_t` from a unidirectional encoder + controller using `[e_t ; f_{t-1}]`.
- **Communication posteriors** `q(m^{j→i}_t | r^j_t)` with separate affine maps for each source-target pair.
- **Structured KL bottleneck** with separate penalties for `g0`, `u`, and `m`, and the paper's staged KL schedule.
- **Gaussian observations** for synthetic data and **Poisson observations** for spike counts.
- **Factor size set equal to neuron count**, as described in Appendix A.1 / Table S1.
- **Light L2 regularization on recurrent GRU weights**.

## One ambiguity in the paper

The public paper does **not** specify how it resolves the within-timestep cycle created by:

- Eq. 2: `g_t` depends on messages `m_t`
- Eq. 7-8: `m_t` is inferred from source-region `r_t`
- `r_t` depends on `g_t`

Because the official MR-LFADS code is not public, this reimplementation exposes two options:

- `message_timing = "lagged"` (default): messages emitted at `t` are consumed at `t+1`
- `message_timing = "fixed_point"`: a small fixed-point loop approximates the instantaneous coupling implied by the equations

Everything else is directly matched to the paper.

## Package layout

- `mr_lfads/model.py` – MR-LFADS architecture
- `mr_lfads/train.py` – training loop and checkpointing
- `mr_lfads/synthetic.py` – synthetic task-trained multi-region data generator
- `mr_lfads/evaluate.py` – effectome and message-content metrics
- `scripts/generate_memory_dataset.py` – generate a synthetic dataset
- `scripts/train_mr_lfads.py` – train MR-LFADS on a `.npz` dataset
- `configs/memory_ring_exp1.json` – paper-like defaults for the synthetic memory dataset

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

### 1) Generate a synthetic dataset

Paper-like 3-region memory network:

```bash
python scripts/generate_memory_dataset.py \
  --out data/memory_ring_exp1.npz \
  --stim_dims 2,3,4 \
  --hidden_size 64 \
  --n_trials 1024 \
  --train_epochs 300
```

### 2) Train MR-LFADS

```bash
python scripts/train_mr_lfads.py \
  --data data/memory_ring_exp1.npz \
  --save_dir runs/memory_ring_exp1 \
  --config_json configs/memory_ring_exp1.json
```

Outputs written to `save_dir`:

- `best_model.pt`
- `last_model.pt`
- `history.json`
- `metrics.json`
- `config_resolved.json`
- `posterior_means_valid.npz`
- `effectome_valid.npy`

## Dataset format

The training script expects a `.npz` file with one array per region:

- `region0`, `region1`, ... or `spikes_region0`, `spikes_region1`, ...
- each array must have shape `[trials, time, neurons]`

Optional keys used for evaluation:

- `tau` – history length used for `q(g0 | x_{-tau:0})`
- `output_type` – `gaussian` or `poisson`
- `gt_input_region{i}` – ground-truth unobserved input targets aligned to the modeled segment
- `gt_messages_{src}to{tgt}` – ground-truth messages aligned to the modeled segment

## Paper defaults included here

The provided config uses the paper defaults where they were explicit:

- `tau = 10`
- modeled segment length `T = 190` (for a total trial length of 200)
- `gen_dim = 2 * n_neurons` (auto-resolved from the dataset)
- `fac_dim = n_neurons`
- `input_dim = 4` and `message_dim = 4` for the memory experiment
- `beta_u = 0.1`, `beta_m = 0.01`
- `beta_g0 = beta_u`
- input/g0 KL starts at epoch 50 and ramps for 200 epochs
- message KL starts at epoch 150 and ramps for 100 epochs
- `l2_alpha = 1e-4`
- `max_epochs = 350`
- initial learning rate `0.004`

## Notes on exactness

Because the paper did not release code, this is a **faithful reimplementation**, not a byte-for-byte copy of the authors' training code. The main things that the paper leaves unspecified are:

- the exact resolution of the within-timestep message cycle
- some hidden dimensions for encoders/controllers
- some optimizer/dropout details not listed in the appendix


