#!/usr/bin/env python3
"""Generate the synthetic memory-ring dataset used to smoke-test MR-LFADS.

Examples
--------
Paper-like 3-region memory network:
    python scripts/generate_memory_dataset.py \
        --out data/memory_ring_exp1.npz \
        --stim_dims 2,3,4 \
        --hidden_size 64 \
        --n_trials 1024 \
        --train_epochs 300

A 5-region ring for quick debugging:
    python scripts/generate_memory_dataset.py \
        --out data/memory_ring_5r.npz \
        --stim_dims 2,2,2,2,2 \
        --hidden_size 32 \
        --n_trials 256 \
        --train_epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mr_lfads.synthetic import (
    MemoryRingConfig,
    generate_memory_ring_dataset,
    save_memory_ring_dataset,
    train_memory_ring_dgn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic multi-region memory-ring dataset.")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--stim_dims", default="2,3,4", help="Comma-separated private stimulus dimensions per region.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden units per DGN region.")
    parser.add_argument("--total_steps", type=int, default=200, help="Total steps per trial (paper uses 200).")
    parser.add_argument("--tau", type=int, default=10, help="History steps reserved for g0 inference (paper uses 10).")
    parser.add_argument("--delay", type=int, default=2, help="Communication delay in stimulus steps (paper exp. 1 uses 2).")
    parser.add_argument("--history_window", type=int, default=5, help="How many past steps each DGN region must remember.")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Std of dynamic noise added to DGN hidden states.")
    parser.add_argument("--n_trials", type=int, default=1024, help="Number of trials to generate.")
    parser.add_argument("--train_epochs", type=int, default=300, help="How long to train the DGN before sampling trials.")
    parser.add_argument("--train_batch_size", type=int, default=2048, help="DGN training batch size.")
    parser.add_argument("--train_batches_per_epoch", type=int, default=1, help="Synthetic mini-batches per DGN epoch.")
    parser.add_argument("--train_lr", type=float, default=1e-3, help="DGN training learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="DGN optimizer weight decay.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stim_dims = [int(tok) for tok in args.stim_dims.split(",") if tok.strip()]
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MemoryRingConfig(
        stim_dims=stim_dims,
        hidden_size=args.hidden_size,
        total_steps=args.total_steps,
        tau=args.tau,
        delay=args.delay,
        history_window=args.history_window,
        noise_std=args.noise_std,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        train_batches_per_epoch=args.train_batches_per_epoch,
        train_lr=args.train_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    dgn = train_memory_ring_dgn(cfg, device=device)
    dataset = generate_memory_ring_dataset(dgn, n_trials=args.n_trials, config=cfg, device=device)
    save_memory_ring_dataset(Path(args.out), dataset, cfg)
    print(f"Saved dataset to {args.out}")
    print(f"Saved metadata to {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
