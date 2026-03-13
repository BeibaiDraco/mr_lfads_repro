#!/usr/bin/env python3
"""Train MR-LFADS on a .npz dataset.

The dataset must contain keys like region0, region1, ... (or spikes_region0, ...), each
with shape [trials, time, neurons]. If a scalar key `tau` exists it will be used; otherwise
pass --tau explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from mr_lfads.train import TrainConfig, train_mr_lfads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MR-LFADS on a multi-region dataset.")
    parser.add_argument("--data", required=True, help="Path to the .npz dataset.")
    parser.add_argument("--save_dir", required=True, help="Directory where outputs are written.")
    parser.add_argument("--config_json", default=None, help="Optional JSON file overriding TrainConfig fields.")
    parser.add_argument("--tau", type=int, default=None, help="Override tau if it is not stored in the dataset.")
    parser.add_argument("--output_type", default=None, choices=[None, "gaussian", "poisson"], help="Override observation model.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto.")
    return parser.parse_args()


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    config_dict = load_config(args.config_json)
    cfg = TrainConfig(**config_dict)
    if args.tau is not None:
        cfg.tau = args.tau
    if args.output_type is not None:
        cfg.output_type = args.output_type
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    results = train_mr_lfads(dataset_path=args.data, save_dir=args.save_dir, train_config=cfg)
    print("Training complete.")
    print(f"Outputs written to {args.save_dir}")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
