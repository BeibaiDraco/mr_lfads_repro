#!/usr/bin/env python3
"""Combine per-region .npz files for a single session into a unified dataset.

Filters for correct trials with RT > 150ms.
Region mapping: FEF→region0, LIP→region1, SC→region2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REGION_ORDER = ["FEF", "LIP", "SC"]
RT_THRESHOLD_MS = 150.0


def find_region_files(data_dir: Path, date: str):
    """Find the 3 region files for a given session date, auto-detecting M/S prefix."""
    files = {}
    for region in REGION_ORDER:
        candidates = list(data_dir.glob(f"rct02_{date}_?{region}_vert_all.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 file for {region} on {date}, found {len(candidates)}: {candidates}"
            )
        files[region] = candidates[0]
    return files


def main():
    parser = argparse.ArgumentParser(description="Prepare combined session dataset")
    parser.add_argument("--data_dir", type=str, default="data/real_data",
                        help="Directory containing per-region .npz files")
    parser.add_argument("--date", type=str, required=True,
                        help="Session date, e.g. 20200929")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path (default: data/rct_{date}.npz)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    date = args.date
    output = Path(args.output) if args.output else data_dir.parent / f"rct_{date}.npz"

    region_files = find_region_files(data_dir, date)
    print(f"Session {date}:")
    for region, path in region_files.items():
        print(f"  {region}: {path.name}")

    ref = np.load(region_files["FEF"], allow_pickle=True)
    is_correct = ref["is_correct"].astype(bool)
    rt_ms = ref["rt_ms"]
    valid_mask = is_correct & (rt_ms > RT_THRESHOLD_MS)
    n_valid = valid_mask.sum()
    print(f"  Total trials: {len(is_correct)}, correct & RT>{RT_THRESHOLD_MS}ms: {n_valid}")

    if n_valid == 0:
        raise RuntimeError(f"No valid trials for session {date}!")

    save_dict = {
        "tau": 10,
        "output_type": "poisson",
        "time_bins_s": np.asarray(ref["time_bins_s"]),
        "category": np.asarray(ref["category"])[valid_mask],
        "direction": np.asarray(ref["direction"])[valid_mask],
        "rt_ms": rt_ms[valid_mask],
        "saccade_sign": np.asarray(ref["saccade_sign"])[valid_mask],
    }

    for idx, region in enumerate(REGION_ORDER):
        d = np.load(region_files[region], allow_pickle=True)
        spikes = np.asarray(d["spikes"], dtype=np.float32)[valid_mask]
        save_dict[f"region{idx}"] = spikes
        print(f"  region{idx} ({region}): {spikes.shape}")

    np.savez(output, **save_dict)
    print(f"Saved: {output}  ({n_valid} trials)")


if __name__ == "__main__":
    main()
