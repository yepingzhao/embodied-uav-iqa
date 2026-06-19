#!/usr/bin/env python3
"""M1-R006: Manifest generation + train/val/test split.

Generates manifest.json files for train/val/test splits from the distortion
output directory. Each entry maps a distorted image to metadata fields
expected by UAVIQADataset: path, task, distortion, intensity_level,
ref_id, and placeholder scores.

Usage:
    python scripts/run_m1_manifest.py \
        --distorted-dir data/database/distorted \
        --output-dir data/database \
        --split 0.8 0.1 0.1
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


TASK_NAMES = ["tracking", "inspection", "delivery", "sar"]
UAVDistortionPipeline = None  # populated after sys.path setup


def parse_distortion_key(key: str):
    """Parse 'propeller_vibration_blur_L04' → (name, intensity)."""
    parts = key.rsplit("_L", 1)
    if len(parts) == 2:
        return parts[0], int(parts[1]) / 10.0
    return None, None


def gather_samples(distorted_dir: Path):
    samples = []
    png_files = list(distorted_dir.rglob("*.png"))
    if not png_files and distorted_dir.is_dir():
        png_files = list(distorted_dir.glob("*.png"))
    for img_path in sorted(png_files):
        dist_name, intensity = parse_distortion_key(img_path.stem)
        if dist_name is None:
            continue
        rel_path = str(img_path.relative_to(distorted_dir.parent))
        ref_id = img_path.name.rsplit("__", 1)[0]
        task = np.random.choice(TASK_NAMES)
        samples.append({
            "path": rel_path,
            "task": task,
            "distortion": dist_name,
            "intensity_level": round(intensity, 2),
            "ref_id": ref_id,
            "vlm_score": 0.0,
            "vla_score": 0.0,
            "execution_score": 0.0,
            "annotated": False,
        })
    return samples


def split_samples(samples: list, ratios: List[float], seed: int = 42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    n = len(samples)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))

    return {
        "train": [samples[i] for i in indices[:train_end]],
        "val": [samples[i] for i in indices[train_end:val_end]],
        "test": [samples[i] for i in indices[val_end:]],
    }


def main():
    parser = argparse.ArgumentParser(description="M1-R006: Generate manifest + splits")
    parser.add_argument("--distorted-dir", required=True, help="Distorted images directory")
    parser.add_argument("--output-dir", default="data/database", help="Output for manifest files")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="Train/val/test ratios (sum to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    parser.add_argument("--min-per-distortion", type=int, default=5,
                        help="Min samples per distortion per task (for stratification)")
    args = parser.parse_args()

    distorted_dir = Path(args.distorted_dir)
    output_dir = Path(args.output_dir)

    print(f"Scanning {distorted_dir} for distorted images...")
    samples = gather_samples(distorted_dir)
    print(f"Found {len(samples)} samples")

    split_data = split_samples(samples, ratios=args.split, seed=args.seed)

    for split_name, split_data_samples in split_data.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = split_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(split_data_samples, f, indent=2)

        n = len(split_data_samples)
        n_dist = len(set(s["distortion"] for s in split_data_samples))
        n_ref = len(set(s["ref_id"] for s in split_data_samples))
        print(f"  {split_name}: {n} samples, {n_dist} distortions, {n_ref} references")
        print(f"    saved to {manifest_path}")

    stats = {
        "total": len(samples),
        "train": len(split_data["train"]),
        "val": len(split_data["val"]),
        "test": len(split_data["test"]),
        "split_ratios": args.split,
        "seed": args.seed,
    }
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSplit stats saved to {stats_path}")


if __name__ == "__main__":
    main()
