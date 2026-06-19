#!/usr/bin/env python3
"""Annotate manifest entries with scores derived from AirCopBench annotations.

Reads AirCopBench human Quality/Usability annotations, maps them to reference
images, and computes distorted image scores using a degradation model:
  distorted_score = ref_score × degradation_factor(distortion_type, intensity)

Usage:
    python scripts/annotate_scores.py \
        --manifest-dir data/processed \
        --aircopbench-dir data/raw/AirCopBench \
        --output-dir data/processed
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# AirCopBench annotation parsing
# ---------------------------------------------------------------------------

def parse_quality_score(quality_str: str) -> float:
    """Parse 'Good (4/5)' → 0.8, 'Excellent (5/5)' → 1.0, etc."""
    if quality_str is None:
        return 0.5
    match = re.search(r"\((\d+)(?:\.\d+)?/(\d+)\)", str(quality_str))
    if match:
        return float(match.group(1)) / float(match.group(2))
    return 0.5


def parse_usability(usability_str: str) -> float:
    """Parse usability string into a [0, 1] score."""
    s = str(usability_str)
    if s.startswith("1"):
        return 1.0
    elif s.startswith("2"):
        return 0.5
    elif s.startswith("3"):
        return 0.25
    return 0.0


# ---------------------------------------------------------------------------
# Degradation factors — per-(distortion, task) degradation at max intensity
# ---------------------------------------------------------------------------

DEGRADATION_FACTORS = {
    # UAV-specific distortions
    "propeller_vibration_blur": {"tracking": 0.55, "inspection": 0.40, "delivery": 0.70, "sar": 0.50},
    "atmospheric_scattering_haze": {"tracking": 0.50, "inspection": 0.35, "delivery": 0.65, "sar": 0.45},
    "six_dof_viewpoint_blur": {"tracking": 0.45, "inspection": 0.35, "delivery": 0.65, "sar": 0.50},
    "communication_packet_loss": {"tracking": 0.60, "inspection": 0.45, "delivery": 0.70, "sar": 0.55},
    "low_res_super_resolution": {"tracking": 0.50, "inspection": 0.30, "delivery": 0.65, "sar": 0.50},
    "propeller_shadow": {"tracking": 0.70, "inspection": 0.65, "delivery": 0.80, "sar": 0.75},
    # Generic distortions
    "gaussian_blur": {"tracking": 0.60, "inspection": 0.55, "delivery": 0.75, "sar": 0.65},
    "lens_blur": {"tracking": 0.55, "inspection": 0.50, "delivery": 0.70, "sar": 0.60},
    "motion_blur": {"tracking": 0.50, "inspection": 0.45, "delivery": 0.65, "sar": 0.55},
    "brighten_max": {"tracking": 0.85, "inspection": 0.80, "delivery": 0.90, "sar": 0.85},
    "brighten_min": {"tracking": 0.90, "inspection": 0.85, "delivery": 0.92, "sar": 0.88},
    "brighten_avg": {"tracking": 0.88, "inspection": 0.83, "delivery": 0.90, "sar": 0.87},
    "darken_max": {"tracking": 0.75, "inspection": 0.65, "delivery": 0.82, "sar": 0.70},
    "darken_min": {"tracking": 0.80, "inspection": 0.72, "delivery": 0.85, "sar": 0.78},
    "darken_avg": {"tracking": 0.78, "inspection": 0.70, "delivery": 0.84, "sar": 0.75},
    "color_diffusion": {"tracking": 0.82, "inspection": 0.72, "delivery": 0.88, "sar": 0.80},
    "color_shift": {"tracking": 0.85, "inspection": 0.75, "delivery": 0.90, "sar": 0.82},
    "color_quantize": {"tracking": 0.78, "inspection": 0.68, "delivery": 0.85, "sar": 0.75},
    "white_noise": {"tracking": 0.72, "inspection": 0.60, "delivery": 0.80, "sar": 0.68},
    "color_noise": {"tracking": 0.70, "inspection": 0.58, "delivery": 0.78, "sar": 0.65},
    "impulse_noise": {"tracking": 0.75, "inspection": 0.62, "delivery": 0.82, "sar": 0.70},
    "multiplicative_noise": {"tracking": 0.73, "inspection": 0.60, "delivery": 0.80, "sar": 0.68},
    "jpeg_compression": {"tracking": 0.78, "inspection": 0.70, "delivery": 0.85, "sar": 0.75},
    "jp2k_compression": {"tracking": 0.76, "inspection": 0.68, "delivery": 0.83, "sar": 0.73},
    "webp_compression": {"tracking": 0.78, "inspection": 0.70, "delivery": 0.85, "sar": 0.75},
    "spatial_warp": {"tracking": 0.60, "inspection": 0.50, "delivery": 0.72, "sar": 0.62},
    "spatial_rotation": {"tracking": 0.70, "inspection": 0.60, "delivery": 0.78, "sar": 0.68},
    "spatial_scale": {"tracking": 0.75, "inspection": 0.65, "delivery": 0.82, "sar": 0.72},
    "spatial_shear": {"tracking": 0.68, "inspection": 0.58, "delivery": 0.76, "sar": 0.65},
    "resolution_limit": {"tracking": 0.55, "inspection": 0.40, "delivery": 0.68, "sar": 0.55},
    "grayscale": {"tracking": 0.80, "inspection": 0.65, "delivery": 0.85, "sar": 0.78},
    "sharpness": {"tracking": 0.88, "inspection": 0.85, "delivery": 0.92, "sar": 0.88},
    "contrast": {"tracking": 0.85, "inspection": 0.82, "delivery": 0.90, "sar": 0.85},
    "none": {"tracking": 0.95, "inspection": 0.95, "delivery": 0.95, "sar": 0.95},
}

_DEFAULT_DEG = {"tracking": 0.75, "inspection": 0.68, "delivery": 0.82, "sar": 0.72}


def degradation_factor(distortion: str, task: str) -> float:
    """Get degradation factor at max intensity for (distortion, task)."""
    per_task = DEGRADATION_FACTORS.get(distortion, _DEFAULT_DEG)
    return per_task.get(task, 0.75)


# ---------------------------------------------------------------------------
# Reference score lookup from AirCopBench
# ---------------------------------------------------------------------------

def build_ref_score_lookup(aircopbench_dir: Path, seed: int = 42) -> dict:
    """Build a dict mapping ref_id → {vlm_score, vla_score, execution_score}."""
    lookup = {}

    ann_dirs = list(aircopbench_dir.rglob("Annotations"))
    for ann_dir in ann_dirs:
        for ann_file in sorted(ann_dir.glob("*.json")):
            if "VQA" in ann_file.name:
                continue
            try:
                with open(ann_file) as f:
                    annotations = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if not isinstance(annotations, list):
                continue

            for entry in annotations:
                img1 = entry.get("img1", "")
                fname = os.path.basename(img1)
                ref_id = os.path.splitext(fname)[0]
                quality = parse_quality_score(entry.get("Quality"))
                usability = parse_usability(entry.get("Usibility"))
                combined = 0.4 * quality + 0.6 * usability
                lookup[ref_id] = {
                    "vlm_score": round(quality, 4),
                    "vla_score": round(usability, 4),
                    "execution_score": round(combined, 4),
                    "annotated": True,
                }

    print(f"Built ref score lookup: {len(lookup)} annotated references")
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def annotate_manifest(manifest_path: Path, ref_lookup: dict, rng: np.random.RandomState):
    """Update manifest entries with real or synthetic scores."""
    with open(manifest_path) as f:
        entries = json.load(f)

    annotated = 0
    synthetic = 0

    for entry in entries:
        ref_id = entry.get("ref_id", "")
        distortion = entry.get("distortion", "none")
        task = entry.get("task", "tracking")
        intensity = entry.get("intensity_level", 0.5)

        if ref_id in ref_lookup:
            ref_scores = ref_lookup[ref_id]
            entry["vlm_score"] = ref_scores["vlm_score"]
            entry["vla_score"] = ref_scores["vla_score"]
            entry["execution_score"] = ref_scores["execution_score"]
            entry["annotated"] = True
            annotated += 1
        else:
            hash_int = int(hashlib.md5(ref_id.encode()).hexdigest(), 16)
            ref_vlm = 0.4 + 0.5 * ((hash_int % 1000) / 1000.0)
            ref_vla = 0.3 + 0.5 * (((hash_int // 1000) % 1000) / 1000.0)
            ref_exec = 0.35 + 0.5 * (((hash_int // 1000000) % 1000) / 1000.0)
            entry["vlm_score"] = round(ref_vlm, 4)
            entry["vla_score"] = round(ref_vla, 4)
            entry["execution_score"] = round(ref_exec, 4)
            entry["annotated"] = False
            synthetic += 1

        # Apply degradation model: score = ref_score × [1 - α × intensity]
        deg = degradation_factor(distortion, task)
        alpha = 1.0 - deg
        degrade = alpha * intensity
        noise = rng.normal(0, 0.02 * intensity)

        for key in ("vlm_score", "vla_score", "execution_score"):
            ref_score = entry[key]
            degraded = ref_score * (1.0 - degrade) + noise
            entry[key] = round(max(0.0, min(1.0, degraded)), 4)

    print(f"  {manifest_path.name}: {annotated} annotated, {synthetic} synthetic "
          f"(total {len(entries)})")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Annotate manifest with scores")
    parser.add_argument("--manifest-dir", default="data/processed",
                        help="Directory containing train/val/test manifest.json files")
    parser.add_argument("--aircopbench-dir", default="data/raw/AirCopBench",
                        help="Path to AirCopBench data directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as manifest-dir)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir) if args.output_dir else manifest_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    ref_lookup = build_ref_score_lookup(Path(args.aircopbench_dir), seed=args.seed)

    for split_name in ("train", "val", "test"):
        manifest_path = manifest_dir / split_name / "manifest.json"
        if not manifest_path.exists():
            print(f"  Skip {split_name}: manifest not found at {manifest_path}")
            continue

        entries = annotate_manifest(manifest_path, ref_lookup, rng)

        out_path = output_dir / split_name / "manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)

    # Print score distribution summary
    print("\nScore distribution:")
    for split_name in ("train", "val", "test"):
        manifest_path = output_dir / split_name / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            entries = json.load(f)
        for key in ("vlm_score", "vla_score", "execution_score"):
            scores = [e[key] for e in entries]
            print(f"  {split_name}/{key}: mean={np.mean(scores):.3f}, "
                  f"std={np.std(scores):.3f}, "
                  f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")


if __name__ == "__main__":
    main()
