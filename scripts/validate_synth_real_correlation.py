#!/usr/bin/env python3
"""M1: C2 Correlation validation — synthetic score vs. real AirCopBench annotation.

Computes SRCC/PLCC between degradation-model scores and real AirCopBench human
annotations. This validates Claim C2: "Synthetic distortions injected into clean
AirCopBench frames correlate with real UAV-degraded image quality."

Usage:
    python scripts/validate_synth_real_correlation.py \
        --manifest-dir data/processed \
        --aircopbench-dir data/raw/AirCopBench \
        --output-dir outputs/c2_correlation
"""

import argparse
import json
from pathlib import Path

import numpy as np

from uav_iqa.annotations import (
    build_ref_score_lookup,
    compute_synthetic_score,
)
from uav_iqa.metrics import (
    compute_plcc,
    compute_rmse,
    compute_srcc,
    per_distortion_category_metrics,
)
from uav_iqa.utils import load_manifest, setup_logging

_log = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(description="C2: Synthetic-real correlation")
    parser.add_argument(
        "--manifest-dir",
        default="data/processed",
        help="Directory containing train/val/test manifest.json files",
    )
    parser.add_argument(
        "--aircopbench-dir",
        default="data/raw/AirCopBench",
        help="Path to AirCopBench data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/c2_correlation",
        help="Output directory for results",
    )
    parser.add_argument("--split", default="test", help="Which split to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_dir) / args.split / "manifest.json"
    if not manifest_path.exists():
        _log.error("Manifest not found: %s", manifest_path)
        return

    entries = load_manifest(manifest_path)
    _log.info("Loaded %d entries from %s", len(entries), manifest_path)

    ref_lookup = build_ref_score_lookup(Path(args.aircopbench_dir))

    synthetic_scores = []
    real_scores = []
    distortion_labels = []

    for entry in entries:
        ref_id = entry.get("ref_id", "")
        distortion = entry.get("distortion", "none")
        task = entry.get("task", "tracking")
        intensity = entry.get("intensity_level", 0.5)

        synth = compute_synthetic_score(distortion, task, intensity)

        if ref_id in ref_lookup:
            real = ref_lookup[ref_id]["execution_score"]
        else:
            continue

        synthetic_scores.append(synth)
        real_scores.append(real)
        distortion_labels.append(distortion)

    synth_arr = np.array(synthetic_scores)
    real_arr = np.array(real_scores)

    _log.info("Pairs with real annotations: %d", len(synth_arr))

    srcc = compute_srcc(synth_arr, real_arr)
    plcc = compute_plcc(synth_arr, real_arr)
    rmse = compute_rmse(synth_arr, real_arr)

    _log.info("=" * 50)
    _log.info("C2: Synthetic-vs-Real Correlation")
    _log.info("=" * 50)
    _log.info("  SRCC:  %.4f", srcc)
    _log.info("  PLCC:  %.4f", plcc)
    _log.info("  RMSE:  %.4f", rmse)

    per_cat = per_distortion_category_metrics(synth_arr, real_arr, distortion_labels)
    _log.info("  By distortion category:")
    for cat, m in per_cat.items():
        _log.info(
            "    %s (n=%d): srcc=%.4f plcc=%.4f",
            cat,
            m.get("n", 0),
            m.get("srcc", 0),
            m.get("plcc", 0),
        )

    results = {
        "split": args.split,
        "n_samples": int(len(synth_arr)),
        "srcc": float(srcc),
        "plcc": float(plcc),
        "rmse": float(rmse),
        "per_distortion_category": per_cat,
    }

    with open(output_dir / "c2_correlation.json", "w") as f:
        json.dump(results, f, indent=2)
    _log.info("Results saved to %s", output_dir / "c2_correlation.json")


if __name__ == "__main__":
    main()
