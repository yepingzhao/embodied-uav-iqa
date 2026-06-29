#!/usr/bin/env python3
"""M1: C2 Correlation validation — VLM scores vs. real AirCopBench annotations.

Computes SRCC/PLCC between VLM-predicted scores and real AirCopBench human
annotations.  Uses description-comparison scoring (paper-consistent methodology):
VLM generates scene descriptions for both reference and distorted images,
then compares them via BLEU/ROUGE-L/CIDEr.

Usage:
    python scripts/validate_synth_real_correlation.py \
        --manifest-dir data/processed \
        --aircopbench-dir data/raw/AirCopBench \
        --ref-dir data/processed/ref_images \
        --scorer-model Qwen2-VL --scorer-device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np

from uav_iqa.annotations import build_ref_score_lookup
from uav_iqa.metrics import (
    compute_plcc,
    compute_rmse,
    compute_srcc,
    per_distortion_category_metrics,
)
from uav_iqa.utils import load_flat_samples, setup_logging

_log = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="C2: VLM-Predicted-vs-Real correlation"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Processed output directory containing train/ test/ with grouped JSONs",
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
    parser.add_argument(
        "--data-root",
        default="data/processed/distorted",
        help="Root directory for resolving image paths",
    )
    parser.add_argument(
        "--scorer-model",
        required=True,
        help="VLM model for direct scoring (e.g. Qwen2-VL).",
    )
    parser.add_argument(
        "--scorer-backend",
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="VLM inference backend (default: auto)",
    )
    parser.add_argument(
        "--scorer-device",
        default="cuda",
        help="Device for VLM inference (default: cuda)",
    )
    parser.add_argument(
        "--ref-dir",
        default="data/processed/ref_images",
        help="Reference images directory for comparison scoring",
    )
    parser.add_argument(
        "--scorer-batch-size",
        type=int,
        default=8,
        help="Batch size for VLM scoring (default: 8)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_flat_samples(Path(args.output_dir), args.split)
    _log.info("Loaded %d entries from %s/%s", len(entries), args.output_dir, args.split)

    ref_lookup = build_ref_score_lookup(Path(args.aircopbench_dir))

    from uav_iqa.vlm import VLMScorer

    scorer = VLMScorer(
        model_name=args.scorer_model,
        backend=args.scorer_backend,
        device=args.scorer_device,
    )

    data_root = Path(args.data_root)
    ref_dir = Path(args.ref_dir)
    image_paths = [str(data_root / entry.get("path", "")) for entry in entries]
    task_labels = [entry.get("task", "tracking") for entry in entries]

    ref_paths = []
    for entry in entries:
        ref_rel = entry.get("ref_path", "")
        ref_id = entry.get("ref_id", "")
        if ref_rel:
            ref_paths.append(str(ref_dir / Path(ref_rel).name))
        elif ref_id:
            ref_paths.append(str(ref_dir / f"{ref_id}.png"))
        else:
            ref_paths.append(str(ref_dir / "unknown.png"))

    _log.info("VLM comparison scoring %d pairs...", len(image_paths))
    results = scorer.score_batch_comparison(
        ref_paths,
        image_paths,
        task_labels,
        show_progress=True,
    )
    predicted_scores = [r.get("cognitive_score", 0.5) for r in results]

    predicted_list = []
    real_list = []
    distortion_labels = []

    for entry, pred in zip(entries, predicted_scores):
        ref_id = entry.get("ref_id", "")
        if ref_id in ref_lookup:
            real = ref_lookup[ref_id]["execution_score"]
        else:
            continue

        predicted_list.append(pred)
        real_list.append(real)
        distortion_labels.append(entry.get("distortion", "none"))

    pred_arr = np.array(predicted_list)
    real_arr = np.array(real_list)

    _log.info("Pairs with real annotations: %d", len(pred_arr))

    srcc = compute_srcc(pred_arr, real_arr)
    plcc = compute_plcc(pred_arr, real_arr)
    rmse = compute_rmse(pred_arr, real_arr)

    _log.info("=" * 50)
    _log.info("C2: VLM-Predicted-vs-Real Correlation")
    _log.info("=" * 50)
    _log.info("  SRCC:  %.4f", srcc)
    _log.info("  PLCC:  %.4f", plcc)
    _log.info("  RMSE:  %.4f", rmse)

    per_cat = per_distortion_category_metrics(pred_arr, real_arr, distortion_labels)
    _log.info("  By distortion category:")
    for cat, m in per_cat.items():
        _log.info(
            "    %s (n=%d): srcc=%.4f plcc=%.4f",
            cat,
            m.get("n", 0),
            m.get("srcc", 0),
            m.get("plcc", 0),
        )

    results_out = {
        "split": args.split,
        "mode": "VLM-comparison",
        "n_samples": int(len(pred_arr)),
        "srcc": float(srcc),
        "plcc": float(plcc),
        "rmse": float(rmse),
        "per_distortion_category": per_cat,
    }

    with open(output_dir / "c2_correlation.json", "w") as f:
        json.dump(results_out, f, indent=2)
    _log.info("Results saved to %s", output_dir / "c2_correlation.json")


if __name__ == "__main__":
    main()
