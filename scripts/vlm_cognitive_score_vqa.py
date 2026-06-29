#!/usr/bin/env python3
"""Compute VLM cognitive quality scores using VQA paradigm with ground truth.

For each VQA quadruplet (ref_img, dist_img, question, gt_answer):
1. Call VLM on clean reference image  → answer R
2. Call VLM on distorted image        → answer D
3. Compute two independent score sets:

   指标1 (GT-normalized):
     BLEU(D,GT)/BLEU(R,GT),  ROUGE(D,GT)/ROUGE(R,GT),  CIDEr(D,GT)/CIDEr(R,GT)
     Ratios clamped [0, 1].  Anchors by the VLM's baseline on the clean reference.

   指标2 (Embodied-IQA style):
     BLEU(D,R),  ROUGE(D,R),  CIDEr(D,R)
     Direct comparison, no GT needed.

Outputs are saved as two separate JSON files.

Input manifest format (JSON array):
  [{
    "ref_img":   "path/to/ref_clean.png",
    "dist_img":  "path/to/distorted.png",
    "question":  "What objects are visible in the scene?",
    "gt_answer": "A red car, two pedestrians, and a traffic light.",
    "task":      "tracking"   // optional, defaults to "vqa"
  }, ...]

Examples:
    # Score a manifest of VQA triplets
    python scripts/vlm_cognitive_score_vqa.py \\
        --manifest data/vqa_triplets.json \\
        --model Qwen2.5-VL

    # Specify GPU and backend
    python scripts/vlm_cognitive_score_vqa.py \\
        --manifest data/vqa_triplets.json \\
        --model Qwen2.5-VL \\
        --backend transformers \\
        --device cuda:0

    # Limit to a subset for testing
    python scripts/vlm_cognitive_score_vqa.py \\
        --manifest data/vqa_triplets.json \\
        --max-items 100

    # Custom output prefix
    python scripts/vlm_cognitive_score_vqa.py \\
        --manifest data/vqa_triplets.json \\
        --output-prefix results/run01
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from uav_iqa.vlm.scorer import VLMScorer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_log = logging.getLogger(__name__)


def _resolve_model_name(raw: str) -> str:
    """Resolve short names like 'Qwen2.5-VL' to full HF model IDs."""
    mapping = {
        "Qwen2.5-VL": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2-VL": "Qwen/Qwen2-VL-7B-Instruct",
        "InternVL2": "OpenGVLab/InternVL2-8B",
        "InternVL2.5": "OpenGVLab/InternVL2.5-8B",
    }
    return mapping.get(raw, raw)


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON array, got {type(data).__name__}")
    return data


def validate_entry(entry: Dict[str, Any], idx: int) -> List[str]:
    errors: List[str] = []
    for key in ("ref_img", "dist_img", "question", "gt_answer"):
        if key not in entry:
            errors.append(f"Entry {idx}: missing required field '{key}'")
        elif not isinstance(entry[key], str) or not entry[key].strip():
            errors.append(f"Entry {idx}: field '{key}' is empty")
    ref = entry.get("ref_img", "")
    dist = entry.get("dist_img", "")
    for label, fp in (("ref_img", ref), ("dist_img", dist)):
        if fp and not os.path.exists(fp):
            errors.append(f"Entry {idx}: {label} not found: {fp}")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Compute VLM cognitive quality scores (指标1 + 指标2) via VQA"
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to VQA triplets JSON manifest."
    )
    parser.add_argument(
        "--model",
        default="Qwen2.5-VL",
        help="VLM model name: short name (Qwen2.5-VL) or full HF ID.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "vllm", "transformers", "none"],
        help="Inference backend (default: auto).",
    )
    parser.add_argument("--device", default="cuda", help="Device for model loading.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens per VLM answer (default: 100).",
    )
    parser.add_argument(
        "--max-items", type=int, default=0, help="Limit to first N entries (0 = all)."
    )
    parser.add_argument(
        "--output-prefix",
        default="vqa_cognitive_scores",
        help="Output directory prefix. Writes {prefix}_metric1.json and {prefix}_metric2.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip entries that already exist in output files (resume).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = load_manifest(str(manifest_path))
    _log.info("Loaded %d entries from %s", len(entries), manifest_path)

    # Validate
    all_errors: List[str] = []
    for i, entry in enumerate(entries):
        all_errors.extend(validate_entry(entry, i))
    if all_errors:
        for err in all_errors[:20]:
            _log.warning(err)
        if len(all_errors) > 20:
            _log.warning("... and %d more validation issues", len(all_errors) - 20)

    # Truncate
    if args.max_items > 0:
        entries = entries[: args.max_items]
        _log.info("Truncated to %d entries", len(entries))

    # Output paths
    out1 = Path(f"{args.output_prefix}_metric1.json")
    out2 = Path(f"{args.output_prefix}_metric2.json")

    # Resume
    existing_refs: set = set()
    if args.skip_existing:
        for out_path in (out1, out2):
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text())
                    if isinstance(existing, list):
                        for item in existing:
                            rid = f"{item.get('ref_path', '')}|{item.get('dist_path', '')}"
                            existing_refs.add(rid)
                except (json.JSONDecodeError, OSError):
                    pass
        if existing_refs:
            _log.info(
                "Found %d existing results, will skip matching entries",
                len(existing_refs),
            )

    # Load VLM
    model_name = _resolve_model_name(args.model)
    _log.info(
        "Loading VLM: %s (backend=%s, device=%s)", model_name, args.backend, args.device
    )
    scorer = VLMScorer(
        model_name=model_name,
        backend=args.backend,
        device=args.device,
        seed=args.seed,
    )

    # Score
    results_metric1: List[dict] = []
    results_metric2: List[dict] = []

    for idx, entry in enumerate(tqdm(entries, desc="VQA scoring", unit="item")):
        ref_path = str(entry["ref_img"])
        dist_path = str(entry["dist_img"])

        # Skip if already scored
        if existing_refs:
            rid = f"{ref_path}|{dist_path}"
            if rid in existing_refs:
                _log.debug("Skipping already-scored: %s", rid)
                continue

        task = entry.get("task", "vqa")
        result = scorer.score_vqa_with_gt(
            ref_image_path=ref_path,
            dist_image_path=dist_path,
            question=entry["question"],
            gt_answer=entry["gt_answer"],
            task=task,
            max_tokens=args.max_tokens,
        )

        # Split into separate records
        meta = result["metadata"]
        common = {
            "ref_path": meta["ref_path"],
            "dist_path": meta["dist_path"],
            "task": meta["task"],
            "question": meta["question"],
            "gt_answer": meta["gt_answer"],
            "ref_answer": meta["ref_answer"],
            "dist_answer": meta["dist_answer"],
            "baseline_failure": meta["baseline_failure"],
        }

        results_metric1.append({**common, **result["metric1_gt_normalized"]})
        results_metric2.append({**common, **result["metric2_ref_vs_dist"]})

    # Write outputs
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)

    with open(out1, "w") as f:
        json.dump(results_metric1, f, indent=2, ensure_ascii=False)
    _log.info("指标1 (GT-normalized) → %s (%d entries)", out1, len(results_metric1))

    with open(out2, "w") as f:
        json.dump(results_metric2, f, indent=2, ensure_ascii=False)
    _log.info("指标2 (Ref-vs-Dist)  → %s (%d entries)", out2, len(results_metric2))

    # Summary
    if results_metric1:
        avg_gt = sum(r.get("cognitive_score", 0) for r in results_metric1) / len(
            results_metric1
        )
        avg_rd = sum(r.get("cognitive_score", 0) for r in results_metric2) / len(
            results_metric2
        )
        n_fail = sum(1 for r in results_metric1 if r.get("baseline_failure"))
        _log.info(
            "Avg 指标1=%.4f  Avg 指标2=%.4f  baseline_failures=%d/%d",
            avg_gt,
            avg_rd,
            n_fail,
            len(results_metric1),
        )


if __name__ == "__main__":
    main()
