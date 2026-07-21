#!/usr/bin/env python3
"""Update target annotation files with multi-model VLM average cognitive scores.

Reads per-model VLM annotations from ``data/annotated/vlm/{model}/{split}/``,
computes the cognitive_score for each entry, then updates:
  - ``vlm_scores``: dict of ``{model_name: cognitive_score}`` for all models
  - ``cognitive_score``: mean of all available per-model cognitive_scores

The cognitive_score formula (matching ``compute_cognitive_score`` in
``src/uav_iqa/text_metrics.py``):
    score = (1.0 * bleu + 1.0 * rouge_l + 0.1 * cider) / 2.1
"""

import json
import os
from pathlib import Path

WEIGHTS = (1.0, 1.0, 0.1)
WEIGHT_SUM = sum(WEIGHTS)

ANNOTATED_DIR = Path(__file__).resolve().parent.parent / "data" / "annotated"
VLM_DIR = ANNOTATED_DIR / "vlm"
SPLITS = ["train", "test"]
DATASETS = ["Real2", "Sim3", "Sim5", "Sim6"]

MODELS = ["Qwen2.5-VL", "Qwen2-VL", "InternVL2", "Mini-InternVL"]


def compute_cognitive_score(bleu: float, rouge_l: float, cider: float) -> float:
    return (WEIGHTS[0] * bleu + WEIGHTS[1] * rouge_l + WEIGHTS[2] * cider) / WEIGHT_SUM


def main():
    for split in SPLITS:
        for dataset in DATASETS:
            filename = f"{dataset}_VQA_{split}.json"
            target_path = ANNOTATED_DIR / split / filename

            if not target_path.exists():
                print(f"SKIP: {target_path} not found")
                continue

            # Load target file
            with open(target_path) as f:
                target_entries = json.load(f)
            print(f"\n{'='*60}")
            print(f"Processing: {split}/{filename} ({len(target_entries)} entries)")

            # Build sample_id → index lookup in target
            target_by_id = {e["sample_id"]: i for i, e in enumerate(target_entries)}

            # Collect per-model cognitive scores
            model_scores: dict[str, dict[str, float]] = {
                m: {} for m in MODELS
            }
            for model in MODELS:
                vlm_path = VLM_DIR / model / split / filename
                if not vlm_path.exists():
                    print(f"  WARNING: {vlm_path} not found, skipping model {model}")
                    continue

                with open(vlm_path) as f:
                    vlm_entries = json.load(f)

                found = 0
                missing = 0
                for entry in vlm_entries:
                    sid = entry["sample_id"]
                    score = compute_cognitive_score(
                        entry["bleu"], entry["rouge_l"], entry["cider"]
                    )
                    if sid in target_by_id:
                        model_scores[model][sid] = round(score, 6)
                        found += 1
                    else:
                        missing += 1

                print(f"  {model}: {found} matched, {missing} not in target")

            # Update target entries
            updated = 0
            for i, entry in enumerate(target_entries):
                sid = entry["sample_id"]
                vlm_scores = {}
                for model in MODELS:
                    if sid in model_scores[model]:
                        vlm_scores[model] = model_scores[model][sid]

                if vlm_scores:
                    entry["vlm_scores"] = vlm_scores
                    scores = list(vlm_scores.values())
                    entry["cognitive_score"] = round(sum(scores) / len(scores), 6)
                    updated += 1

            print(f"  Updated: {updated}/{len(target_entries)} entries")

            # Write back
            with open(target_path, "w") as f:
                json.dump(target_entries, f, indent=2, ensure_ascii=False)
            print(f"  Wrote: {target_path}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
