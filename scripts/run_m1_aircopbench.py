#!/usr/bin/env python3
"""
M1 AirCopBench-specific pipeline:
1. Build image index from AirCopBench data directory
2. Read human annotations from JSON files (Quality, Usibility, Degradation)
3. Run distortion injection on all images (24 types × 5 levels)
4. Generate manifest.json with real quality scores from AirCopBench annotations

Usage:
    python scripts/run_m1_aircopbench.py \
        --data-dir data/AirCopBench \
        --output-dir outputs/m1_aircopbench \
        --dry-run --workers 4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.uav_iqa.distortion import UAVDistortionPipeline


def parse_quality_score(quality_str):
    """Parse 'Good (4/5)' → 0.8, 'Excellent (5/5)' → 1.0, etc."""
    match = re.search(r'\((\d+)(?:\.\d+)?/(\d+)\)', str(quality_str))
    if match:
        return float(match.group(1)) / float(match.group(2))
    return 0.5  # default


def parse_usability(usability_str):
    """Parse '1 (Available)' → 1.0, '2 (Partially available)' → 0.5, etc."""
    s = str(usability_str)
    if s.startswith('1'):
        return 1.0
    elif s.startswith('2'):
        return 0.5
    elif s.startswith('3'):
        return 0.25
    return 0.0


def get_degradation_types(entry):
    """Extract degradation category string(s) from an annotation entry."""
    degs = []
    # Sim annotations: Degradation field is a string
    deg_field = entry.get("Degradation", "")
    if isinstance(deg_field, dict):
        degs.extend(deg_field.get("choices", []))
    elif isinstance(deg_field, str) and deg_field.strip():
        degs.append(deg_field)

    # Real annotations: PerceptionIssues list
    for issue in entry.get("PerceptionIssues", []):
        labels = issue.get("rectanglelabels", [])
        degs.extend(labels)

    # Other Degradation field (sim)
    other = entry.get("Other Degradation", "")
    if other and other.strip():
        degs.append(other)

    if not degs:
        degs.append("none")
    return degs


def _extract_scene_and_frame(basename):
    """Extract (scene, frame) from annotation basenames.

    Sim3 annotations:  '8f2a9605-scene_004-UAV1_frame_005.jpg' → ('scene_004', 'UAV1_frame_005.jpg')
    Real2 annotations: '23-00000001-UAV1.jpg' → (None, '23-00000001-UAV1.jpg')
    """
    m = re.search(r'(scene_\d+)-(UAV\d+_frame_\d+\.jpg)', basename)
    if m:
        return m.group(1), m.group(2)
    return None, basename  # Real data: no scene prefix


def build_image_index(data_dir):
    """Scan AirCopBench data directory, build image → annotation mapping."""
    data_dir = Path(data_dir)
    image_index = {}

    for img_path in data_dir.rglob("*.jpg"):
        if '.cache' in str(img_path) or 'distorted' in str(img_path):
            continue
        rel = str(img_path.relative_to(data_dir))
        image_index[str(img_path)] = {"rel_path": rel, "annotation": None}

    annotation_files = list(data_dir.rglob("Annotations/*.json"))
    print(f"Found {len(image_index)} images, {len(annotation_files)} annotation files")

    ann_lookup = {}
    for ann_file in annotation_files:
        with open(ann_file) as f:
            annotations = json.load(f)
        for entry in annotations:
            img1 = entry.get("img1", "")
            scene, frame = _extract_scene_and_frame(os.path.basename(img1))
            key = (scene, frame) if scene else (None, frame)
            ann_lookup[key] = entry

    matched = 0
    for img_path_str, info in image_index.items():
        fname = os.path.basename(img_path_str)
        # Extract scene from relative path (e.g., Sim_3_UAVs/Samples/images/scene_004/UAV2/frame.jpg)
        rel = info["rel_path"]
        scene_m = re.search(r'(scene_\d+)', rel)
        scene = scene_m.group(1) if scene_m else None
        # Try scene-specific match first, then unscoped
        key = (scene, fname) if scene else (None, fname)
        entry = ann_lookup.get(key)
        if not entry and scene:
            entry = ann_lookup.get((None, fname))
        if entry:
            info["annotation"] = entry
            matched += 1

    print(f"Matched {matched}/{len(image_index)} images to annotations")
    return image_index


def assign_task_label(entry, img_rel_path):
    """Assign task category — randomized with seed for reproducibility."""
    import hashlib

    # Use image path hash for deterministic assignment per image
    img_name = os.path.basename(img_rel_path)
    hash_int = int(hashlib.md5(img_name.encode()).hexdigest(), 16)
    tasks = ["tracking", "inspection", "delivery", "sar"]
    return tasks[hash_int % 4]


def main():
    parser = argparse.ArgumentParser(description="M1 AirCopBench pipeline")
    parser.add_argument("--data-dir", default="data/AirCopBench",
                        help="Path to AirCopBench data directory")
    parser.add_argument("--output-dir", default="outputs/m1_aircopbench",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 5 images for verification")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for distortion injection")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--compress", action="store_true", default=True,
                        help="Save distorted images as compressed PNG")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build image index
    print("=== Step 1: Building image index ===")
    image_index = build_image_index(args.data_dir)

    if args.dry_run:
        import random
        random.seed(args.seed)
        # Prefer annotated images for dry-run verification
        annotated = {k: v for k, v in image_index.items() if v.get("annotation")}
        unannotated = {k: v for k, v in image_index.items() if not v.get("annotation")}
        keys = list(annotated.keys())[:4] + list(unannotated.keys())[:1]
        random.shuffle(keys)
        image_index = {k: image_index[k] for k in keys}
        num_ann = sum(1 for v in image_index.values() if v.get("annotation"))
        print(f"DRY RUN: processing {len(image_index)} images ({num_ann} with annotations)")

    # Step 2: Save annotation summary
    print("=== Step 2: Extracting annotation metadata ===")
    annotation_summary = {}
    for img_path, info in image_index.items():
        ann = info.get("annotation") or {}
        annotation_summary[os.path.basename(img_path)] = {
            "quality": parse_quality_score(ann.get("Quality", "")),
            "usability": parse_usability(ann.get("Usibility", "")),
            "degradations": get_degradation_types(ann),
            "task": assign_task_label(info, info["rel_path"]),
        }

    with open(output_dir / "aircopbench_annotations.json", "w") as f:
        json.dump(annotation_summary, f, indent=2)
    print(f"Saved annotation summary: {len(annotation_summary)} entries")

    # Step 3: Run distortion injection
    print("=== Step 3: Distortion injection ===")
    distorted_dir = output_dir / "distorted"
    distorted_dir.mkdir(parents=True, exist_ok=True)

    pipeline = UAVDistortionPipeline(seed=args.seed)
    pipeline.inject_directory(
        image_paths=list(image_index.keys()),
        output_dir=str(distorted_dir),
        compress=args.compress,
        max_workers=args.workers,
    )

    log_file = distorted_dir / "distortion_log.json"
    if log_file.exists():
        with open(log_file) as f:
            log_data = json.load(f)
        total = log_data.get("total_distorted", 0)
        failed = log_data.get("failed", 0)
        print(f"Distortion complete: {total} generated, {failed} failed")

    # Step 4: Generate manifest.json
    print("=== Step 4: Generating manifest ===")
    manifest_entries = []
    for dist_file in distorted_dir.glob("**/*.png"):
        fname = dist_file.name
        # Parse distortion name and intensity from filename
        # Format: {original_name}__{distortion_name}_L0{intensity}[-{index}].png
        parts = fname.split("__")
        if len(parts) < 2:
            continue

        orig_name = parts[0] + ".jpg"  # reconstruct original filename
        dist_info = parts[1].replace(".png", "")

        # Extract intensity level from the distortion part
        intensity_match = re.search(r'_L0?(\d)(?:\.0+)?-', dist_info + "-")
        if intensity_match:
            intensity_level = float(intensity_match.group(1)) / 10.0
        else:
            intensity_level = 0.5

        # Extract distortion name
        distortion_name = re.sub(r'_L\d+.*$', '', dist_info)

        # Get original annotation
        ann_info = annotation_summary.get(orig_name, {})
        if not ann_info and orig_name.replace(".jpg", "") in annotation_summary:
            ann_info = annotation_summary.get(orig_name.replace(".jpg", ""), {})

        manifest_entries.append({
            "path": str(dist_file.relative_to(output_dir)),
            "original": orig_name,
            "distortion": distortion_name,
            "intensity_level": intensity_level,
            "task": ann_info.get("task", "inspection"),
            "vlm_score": ann_info.get("quality", 0.0),
            "vla_score": ann_info.get("usability", 0.0),
            "execution_score": 0.0,
            "annotated": True,
            "degradation_types": ann_info.get("degradations", []),
        })

    print(f"Generated {len(manifest_entries)} manifest entries")

    # Shuffle and split
    import random
    random.seed(args.seed)
    random.shuffle(manifest_entries)

    n = len(manifest_entries)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    splits = {
        "train": manifest_entries[:train_end],
        "val": manifest_entries[train_end:val_end],
        "test": manifest_entries[val_end:],
    }

    manifest = {
        "splits": splits,
        "total": n,
        "distortion_types": sorted(set(e["distortion"] for e in manifest_entries)),
        "tasks": sorted(set(e["task"] for e in manifest_entries)),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved: {manifest_path}")
    print(f"  Total entries: {n}")
    for split_name, entries in splits.items():
        print(f"  {split_name}: {len(entries)} entries")
    print(f"  Distortion types: {len(manifest['distortion_types'])}")
    print(f"  Tasks: {manifest['tasks']}")


if __name__ == "__main__":
    main()
