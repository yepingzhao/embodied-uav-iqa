#!/usr/bin/env python3
"""Extract clean reference frames from AirCopBench directory structure.

AirCopBench stores clean images mixed with pre-degraded variants (loss/, noise/
subdirs). This script gathers only the clean reference frames and copies them
into a unified flat directory ready for distortion injection.

Usage:
    python scripts/extract_aircopbench_refs.py \
        --aircopbench-root data/raw/AirCopBench \
        --output-dir data/processed/ref_images
"""

import argparse
import shutil
from pathlib import Path

# Subdirectories that contain pre-degraded (non-reference) images
EXCLUDE_DIRS = {
    "loss",
    "noise",
    "point_clouds",
    "Annotations",
    "original_json",
    "original_xml",
    "train",
    "test",
    ".git",
}

# Image extensions to collect
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_clean_images(root: Path) -> list[Path]:
    """Walk the AirCopBench tree and return clean reference image paths."""
    images = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        # Exclude paths that pass through a degraded/annotation directory
        parts = set(p.relative_to(root).parts)
        if parts & EXCLUDE_DIRS:
            continue
        images.append(p)
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean reference frames from AirCopBench"
    )
    parser.add_argument(
        "--aircopbench-root",
        required=True,
        help="Root directory of the AirCopBench dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/ref_images",
        help="Output directory for unified reference images",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy files (default is symlink to save disk space)",
    )
    args = parser.parse_args()

    root = Path(args.aircopbench_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    images = find_clean_images(root)
    print(f"Found {len(images)} clean reference images in {root}")

    copied = 0
    for img in images:
        # Preserve source identity in filename: Sim3_scene_001_UAV1_00001.jpg
        rel = img.relative_to(root)
        stem = "_".join(rel.parts).replace("/", "_").replace("\\", "_")
        dest = output / stem

        if not dest.exists():
            if args.copy:
                shutil.copy2(img, dest)
            else:
                dest.symlink_to(img.resolve())
        copied += 1

    op = "Copied" if args.copy else "Symlinked"
    print(f"{op} {copied} images to {output}")
    print(f"Next: python scripts/run_m1_inject.py --image-dir {output}")


if __name__ == "__main__":
    main()
