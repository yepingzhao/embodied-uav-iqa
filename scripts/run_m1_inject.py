#!/usr/bin/env python3
"""M1-R005: Batch distortion injection pipeline.

Applies all 24 distortion types at 5 intensity levels to a directory of
reference images, saving distorted variants to the database directory.

Usage:
    python scripts/run_m1_inject.py \
        --image-dir data/raw/AirCopBench/Sim_3_UAVs/Samples/images/scene_001 \
        --output-dir data/processed/distorted \
        --workers 8

Dry run (process 5 images only):
    python scripts/run_m1_inject.py --image-dir ... --dry-run
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uav_iqa.distortion import UAVDistortionPipeline


def find_images(image_dir: Path, exts: tuple = (".jpg", ".jpeg", ".png")):
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])


def process_image(args_tuple):
    img_path, output_dir, pipeline_kwargs, compress = args_tuple
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return {"path": str(img_path), "status": "read_error"}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pipeline = UAVDistortionPipeline(**pipeline_kwargs)
        results = pipeline.generate_all(image)

        ref_name = img_path.stem
        out_subdir = output_dir / ref_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        saved = {}
        for key, distorted in results.items():
            out_path = out_subdir / f"{key}.png"
            save_img = distorted
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            params = [cv2.IMWRITE_PNG_COMPRESSION, compress]
            cv2.imwrite(str(out_path), save_img, params)
            saved[key] = str(out_path.relative_to(output_dir.parent))

        return {"path": str(img_path), "status": "ok", "saved": saved}
    except Exception as e:
        return {"path": str(img_path), "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="M1-R005: Batch distortion injection")
    parser.add_argument("--image-dir", required=True, help="Directory of reference images")
    parser.add_argument("--output-dir", default="data/processed/distorted", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Process first 5 images only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compress", type=int, default=6, help="PNG compression level (0-9)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir)
    if args.dry_run:
        images = images[:5]
        print(f"[DRY RUN] Processing {len(images)} images")

    print(f"Found {len(images)} reference images")
    print("24 distortions × 5 levels = 120 variants per image")
    print(f"Total expected pairs: {len(images) * 120}")
    print(f"Output: {output_dir}")

    pipeline_kwargs = {"seed": args.seed}
    tasks = [(p, output_dir, pipeline_kwargs, args.compress) for p in images]

    t0 = time.time()
    ok, errors = 0, 0
    log_entries = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            log_entries.append(result)
            if result["status"] == "ok":
                ok += 1
            else:
                errors += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] {ok} ok, {errors} err | "
                      f"{rate:.1f} img/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {ok} succeeded, {errors} failed.")

    log_path = output_dir / "distortion_log.json"
    with open(log_path, "w") as f:
        json.dump({"total": len(images), "ok": ok, "errors": errors,
                    "elapsed_s": elapsed, "entries": log_entries}, f, indent=2)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
