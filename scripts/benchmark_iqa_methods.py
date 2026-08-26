#!/usr/bin/env python3
"""Benchmark 15 IQA methods from the Embodied-IQA paper (NeurIPS 2025).

Thin CLI wrapper — see uav_iqa.baselines.evaluator for the implementation.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from uav_iqa.baselines.evaluator import AVAILABLE_METHODS, run_benchmark, run_benchmark_parallel
from uav_iqa.data.samples import load_benchmark_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark IQA methods")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--methods", nargs="+", default=[k for k in AVAILABLE_METHODS])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel mode (e.g. '0,1,2,3,4,5,6,7')")
    parser.add_argument("--num-workers-per-gpu", type=int, default=4,
                        help="Max concurrent methods per GPU (default: 4)")
    parser.add_argument("--parallel", action="store_true", default=None,
                        help="Force parallel mode (auto-enabled when --gpu-ids is set)")
    parser.add_argument("--no-parallel", action="store_true", default=None,
                        help="Force sequential mode")
    parser.add_argument("--image-dir", default="data/processed",
                        help="Base directory for image files (distorted + ref). "
                             "Default: data/processed.")
    parser.add_argument("--finetuned-dir", default=None)
    parser.add_argument("--ref-image-dir", default=None,
                        help="Root directory for reference (original) images. "
                             "FR methods load ref images from this directory. "
                             "Default: same as --image-dir.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for pyiqa/LPIPS methods (default: 1, no batching). "
                             "Larger values (e.g. 32, 64) accelerate GPU inference.")
    args = parser.parse_args()

    # Reduce CUDA memory fragmentation to avoid OOM with large batch sizes
    # (particularly AHIQ which uses CFANet with large intermediate tensors).
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether to use parallel mode
    use_parallel = False
    if args.no_parallel:
        use_parallel = False
    elif args.parallel or args.gpu_ids is not None:
        use_parallel = True

    entries = load_benchmark_samples(data_dir, "test")
    if args.max_samples > 0:
        entries = entries[: args.max_samples]
    _log.info("Evaluating %d samples", len(entries))

    finetuned_dir = Path(args.finetuned_dir) if args.finetuned_dir else None
    image_dir = Path(args.image_dir)
    ref_image_dir = Path(args.ref_image_dir) if args.ref_image_dir else None

    if use_parallel:
        gpu_ids = [int(x.strip()) for x in (args.gpu_ids or "0").split(",")]
        _log.info("Parallel mode: %d methods across GPUs %s (max %d per GPU)",
                  len(args.methods), gpu_ids, args.num_workers_per_gpu)
        results = run_benchmark_parallel(
            entries=entries,
            data_dir=data_dir,
            gpu_ids=gpu_ids,
            methods=args.methods,
            image_size=args.image_size,
            num_workers_per_gpu=args.num_workers_per_gpu,
            finetuned_dir=finetuned_dir,
            image_dir=image_dir,
            ref_image_dir=ref_image_dir,
            batch_size=args.batch_size,
        )
    else:
        device = torch.device(args.device)
        _log.info("Sequential mode on device: %s", device)
        results = run_benchmark(
            entries=entries,
            data_dir=data_dir,
            device=device,
            methods=args.methods,
            image_size=args.image_size,
            finetuned_dir=finetuned_dir,
            image_dir=image_dir,
            ref_image_dir=ref_image_dir,
            batch_size=args.batch_size,
        )

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"n_samples": len(entries), "methods": results}, f, indent=2)

    _log.info("=" * 80)
    _log.info("Benchmark Complete")
    _log.info("=" * 80)
    for name in sorted(results.keys(), key=lambda n: results[n]["srcc"], reverse=True):
        r = results[name]
        uav_srcc = r.get("per_distortion_category", {}).get("UAV", {}).get("srcc", 0)
        gen_srcc = r.get("per_distortion_category", {}).get("Generic", {}).get("srcc", 0)
        ft_label = "Yes" if r.get("finetuned") else "No"
        _log.info("%-24s %-5s %-8s %8.4f %8.4f %12.4f %12.4f",
                  name, r["category"], ft_label, r["srcc"], r["plcc"],
                  uav_srcc, gen_srcc)
    _log.info("Results: %s", output_dir / "benchmark_results.json")


if __name__ == "__main__":
    main()
