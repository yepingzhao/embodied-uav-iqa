#!/usr/bin/env python3
"""UAV-IQA data synthesis pipeline — unified CLI.

Replaces the 5 previous scripts (extract_aircopbench_refs, run_m1_inject,
run_m1_manifest, annotate_scores, run_m1_aircopbench) with a single
entry point backed by ``uav_iqa.data_synthesis``.

Subcommands:
    extract     Extract clean reference frames from a dataset root.
    inject      Apply distortions to reference images.
    manifest    Generate train/val/test manifest.json from distorted images.
    annotate    Annotate manifest entries with degradation-model scores.
    all         Run the full pipeline end-to-end.

Examples:
    # Full AirCopBench pipeline
    python scripts/synthesize_data.py all \\
        --dataset aircopbench \\
        --input-root data/raw/AirCopBench \\
        --output-dir data/processed

    # Single step: inject only
    python scripts/synthesize_data.py inject \\
        --image-dir data/processed/ref_images \\
        --output-dir data/processed/distorted \\
        --workers 8

    # Generic image directory (no annotations)
    python scripts/synthesize_data.py all \\
        --dataset generic \\
        --input-root /path/to/images \\
        --output-dir data/processed
"""

import argparse

from uav_iqa.data_synthesis import DatasetFormat, create_pipeline
from uav_iqa.utils import load_task_map, setup_logging

_log = setup_logging(__name__)


def _add_dataset_arg(parser, required: bool = False):
    parser.add_argument(
        "--dataset",
        default="aircopbench",
        choices=DatasetFormat.list_formats(),
        required=required,
        help="Dataset format (default: aircopbench)",
    )


def _add_seed_arg(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")


def _add_output_dir_arg(parser, default: str):
    parser.add_argument("--output-dir", default=default, help="Output directory")


def cmd_extract(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    pipeline.extract_references(
        input_root=args.input_root,
        output_dir=args.output_dir,
        copy=args.copy,
    )


def cmd_inject(args):
    pipeline = create_pipeline("generic", seed=args.seed)
    pipeline.inject_distortions(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        compress=not args.no_compress,
        dry_run=args.dry_run,
    )


def cmd_manifest(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    task_map = load_task_map(args.task_map)
    pipeline.generate_manifests(
        distorted_dir=args.distorted_dir,
        output_dir=args.output_dir,
        ref_dir=args.ref_dir,
        split=tuple(args.split),
        task_map=task_map,
    )


def cmd_annotate(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    pipeline.annotate_scores(
        manifest_dir=args.manifest_dir,
        input_root=args.input_root,
        output_dir=args.output_dir,
        noise_scale=args.noise_scale,
    )


def cmd_all(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    task_map = load_task_map(args.task_map)
    pipeline.run_full(
        input_root=args.input_root,
        output_dir=args.output_dir,
        steps=args.steps,
        copy=args.copy,
        workers=args.workers,
        compress=not args.no_compress,
        split=tuple(args.split),
        task_map=task_map,
        noise_scale=args.noise_scale,
        dry_run=args.dry_run,
    )


def main():
    parser = argparse.ArgumentParser(
        description="UAV-IQA data synthesis pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- extract ----
    p_extract = sub.add_parser("extract", help="Extract clean reference frames")
    _add_dataset_arg(p_extract, required=True)
    p_extract.add_argument("--input-root", required=True, help="Dataset root directory")
    _add_output_dir_arg(p_extract, "data/processed/ref_images")
    p_extract.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    _add_seed_arg(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    # ---- inject ----
    p_inject = sub.add_parser("inject", help="Apply distortions to reference images")
    p_inject.add_argument("--image-dir", required=True, help="Reference images directory")
    _add_output_dir_arg(p_inject, "data/processed/distorted")
    p_inject.add_argument("--workers", type=int, default=4, help="Parallel workers")
    p_inject.add_argument("--no-compress", action="store_true", help="Disable PNG compression")
    p_inject.add_argument("--dry-run", action="store_true", help="Process first 5 images only")
    _add_seed_arg(p_inject)
    p_inject.set_defaults(func=cmd_inject)

    # ---- manifest ----
    p_manifest = sub.add_parser("manifest", help="Generate train/val/test manifest.json")
    _add_dataset_arg(p_manifest)
    p_manifest.add_argument("--distorted-dir", required=True, help="Distorted images directory")
    _add_output_dir_arg(p_manifest, "data/processed")
    p_manifest.add_argument("--ref-dir", default=None, help="Reference images directory")
    p_manifest.add_argument(
        "--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        help="Train/val/test ratios",
    )
    p_manifest.add_argument("--task-map", default=None, help="JSON task mapping file")
    _add_seed_arg(p_manifest)
    p_manifest.set_defaults(func=cmd_manifest)

    # ---- annotate ----
    p_annotate = sub.add_parser("annotate", help="Annotate manifest with scores")
    _add_dataset_arg(p_annotate)
    p_annotate.add_argument("--manifest-dir", default="data/processed", help="Manifest directory")
    p_annotate.add_argument("--input-root", default=None, help="Dataset root for annotation lookup")
    _add_output_dir_arg(p_annotate, None)
    p_annotate.add_argument("--noise-scale", type=float, default=0.02, help="Noise std multiplier")
    _add_seed_arg(p_annotate)
    p_annotate.set_defaults(func=cmd_annotate)

    # ---- all ----
    p_all = sub.add_parser("all", help="Run full pipeline end-to-end")
    _add_dataset_arg(p_all, required=True)
    p_all.add_argument("--input-root", required=True, help="Dataset root directory")
    _add_output_dir_arg(p_all, "data/processed")
    p_all.add_argument(
        "--steps", default="all",
        help="Comma-separated steps: extract,inject,manifest,annotate (or 'all')",
    )
    p_all.add_argument("--copy", action="store_true", help="Copy ref images instead of symlink")
    p_all.add_argument("--workers", type=int, default=4, help="Parallel workers for injection")
    p_all.add_argument("--no-compress", action="store_true", help="Disable PNG compression")
    p_all.add_argument(
        "--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        help="Train/val/test ratios",
    )
    p_all.add_argument("--task-map", default=None, help="JSON task mapping file")
    p_all.add_argument("--noise-scale", type=float, default=0.02, help="Noise std multiplier")
    p_all.add_argument("--dry-run", action="store_true", help="Inject only 5 images")
    _add_seed_arg(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
