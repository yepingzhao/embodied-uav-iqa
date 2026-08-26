#!/usr/bin/env python3
"""UAV-IQA distortion synthesis pipeline — CLI.

Backed by ``uav_iqa.data``.

Subcommands:
    inject     Group by scene+frame, apply 36 distortions to all UAVs.

Examples:
    python scripts/distortion_synthesis.py inject \
        --dataset aircopbench \
        --input-root data/raw/AirCopBench \
        --output-dir data/processed
"""

import argparse

from uav_iqa.data import DatasetAdapter, create_pipeline
from uav_iqa.utils import setup_logging

_log = setup_logging(__name__)


def _add_dataset_arg(parser, required: bool = False):
    parser.add_argument(
        "--dataset",
        default="aircopbench",
        choices=DatasetAdapter.list_adapters(),
        required=required,
        help="Dataset format (default: aircopbench)",
    )


def _add_seed_arg(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")


def _add_output_dir_arg(parser, default: str):
    parser.add_argument("--output-dir", default=default, help="Output directory")


def cmd_inject(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    pipeline.inject_distortions(
        input_root=args.input_root,
        output_dir=args.output_dir,
        workers=args.workers,
        compress=not args.no_compress,
        dry_run=args.dry_run,
        fmt=args.format,
    )


def main():
    parser = argparse.ArgumentParser(description="UAV-IQA distortion synthesis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- inject ----
    p_inject = sub.add_parser("inject", help="Group by scene+frame and inject distortions")
    _add_dataset_arg(p_inject, required=True)
    p_inject.add_argument("--input-root", required=True, help="Raw dataset root directory")
    _add_output_dir_arg(p_inject, "data/processed")
    p_inject.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    p_inject.add_argument("--no-compress", action="store_true", help="Disable compression")
    p_inject.add_argument(
        "--format", default="png", choices=["png", "jpeg"],
        help="Output image format (default: png)",
    )
    p_inject.add_argument(
        "--dry-run", action="store_true",
        help="Process only 2 groups with 3 distortions",
    )
    _add_seed_arg(p_inject)
    p_inject.set_defaults(func=cmd_inject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
