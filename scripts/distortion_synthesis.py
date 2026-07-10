#!/usr/bin/env python3
"""UAV-IQA data synthesis pipeline — unified CLI.

Backed by ``uav_iqa.data_synthesis``.

Subcommands:
    inject     Group by scene+frame, apply 36 distortions to all UAVs.
    annotate   VLM multi-image inference for cognitive scores.
    aggregate  Merge per-model scores into final cognitive_score.
    all        Run the full pipeline end-to-end.

Examples:
    # Full AirCopBench pipeline
    python scripts/data_synthesis.py all \
        --dataset aircopbench \
        --input-root data/raw/AirCopBench \
        --output-dir data/processed

    # Single step: inject only
    python scripts/data_synthesis.py inject \
        --dataset aircopbench \
        --input-root data/raw/AirCopBench \
        --output-dir data/processed \
        --workers 8

    # Annotate with a specific VLM
    python scripts/data_synthesis.py annotate \
        --output-dir data/processed \
        --scorer-model Qwen2.5-VL
"""

import argparse

from uav_iqa.data_synthesis import DatasetFormat, create_pipeline
from uav_iqa.utils import setup_logging

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


def _add_scorer_args(parser):
    parser.add_argument(
        "--scorer-model",
        default=None,
        help="VLM model for scoring (e.g. Qwen2.5-VL). "
        "If not set, the annotation step is skipped.",
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
        "--scorer-batch-size",
        type=int,
        default=8,
        help="Batch size for VLM scoring (default: 8)",
    )


def _add_output_dir_arg(parser, default: str):
    parser.add_argument("--output-dir", default=default, help="Output directory")


def _make_scorer(args):
    if not args.scorer_model:
        return None
    from uav_iqa.vlm import VLMScorer

    return VLMScorer(
        model_name=args.scorer_model,
        backend=args.scorer_backend,
        device=args.scorer_device,
        seed=args.seed,
    )


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


def cmd_annotate(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    scorer = _make_scorer(args)
    pipeline.annotate_scores(
        output_dir=args.output_dir,
        scorer=scorer,
        input_root=args.input_root,
        max_entries=args.max_entries,
        files=args.files,
    )


def cmd_aggregate(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    pipeline.aggregate_scores(
        output_dir=args.output_dir,
        strategy=args.strategy,
    )


def cmd_merge(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    pipeline.merge_sidecar_scores(
        output_dir=args.output_dir,
        strategy=args.strategy,
        model_names=args.models,
    )


def cmd_all(args):
    pipeline = create_pipeline(args.dataset, seed=args.seed)
    scorer = _make_scorer(args)

    pipeline.run_full(
        input_root=args.input_root,
        output_dir=args.output_dir,
        steps=args.steps,
        workers=args.workers,
        compress=not args.no_compress,
        dry_run=args.dry_run,
        fmt=args.format,
        scorer=scorer,
        max_annotate_entries=args.max_annotate_entries,
    )


def main():
    parser = argparse.ArgumentParser(description="UAV-IQA data synthesis pipeline")
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

    # ---- annotate ----
    p_annotate = sub.add_parser("annotate", help="VLM multi-image annotation")
    _add_dataset_arg(p_annotate)
    p_annotate.add_argument("--output-dir", default="data/processed", help="Processed output directory")
    p_annotate.add_argument("--input-root", help="Raw dataset root (for resolving reference image paths)")
    p_annotate.add_argument(
        "--max-entries", type=int, default=None,
        help="Limit entries per split (debugging)",
    )
    p_annotate.add_argument(
        "--files", nargs="+", default=None,
        help="Specific file paths relative to output-dir (e.g. train/Sim3_VQA_train.json). "
             "Use for multi-GPU: assign different files to each GPU.",
    )
    _add_seed_arg(p_annotate)
    _add_scorer_args(p_annotate)
    p_annotate.set_defaults(func=cmd_annotate)

    # ---- aggregate ----
    p_aggregate = sub.add_parser("aggregate", help="Merge per-model VLM scores")
    _add_dataset_arg(p_aggregate)
    p_aggregate.add_argument("--output-dir", default="data/processed", help="Processed output directory")
    p_aggregate.add_argument(
        "--strategy", default="mean", choices=["mean"],
        help="Aggregation strategy (default: mean)",
    )
    _add_seed_arg(p_aggregate)
    p_aggregate.set_defaults(func=cmd_aggregate)

    # ---- merge ----
    p_merge = sub.add_parser(
        "merge", help="Merge sidecar VLM scores back into main annotated JSONs",
    )
    _add_dataset_arg(p_merge)
    p_merge.add_argument("--output-dir", default="data/processed", help="Processed output directory")
    p_merge.add_argument(
        "--strategy", default="mean", choices=["mean"],
        help="Aggregation strategy (default: mean)",
    )
    p_merge.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model names to merge (default: all models under vlm/)",
    )
    _add_seed_arg(p_merge)
    p_merge.set_defaults(func=cmd_merge)

    # ---- all ----
    p_all = sub.add_parser("all", help="Run full pipeline end-to-end")
    _add_dataset_arg(p_all, required=True)
    p_all.add_argument("--input-root", required=True, help="Dataset root directory")
    _add_output_dir_arg(p_all, "data/processed")
    p_all.add_argument(
        "--steps", default="all",
        help="Comma-separated steps: inject,annotate,aggregate (or 'all')",
    )
    p_all.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    p_all.add_argument("--no-compress", action="store_true", help="Disable compression")
    p_all.add_argument(
        "--format", default="png", choices=["png", "jpeg"],
        help="Output image format (default: png)",
    )
    p_all.add_argument(
        "--dry-run", action="store_true",
        help="Inject only 2 groups with 3 distortions",
    )
    p_all.add_argument(
        "--max-annotate-entries", type=int, default=None,
        help="Limit annotation entries (debugging)",
    )
    _add_seed_arg(p_all)
    _add_scorer_args(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
