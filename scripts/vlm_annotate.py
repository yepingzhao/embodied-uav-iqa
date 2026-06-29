#!/usr/bin/env python3
"""Batch VLM annotation CLI — score grouped processed JSONs with VLM models.

Replaces the placeholder cognitive_scores with actual VLM multi-image scores.

Examples:
    python scripts/vlm_annotate.py \\
        --output-dir data/processed \\
        --model Qwen2.5-VL \\
        --backend vllm

    python scripts/vlm_annotate.py \\
        --output-dir data/processed \\
        --all-models

    python scripts/vlm_annotate.py \\
        --output-dir data/processed \\
        --model Qwen2.5-VL \\
        --max-entries 100

    python scripts/vlm_annotate.py \\
        --output-dir data/processed \\
        --models Qwen2.5-VL InternVL2 \\
        --splits train
"""

import argparse
from pathlib import Path

from uav_iqa.batch_annotator import BatchAnnotator
from uav_iqa.utils import setup_logging
from uav_iqa.vlm import MODEL_REGISTRY, VLMScorer

_log = setup_logging(__name__)


def cmd_annotate(args):
    output_dir = Path(args.output_dir)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else output_dir / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    models = args.models if args.models else [args.model]
    if args.all_models:
        models = sorted(MODEL_REGISTRY)
    _log.info("Models to run: %s", ", ".join(models))

    for model_name in models:
        _log.info("=" * 60)
        _log.info("Model: %s", model_name)

        scorer = VLMScorer(
            model_name=model_name,
            backend=args.backend,
            device=args.device,
            seed=args.seed,
            cache_dir=args.cache_dir,
            vqa_dir=str(output_dir),
        )

        annotator = BatchAnnotator(scorer, output_dir)
        results = annotator.annotate_splits(
            splits=args.splits,
            max_entries=args.max_entries,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
        )

        for key, info in results.items():
            _log.info("  %s: scored=%d total=%d", key, info["scored"], info["total"])

    _log.info("All done.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch VLM annotation for grouped processed JSONs"
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Processed output directory containing train/ and test/ subdirs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory (default: --output-dir/checkpoints)",
    )

    parser.add_argument(
        "--model",
        default="Qwen2.5-VL",
        help="VLM model name (short name from registry or HF model ID). "
        "Ignored if --models is set.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Space-separated model names for ensemble scoring",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Score with all models registered in MODEL_REGISTRY",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "vllm", "transformers", "none"],
        help="VLM backend (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model loading (default: cuda)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory for offline model loading",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to annotate (default: train test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Cap on entries to score per file (for testing)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N entries (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available",
    )

    args = parser.parse_args()
    cmd_annotate(args)


if __name__ == "__main__":
    main()
