#!/usr/bin/env python
"""CLI for the multi-GPU offline inference framework.

Usage::

    python scripts/inference.py \
        --input-dir data/processed/train \
        --output-dir data/output/train \
        --checkpoint-path data/checkpoint.db \
        --num-gpus 4 \
        --model Qwen2.5-VL \
        --backend vllm \
        --chunk-size 256 \
        --batch-size 16

    # Resume from checkpoint:
    python scripts/inference.py --resume [same args]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-GPU offline VLM inference for UAV-IQA annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    p.add_argument("--input-dir", required=True, type=Path, help="Input directory with *_VQA_*.json")
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory for annotated JSONs")
    p.add_argument("--checkpoint-path", required=True, type=Path, help="SQLite checkpoint database path")

    # Scheduling
    p.add_argument("--chunk-size", type=int, default=256, help="Records per task (scheduling unit)")
    p.add_argument("--batch-size", type=int, default=16, help="Samples per GPU forward pass")
    p.add_argument("--max-entries", type=int, default=0, help="Cap total entries (0 = no limit, for testing)")

    # GPU
    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    p.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU IDs (default: 0,1,2,...)")

    # VLM
    p.add_argument("--model", type=str, default="Qwen2.5-VL", help="VLM model short name")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "transformers", "none"])
    p.add_argument("--device", type=str, default="", help="Device override (e.g. cuda:0)")
    p.add_argument("--seed", type=int, default=42)

    # Misc
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Use DummyExecutor (no model loaded)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _log = logging.getLogger("inference_cli")

    # Late import to avoid loading torch for --help
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from uav_iqa.inference import InferenceConfig, InferenceEngine
    from uav_iqa.inference.executor import DummyExecutor, VLMExecutor

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))

    # Build executor factory
    if args.dry_run:
        def factory(_gpu_idx: int) -> DummyExecutor:
            return DummyExecutor(model_name="dry_run", score=0.5)
    else:
        def factory(gpu_idx: int) -> VLMExecutor:
            gpu_id = gpu_ids[gpu_idx] if gpu_idx < len(gpu_ids) else gpu_idx
            return VLMExecutor(
                model_name=args.model,
                backend=args.backend,
                device=args.device or f"cuda:{gpu_id}",
                seed=args.seed,
            )

    cfg = InferenceConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        seed=args.seed,
        max_entries=args.max_entries,
    )

    engine = InferenceEngine(cfg, executor_factory=factory)
    result = engine.run(resume=args.resume)

    _log.info("=" * 60)
    _log.info("Inference complete")
    _log.info("  Files written: %s", list(result["files_written"].keys()))
    _log.info("  Validation: %s", result["validation"])
    _log.info("  Metrics: %s", result["metrics"])
    _log.info("=" * 60)

    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
