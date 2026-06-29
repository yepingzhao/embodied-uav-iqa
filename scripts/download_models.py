#!/usr/bin/env python3
"""Download VLM models from HuggingFace Hub for offline use.

Downloads models registered in MODEL_REGISTRY using huggingface_hub.snapshot_download.
Supports per-family model classes and trust_remote_code.

Models (15 total, 6 families):
  qwen:        Qwen2-VL, Qwen2.5-VL
  internvl:    Mini-InternVL, InternVL2, InternVL2.5, InternVL3
  internlm_xc: InternLM-Xcomposer2, InternLM-Xcomposer2.5
  ovis:        Ovis1.5-Gemma, Ovis1.6-Llama, Ovis2
  phi:         Phi3-Vision, Phi3.5-Vision, Phi4-Multimodal
  mplug:       MPlugOwl3

Examples:
    # Download all 15 models
    python scripts/download_models.py --all

    # Download specific models by short name
    python scripts/download_models.py --models Qwen2-VL InternVL2 Phi3-Vision

    # Download to custom directory with auth token (for gated models)
    python scripts/download_models.py --all --cache-dir /data/models --token hf_xxx

    # Download + validate each model loads correctly
    python scripts/download_models.py --all --validate
"""

import argparse
import importlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("download_models")


def _load_registry():
    """Lazy-load MODEL_REGISTRY without pulling in the full uav_iqa package.

    Uses importlib.util to load just vlm_vla_scorer.py (stdlib-only imports),
    avoiding the heavyweight lightning/torch import chain in uav_iqa.__init__.
    """
    module_path = (
        Path(__file__).resolve().parent.parent / "src" / "uav_iqa" / "vlm_vla_scorer.py"
    )

    spec = importlib.util.spec_from_file_location(
        "uav_iqa.vlm_vla_scorer", str(module_path)
    )
    if spec is None or spec.loader is None:
        _log.error("Cannot find vlm_vla_scorer.py at %s", module_path)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download VLM models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true", help="Download all 15 registered models"
    )
    group.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Specific model short names to download",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom cache directory (default: HF_HOME or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token for gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each model by loading via transformers after download",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip download, only validate models already in cache",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for validation loading (default: cuda, use cpu if no GPU)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Limit to first N models (for testing)",
    )
    return parser


def _get_model_names(args, registry: dict) -> list:
    if args.all:
        names = sorted(registry)
    else:
        names = args.models
        for name in names:
            if name not in registry:
                valid = ", ".join(sorted(registry))
                _log.error("Unknown model '%s'. Valid names: %s", name, valid)
                sys.exit(1)
    if args.max_models:
        names = names[: args.max_models]
    return names


def download_model(config, cache_dir: str | None = None, token: str | None = None):
    """Download a single model from HuggingFace Hub.

    Uses snapshot_download to fetch all repository files without loading into memory.
    """
    import huggingface_hub

    _log.info("Downloading %s (%s) ...", config.short_name, config.hf_model_id)

    kwargs = dict(
        repo_id=config.hf_model_id,
        cache_dir=cache_dir,
        token=token or os.environ.get("HF_TOKEN"),
        resume_download=True,
    )
    # Only pass ignore_patterns for models that don't need safetensors variants
    # None by default — snapshot_download handles this correctly

    try:
        local_path = huggingface_hub.snapshot_download(**kwargs)
        _log.info("  -> %s", local_path)
        return local_path
    except Exception as exc:
        _log.error("  Download failed: %s", exc)
        raise


def validate_model(config, cache_dir: str | None = None, device: str = "cuda"):
    """Load a model via transformers to verify it works.

    Uses the per-family model_class_name and processor_class_name from VLMConfig.
    """
    import torch

    tf_mod = importlib.import_module("transformers")
    model_cls = getattr(tf_mod, config.model_class_name)
    proc_cls = getattr(tf_mod, config.processor_class_name)

    _log.info(
        "  Validating %s (class=%s, device=%s) ...",
        config.short_name,
        config.model_class_name,
        device,
    )

    kwargs: dict = dict(
        trust_remote_code=config.trust_remote_code,
    )
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        processor = proc_cls.from_pretrained(config.hf_model_id, **kwargs)
        _log.info("    processor: OK (%s)", type(processor).__name__)
    except Exception as exc:
        _log.warning("    processor load failed: %s", exc)
        processor = None

    try:
        model = model_cls.from_pretrained(
            config.hf_model_id,
            torch_dtype=torch.float16,
            **kwargs,
        )
        if device == "cpu":
            model = model.to("cpu")
        elif device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        _log.info(
            "    model: OK (%s, %s params)", type(model).__name__, _count_params(model)
        )
    except Exception as exc:
        _log.warning("    model load failed: %s (may need more GPU RAM)", exc)
        model = None

    return processor is not None and model is not None


def _count_params(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    if total >= 1e9:
        return f"{total / 1e9:.1f}B"
    return f"{total / 1e6:.0f}M"


def main():
    args = build_parser().parse_args()

    # Validate dependencies
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        _log.error(
            "huggingface_hub not installed. Run: uv sync --group dev --extra vlm"
        )
        sys.exit(1)

    if args.validate or args.validate_only:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            _log.error(
                "transformers/torch not installed. Run: uv sync --group dev --extra vlm"
            )
            sys.exit(1)

    MODEL_REGISTRY = _load_registry()
    model_names = _get_model_names(args, MODEL_REGISTRY)
    _log.info("Target: %d model(s)", len(model_names))
    for i, name in enumerate(model_names, 1):
        _log.info("[%d/%d] %s", i, len(model_names), name)

    results: dict[str, dict] = {}
    total_start = time.monotonic()

    for name in model_names:
        config = MODEL_REGISTRY[name]
        start = time.monotonic()
        status = {"downloaded": False, "validated": False, "path": None, "error": None}

        try:
            if not args.validate_only:
                local_path = download_model(
                    config, cache_dir=args.cache_dir, token=args.token
                )
                status["downloaded"] = True
                status["path"] = local_path

            if args.validate or args.validate_only:
                ok = validate_model(
                    config, cache_dir=args.cache_dir, device=args.device
                )
                status["validated"] = ok
                if not ok:
                    status["error"] = "validation: model or processor load failed"

            elapsed = time.monotonic() - start
            _log.info("  %s done in %.0fs", name, elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - start
            status["error"] = str(exc)
            _log.error("  %s FAILED after %.0fs: %s", name, elapsed, exc)

        results[name] = status

    total_elapsed = time.monotonic() - total_start

    # Summary
    n_downloaded = sum(1 for r in results.values() if r["downloaded"])
    n_validated = sum(1 for r in results.values() if r["validated"])
    n_failed = sum(1 for r in results.values() if r["error"])

    _log.info("=" * 60)
    _log.info(
        "Done: %d downloaded, %d validated, %d failed in %.0fs",
        n_downloaded,
        n_validated,
        n_failed,
        total_elapsed,
    )
    if n_failed:
        _log.warning("Failed models:")
        for name, r in results.items():
            if r["error"]:
                _log.warning("  %s: %s", name, r["error"])


if __name__ == "__main__":
    main()
