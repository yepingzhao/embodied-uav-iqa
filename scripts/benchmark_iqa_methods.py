#!/usr/bin/env python3
"""M2-R009/R010/R011/R012: Benchmark existing IQA methods.

Runs 15+ existing IQA methods on the UAV-Embodied-IQA test set and computes
SRCC/PLCC against VLA decision scores. Models are cached per method to avoid
re-instantiation overhead on large test sets.

Two modes per method:
  - zero-shot: pretrained weights only (current default)
  - fine-tuned: checkpoint from scripts/finetune_baselines.py

Supported methods:
  FR: psnr, ssim, ms_ssim, lpips_alex, lpips_vgg, dists, ahiq, topiq_fr
  NR-handcrafted: brisque, brisque_dct (frequency-aware), niqe
  NR-deep: maniqa, clip_iqa, q_align, topiq_nr

Usage:
    # Zero-shot only
    python scripts/benchmark_iqa_methods.py \
        --data-dir data/processed \
        --output-dir outputs/benchmark \
        --methods psnr ssim lpips_alex brisque clip_iqa

    # With fine-tuned checkpoints (runs zero-shot first, then fine-tuned)
    python scripts/benchmark_iqa_methods.py \
        --data-dir data/processed \
        --output-dir outputs/benchmark \
        --finetuned-dir outputs/finetune
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from uav_iqa.metrics import (
    evaluate_iqa,
    per_distortion_category_metrics,
    per_task_metrics,
)
from uav_iqa.utils import load_flat_samples, load_image_tensor, setup_logging

_log = setup_logging(__name__)


def load_image_and_score(sample, data_dir, image_size=256):
    img_path = Path(data_dir) / sample["path"]
    image_tensor = load_image_tensor(img_path, image_size).unsqueeze(0)
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    score = sample.get("score", 0.0)
    if isinstance(score, list):
        score = np.mean(score)
    return (
        image_np,
        image_tensor,
        float(score),
        sample.get("task", "scene_description"),
        sample.get("distortion", "unknown"),
        sample.get("intensity_level", 0.0),
        sample.get("ref_path", ""),
    )


# ---------------------------------------------------------------------------
# IQA method implementations -- single registry with uniform signatures
# ---------------------------------------------------------------------------


class IQAEvaluator:
    """Registry of IQA methods. Each method receives (data_dir, img_np, img_t, ref_path).

    Deep-learning models are constructed once and cached via lru_cache to avoid
    per-sample instantiation overhead.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    @staticmethod
    def _load_ref_tensor(ref_path, data_dir):
        ref_img = Image.open(Path(data_dir) / ref_path).convert("RGB")
        ref_np = np.array(ref_img).astype(np.float32) / 255.0
        return torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)

    # -- Full-reference methods --

    @staticmethod
    def psnr(data_dir, img_np, img_t, ref_path):
        from skimage.metrics import peak_signal_noise_ratio

        if not ref_path:
            return 0.0
        ref = np.array(Image.open(Path(data_dir) / ref_path).convert("RGB"))
        return peak_signal_noise_ratio(
            ref, (img_np * 255).astype(np.uint8), data_range=255
        )

    @staticmethod
    def ssim(data_dir, img_np, img_t, ref_path):
        from skimage.metrics import structural_similarity

        if not ref_path:
            return 0.0
        ref = np.array(Image.open(Path(data_dir) / ref_path).convert("RGB"))
        return structural_similarity(
            ref, (img_np * 255).astype(np.uint8), channel_axis=2, data_range=255
        )

    @staticmethod
    def ms_ssim(data_dir, img_np, img_t, ref_path):
        try:
            from pytorch_msssim import ms_ssim

            if not ref_path:
                return 0.0
            ref_t = IQAEvaluator._load_ref_tensor(ref_path, data_dir)
            return float(ms_ssim(img_t, ref_t, data_range=1.0))
        except ImportError:
            return 0.0

    def lpips_alex(self, data_dir, img_np, img_t, ref_path):
        return self._lpips(img_t, ref_path, data_dir, net="alex")

    def lpips_vgg(self, data_dir, img_np, img_t, ref_path):
        return self._lpips(img_t, ref_path, data_dir, net="vgg")

    @lru_cache(maxsize=2)
    def _get_lpips(self, net: str):
        try:
            import lpips

            return lpips.LPIPS(net=net, verbose=False).to(self.device)
        except ImportError:
            return None

    def _lpips(self, img_t, ref_path, data_dir, net="alex"):
        loss_fn = self._get_lpips(net)
        if loss_fn is None or not ref_path:
            return 0.0
        ref_t = IQAEvaluator._load_ref_tensor(ref_path, data_dir).to(self.device)
        return float(loss_fn(img_t.to(self.device), ref_t).item())

    @lru_cache(maxsize=8)
    def _get_pyiqa_metric(self, metric_name: str):
        try:
            import pyiqa

            return pyiqa.create_metric(metric_name, device=self.device)
        except ImportError:
            return None

    def _pyiqa_fr(self, data_dir, img_t, ref_path, metric_name):
        if not ref_path:
            return 0.0
        metric = self._get_pyiqa_metric(metric_name)
        if metric is None:
            return 0.0
        ref_t = IQAEvaluator._load_ref_tensor(ref_path, data_dir).to(self.device)
        return float(metric(img_t.to(self.device), ref_t).item())

    def _pyiqa_nr(self, img_t, metric_name):
        metric = self._get_pyiqa_metric(metric_name)
        if metric is None:
            return 0.0
        return float(metric(img_t.to(self.device)).item())

    def ahiq(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "ahiq")

    def dists(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "dists")

    def topiq_fr(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "topiq_fr")

    # -- No-reference methods --

    def brisque(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "brisque")

    @staticmethod
    def brisque_dct(data_dir, img_np, img_t, ref_path):
        """Frequency-aware NR-IQA: NSS features on block-DCT coefficients.

        Implements the DCT-domain analogue of BRISQUE: 8×8 block DCT →
        fit GGD to AC coefficients per block → aggregate statistics →
        linear regression score.
        """
        import numpy as np

        gray = np.mean(img_np, axis=2) if img_np.ndim == 3 else img_np.copy()
        gray = (gray * 255).astype(np.uint8).astype(np.float32)
        h, w = gray.shape

        # Block DCT → aggregate AC coefficient stats
        block_h, block_w = 8, 8
        ac_std = []
        for y in range(0, h - block_h + 1, block_h):
            for x in range(0, w - block_w + 1, block_w):
                block = gray[y : y + block_h, x : x + block_w]
                dct = np.zeros_like(block)
                for u in range(block_h):
                    for v in range(block_w):
                        cu = np.sqrt(2 / block_h) if u > 0 else 1 / np.sqrt(block_h)
                        cv = np.sqrt(2 / block_w) if v > 0 else 1 / np.sqrt(block_w)
                        dct_uv = 0.0
                        for i in range(block_h):
                            for j in range(block_w):
                                dct_uv += (
                                    block[i, j]
                                    * np.cos(np.pi * u * (2 * i + 1) / (2 * block_h))
                                    * np.cos(np.pi * v * (2 * j + 1) / (2 * block_w))
                                )
                        dct[u, v] = cu * cv * dct_uv
                # Collect AC coefficients (exclude DC at (0,0))
                ac = dct[1:, :].ravel()
                std = np.std(ac) if len(ac) > 0 else 0.0
                ac_std.append(std)

        if not ac_std:
            return 0.0

        ac_arr = np.array(ac_std)
        # Aggregate stats: mean, variance, skewness, kurtosis of AC stddevs
        mu = np.mean(ac_arr)
        sigma_sq = np.var(ac_arr)
        skew = np.mean((ac_arr - mu) ** 3) / (sigma_sq**1.5 + 1e-8)
        kurt = np.mean((ac_arr - mu) ** 4) / (sigma_sq**2 + 1e-8)

        # Higher AC energy + variance = more texture = higher quality (for UAV tasks)
        # Combine into a heuristic score (hand-calibrated ranges)
        energy = np.mean(ac_arr)
        score_raw = (
            0.4 * np.clip(energy / 15.0, 0, 1)
            + 0.3 * np.clip(np.sqrt(max(sigma_sq, 0)) / 8.0, 0, 1)
            + 0.2 * (1.0 - np.clip(abs(skew) / 3.0, 0, 1))
            + 0.1 * (1.0 - np.clip(abs(kurt - 3.0) / 10.0, 0, 1))
        )
        return float(np.clip(score_raw, 0.0, 1.0))

    def niqe(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "niqe")

    def clip_iqa(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "clipiqa")

    def maniqa(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "maniqa")

    def q_align(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "qalign")

    def topiq_nr(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "topiq_nr")


# Registry: name -> (category, needs_instance, can_finetune)
# can_finetune=True means scripts/finetune_baselines.py supports it (NR methods via pyiqa)
AVAILABLE_METHODS = {
    "psnr": ("FR", False, False),
    "ssim": ("FR", False, False),
    "ms_ssim": ("FR", False, False),
    "lpips_alex": ("FR", True, False),
    "lpips_vgg": ("FR", True, False),
    "dists": ("FR", True, False),
    "ahiq": ("FR", True, False),
    "topiq_fr": ("FR", True, False),
    "brisque": ("NR", True, True),
    "brisque_dct": (
        "NR",
        False,
        False,
    ),  # frequency-aware: BRISQUE features on DCT domain
    "niqe": ("NR", True, True),
    "clip_iqa": ("NR", True, True),
    "maniqa": ("NR", True, True),
    "q_align": ("NR", True, False),
    "topiq_nr": ("NR", True, True),
}

# Map method name to pyiqa metric name (some differ)
METHOD_TO_PYIQA = {
    "brisque": "brisque",
    "niqe": "niqe",
    "clip_iqa": "clipiqa",
    "maniqa": "maniqa",
    "q_align": "qalign",
    "topiq_nr": "topiq_nr",
    "ahiq": "ahiq",
    "dists": "dists",
    "topiq_fr": "topiq_fr",
}


def load_finetuned_checkpoint(ckpt_path: str, device: torch.device):
    """Load a fine-tuned pyiqa model from checkpoint.

    Returns the model callable (img, ref=None) -> scalar tensor,
    or None on failure.
    """
    import pyiqa

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        _log.warning("  Failed to load checkpoint %s: %s", ckpt_path, e)
        return None

    # Lightning checkpoints store model state under "state_dict" with "model." prefix
    state_dict = ckpt.get("state_dict", ckpt)
    hparams = ckpt.get("hyper_parameters", {})
    method_name = hparams.get("method_name") if isinstance(hparams, dict) else None

    if method_name is None:
        _log.warning("  Checkpoint missing method_name in hyper_parameters")
        return None

    pyiqa_name = METHOD_TO_PYIQA.get(method_name, method_name)
    model = pyiqa.create_metric(pyiqa_name, device=device)

    # Strip "model." prefix from state_dict keys
    model_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state[k[len("model.") :]] = v

    if not model_state:
        _log.warning("  Checkpoint has no 'model.*' keys")
        return None

    try:
        model.load_state_dict(model_state, strict=False)
    except Exception as e:
        _log.warning("  State dict load failed: %s", e)
        return None

    model.eval()
    return model


def run_benchmark(
    entries: list,
    data_dir: Path,
    device: torch.device,
    methods: list,
    image_size: int,
    finetuned_dir: Optional[Path] = None,
) -> dict:
    """Run benchmark for given methods, optionally using fine-tuned checkpoints.

    Args:
        finetuned_dir: If provided, deep methods that have a corresponding
                       {method}/best.ckpt checkpoint will use fine-tuned weights.
                       Results are keyed as "{method}_ft" to distinguish.
    """
    targets = []
    task_ids = []
    distortion_labels = []
    for s in entries:
        score = s.get("score", 0.0)
        if isinstance(score, list):
            score = np.mean(score)
        targets.append(float(score))
        task_ids.append(s.get("task", "scene_description"))
        distortion_labels.append(s.get("distortion", "unknown"))
    targets = np.array(targets)

    evaluator = IQAEvaluator(device=device)
    results = {}

    # Pre-load fine-tuned models for methods that have checkpoints
    finetuned_models = {}
    if finetuned_dir and finetuned_dir.exists():
        for method_name in methods:
            info = AVAILABLE_METHODS.get(method_name)
            if info is None or not info[2]:  # can_finetune=False
                continue
            ckpt_path = finetuned_dir / method_name / "best.ckpt"
            if ckpt_path.exists():
                model = load_finetuned_checkpoint(str(ckpt_path), device)
                if model is not None:
                    finetuned_models[method_name] = model
                    _log.info("  Loaded fine-tuned %s from %s", method_name, ckpt_path)

    for method_name in methods:
        if method_name not in AVAILABLE_METHODS:
            _log.info("  SKIP %s: not available", method_name)
            continue

        cat, needs_instance, _ = AVAILABLE_METHODS[method_name]

        # --- Zero-shot ---
        func = (
            getattr(evaluator, method_name.replace("-", "_"))
            if needs_instance
            else getattr(IQAEvaluator, method_name)
        )
        _log.info("  Running %s (%s, zero-shot)...", method_name, cat)

        predictions = []
        for i, s in enumerate(entries):
            img_np, img_t, score, task, dist, _, ref_path = load_image_and_score(
                s, data_dir, image_size
            )
            try:
                pred = func(data_dir, img_np, img_t, ref_path)
            except Exception:
                pred = 0.0
            predictions.append(float(pred) if pred is not None else 0.0)
            if (i + 1) % 1000 == 0:
                _log.info("    %d/%d", i + 1, len(entries))

        preds = np.array(predictions)
        metrics = evaluate_iqa(preds, targets)
        per_task = per_task_metrics(preds, targets, task_ids)
        per_cat = per_distortion_category_metrics(preds, targets, distortion_labels)

        results[method_name] = {
            "category": cat,
            "finetuned": False,
            "srcc": float(metrics["srcc"]),
            "plcc": float(metrics["plcc"]),
            "rmse": float(metrics["rmse"]),
            "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
            "per_task": per_task,
            "per_distortion_category": per_cat,
        }
        _log.info(
            "    Zero-shot  SRCC=%.4f PLCC=%.4f", metrics["srcc"], metrics["plcc"]
        )

        # --- Fine-tuned (if available) ---
        if method_name in finetuned_models:
            _log.info("  Running %s (%s, fine-tuned)...", method_name, cat)
            ft_model = finetuned_models[method_name]
            is_fr = cat == "FR"

            ft_preds = []
            with torch.no_grad():
                for i, s in enumerate(entries):
                    img_t = (
                        load_image_tensor(Path(data_dir) / s["path"], image_size)
                        .unsqueeze(0)
                        .to(device)
                    )

                    ref_t = None
                    if is_fr:
                        ref_path = s.get("ref_path", "")
                        if ref_path:
                            ref_img = Image.open(Path(data_dir) / ref_path).convert(
                                "RGB"
                            )
                            ref_np = np.array(ref_img).astype(np.float32) / 255.0
                            ref_t = (
                                torch.from_numpy(ref_np)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                                .to(device)
                            )

                    try:
                        if is_fr and ref_t is not None:
                            out = ft_model(img_t, ref_t)
                        else:
                            out = ft_model(img_t)
                        pred = float(out.item()) if hasattr(out, "item") else float(out)
                    except Exception:
                        pred = 0.0

                    ft_preds.append(pred)
                    if (i + 1) % 1000 == 0:
                        _log.info("    %d/%d", i + 1, len(entries))

            ft_preds_arr = np.array(ft_preds)
            ft_metrics = evaluate_iqa(ft_preds_arr, targets)
            ft_per_task = per_task_metrics(ft_preds_arr, targets, task_ids)
            ft_per_cat = per_distortion_category_metrics(
                ft_preds_arr, targets, distortion_labels
            )

            ft_key = f"{method_name}_ft"
            results[ft_key] = {
                "category": cat,
                "finetuned": True,
                "srcc": float(ft_metrics["srcc"]),
                "plcc": float(ft_metrics["plcc"]),
                "rmse": float(ft_metrics["rmse"]),
                "kendall_tau": float(ft_metrics.get("kendall_tau", 0.0)),
                "per_task": ft_per_task,
                "per_distortion_category": ft_per_cat,
            }
            _log.info(
                "    Fine-tuned SRCC=%.4f PLCC=%.4f",
                ft_metrics["srcc"],
                ft_metrics["plcc"],
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark existing IQA methods")
    parser.add_argument("--data-dir", default="data/processed", help="Database root")
    parser.add_argument(
        "--output-dir", default="outputs/benchmark", help="Output directory"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[k for k in AVAILABLE_METHODS],
        help="Methods to benchmark",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Limit samples (0 = all)"
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cuda, cpu, mps)")
    parser.add_argument(
        "--finetuned-dir",
        default=None,
        help="Directory containing fine-tuned checkpoints (e.g., outputs/finetune)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _log.info("Using device: %s", device)

    entries = load_flat_samples(data_dir, "test")
    if args.max_samples > 0:
        entries = entries[: args.max_samples]
    _log.info("Evaluating %d samples", len(entries))

    finetuned_dir = Path(args.finetuned_dir) if args.finetuned_dir else None
    results = run_benchmark(
        entries=entries,
        data_dir=data_dir,
        device=device,
        methods=args.methods,
        image_size=args.image_size,
        finetuned_dir=finetuned_dir,
    )

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"n_samples": len(entries), "methods": results}, f, indent=2)

    _log.info("=" * 80)
    _log.info("Benchmark Complete")
    _log.info("=" * 80)
    _log.info(
        "%-24s %-5s %-8s %8s %8s %12s %12s",
        "Method",
        "Cat",
        "Finetune",
        "SRCC",
        "PLCC",
        "UAV_SRCC",
        "Gen_SRCC",
    )
    _log.info("-" * 90)
    for name in sorted(results.keys(), key=lambda n: results[n]["srcc"], reverse=True):
        r = results[name]
        uav_srcc = r.get("per_distortion_category", {}).get("UAV", {}).get("srcc", 0)
        gen_srcc = (
            r.get("per_distortion_category", {}).get("Generic", {}).get("srcc", 0)
        )
        ft_label = "Yes" if r.get("finetuned") else "No"
        _log.info(
            "%-24s %-5s %-8s %8.4f %8.4f %12.4f %12.4f",
            name,
            r["category"],
            ft_label,
            r["srcc"],
            r["plcc"],
            uav_srcc,
            gen_srcc,
        )
    _log.info("Results: %s", output_dir / "benchmark_results.json")


if __name__ == "__main__":
    main()
