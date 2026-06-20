#!/usr/bin/env python3
"""M2-R009/R010/R011/R012: Benchmark existing IQA methods.

Runs 15+ existing IQA methods on the UAV-Embodied-IQA test set and computes
SRCC/PLCC against VLA decision scores. Models are cached per method to avoid
re-instantiation overhead on large test sets.

Supported methods:
  FR: psnr, ssim, ms_ssim, lpips_alex, lpips_vgg, dists, ahiq, topiq_fr
  NR-handcrafted: brisque, niqe, ilniqe
  NR-deep: maniqa, clip_iqa, q_align, topiq_nr

Usage:
    python scripts/benchmark_iqa_methods.py \
        --data-dir data/processed \
        --output-dir outputs/benchmark \
        --methods psnr ssim lpips_alex brisque clip_iqa
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uav_iqa.metrics import (
    evaluate_iqa,
    per_task_metrics,
    per_distortion_category_metrics,
)
from uav_iqa.utils import load_image_tensor, load_manifest, setup_logging

_log = setup_logging(__name__)


def load_image_and_score(sample, data_dir, image_size=256):
    img_path = Path(data_dir) / sample["path"]
    image_tensor = load_image_tensor(img_path, image_size).unsqueeze(0)
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    score = sample.get("vla_score", 0.0)
    if isinstance(score, list):
        score = np.mean(score)
    return (
        image_np,
        image_tensor,
        float(score),
        sample.get("task", "tracking"),
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

    def topiq_fr(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "topiq_fr")

    # -- No-reference methods --

    def brisque(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "brisque")

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


# Registry: name -> (category, needs_instance)
AVAILABLE_METHODS = {
    "psnr": ("FR", False),
    "ssim": ("FR", False),
    "ms_ssim": ("FR", False),
    "lpips_alex": ("FR", True),
    "lpips_vgg": ("FR", True),
    "ahiq": ("FR", True),
    "topiq_fr": ("FR", True),
    "brisque": ("NR", True),
    "niqe": ("NR", True),
    "clip_iqa": ("NR", True),
    "maniqa": ("NR", True),
    "q_align": ("NR", True),
    "topiq_nr": ("NR", True),
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark existing IQA methods")
    parser.add_argument("--data-dir", default="data/processed", help="Database root")
    parser.add_argument(
        "--output-dir", default="outputs/benchmark", help="Output directory"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(AVAILABLE_METHODS.keys()),
        help="Methods to benchmark",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="Limit samples (0 = all)"
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cuda, cpu, mps)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _log.info("Using device: %s", device)

    manifest_path = data_dir / "test" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    _log.info("Evaluating %d samples", len(manifest))

    targets, task_ids, distortion_labels = [], [], []
    for s in manifest:
        score = s.get("vla_score", 0.0)
        if isinstance(score, list):
            score = np.mean(score)
        targets.append(float(score))
        task_ids.append(s.get("task", "tracking"))
        distortion_labels.append(s.get("distortion", "unknown"))
    targets = np.array(targets)

    evaluator = IQAEvaluator(device=device)

    results = {}
    for method_name in args.methods:
        if method_name not in AVAILABLE_METHODS:
            _log.info("  SKIP %s: not available", method_name)
            continue

        cat, needs_instance = AVAILABLE_METHODS[method_name]
        func = (
            getattr(evaluator, method_name.replace("-", "_"))
            if needs_instance
            else getattr(IQAEvaluator, method_name)
        )
        _log.info("  Running %s (%s)...", method_name, cat)

        predictions = []
        for i, s in enumerate(manifest):
            img_np, img_t, score, task, dist, intens, ref_path = load_image_and_score(
                s, data_dir, args.image_size
            )

            try:
                pred = func(data_dir, img_np, img_t, ref_path)
            except Exception:
                pred = 0.0

            predictions.append(float(pred) if pred is not None else 0.0)
            if (i + 1) % 1000 == 0:
                _log.info("    %d/%d", i + 1, len(manifest))

        preds = np.array(predictions)
        metrics = evaluate_iqa(preds, targets)
        per_task = per_task_metrics(preds, targets, task_ids)
        per_cat = per_distortion_category_metrics(preds, targets, distortion_labels)

        results[method_name] = {
            "category": cat,
            "srcc": float(metrics["srcc"]),
            "plcc": float(metrics["plcc"]),
            "rmse": float(metrics["rmse"]),
            "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
            "per_task": per_task,
            "per_distortion_category": per_cat,
        }
        _log.info("    SRCC=%.4f PLCC=%.4f", metrics["srcc"], metrics["plcc"])

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"n_samples": len(manifest), "methods": results}, f, indent=2)

    _log.info("=" * 60)
    _log.info("Benchmark Complete")
    _log.info("=" * 60)
    _log.info(
        "%-20s %-5s %8s %8s %12s %12s",
        "Method",
        "Cat",
        "SRCC",
        "PLCC",
        "UAV_SRCC",
        "Gen_SRCC",
    )
    _log.info("-" * 75)
    for name in sorted(results.keys(), key=lambda n: results[n]["srcc"], reverse=True):
        r = results[name]
        uav_srcc = r.get("per_distortion_category", {}).get("UAV", {}).get("srcc", 0)
        gen_srcc = (
            r.get("per_distortion_category", {}).get("Generic", {}).get("srcc", 0)
        )
        _log.info(
            "%-20s %-5s %8.4f %8.4f %12.4f %12.4f",
            name,
            r["category"],
            r["srcc"],
            r["plcc"],
            uav_srcc,
            gen_srcc,
        )
    _log.info("Results: %s", output_dir / "benchmark_results.json")


if __name__ == "__main__":
    main()
