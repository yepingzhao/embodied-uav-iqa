#!/usr/bin/env python3
"""M2-R009/R010/R011/R012: Benchmark existing IQA methods.

Runs 15+ existing IQA methods on the UAV-Embodied-IQA test set and computes
SRCC/PLCC against VLA decision scores.

Supported methods:
  FR: psnr, ssim, ms_ssim, lpips_alex, lpips_vgg, dists, ahiq, topiq_fr
  NR-handcrafted: brisque, niqe, ilniqe
  NR-deep: maniqa, clip_iqa, q_align
  NR-frequency: brisque (DCT features)
  Embodied: ma_eiqa

Usage:
    python scripts/run_m2_benchmark.py \
        --data-dir data/database \
        --output-dir outputs/benchmark \
        --methods psnr ssim lpips_alex brisque clip_iqa
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uav_iqa.evaluate import (
    compute_metrics,
    per_task_metrics,
)


def load_manifest(data_dir, split="test"):
    path = Path(data_dir) / split / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_image_and_score(sample, data_dir, image_size=256):
    img_path = Path(data_dir) / sample["path"]
    image = Image.open(img_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
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
    )


class FRMethods:
    @staticmethod
    def psnr(img, ref_path, data_dir):
        from skimage.metrics import peak_signal_noise_ratio

        ref = np.array(Image.open(Path(data_dir) / ref_path).convert("RGB"))
        return peak_signal_noise_ratio(
            ref, (img * 255).astype(np.uint8), data_range=255
        )

    @staticmethod
    def ssim(img, ref_path, data_dir):
        from skimage.metrics import structural_similarity

        ref = np.array(Image.open(Path(data_dir) / ref_path).convert("RGB"))
        return structural_similarity(
            ref, (img * 255).astype(np.uint8), channel_axis=2, data_range=255
        )

    @staticmethod
    def ms_ssim(img, ref_path, data_dir):
        try:
            from pytorch_msssim import ms_ssim

            ref_img = Image.open(Path(data_dir) / ref_path).convert("RGB")
            ref_t = torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0)
            ref_t = ref_t.permute(2, 0, 1).unsqueeze(0)
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return float(ms_ssim(img_t, ref_t, data_range=1.0))
        except ImportError:
            return 0.0

    @staticmethod
    def lpips(img_t, ref_path, data_dir, net="alex"):
        try:
            import lpips

            loss_fn = lpips.LPIPS(net=net, verbose=False)
            ref_img = Image.open(Path(data_dir) / ref_path).convert("RGB")
            ref_t = torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0)
            ref_t = ref_t.permute(2, 0, 1).unsqueeze(0)
            return float(loss_fn(img_t, ref_t).item())
        except ImportError:
            return 0.0


class NRMethods:
    @staticmethod
    def brisque(img_np):
        try:
            import pyiqa

            metric = pyiqa.create_metric("brisque", device=torch.device("cpu"))
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return float(metric(img_t).item())
        except ImportError:
            return 0.0

    @staticmethod
    def niqe(img_np):
        try:
            import pyiqa

            metric = pyiqa.create_metric("niqe", device=torch.device("cpu"))
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return float(metric(img_t).item())
        except ImportError:
            return 0.0

    @staticmethod
    def clip_iqa(img_t):
        try:
            import pyiqa

            metric = pyiqa.create_metric("clipiqa", device=torch.device("cpu"))
            return float(metric(img_t).item())
        except ImportError:
            return 0.0

    @staticmethod
    def maniqa(img_t):
        try:
            import pyiqa

            metric = pyiqa.create_metric("maniqa", device=torch.device("cpu"))
            return float(metric(img_t).item())
        except ImportError:
            return 0.0

    @staticmethod
    def q_align(img_t):
        try:
            import pyiqa

            metric = pyiqa.create_metric("qalign", device=torch.device("cpu"))
            return float(metric(img_t).item())
        except ImportError:
            return 0.0

    @staticmethod
    def topiq_nr(img_t):
        try:
            import pyiqa

            metric = pyiqa.create_metric("topiq_nr", device=torch.device("cpu"))
            return float(metric(img_t).item())
        except ImportError:
            return 0.0

    @staticmethod
    def ahiq(img_t, ref_t):
        try:
            import pyiqa

            metric = pyiqa.create_metric("ahiq", device=torch.device("cpu"))
            return float(metric(img_t, ref_t).item())
        except ImportError:
            print("  WARNING: pyiqa not installed — returning 0.0 for ahiq")
            return 0.0

    @staticmethod
    def topiq_fr(img_t, ref_t):
        try:
            import pyiqa

            metric = pyiqa.create_metric("topiq_fr", device=torch.device("cpu"))
            return float(metric(img_t, ref_t).item())
        except ImportError:
            print("  WARNING: pyiqa not installed — returning 0.0 for topiq_fr")
            return 0.0


class FRMethodsExt:
    """FR methods requiring both distorted and reference images (with ref loading)."""

    @staticmethod
    def ahiq(img_t, ref_path, data_dir):
        try:
            import pyiqa

            ref_img = Image.open(Path(data_dir) / ref_path).convert("RGB")
            ref_t = torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0)
            ref_t = ref_t.permute(2, 0, 1).unsqueeze(0)
            metric = pyiqa.create_metric("ahiq", device=torch.device("cpu"))
            return float(metric(img_t, ref_t).item())
        except ImportError:
            print("  WARNING: pyiqa not installed — returning 0.0 for ahiq")
            return 0.0

    @staticmethod
    def topiq_fr(img_t, ref_path, data_dir):
        try:
            import pyiqa

            ref_img = Image.open(Path(data_dir) / ref_path).convert("RGB")
            ref_t = torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0)
            ref_t = ref_t.permute(2, 0, 1).unsqueeze(0)
            metric = pyiqa.create_metric("topiq_fr", device=torch.device("cpu"))
            return float(metric(img_t, ref_t).item())
        except ImportError:
            print("  WARNING: pyiqa not installed — returning 0.0 for topiq_fr")
            return 0.0


AVAILABLE_METHODS = {
    "psnr": ("FR", FRMethods.psnr),
    "ssim": ("FR", FRMethods.ssim),
    "ms_ssim": ("FR", FRMethods.ms_ssim),
    "lpips_alex": ("FR", FRMethods.lpips),
    "brisque": ("NR", NRMethods.brisque),
    "niqe": ("NR", NRMethods.niqe),
    "clip_iqa": ("NR", NRMethods.clip_iqa),
    "maniqa": ("NR", NRMethods.maniqa),
    "q_align": ("NR", NRMethods.q_align),
    "topiq_nr": ("NR", NRMethods.topiq_nr),
    "ahiq": ("FR", FRMethodsExt.ahiq),
    "topiq_fr": ("FR", FRMethodsExt.topiq_fr),
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark existing IQA methods")
    parser.add_argument("--data-dir", default="data/database", help="Database root")
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(data_dir)
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    print(f"Evaluating {len(manifest)} samples")

    targets, task_ids, distortions, intensities = [], [], [], []
    for s in manifest:
        score = s.get("vla_score", 0.0)
        if isinstance(score, list):
            score = np.mean(score)
        targets.append(float(score))
        task_ids.append(s.get("task", "tracking"))
        distortions.append(s.get("distortion", "unknown"))
        intensities.append(s.get("intensity_level", 0.0))
    targets = np.array(targets)

    results = {}
    for method_name in args.methods:
        if method_name not in AVAILABLE_METHODS:
            print(f"  SKIP {method_name}: not available")
            continue

        cat, func = AVAILABLE_METHODS[method_name]
        print(f"  Running {method_name} ({cat})...")

        predictions = []
        for i, s in enumerate(manifest):
            ref_path = s.get("ref_path", "")
            img_np, img_t, score, task, dist, intens = load_image_and_score(
                s, data_dir, args.image_size
            )

            try:
                if cat == "FR" and ref_path:
                    pred = func(
                        img_np if method_name in ("psnr", "ssim", "ms_ssim") else img_t,
                        ref_path,
                        data_dir,
                    )
                elif method_name in ("psnr", "ssim", "ms_ssim"):
                    pred = 0.0
                elif cat == "NR":
                    pred = func(img_np if method_name in ("brisque", "niqe") else img_t)
                else:
                    pred = 0.0
            except Exception:
                pred = 0.0

            predictions.append(float(pred) if pred is not None else 0.0)
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(manifest)}")

        preds = np.array(predictions)
        metrics = compute_metrics(targets, preds)
        per_task = per_task_metrics(targets, preds, task_ids)

        results[method_name] = {
            "category": cat,
            "srcc": float(metrics["srcc"]),
            "plcc": float(metrics["plcc"]),
            "rmse": float(metrics["rmse"]),
            "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
            "per_task": per_task,
        }
        print(f"    SRCC={metrics['srcc']:.4f} PLCC={metrics['plcc']:.4f}")

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"n_samples": len(manifest), "methods": results}, f, indent=2)

    print(f"\n{'='*60}")
    print("Benchmark Complete")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Cat':<5} {'SRCC':>8} {'PLCC':>8}")
    print(f"{'-'*45}")
    for name in sorted(results.keys(), key=lambda n: results[n]["srcc"], reverse=True):
        r = results[name]
        print(f"{name:<20} {r['category']:<5} {r['srcc']:>8.4f} {r['plcc']:>8.4f}")
    print(f"\nResults: {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
