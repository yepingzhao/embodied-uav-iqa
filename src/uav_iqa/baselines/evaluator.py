"""IQA baseline evaluator — method registry and zero-shot/fine-tuned benchmarking.

Provides the 15 baseline methods from the Embodied-IQA paper (NeurIPS 2025):
  Zero-shot (5): PSNR, SSIM, Brisque, Q-Align, Q-Align+
  FR (5): AHIQ, CKDN, DISTS, LPIPS, TOPIQ-FR
  NR (5): CLIPIQA, CNNIQA, DBCNN, QualiClip, TOPIQ-NR
"""

import json
import logging
import multiprocessing as mp
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from uav_iqa.evaluation import (
    evaluate_iqa,
    per_distortion_category_metrics,
    per_task_metrics,
)
from uav_iqa.utils import load_image_tensor

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_image_and_score(sample, data_dir, image_size=256, image_dir=None):
    img_base = Path(image_dir) if image_dir else Path(data_dir)
    img_path = img_base / sample["path"]
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
# Method registry
# ---------------------------------------------------------------------------

# Registry: name -> (category, needs_instance, can_finetune)
# Matches the 15 baseline methods in the Embodied-IQA paper (NeurIPS 2025):
AVAILABLE_METHODS = {
    "psnr": ("Zero-shot", True, False),
    "ssim": ("Zero-shot", True, False),
    "brisque": ("Zero-shot", True, True),
    "q_align": ("Zero-shot", True, False),
    "q_align_plus": ("Zero-shot", True, False),
    "ahiq": ("FR", True, True),
    "ckdn": ("FR", True, True),
    "dists": ("FR", True, True),
    "lpips": ("FR", True, True),
    "topiq_fr": ("FR", True, True),
    "clip_iqa": ("NR", True, True),
    "cnniqa": ("NR", True, True),
    "dbcnn": ("NR", True, True),
    "qualiclip": ("NR", True, True),
    "topiq_nr": ("NR", True, True),
}

METHOD_TO_PYIQA = {
    "brisque": "brisque",
    "clip_iqa": "clipiqa",
    "q_align": "qalign",
    "q_align_plus": "qalign",
    "topiq_nr": "topiq_nr",
    "ahiq": "ahiq",
    "ckdn": "ckdn",
    "dists": "dists",
    "lpips": "lpips",
    "topiq_fr": "topiq_fr",
    "cnniqa": "cnniqa",
    "dbcnn": "dbcnn",
    "qualiclip": "qualiclip",
}


# ---------------------------------------------------------------------------
# IQAEvaluator
# ---------------------------------------------------------------------------


class IQAEvaluator:
    """Registry of IQA methods with uniform (data_dir, img_np, img_t, ref_path) -> float."""

    def __init__(self, device: torch.device = None, ref_image_dir: str | Path | None = None):
        self.device = device or torch.device("cpu")
        self.ref_image_dir = Path(ref_image_dir) if ref_image_dir else None

    def _ref_base(self, fallback_dir):
        """Directory to resolve reference image paths against.

        Uses ``ref_image_dir`` when set on the evaluator; falls back to
        *fallback_dir* for backward compatibility.
        """
        return self.ref_image_dir or Path(fallback_dir)

    def _load_ref_tensor(self, ref_path, fallback_dir="", target_size=None):
        """Load a reference image as a ``[1, 3, H, W]`` float32 tensor.

        If *target_size* ``(H, W)`` is given, the reference is resized to
        match before conversion.
        """
        base = self._ref_base(fallback_dir)
        ref_img = Image.open(base / ref_path).convert("RGB")
        if target_size is not None:
            ref_img = ref_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        ref_np = np.array(ref_img).astype(np.float32) / 255.0
        return torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)

    # -- Full-reference --

    def psnr(self, data_dir, img_np, img_t, ref_path):
        from skimage.metrics import peak_signal_noise_ratio
        if not ref_path:
            return 0.0
        base = self._ref_base(data_dir)
        h, w = img_np.shape[:2]
        ref_img = Image.open(base / ref_path).convert("RGB").resize((w, h), Image.BILINEAR)
        ref = np.array(ref_img)
        return peak_signal_noise_ratio(ref, (img_np * 255).astype(np.uint8), data_range=255)

    def ssim(self, data_dir, img_np, img_t, ref_path):
        from skimage.metrics import structural_similarity
        if not ref_path:
            return 0.0
        base = self._ref_base(data_dir)
        h, w = img_np.shape[:2]
        ref_img = Image.open(base / ref_path).convert("RGB").resize((w, h), Image.BILINEAR)
        ref = np.array(ref_img)
        return structural_similarity(ref, (img_np * 255).astype(np.uint8), channel_axis=2, data_range=255)

    # -- LPIPS --

    def lpips(self, data_dir, img_np, img_t, ref_path):
        return self._lpips(img_t, ref_path, data_dir, net="alex")

    @lru_cache(maxsize=2)
    def _get_lpips(self, net: str):
        try:
            import lpips
            return lpips.LPIPS(net=net, verbose=False).to(self.device)
        except ImportError:
            return None
        except Exception as e:
            _log.warning("Failed to create LPIPS metric (net=%s): %s", net, e)
            return None

    def _lpips(self, img_t, ref_path, data_dir, net="alex"):
        loss_fn = self._get_lpips(net)
        if loss_fn is None or not ref_path:
            return 0.0
        _, _, h, w = img_t.shape
        ref_t = self._load_ref_tensor(ref_path, data_dir, target_size=(h, w)).to(self.device)
        return float(loss_fn(img_t.to(self.device), ref_t).item())

    # -- pyiqa helpers --

    @lru_cache(maxsize=8)
    def _get_pyiqa_metric(self, metric_name: str):
        try:
            import pyiqa
            return pyiqa.create_metric(metric_name, device=self.device)
        except ImportError:
            return None
        except Exception as e:
            _log.warning("Failed to create pyiqa metric '%s': %s", metric_name, e)
            return None

    def _pyiqa_fr(self, data_dir, img_t, ref_path, metric_name):
        if not ref_path:
            return 0.0
        metric = self._get_pyiqa_metric(metric_name)
        if metric is None:
            return 0.0
        _, _, h, w = img_t.shape
        ref_t = self._load_ref_tensor(ref_path, data_dir, target_size=(h, w)).to(self.device)
        return float(metric(img_t.to(self.device), ref_t).item())

    def _pyiqa_nr(self, img_t, metric_name):
        metric = self._get_pyiqa_metric(metric_name)
        if metric is None:
            return 0.0
        return float(metric(img_t.to(self.device)).item())

    # -- FR methods --

    def ahiq(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "ahiq")

    def dists(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "dists")

    def topiq_fr(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "topiq_fr")

    def ckdn(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_fr(data_dir, img_t, ref_path, "ckdn")

    # -- NR methods --

    def brisque(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "brisque")

    def clip_iqa(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "clipiqa")

    def q_align(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "qalign")

    def q_align_plus(self, data_dir, img_np, img_t, ref_path):
        metric = self._get_pyiqa_metric("qalign")
        if metric is None:
            return 0.0
        with torch.no_grad():
            return float(metric.forward(img_t.to(self.device), task_="aesthetic").item())

    def topiq_nr(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "topiq_nr")

    def cnniqa(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "cnniqa")

    def dbcnn(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "dbcnn")

    def qualiclip(self, data_dir, img_np, img_t, ref_path):
        return self._pyiqa_nr(img_t, "qualiclip")


# ---------------------------------------------------------------------------
# PyTorch batch PSNR / SSIM (equivalent to skimage, no extra dependencies)
# ---------------------------------------------------------------------------


def _psnr_batch(img: torch.Tensor, ref: torch.Tensor) -> list[float]:
    """PSNR for ``[B, C, H, W]`` tensors in [0, 1].  Returns list of *B* floats."""
    mse = torch.mean((img - ref) ** 2, dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-8))).tolist()


def _ssim_batch(img: torch.Tensor, ref: torch.Tensor,
                window_size: int = 11, sigma: float = 1.5) -> list[float]:
    """SSIM for ``[B, C, H, W]`` tensors in [0, 1].  Returns list of *B* floats."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = img.shape

    coords = torch.arange(window_size, dtype=torch.float32, device=img.device)
    coords = coords - window_size / 2 + 0.5
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    g2d = g.unsqueeze(0) * g.unsqueeze(1)
    window = g2d.expand(C, 1, window_size, window_size)

    pad = window_size // 2

    def _conv(t):
        return torch.nn.functional.conv2d(
            torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="reflect"),
            window, groups=C,
        )

    mu1 = _conv(img)
    mu2 = _conv(ref)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = _conv(img * img) - mu1_sq
    sigma2_sq = _conv(ref * ref) - mu2_sq
    sigma12 = _conv(img * ref) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3]).tolist()


# ---------------------------------------------------------------------------
# Batch scoring helpers
# ---------------------------------------------------------------------------


def _should_use_batching(method_name: str, batch_size: int) -> bool:
    """Return ``True`` when *method_name* supports batched inference."""
    if method_name in ("q_align", "q_align_plus"):
        return False  # q_align asserts batch_size == 1
    pyiqa_name = METHOD_TO_PYIQA.get(method_name)
    return batch_size > 1 and (
        pyiqa_name is not None or method_name in ("lpips", "psnr", "ssim")
    )


def _score_method_entries(
    evaluator: "IQAEvaluator",
    entries: list,
    data_dir: str,
    image_dir: str,
    image_base: str,
    image_size: int,
    method_name: str,
    cat: str,
    batch_size: int,
    func,
    *,
    log_prefix: str = "",
    log_interval: int = 1000,
) -> list[float]:
    """Score *entries* with *method_name* using either batch or per-sample path.

    Routes to :func:`_score_batched` when batching is viable, otherwise falls
    back to a per-sample loop calling *func*.
    """
    if _should_use_batching(method_name, batch_size):
        return _score_batched(
            evaluator, entries, image_base, image_size,
            method_name, cat, batch_size,
        )

    predictions: list[float] = []
    for i, s in enumerate(entries):
        img_np, img_t, _score, _task, _dist, _int, ref_path = load_image_and_score(
            s, data_dir, image_size, image_dir=image_dir,
        )
        try:
            pred = func(image_base, img_np, img_t, ref_path)
        except Exception:
            pred = 0.0
        predictions.append(float(pred) if pred is not None else 0.0)
        if (i + 1) % log_interval == 0:
            _log.info("%s%d/%d", log_prefix, i + 1, len(entries))
    return predictions


def _score_batched(
    evaluator: "IQAEvaluator",
    entries: list,
    image_dir: str,
    image_size: int,
    method_name: str,
    cat: str,
    batch_size: int,
) -> list[float]:
    """Score *entries* in batches using a pyiqa metric or LPIPS.

    Returns a flat list of scalar scores (one per entry).
    """
    predictions: list[float] = []
    pyiqa_name = METHOD_TO_PYIQA.get(method_name)
    is_qalign_plus = method_name == "q_align_plus"
    is_lpips = method_name == "lpips"
    is_skimage = method_name in ("psnr", "ssim")
    is_fr = cat == "FR" or is_skimage  # PSNR/SSIM need reference images

    # -- acquire metric once (cached inside evaluator) --
    metric = None
    if is_lpips:
        metric = evaluator._get_lpips("alex")
    elif not is_skimage:
        metric = evaluator._get_pyiqa_metric(pyiqa_name)

    if metric is None and not is_skimage:
        return [0.0] * len(entries)

    total = len(entries)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_slice = entries[batch_start:batch_end]
        count = len(batch_slice)

        if batch_start % 5000 == 0:
            _log.info("    %d/%d", batch_start, total)

        img_batch: list[torch.Tensor] = []
        ref_batch: list[torch.Tensor] = [] if is_fr else None  # type: ignore[assignment]

        for s in batch_slice:
            img_path = Path(image_dir) / s["path"]
            img_t = load_image_tensor(img_path, image_size)
            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)
            img_batch.append(img_t)

            if is_fr:
                ref_path = s.get("ref_path", "")
                if ref_path:
                    ref_t = evaluator._load_ref_tensor(
                        ref_path, image_dir, target_size=(image_size, image_size)
                    )
                else:
                    ref_t = torch.zeros(1, 3, image_size, image_size)
                ref_batch.append(ref_t)  # type: ignore[union-attr]

        batch_t = torch.cat(img_batch, dim=0).to(evaluator.device)

        try:
            if is_lpips:
                batch_ref = torch.cat(ref_batch, dim=0).to(evaluator.device)  # type: ignore[arg-type]
                out = metric(batch_t, batch_ref)
            elif is_skimage:
                batch_ref = torch.cat(ref_batch, dim=0).to(evaluator.device)  # type: ignore[arg-type]
                if method_name == "psnr":
                    out = _psnr_batch(batch_t, batch_ref)
                else:
                    out = _ssim_batch(batch_t, batch_ref)
            elif is_fr:
                batch_ref = torch.cat(ref_batch, dim=0).to(evaluator.device)  # type: ignore[arg-type]
                out = metric(batch_t, batch_ref)
            elif is_qalign_plus:
                out = metric.forward(batch_t, task_="aesthetic")
            else:
                out = metric(batch_t)

            if isinstance(out, list):
                predictions.extend(out)
            elif hasattr(out, "squeeze"):
                vals = out.squeeze()
                if vals.dim() == 0:
                    vals = vals.unsqueeze(0)
                predictions.extend([float(v) for v in vals.detach().cpu()])
            else:
                predictions.extend([float(out)])
        except Exception as e:
            _log.warning("  Batch [%d:%d] failed for %s: %s: %s",
                         batch_start, batch_end, method_name, type(e).__name__, e)
            predictions.extend([0.0] * count)

    return predictions


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_finetuned_checkpoint(ckpt_path: str, device: torch.device):
    import pyiqa
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        _log.warning("  Failed to load checkpoint %s: %s", ckpt_path, e)
        return None
    state_dict = ckpt.get("state_dict", ckpt)
    hparams = ckpt.get("hyper_parameters", {})
    method_name = hparams.get("method_name") if isinstance(hparams, dict) else None
    if method_name is None:
        _log.warning("  Checkpoint missing method_name in hyper_parameters")
        return None
    pyiqa_name = METHOD_TO_PYIQA.get(method_name, method_name)
    model = pyiqa.create_metric(pyiqa_name, device=device)
    model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    entries: list,
    data_dir: Path,
    device: torch.device,
    methods: list,
    image_size: int,
    finetuned_dir: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    ref_image_dir: Optional[Path] = None,
    batch_size: int = 1,
) -> dict:
    # Reduce CUDA memory fragmentation to avoid OOM with large batch sizes.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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

    evaluator = IQAEvaluator(device=device, ref_image_dir=ref_image_dir)
    results = {}

    finetuned_models = {}
    if finetuned_dir and finetuned_dir.exists():
        for method_name in methods:
            info = AVAILABLE_METHODS.get(method_name)
            if info is None or not info[2]:
                continue
            ckpt_path = finetuned_dir / method_name / "best.ckpt"
            if ckpt_path.exists():
                model = load_finetuned_checkpoint(str(ckpt_path), device)
                if model is not None:
                    finetuned_models[method_name] = model
                    _log.info("  Loaded fine-tuned %s from %s", method_name, ckpt_path)

    img_base_dir = image_dir or data_dir
    ref_base = ref_image_dir or img_base_dir

    for method_name in methods:
        info = AVAILABLE_METHODS.get(method_name)
        if info is None:
            _log.info("  SKIP %s: not available", method_name)
            continue
        cat, needs_instance, _can_finetune = info

        func = (
            getattr(evaluator, method_name.replace("-", "_"))
            if needs_instance
            else getattr(IQAEvaluator, method_name)
        )
        _log.info("  Running %s (%s, zero-shot)...", method_name, cat)

        predictions = _score_method_entries(
            evaluator, entries, str(data_dir), str(image_dir or data_dir),
            str(img_base_dir), image_size, method_name, cat, batch_size,
            func, log_prefix="    ", log_interval=1000,
        )

        preds = np.array(predictions)
        metrics = evaluate_iqa(preds, targets)
        per_task = per_task_metrics(preds, targets, task_ids)
        per_cat = per_distortion_category_metrics(preds, targets, distortion_labels)
        results[method_name] = {
            "category": cat, "finetuned": False,
            "srcc": float(metrics["srcc"]), "plcc": float(metrics["plcc"]),
            "rmse": float(metrics["rmse"]),
            "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
            "per_task": per_task, "per_distortion_category": per_cat,
        }
        _log.info("    Zero-shot  SRCC=%.4f PLCC=%.4f", metrics["srcc"], metrics["plcc"])

        if method_name in finetuned_models:
            _log.info("  Running %s (%s, fine-tuned)...", method_name, cat)
            ft_model = finetuned_models[method_name]
            is_fr = cat == "FR"
            ft_preds = []
            with torch.no_grad():
                for i, s in enumerate(entries):
                    img_t = load_image_tensor(Path(img_base_dir) / s["path"], image_size).unsqueeze(0).to(device)
                    ref_t = None
                    if is_fr:
                        ref_path = s.get("ref_path", "")
                        if ref_path:
                            ref_img = Image.open(ref_base / ref_path).convert("RGB")
                            ref_np = np.array(ref_img).astype(np.float32) / 255.0
                            ref_t = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)
                    try:
                        out = ft_model(img_t, ref_t) if (is_fr and ref_t is not None) else ft_model(img_t)
                        pred = float(out.item()) if hasattr(out, "item") else float(out)
                    except Exception:
                        pred = 0.0
                    ft_preds.append(pred)
                    if (i + 1) % 1000 == 0:
                        _log.info("    %d/%d", i + 1, len(entries))
            ft_preds_arr = np.array(ft_preds)
            ft_metrics = evaluate_iqa(ft_preds_arr, targets)
            ft_per_task = per_task_metrics(ft_preds_arr, targets, task_ids)
            ft_per_cat = per_distortion_category_metrics(ft_preds_arr, targets, distortion_labels)
            ft_key = f"{method_name}_ft"
            results[ft_key] = {
                "category": cat, "finetuned": True,
                "srcc": float(ft_metrics["srcc"]), "plcc": float(ft_metrics["plcc"]),
                "rmse": float(ft_metrics["rmse"]),
                "kendall_tau": float(ft_metrics.get("kendall_tau", 0.0)),
                "per_task": ft_per_task, "per_distortion_category": ft_per_cat,
            }
            _log.info("    Fine-tuned SRCC=%.4f PLCC=%.4f", ft_metrics["srcc"], ft_metrics["plcc"])

    return results


# ---------------------------------------------------------------------------
# Parallel multi-GPU benchmark
# ---------------------------------------------------------------------------


def _benchmark_worker(
    method_name: str,
    gpu_id: int,
    gpu_semaphore: mp.Semaphore,
    entries: list,
    data_dir: str,
    image_dir: str,
    image_size: int,
    finetuned_dir: str | None,
    result_queue: mp.Queue,
    ref_image_dir: str | None = None,
    batch_size: int = 1,
) -> None:
    """Run one IQA method on one GPU (called via ``multiprocessing.Process``).

    Sets ``CUDA_VISIBLE_DEVICES`` to isolate the assigned GPU, then runs
    zero-shot and (optionally) fine-tuned scoring.  Results are sent back
    through *result_queue* as ``(method_name, predictions, ft_predictions, error)``.
    """
    gpu_semaphore.acquire()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Reduce CUDA memory fragmentation to avoid OOM with large batch sizes
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        info = AVAILABLE_METHODS.get(method_name)
        if info is None:
            result_queue.put((method_name, None, None, f"Unknown method: {method_name}"))
            return
        cat, needs_instance, can_finetune = info

        _log.info("[%s] GPU %d, %d samples", method_name, gpu_id, len(entries))

        ref_base = Path(ref_image_dir) if ref_image_dir else Path(image_dir)

        # -- Zero-shot scoring --
        evaluator = IQAEvaluator(device=device, ref_image_dir=ref_image_dir)
        func = (
            getattr(evaluator, method_name.replace("-", "_"))
            if needs_instance
            else getattr(IQAEvaluator, method_name)
        )

        predictions = _score_method_entries(
            evaluator, entries, data_dir, image_dir, image_dir,
            image_size, method_name, cat, batch_size, func,
            log_prefix=f"[{method_name}] ", log_interval=5000,
        )

        # -- Fine-tuned scoring --
        ft_predictions = None
        if finetuned_dir and can_finetune:
            ckpt_path = Path(finetuned_dir) / method_name / "best.ckpt"
            if ckpt_path.exists():
                ft_model = load_finetuned_checkpoint(str(ckpt_path), device)
                if ft_model is not None:
                    is_fr = cat == "FR"
                    ft_predictions = []
                    with torch.no_grad():
                        for i, s in enumerate(entries):
                            img_t = load_image_tensor(
                                Path(image_dir) / s["path"], image_size
                            ).unsqueeze(0).to(device)
                            ref_t = None
                            if is_fr:
                                ref_path = s.get("ref_path", "")
                                if ref_path:
                                    ref_img = Image.open(ref_base / ref_path).convert("RGB")
                                    ref_np = np.array(ref_img).astype(np.float32) / 255.0
                                    ref_t = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)
                            try:
                                out = ft_model(img_t, ref_t) if (is_fr and ref_t is not None) else ft_model(img_t)
                                pred = float(out.item()) if hasattr(out, "item") else float(out)
                            except Exception:
                                pred = 0.0
                            ft_predictions.append(pred)
                            if (i + 1) % 5000 == 0:
                                _log.info("[%s_ft] %d/%d", method_name, i + 1, len(entries))

        result_queue.put((method_name, predictions, ft_predictions, None))
        _log.info("[%s] Done", method_name)

    except Exception as e:
        _log.error("[%s] Failed: %s", method_name, e, exc_info=True)
        result_queue.put((method_name, None, None, str(e)))
    finally:
        gpu_semaphore.release()


def run_benchmark_parallel(
    entries: list,
    data_dir: Path,
    gpu_ids: list[int],
    methods: list,
    image_size: int = 256,
    num_workers_per_gpu: int = 4,
    finetuned_dir: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    ref_image_dir: Optional[Path] = None,
    batch_size: int = 1,
) -> dict:
    """Run benchmark with methods distributed across GPUs in parallel.

    Each method runs in its own process on an assigned GPU.  Methods are
    distributed round-robin across *gpu_ids*.  A per-GPU semaphore caps
    the number of concurrent methods on each GPU to *num_workers_per_gpu*.

    Returns the same ``dict`` structure as :func:`run_benchmark`.
    """
    # Reduce CUDA memory fragmentation (avoids OOM with large batch sizes,
    # particularly for AHIQ which uses CFANet with large intermediate tensors).
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Pre-compute ground truth (CPU)
    targets = np.array([_extract_score(s) for s in entries])
    task_ids = [s.get("task", "scene_description") for s in entries]
    distortion_labels = [s.get("distortion", "unknown") for s in entries]

    # Round-robin method → GPU assignment
    assignments: list[tuple[int, str, int, tuple]] = []
    method_order: list[str] = []
    for i, method_name in enumerate(methods):
        info = AVAILABLE_METHODS.get(method_name)
        if info is None:
            _log.warning("SKIP %s: not available", method_name)
            continue
        gpu_id = gpu_ids[i % len(gpu_ids)]
        assignments.append((i, method_name, gpu_id, info))
        method_order.append(method_name)

    if not assignments:
        return {}

    _log.info(
        "Launching %d methods across GPUs %s (max %d per GPU)",
        len(assignments), gpu_ids, num_workers_per_gpu,
    )

    # Per-GPU semaphores
    gpu_semaphores: dict[int, mp.Semaphore] = {
        gid: mp.Semaphore(num_workers_per_gpu) for gid in gpu_ids
    }

    result_queue: mp.Queue = mp.Queue()
    processes: list[mp.Process] = []

    data_dir_str = str(data_dir)
    image_dir_str = str(image_dir) if image_dir else data_dir_str
    ft_dir_str = str(finetuned_dir) if finetuned_dir else None
    ref_image_dir_str = str(ref_image_dir) if ref_image_dir else None

    for _, method_name, gpu_id, info in assignments:
        p = mp.Process(
            target=_benchmark_worker,
            args=(
                method_name,
                gpu_id,
                gpu_semaphores[gpu_id],
                entries,
                data_dir_str,
                image_dir_str,
                image_size,
                ft_dir_str,
                result_queue,
                ref_image_dir_str,
                batch_size,
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Collect results — compute metrics and save incrementally
    results: dict = {}
    results_file = Path(os.environ.get("BENCHMARK_RESULTS_JSON",
                                       "benchmark_results_partial.json"))
    for _ in range(len(assignments)):
        method_name, preds, ft_preds, error = result_queue.get()
        info = AVAILABLE_METHODS.get(method_name)
        cat = info[0] if info else "?"
        if error or preds is None:
            _log.error("Method %s failed: %s", method_name, error)
            results[method_name] = {
                "category": cat, "finetuned": False,
                "error": error or "Method failed",
                "srcc": 0.0, "plcc": 0.0, "rmse": 0.0, "kendall_tau": 0.0,
                "per_task": {}, "per_distortion_category": {},
            }
        else:
            preds_arr = np.array(preds)
            metrics = evaluate_iqa(preds_arr, targets)
            per_task = per_task_metrics(preds_arr, targets, task_ids)
            per_cat = per_distortion_category_metrics(preds_arr, targets, distortion_labels)
            results[method_name] = {
                "category": cat, "finetuned": False,
                "srcc": float(metrics["srcc"]), "plcc": float(metrics["plcc"]),
                "rmse": float(metrics["rmse"]),
                "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
                "per_task": per_task, "per_distortion_category": per_cat,
            }
            _log.info("  %-24s %-5s  Zero-shot  %8.4f %8.4f",
                      method_name, cat, metrics["srcc"], metrics["plcc"])
            # Save incrementally — survives interruption
            results_file.write_text(json.dumps(
                {"n_samples": len(entries), "methods": results}, indent=2,
            ))

    # Wait for all processes
    for p in processes:
        p.join()

    return results


def _extract_score(sample: dict) -> float:
    """Extract a scalar ground-truth score from a flat sample dict."""
    score = sample.get("score", 0.0)
    if isinstance(score, list):
        score = float(np.mean(score))
    return float(score)
