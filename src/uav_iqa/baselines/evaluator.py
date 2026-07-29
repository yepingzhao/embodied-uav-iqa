"""IQA baseline evaluator — method registry and zero-shot/fine-tuned benchmarking.

Provides the 15 baseline methods from the Embodied-IQA paper (NeurIPS 2025):
  Zero-shot (5): PSNR, SSIM, Brisque, Q-Align, Q-Align+
  FR (5): AHIQ, CKDN, DISTS, LPIPS, TOPIQ-FR
  NR (5): CLIPIQA, CNNIQA, DBCNN, QualiClip, TOPIQ-NR
"""

import logging
import multiprocessing as mp
import os
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
    "psnr": ("Zero-shot", False, False),
    "ssim": ("Zero-shot", False, False),
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

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    @staticmethod
    def _load_ref_tensor(ref_path, data_dir):
        ref_img = Image.open(Path(data_dir) / ref_path).convert("RGB")
        ref_np = np.array(ref_img).astype(np.float32) / 255.0
        return torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)

    # -- Full-reference (static) --

    @staticmethod
    def psnr(data_dir, img_np, img_t, ref_path):
        from skimage.metrics import peak_signal_noise_ratio
        if not ref_path:
            return 0.0
        ref = np.array(Image.open(Path(data_dir) / ref_path).convert("RGB"))
        return peak_signal_noise_ratio(ref, (img_np * 255).astype(np.uint8), data_range=255)

    @staticmethod
    def ssim(data_dir, img_np, img_t, ref_path):
        from skimage.metrics import structural_similarity
        if not ref_path:
            return 0.0
        ref = np.array(Image.open(Path(data_dir) / ref_path).convert("RGB"))
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

    def _lpips(self, img_t, ref_path, data_dir, net="alex"):
        loss_fn = self._get_lpips(net)
        if loss_fn is None or not ref_path:
            return 0.0
        ref_t = IQAEvaluator._load_ref_tensor(ref_path, data_dir).to(self.device)
        return float(loss_fn(img_t.to(self.device), ref_t).item())

    # -- pyiqa helpers --

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
) -> dict:
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

        predictions = []
        for i, s in enumerate(entries):
            img_np, img_t, _score, _task, _dist, _int, ref_path = load_image_and_score(
                s, data_dir, image_size, image_dir=image_dir
            )
            try:
                pred = func(img_base_dir, img_np, img_t, ref_path)
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
                            ref_img = Image.open(Path(img_base_dir) / ref_path).convert("RGB")
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
) -> None:
    """Run one IQA method on one GPU (called via ``multiprocessing.Process``).

    Sets ``CUDA_VISIBLE_DEVICES`` to isolate the assigned GPU, then runs
    zero-shot and (optionally) fine-tuned scoring.  Results are sent back
    through *result_queue* as ``(method_name, predictions, ft_predictions, error)``.
    """
    gpu_semaphore.acquire()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        info = AVAILABLE_METHODS.get(method_name)
        if info is None:
            result_queue.put((method_name, None, None, f"Unknown method: {method_name}"))
            return
        cat, needs_instance, can_finetune = info

        _log.info("[%s] GPU %d, %d samples", method_name, gpu_id, len(entries))

        # -- Zero-shot scoring --
        evaluator = IQAEvaluator(device=device)
        func = (
            getattr(evaluator, method_name.replace("-", "_"))
            if needs_instance
            else getattr(IQAEvaluator, method_name)
        )

        predictions = []
        for i, s in enumerate(entries):
            img_np, img_t, _score, _task, _dist, _int, ref_path = load_image_and_score(
                s, data_dir, image_size, image_dir=image_dir
            )
            try:
                pred = func(image_dir, img_np, img_t, ref_path)
            except Exception:
                pred = 0.0
            predictions.append(float(pred) if pred is not None else 0.0)
            if (i + 1) % 5000 == 0:
                _log.info("[%s] %d/%d", method_name, i + 1, len(entries))

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
                                    ref_img = Image.open(Path(image_dir) / ref_path).convert("RGB")
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
) -> dict:
    """Run benchmark with methods distributed across GPUs in parallel.

    Each method runs in its own process on an assigned GPU.  Methods are
    distributed round-robin across *gpu_ids*.  A per-GPU semaphore caps
    the number of concurrent methods on each GPU to *num_workers_per_gpu*.

    Returns the same ``dict`` structure as :func:`run_benchmark`.
    """
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

    for worker_id, method_name, gpu_id, info in assignments:
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
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Collect results
    gathered: dict[str, tuple] = {}
    for _ in range(len(assignments)):
        method_name, preds, ft_preds, error = result_queue.get()
        if error:
            _log.error("Method %s failed: %s", method_name, error)
            gathered[method_name] = (None, None)
        else:
            gathered[method_name] = (preds, ft_preds)

    # Wait for all processes
    for p in processes:
        p.join()

    # Compute metrics (main process, CPU)
    results: dict = {}
    for method_name in method_order:
        info = AVAILABLE_METHODS.get(method_name)
        if info is None:
            continue
        cat = info[0]
        pair = gathered.get(method_name)
        if pair is None or pair[0] is None:
            results[method_name] = {
                "category": cat, "finetuned": False,
                "error": "Method failed or not found",
                "srcc": 0.0, "plcc": 0.0, "rmse": 0.0, "kendall_tau": 0.0,
                "per_task": {}, "per_distortion_category": {},
            }
            continue

        preds_arr, ft_preds_arr = np.array(pair[0]), pair[1]
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
        _log.info(
            "  %-24s %-5s   No  %8.4f %8.4f",
            method_name, cat, metrics["srcc"], metrics["plcc"],
        )

        if ft_preds_arr is not None:
            ft_preds_arr = np.array(ft_preds_arr)
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
            _log.info(
                "  %-24s %-5s  Yes  %8.4f %8.4f",
                method_name, cat, ft_metrics["srcc"], ft_metrics["plcc"],
            )

    return results


def _extract_score(sample: dict) -> float:
    """Extract a scalar ground-truth score from a flat sample dict."""
    score = sample.get("score", 0.0)
    if isinstance(score, list):
        score = float(np.mean(score))
    return float(score)

