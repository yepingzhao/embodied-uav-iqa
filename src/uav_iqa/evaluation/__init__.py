"""IQA and text-similarity evaluation APIs."""

from .iqa import (
    compute_kendall_tau,
    compute_plcc,
    compute_rmse,
    compute_srcc,
    evaluate_iqa,
    per_distortion_category_metrics,
    per_distortion_metrics,
    per_task_metrics,
)
from .text import compute_bleu, compute_cider, compute_cognitive_score, compute_rouge_l

__all__ = [
    "compute_kendall_tau",
    "compute_bleu",
    "compute_cider",
    "compute_cognitive_score",
    "compute_plcc",
    "compute_rmse",
    "compute_rouge_l",
    "compute_srcc",
    "evaluate_iqa",
    "per_distortion_category_metrics",
    "per_distortion_metrics",
    "per_task_metrics",
]
