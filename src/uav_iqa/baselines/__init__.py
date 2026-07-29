"""Baseline IQA methods from the Embodied-IQA paper (NeurIPS 2025).

Package structure:
- evaluator: IQAEvaluator (15 methods), run_benchmark, AVAILABLE_METHODS
- finetune: BaselineLightningModule, BaselineDataModule, FINETUNABLE_METHODS
"""

from uav_iqa.baselines.evaluator import (
    AVAILABLE_METHODS,
    METHOD_TO_PYIQA,
    IQAEvaluator,
    load_finetuned_checkpoint,
    run_benchmark,
    run_benchmark_parallel,
)
from uav_iqa.baselines.finetune import (
    FINETUNABLE_METHODS,
    PYIQA_NAME_MAP,
    BaselineDataModule,
    BaselineFRDataset,
    BaselineLightningModule,
    evaluate_checkpoint,
)

__all__ = [
    # evaluator
    "AVAILABLE_METHODS",
    "METHOD_TO_PYIQA",
    "IQAEvaluator",
    "load_finetuned_checkpoint",
    "run_benchmark",
    "run_benchmark_parallel",
    # finetune
    "FINETUNABLE_METHODS",
    "PYIQA_NAME_MAP",
    "BaselineDataModule",
    "BaselineFRDataset",
    "BaselineLightningModule",
    "evaluate_checkpoint",
]
