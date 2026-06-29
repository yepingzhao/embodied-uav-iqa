"""Base scoring abstractions for UAV-IQA.

Provides:
- Task type constants
- BaseScorer: abstract interface for VLM/VLA/execution scoring
- extract_ref_id: utility for extracting stable reference IDs from image paths
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

_log = logging.getLogger(__name__)

# Task type constants
TASK_TRACKING = "tracking"
TASK_INSPECTION = "inspection"
TASK_DELIVERY = "delivery"
TASK_SAR = "sar"
ALL_TASKS = (TASK_TRACKING, TASK_INSPECTION, TASK_DELIVERY, TASK_SAR)


def extract_ref_id(image_path: Union[str, Path]) -> str:
    """Extract a stable ref_id from an image path.

    Strips '_clean' suffix if present (AirCopBench convention).
    """
    stem = Path(image_path).stem
    if stem.endswith("_clean"):
        stem = stem[: -len("_clean")]
    return stem


class BaseScorer(ABC):
    """Abstract interface for VLM/VLA/execution scoring.

    Each scorer produces a dict with:
        vlm_score: float          — cognitive quality score [0.0, 1.0]
        vla_score: float          — decision usability score [0.0, 1.0]
        execution_score: float    — execution success score [0.0, 1.0]
        annotated: bool           — whether scores are real (vs synthetic)
    """

    @abstractmethod
    def score_image(self, image_path: Union[str, Path], task: str) -> dict:
        """Score a single image for a given task type.

        Returns dict with keys: vlm_score, vla_score, execution_score, annotated.
        """
        ...

    @abstractmethod
    def score_batch(
        self, image_paths: List[Union[str, Path]], tasks: List[str]
    ) -> List[dict]:
        """Score a batch of images efficiently.

        Args:
            image_paths: List of paths to image files.
            tasks: List of task type strings, one per image (same length as image_paths).

        Returns:
            List of score dicts, one per image.
        """
        ...

    @abstractmethod
    def build_ref_lookup(self, reference_dir: Union[str, Path]) -> Dict[str, dict]:
        """Build a ref_id -> scores lookup from a directory of reference images.

        Args:
            reference_dir: Path to directory containing reference images.

        Returns:
            Dict mapping ref_id (stem) to score dict.
        """
        ...
