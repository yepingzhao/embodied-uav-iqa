"""Placeholder module for future VLA (Vision-Language-Action) scoring.

Currently provides task type constants and a deprecated abstract BaseScorer
interface.  No concrete scorer inherits from BaseScorer — VLM scoring lives
in ``uav_iqa.vlm`` and VLA/execution scoring is planned for a future release.
"""

import warnings

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

from .annotations import SUBTASK_NAMES

_log = logging.getLogger(__name__)

ALL_TASKS = tuple(SUBTASK_NAMES.values())


class BaseScorer(ABC):
    """Abstract interface for VLM/VLA/execution scoring.

    .. deprecated::
        No concrete scorer inherits from this class.  VLM scoring lives in
        ``uav_iqa.vlm`` and VLA/execution scoring is planned for a future
        release.  This class is retained as a placeholder and will be removed.

    Each scorer produces a dict with:
        cognitive_score: float    — unified quality score [0.0, 1.0]
        annotated: bool           — whether scores are real (vs synthetic)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            "BaseScorer is deprecated — no concrete scorer inherits from it. "
            "Use uav_iqa.vlm for VLM scoring.  VLA/execution scoring is planned.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def score_image(self, image_path: Union[str, Path], task: str) -> dict:
        """Score a single image for a given task type.

        Returns dict with keys: cognitive_score (float), annotated (bool).
        """
        ...

    @abstractmethod
    def score_batch(self, image_paths: List[Union[str, Path]], tasks: List[str]) -> List[dict]:
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
