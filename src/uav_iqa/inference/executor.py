"""Executor layer: wraps the VLM scorer and manages its lifecycle.

The executor is the only module that knows about VLM-specific APIs.
Workers call ``prepare`` once, then ``infer`` for each batch, and
``finalize`` when done. This separation keeps the worker stateless
with respect to model internals.

For testing, a :class:`DummyExecutor` is provided that returns a fixed
score without loading any model.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Abstract inference executor with a 3-phase lifecycle."""

    @abstractmethod
    def prepare(self) -> None:
        """Load the model and allocate resources. Called once per worker."""

    @abstractmethod
    def infer(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run inference on *batch* and return one output dict per input.

        The returned list must have the same length and order as *batch*.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Release resources. Called once when the worker exits."""

    @staticmethod
    def _empty_model_details() -> dict[str, float | str]:
        return {
            "prompt": "",
            "ref_answer": "",
            "dist_answer": "",
            "bleu": 0.0,
            "rouge_l": 0.0,
            "cider": 0.0,
        }


class VLMExecutor(BaseExecutor):
    """Wraps :class:`uav_iqa.vlm.VLMScorer` for multi-image scoring.

    Each input dict is expected to have:
      - ``uav_paths``: ``dict[str, str]`` — clean reference paths
      - ``distorted_uav_paths``: ``dict[str, str]`` — distorted paths
      - ``question``: ``str`` — VQA question used as prompt
      - ``subtask_type``: ``str`` — subtask identifier

    The output dict adds ``vlm_scores`` and ``cognitive_score`` fields.
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "auto",
        device: str = "",
        seed: int = 42,
        raw_data_dir: str | None = None,
        distorted_data_dir: str | None = None,
        **scorer_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.seed = seed
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        self.distorted_data_dir = Path(distorted_data_dir) if distorted_data_dir else None
        self.scorer_kwargs = scorer_kwargs
        self._scorer: Any = None

    def prepare(self) -> None:
        from uav_iqa.vlm import VLMScorer

        kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "backend": self.backend,
            "seed": self.seed,
        }
        if self.device:
            kwargs["device"] = self.device
        kwargs.update(self.scorer_kwargs)
        self._scorer = VLMScorer(**kwargs)
        _log.info("VLMExecutor ready: model=%s backend=%s", self.model_name, self.backend)

    def infer(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._scorer is None:
            raise RuntimeError("VLMExecutor.infer called before prepare()")

        results: list[dict[str, Any]] = []
        for entry in batch:
            output = self._score_entry(entry)
            results.append(output)
        return results

    def finalize(self) -> None:
        self._scorer = None
        _log.info("VLMExecutor finalized")

    def _score_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        uav_paths = entry.get("uav_paths") or {}
        dist_paths = entry.get("distorted_uav_paths") or {}
        uav_keys = sorted(uav_paths.keys())
        ref_list = [str(self._resolve_ref(p)) for p in (uav_paths[k] for k in uav_keys)]
        dist_list = [str(self._resolve_dist(p)) for p in (dist_paths.get(k, "") for k in uav_keys)]
        question = entry.get("question", "")
        subtask = entry.get("subtask_type", "")

        if not ref_list or not dist_list or not question:
            return {
                **entry,
                "vlm_scores": {self.model_name: 0.0},
                "cognitive_score": 0.0,
                "_model_details": {self.model_name: self._empty_model_details()},
            }

        try:
            result = self._scorer.score_multi_image(
                ref_image_paths=ref_list,
                dist_image_paths=dist_list,
                question=question,
                subtask_type=subtask,
            )
            score = float(result.get("cognitive_score", 0.0))
            # Map scorer output fields → model detail record.
            # Expected scorer return: cognitive_score, prompt, ref_description,
            # dist_description, bleu, rouge_l, cider.
            model_details = {
                "prompt": result.get("prompt", ""),
                "ref_answer": result.get("ref_description", ""),
                "dist_answer": result.get("dist_description", ""),
                "bleu": result.get("bleu", 0.0),
                "rouge_l": result.get("rouge_l", 0.0),
                "cider": result.get("cider", 0.0),
            }
        except Exception:
            _log.warning("score_multi_image failed for %s", entry.get("sample_id", "?"), exc_info=True)
            score = 0.0
            model_details = self._empty_model_details()

        vlm_scores = dict(entry.get("vlm_scores") or {})
        vlm_scores[self.model_name] = score
        return {
            **entry,
            "vlm_scores": vlm_scores,
            "cognitive_score": score,
            "_model_details": {self.model_name: model_details},
        }

    def _resolve_ref(self, rel_path: str) -> Path:
        """Resolve a clean reference image path against ``raw_data_dir``."""
        if self.raw_data_dir is not None:
            return self.raw_data_dir / rel_path
        return Path(rel_path)

    def _resolve_dist(self, rel_path: str) -> Path:
        """Resolve a distorted image path against ``distorted_data_dir``."""
        if self.distorted_data_dir is not None:
            return self.distorted_data_dir / rel_path
        return Path(rel_path)


class DummyExecutor(BaseExecutor):
    """A no-op executor for testing. Returns a deterministic score.

    The score is derived from the entry's ``sample_id`` hash so that
    tests can verify ordering and data flow without loading any model.
    """

    def __init__(self, model_name: str = "dummy", score: float = 0.5) -> None:
        self.model_name = model_name
        self._score = score
        self._prepared = False

    def prepare(self) -> None:
        self._prepared = True

    def infer(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._prepared:
            raise RuntimeError("DummyExecutor.infer called before prepare()")
        results: list[dict[str, Any]] = []
        for entry in batch:
            vlm_scores = dict(entry.get("vlm_scores") or {})
            vlm_scores[self.model_name] = self._score
            results.append({
                **entry,
                "vlm_scores": vlm_scores,
                "cognitive_score": self._score,
                "_model_details": {
                    self.model_name: self._empty_model_details(),
                },
            })
        return results

    def finalize(self) -> None:
        self._prepared = False
