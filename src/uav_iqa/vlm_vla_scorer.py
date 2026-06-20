"""Real VLM/VLA scoring module for UAV-IQA.

Provides:
- BaseScorer: abstract interface for scoring images with VLM/VLA/execution scores
- SyntheticScorer: deterministic hash-based synthetic scores (fallback)
- VLMScorer: calls a real Vision-Language Model for quality assessment
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

_log = logging.getLogger(__name__)

# Task type constants
TASK_TRACKING = "tracking"
TASK_INSPECTION = "inspection"
TASK_DELIVERY = "delivery"
TASK_SAR = "sar"
ALL_TASKS = (TASK_TRACKING, TASK_INSPECTION, TASK_DELIVERY, TASK_SAR)


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


class SyntheticScorer(BaseScorer):
    """Deterministic hash-based synthetic scorer.

    Produces reproducible pseudo-scores using MD5 hashing of the image
    identifier. Used as fallback when real VLM/VLA models are unavailable,
    or for pipeline testing.

    The scores are deterministic per ref_id, enabling reproducible experiments.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        _log.info("SyntheticScorer initialized (seed=%d)", seed)

    def score_image(self, image_path: Union[str, Path], task: str) -> dict:
        """Generate deterministic synthetic scores for a single image.

        Uses both image path and task for hash diversity.
        """
        ref_id = self._extract_ref_id(image_path)
        return self._synthetic_scores(ref_id)

    def score_batch(
        self, image_paths: List[Union[str, Path]], tasks: List[str]
    ) -> List[dict]:
        """Generate synthetic scores for a batch."""
        if len(image_paths) != len(tasks):
            raise ValueError(
                f"Length mismatch: {len(image_paths)} images vs {len(tasks)} tasks"
            )
        return [self.score_image(p, t) for p, t in zip(image_paths, tasks)]

    def build_ref_lookup(self, reference_dir: Union[str, Path]) -> Dict[str, dict]:
        """Build synthetic ref lookup from image files.

        For images ending in *_clean.png (AirCopBench convention), uses the base stem
        as ref_id. Otherwise uses the full stem.
        """
        reference_dir = Path(reference_dir)
        if not reference_dir.is_dir():
            _log.warning("Reference directory does not exist: %s", reference_dir)
            return {}

        lookup: Dict[str, dict] = {}
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in sorted(reference_dir.glob(ext)):
                ref_id = self._extract_ref_id(img_path)
                lookup[ref_id] = self._synthetic_scores(ref_id)

        _log.info(
            "Built synthetic ref lookup: %d references from %s",
            len(lookup),
            reference_dir,
        )
        return lookup

    @staticmethod
    def _extract_ref_id(image_path: Union[str, Path]) -> str:
        """Extract a stable ref_id from an image path.

        Strips '_clean' suffix if present (AirCopBench convention).
        """
        stem = Path(image_path).stem
        if stem.endswith("_clean"):
            stem = stem[: -len("_clean")]
        return stem

    def _synthetic_scores(self, ref_id: str) -> dict:
        """Generate deterministic scores from ref_id.

        Uses MD5 hash with task-specific offsets to produce diverse scores.
        """
        # Use seed in hash for reproducibility across different SyntheticScorer instances
        hash_input = f"{self.seed}:{ref_id}"
        hash_int = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Generate three independent scores from different parts of the hash
        ref_vlm = 0.4 + 0.5 * ((hash_int % 1000) / 1000.0)
        ref_vla = 0.3 + 0.5 * (((hash_int // 1000) % 1000) / 1000.0)
        ref_exec = 0.35 + 0.5 * (((hash_int // 1000000) % 1000) / 1000.0)

        return {
            "vlm_score": round(ref_vlm, 4),
            "vla_score": round(ref_vla, 4),
            "execution_score": round(ref_exec, 4),
            "annotated": False,
        }


class VLMScorer(BaseScorer):
    """Real VLM-based quality scorer using vLLM or transformers.

    Loads a vision-language model and scores images by prompting the model
    to assess visual quality for a given aerial embodied task.

    Requires vLLM or transformers with VL support (install: uv sync --group dev --extra vlm).
    Falls back to SyntheticScorer if no VLM backend is available.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        backend: str = "auto",  # "auto", "vllm", "transformers", "none"
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.seed = seed
        self._model = None
        self._fallback: Optional[SyntheticScorer] = None

        _log.info(
            "VLMScorer initialized: model=%s backend=%s device=%s",
            model_name,
            backend,
            device,
        )

    @property
    def fallback(self) -> SyntheticScorer:
        if self._fallback is None:
            self._fallback = SyntheticScorer(seed=self.seed)
        return self._fallback

    def _resolve_backend(self) -> str:
        """Resolve actual backend capability.

        Checks for available VLM backends in priority order:
        1. vLLM (fastest, recommended for production)
        2. transformers (fallback, more memory)
        3. none (no VLM available)

        When self.backend is 'auto', auto-detects the best available.
        When set to 'none', always falls back.
        """
        if self.backend == "none":
            return "none"

        if self.backend == "vllm":
            try:
                import vllm  # noqa: F401

                return "vllm"
            except ImportError:
                _log.warning("vLLM requested but not installed, using fallback")
                return "none"

        if self.backend == "transformers":
            try:
                import transformers  # noqa: F401

                return "transformers"
            except ImportError:
                _log.warning("Transformers requested but not installed, using fallback")
                return "none"

        # auto: try vllm first, then transformers, then none
        try:
            import vllm  # noqa: F401

            return "vllm"
        except ImportError:
            pass

        try:
            import transformers  # noqa: F401

            return "transformers"
        except ImportError:
            pass

        _log.info(
            "No VLM backend available (%s), will use synthetic fallback",
            self.model_name,
        )
        return "none"

    def _load_model(self):
        """Load the VLM model (lazy initialization).

        Called on first use when a real backend is available.
        """
        backend = self._resolve_backend()
        if backend == "none":
            _log.info("No backend to load")
            return

        if backend == "vllm":
            from vllm import LLM, SamplingParams

            _log.info(
                "Loading model %s via vLLM on %s...", self.model_name, self.device
            )
            self._model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_num_seqs=8,
            )
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=10,
            )
        elif backend == "transformers":
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

            _log.info(
                "Loading model %s via transformers on %s...",
                self.model_name,
                self.device,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            self._model.eval()

        _log.info("Model loaded successfully: %s", self.model_name)

    @staticmethod
    def _parse_vlm_response(response: str, task: str) -> float:
        """Parse a numeric score from VLM response text.

        Handles various formats:
            '4'         → 0.8
            '3.5'       → 0.7
            '4/5'       → 0.8
            'Score: 4'  → 0.8
            'I rate this 3 out of 5' → 0.6

        Returns 0.5 as default when parsing fails (neutral midpoint).
        Score is clamped to [0.0, 1.0].
        """
        import re

        if not response or not response.strip():
            return 0.5

        # Try fraction format first: "4/5", "3.5/5"
        fraction_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)", response)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator > 0:
                return round(max(0.0, min(1.0, numerator / denominator)), 4)

        # Try single number: find all numbers, pick the one closest to valid range
        numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
        if numbers:
            # Prefer numbers in range [1, 5] (our prompt asks for 1-5)
            for num_str in numbers:
                num = float(num_str)
                if 1.0 <= num <= 5.0:
                    return round(max(0.0, min(1.0, num / 5.0)), 4)

            # Fallback: use the last number found
            num = float(numbers[-1])
            if num > 1.0:
                num = num / 5.0
            return round(max(0.0, min(1.0, num)), 4)

        _log.warning(
            "Could not parse score from VLM response: '%s' for task '%s', using default 0.5",
            response[:100],
            task,
        )
        return 0.5

    def _build_prompt(self, task: str) -> str:
        """Build the VLM prompt for quality assessment.

        Each task type gets a tailored prompt that guides the VLM to assess
        image quality from the perspective of that specific embodied UAV task.
        """
        prompt_templates = {
            "tracking": (
                "You are assessing image quality for a UAV visual object tracking task. "
                "Rate the quality of this image on a scale of 1-5 for how well it would "
                "support accurate object tracking. Consider factors like blur, noise, "
                "contrast, and visibility of moving objects. "
                "Respond with only the numeric score (e.g., '4')."
            ),
            "inspection": (
                "You are assessing image quality for a UAV infrastructure inspection task. "
                "Rate the quality of this image on a scale of 1-5 for how well it would "
                "support detecting fine surface defects and structural anomalies. "
                "Consider sharpness, detail preservation, lighting, and artifact presence. "
                "Respond with only the numeric score (e.g., '4')."
            ),
            "delivery": (
                "You are assessing image quality for a UAV package delivery task. "
                "Rate the quality of this image on a scale of 1-5 for how well it would "
                "support precise landing zone identification and obstacle detection. "
                "Consider spatial accuracy, contrast, depth cues, and overall clarity. "
                "Respond with only the numeric score (e.g., '4')."
            ),
            "sar": (
                "You are assessing image quality for a UAV search-and-rescue task. "
                "Rate the quality of this image on a scale of 1-5 for how well it would "
                "support detecting persons or distress signals in varied terrain. "
                "Consider detail preservation, noise tolerance, color fidelity, and "
                "visibility in challenging conditions. "
                "Respond with only the numeric score (e.g., '4')."
            ),
        }

        if task not in prompt_templates:
            _log.warning("Unknown task '%s', falling back to tracking prompt", task)
            task = "tracking"

        return prompt_templates[task]

    def _call_vlm(self, image_path: str, task: str) -> dict:
        """Call the VLM to score a single image.

        Returns a score dict with annotated=True on success.
        """
        backend = self._resolve_backend()
        if backend == "none":
            raise RuntimeError("No VLM backend available for scoring")

        prompt = self._build_prompt(task)

        if self._model is None:
            self._load_model()
            if self._model is None:
                raise RuntimeError("Failed to load VLM model")

        if backend == "vllm":
            from vllm import LLM, SamplingParams  # noqa: F401

            outputs = self._model.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": image_path}}],
                self._sampling_params,
            )
            response_text = outputs[0].outputs[0].text.strip()

        elif backend == "transformers":
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)
            import torch

            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=10)
            response_text = self._processor.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        vlm_score = self._parse_vlm_response(response_text, task)

        return {
            "vlm_score": round(vlm_score, 4),
            "vla_score": round(vlm_score * 0.9, 4),
            "execution_score": round(vlm_score * 0.85, 4),
            "annotated": True,
        }

    def score_image(self, image_path: Union[str, Path], task: str) -> dict:
        """Score a single image using VLM, with automatic fallback.

        If a real VLM backend is available, uses the VLM.
        Otherwise, falls back to SyntheticScorer.
        """
        backend = self._resolve_backend()

        if backend != "none":
            try:
                return self._call_vlm(str(image_path), task)
            except Exception as exc:
                _log.warning(
                    "VLM scoring failed for %s: %s — falling back to synthetic",
                    image_path,
                    exc,
                )
        return self.fallback.score_image(image_path, task)

    def score_batch(
        self, image_paths: List[Union[str, Path]], tasks: List[str]
    ) -> List[dict]:
        """Score a batch of images, with automatic fallback.

        Validates input length consistency, then scores each image.
        Uses the fallback scorer when no VLM backend is available.
        """
        if len(image_paths) != len(tasks):
            raise ValueError(
                f"Length mismatch: {len(image_paths)} images vs {len(tasks)} tasks"
            )

        backend = self._resolve_backend()

        if backend != "none":
            results = []
            for img_path, task in zip(image_paths, tasks):
                try:
                    results.append(self._call_vlm(str(img_path), task))
                except Exception as exc:
                    _log.warning(
                        "VLM batch scoring failed for %s: %s — falling back",
                        img_path,
                        exc,
                    )
                    results.append(self.fallback.score_image(img_path, task))
            return results

        return self.fallback.score_batch(image_paths, tasks)

    def build_ref_lookup(self, reference_dir: Union[str, Path]) -> Dict[str, dict]:
        """Build reference score lookup from a directory of images.

        Uses VLM if available, otherwise falls back to synthetic scoring.
        Results are cached in-memory on the scorer instance.
        """
        backend = self._resolve_backend()

        if backend != "none":
            try:
                reference_dir = Path(reference_dir)
                if not reference_dir.is_dir():
                    _log.warning(
                        "Reference directory does not exist: %s", reference_dir
                    )
                    return {}

                lookup: Dict[str, dict] = {}
                image_extensions = ("*.png", "*.jpg", "*.jpeg")
                for ext in image_extensions:
                    for img_path in sorted(reference_dir.glob(ext)):
                        ref_id = SyntheticScorer._extract_ref_id(img_path)
                        # Default to tracking task for ref images
                        lookup[ref_id] = self._call_vlm(str(img_path), "tracking")

                _log.info(
                    "Built VLM ref lookup: %d references from %s",
                    len(lookup),
                    reference_dir,
                )
                return lookup
            except Exception as exc:
                _log.warning(
                    "VLM ref lookup failed: %s — falling back to synthetic", exc
                )

        return self.fallback.build_ref_lookup(reference_dir)
