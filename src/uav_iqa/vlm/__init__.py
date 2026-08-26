"""VLM scoring, annotation, configuration, and VQA indexing."""

from uav_iqa.vlm.annotator import BatchAnnotator
from uav_iqa.vlm.config import MODEL_REGISTRY, VLMConfig
from uav_iqa.vlm.scorer import VLMScorer
from uav_iqa.vlm.vqa_index import VQAIndex

__all__ = ["BatchAnnotator", "VLMConfig", "MODEL_REGISTRY", "VQAIndex", "VLMScorer"]
