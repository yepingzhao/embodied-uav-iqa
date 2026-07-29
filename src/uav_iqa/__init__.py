from .batch_annotator import BatchAnnotator

from .annotations import (
    SUBTASK_NAMES,
    SUBTASK_TO_ID,
    SUBTASK_NAME_LIST,
    SUBTASK_NAME_TO_ID,
    NUM_SUBTASKS,
    build_ref_score_lookup,
    build_sample_id,
    build_vqa_split_lookup,
    extract_subtask_type,
    extract_subtask_id,
    extract_uav_id_from_question_id,
    get_dataset_name,
    normalize_subtask_type,
    group_by_scene_frame,
    parse_distortion_key,
    seed_for_distortion,
)
from .data_module import UAVIQADataModule
from .distortion_synthesis import (
    DatasetFormat,
    DataSynthesisPipeline,
    create_pipeline,
)
from .dataset import UAVIQADataset, validate_manifest
from .distortion import (
    AtmosphericScatteringHaze,
    CommunicationPacketLoss,
    LowResSuperResolution,
    PropellerShadow,
    PropellerVibrationBlur,
    SixDoFViewpointBlur,
    UAVDistortionPipeline,
)
from .lightning_module import UAVIQALightningModule
from .losses import CrossTaskRegularization, ListMLELoss
from .metrics import (
    compute_plcc,
    compute_srcc,
    evaluate_iqa,
    per_distortion_category_metrics,
)
from .model import UAVIQANet
from .utils import (
    find_images,
    load_flat_samples,
    setup_logging,
)
from uav_iqa.vlm import VLMScorer
# UAVIQACLI removed (2026-06) — use vanilla lightning.pytorch.cli.LightningCLI

__all__ = [
    "UAVDistortionPipeline",
    "PropellerVibrationBlur",
    "AtmosphericScatteringHaze",
    "SixDoFViewpointBlur",
    "CommunicationPacketLoss",
    "LowResSuperResolution",
    "PropellerShadow",
    "UAVIQANet",
    "UAVIQADataset",
    "compute_srcc",
    "compute_plcc",
    "evaluate_iqa",
    "per_distortion_category_metrics",
    "validate_manifest",
    "UAVIQALightningModule",
    "UAVIQADataModule",
    "ListMLELoss",
    "CrossTaskRegularization",
    "parse_distortion_key",
    "build_ref_score_lookup",
    "build_vqa_split_lookup",
    "group_by_scene_frame",
    "extract_subtask_type",
    "extract_subtask_id",
    "extract_uav_id_from_question_id",
    "normalize_subtask_type",
    "SUBTASK_NAMES",
    "SUBTASK_TO_ID",
    "SUBTASK_NAME_LIST",
    "SUBTASK_NAME_TO_ID",
    "NUM_SUBTASKS",
    "build_sample_id",
    "get_dataset_name",
    "seed_for_distortion",
    "setup_logging",
    "load_flat_samples",
    "find_images",
    "DatasetFormat",
    "DataSynthesisPipeline",
    "create_pipeline",
    "VLMScorer",
    "BatchAnnotator",
]
