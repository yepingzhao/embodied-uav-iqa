from .annotations import (
    assign_task_label,
    build_ref_score_lookup,
    compute_synthetic_score,
    degradation_factor,
    parse_distortion_key,
    parse_quality_score,
    parse_usability,
    synthetic_ref_scores,
)
from .data_module import UAVIQDataModule
from .data_synthesis import (
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
    load_manifest,
    load_task_map,
    setup_logging,
    split_samples,
    write_manifest,
)
from .vlm_vla_scorer import (
    BaseScorer,
    SyntheticScorer,
    VLMScorer,
)

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
    "UAVIQDataModule",
    "ListMLELoss",
    "CrossTaskRegularization",
    "parse_distortion_key",
    "parse_quality_score",
    "parse_usability",
    "degradation_factor",
    "compute_synthetic_score",
    "build_ref_score_lookup",
    "assign_task_label",
    "synthetic_ref_scores",
    "setup_logging",
    "load_task_map",
    "load_manifest",
    "split_samples",
    "write_manifest",
    "find_images",
    "DatasetFormat",
    "DataSynthesisPipeline",
    "create_pipeline",
    "BaseScorer",
    "SyntheticScorer",
    "VLMScorer",
]
