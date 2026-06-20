from .distortion import (
    UAVDistortionPipeline,
    PropellerVibrationBlur,
    AtmosphericScatteringHaze,
    SixDoFViewpointBlur,
    CommunicationPacketLoss,
    LowResSuperResolution,
    PropellerShadow,
)
from .model import UAVIQANet
from .dataset import UAVIQADataset, validate_manifest
from .metrics import (
    compute_srcc,
    compute_plcc,
    evaluate_iqa,
    per_distortion_category_metrics,
)
from .lightning_module import UAVIQALightningModule
from .data_module import UAVIQDataModule
from .losses import ListMLELoss, CrossTaskRegularization
from .annotations import (
    parse_distortion_key,
    parse_quality_score,
    parse_usability,
    degradation_factor,
    compute_synthetic_score,
    build_ref_score_lookup,
    assign_task_label,
    synthetic_ref_scores,
)
from .utils import (
    setup_logging,
    load_task_map,
    load_manifest,
    split_samples,
    write_manifest,
    find_images,
)
from .data_synthesis import (
    DatasetFormat,
    DataSynthesisPipeline,
    create_pipeline,
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
]
