"""Utilities for parsing AirCopBench human annotations into numeric scores."""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

_log = logging.getLogger(__name__)

TASK_NAMES = ["tracking", "inspection", "delivery", "sar"]


# ---------------------------------------------------------------------------
# Distortion key parsing
# ---------------------------------------------------------------------------


def parse_distortion_key(key: str) -> Tuple[Optional[str], Optional[float]]:
    """Parse a distortion filename into (distortion_name, intensity_level).

    Handles these formats:
        'gaussian_blur_L04'                     → ('gaussian_blur', 0.4)
        'ref_id__propeller_shadow_L4.png'       → ('propeller_shadow', 0.4)
        'path/to/scene__color_noise_L0_5.jpeg'  → ('color_noise', 0.5)

    Returns (None, None) if the string cannot be parsed.
    """
    stem = os.path.splitext(os.path.basename(key))[0]

    if "__" in stem:
        dist_part = stem.rsplit("__", 1)[1]
    else:
        dist_part = stem

    m = re.search(r"_L(\d+(?:\.\d+)?)$", dist_part)
    if m:
        intensity = float(m.group(1))
        return dist_part[: m.start()], intensity / 10.0 if intensity >= 1 else intensity

    m = re.search(r"_L0?(\d)$", dist_part)
    if m:
        return dist_part[: m.start()], int(m.group(1)) / 10.0

    return None, None


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def parse_quality_score(quality_str: Optional[str]) -> float:
    """Parse 'Good (4/5)' → 0.8, 'Excellent (5/5)' → 1.0, etc."""
    if quality_str is None:
        return 0.5
    match = re.search(r"\((\d+)(?:\.\d+)?/(\d+)\)", str(quality_str))
    if match:
        return float(match.group(1)) / float(match.group(2))
    return 0.5


def parse_usability(usability_str: Optional[str]) -> float:
    """Parse '1 (Available)' → 1.0, '2 (Partially available)' → 0.5, etc."""
    if usability_str is None:
        return 0.0
    s = str(usability_str)
    if s.startswith("1"):
        return 1.0
    if s.startswith("2"):
        return 0.5
    if s.startswith("3"):
        return 0.25
    return 0.0


# ---------------------------------------------------------------------------
# Degradation factors — per-(distortion, task) degradation at max intensity
# ---------------------------------------------------------------------------

DEGRADATION_FACTORS = {
    # UAV-specific distortions
    "propeller_vibration_blur": {
        "tracking": 0.55,
        "inspection": 0.40,
        "delivery": 0.70,
        "sar": 0.50,
    },
    "atmospheric_scattering_haze": {
        "tracking": 0.50,
        "inspection": 0.35,
        "delivery": 0.65,
        "sar": 0.45,
    },
    "six_dof_viewpoint_blur": {
        "tracking": 0.45,
        "inspection": 0.35,
        "delivery": 0.65,
        "sar": 0.50,
    },
    "communication_packet_loss": {
        "tracking": 0.60,
        "inspection": 0.45,
        "delivery": 0.70,
        "sar": 0.55,
    },
    "low_res_super_resolution": {
        "tracking": 0.50,
        "inspection": 0.30,
        "delivery": 0.65,
        "sar": 0.50,
    },
    "propeller_shadow": {
        "tracking": 0.70,
        "inspection": 0.65,
        "delivery": 0.80,
        "sar": 0.75,
    },
    # Generic distortions
    "gaussian_blur": {
        "tracking": 0.60,
        "inspection": 0.55,
        "delivery": 0.75,
        "sar": 0.65,
    },
    "lens_blur": {"tracking": 0.55, "inspection": 0.50, "delivery": 0.70, "sar": 0.60},
    "motion_blur": {
        "tracking": 0.50,
        "inspection": 0.45,
        "delivery": 0.65,
        "sar": 0.55,
    },
    "brighten_max": {
        "tracking": 0.85,
        "inspection": 0.80,
        "delivery": 0.90,
        "sar": 0.85,
    },
    "brighten_min": {
        "tracking": 0.90,
        "inspection": 0.85,
        "delivery": 0.92,
        "sar": 0.88,
    },
    "brighten_avg": {
        "tracking": 0.88,
        "inspection": 0.83,
        "delivery": 0.90,
        "sar": 0.87,
    },
    "darken_max": {"tracking": 0.75, "inspection": 0.65, "delivery": 0.82, "sar": 0.70},
    "darken_min": {"tracking": 0.80, "inspection": 0.72, "delivery": 0.85, "sar": 0.78},
    "darken_avg": {"tracking": 0.78, "inspection": 0.70, "delivery": 0.84, "sar": 0.75},
    "color_diffusion": {
        "tracking": 0.82,
        "inspection": 0.72,
        "delivery": 0.88,
        "sar": 0.80,
    },
    "color_shift": {
        "tracking": 0.85,
        "inspection": 0.75,
        "delivery": 0.90,
        "sar": 0.82,
    },
    "color_quantize": {
        "tracking": 0.78,
        "inspection": 0.68,
        "delivery": 0.85,
        "sar": 0.75,
    },
    "white_noise": {
        "tracking": 0.72,
        "inspection": 0.60,
        "delivery": 0.80,
        "sar": 0.68,
    },
    "color_noise": {
        "tracking": 0.70,
        "inspection": 0.58,
        "delivery": 0.78,
        "sar": 0.65,
    },
    "impulse_noise": {
        "tracking": 0.75,
        "inspection": 0.62,
        "delivery": 0.82,
        "sar": 0.70,
    },
    "multiplicative_noise": {
        "tracking": 0.73,
        "inspection": 0.60,
        "delivery": 0.80,
        "sar": 0.68,
    },
    "jpeg_compression": {
        "tracking": 0.78,
        "inspection": 0.70,
        "delivery": 0.85,
        "sar": 0.75,
    },
    "jp2k_compression": {
        "tracking": 0.76,
        "inspection": 0.68,
        "delivery": 0.83,
        "sar": 0.73,
    },
    "webp_compression": {
        "tracking": 0.78,
        "inspection": 0.70,
        "delivery": 0.85,
        "sar": 0.75,
    },
    "spatial_warp": {
        "tracking": 0.60,
        "inspection": 0.50,
        "delivery": 0.72,
        "sar": 0.62,
    },
    "spatial_rotation": {
        "tracking": 0.70,
        "inspection": 0.60,
        "delivery": 0.78,
        "sar": 0.68,
    },
    "spatial_scale": {
        "tracking": 0.75,
        "inspection": 0.65,
        "delivery": 0.82,
        "sar": 0.72,
    },
    "spatial_shear": {
        "tracking": 0.68,
        "inspection": 0.58,
        "delivery": 0.76,
        "sar": 0.65,
    },
    "resolution_limit": {
        "tracking": 0.55,
        "inspection": 0.40,
        "delivery": 0.68,
        "sar": 0.55,
    },
    "grayscale": {"tracking": 0.80, "inspection": 0.65, "delivery": 0.85, "sar": 0.78},
    "sharpness": {"tracking": 0.88, "inspection": 0.85, "delivery": 0.92, "sar": 0.88},
    "contrast": {"tracking": 0.85, "inspection": 0.82, "delivery": 0.90, "sar": 0.85},
    "none": {"tracking": 0.95, "inspection": 0.95, "delivery": 0.95, "sar": 0.95},
}

_DEFAULT_DEG = {"tracking": 0.75, "inspection": 0.68, "delivery": 0.82, "sar": 0.72}


def degradation_factor(distortion: str, task: str) -> float:
    """Get degradation factor at max intensity for (distortion, task)."""
    per_task = DEGRADATION_FACTORS.get(distortion, _DEFAULT_DEG)
    return per_task.get(task, 0.75)


# ---------------------------------------------------------------------------
# Reference score lookup from AirCopBench
# ---------------------------------------------------------------------------


def build_ref_score_lookup(aircopbench_dir: Path) -> dict:
    """Build a dict mapping ref_id → {vlm_score, vla_score, execution_score}."""
    lookup = {}

    ann_dirs = list(aircopbench_dir.rglob("Annotations"))
    for ann_dir in ann_dirs:
        for ann_file in sorted(ann_dir.glob("*.json")):
            if "VQA" in ann_file.name:
                continue
            try:
                with open(ann_file) as f:
                    annotations = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if not isinstance(annotations, list):
                continue

            for entry in annotations:
                img1 = entry.get("img1", "")
                fname = os.path.basename(img1)
                ref_id = os.path.splitext(fname)[0]
                quality = parse_quality_score(entry.get("Quality"))
                usability = parse_usability(entry.get("Usibility"))
                combined = 0.4 * quality + 0.6 * usability
                lookup[ref_id] = {
                    "vlm_score": round(quality, 4),
                    "vla_score": round(usability, 4),
                    "execution_score": round(combined, 4),
                    "annotated": True,
                }

    _log.info("Built ref score lookup: %d annotated references", len(lookup))
    return lookup


def assign_task_label(img_name: str, task_map: Optional[dict] = None) -> str:
    """Extract task label from path structure, with configurable mapping.

    Parses scene identifiers from file paths (e.g., 'scene_001', 'UAV1') and maps them
    to task types. If no scene identifier is found, falls back to deterministic hash
    with a logged warning.

    Args:
        img_name: File path or basename to extract task from.
        task_map: Optional dict mapping scene/identifier strings to task names.
                  Default mapping rotates scene_001→tracking, scene_002→inspection, etc.

    Default scene mapping (can be overridden):
        {'scene_001': 'tracking', 'scene_002': 'inspection',
         'scene_003': 'delivery', 'scene_004': 'sar'}
    """
    if task_map is not None:
        for key, task in task_map.items():
            if key in img_name:
                return task

    scene_match = re.search(r"scene_(\d+)", img_name)
    if scene_match:
        scene_num = int(scene_match.group(1))
        return TASK_NAMES[(scene_num - 1) % len(TASK_NAMES)]

    task_match = re.search(
        r"(tracking|inspection|delivery|sar)", img_name, re.IGNORECASE
    )
    if task_match:
        return task_match.group(1).lower()

    _log.warning(
        "No scene or task identifier in path '%s' — falling back to hash-based "
        "assignment. Consider providing an explicit --task-map.",
        img_name,
    )
    hash_int = int(hashlib.md5(img_name.encode()).hexdigest(), 16)
    return TASK_NAMES[hash_int % len(TASK_NAMES)]


def compute_synthetic_score(distortion: str, task: str, intensity: float) -> float:
    """Noiseless degradation-model score for a given (distortion, task, intensity).

    Formula: score = 1.0 * (1.0 - alpha * intensity)
    where alpha = 1.0 - degradation_factor(distortion, task)

    Used by both score annotation and C2 correlation validation.
    """
    base = degradation_factor(distortion, task)
    alpha = 1.0 - base
    return max(0.0, min(1.0, 1.0 * (1.0 - alpha * intensity)))


def synthetic_ref_scores(ref_id: str) -> dict:
    """Generate deterministic synthetic scores for a reference without annotations."""
    hash_int = int(hashlib.md5(ref_id.encode()).hexdigest(), 16)
    ref_vlm = 0.4 + 0.5 * ((hash_int % 1000) / 1000.0)
    ref_vla = 0.3 + 0.5 * (((hash_int // 1000) % 1000) / 1000.0)
    ref_exec = 0.35 + 0.5 * (((hash_int // 1000000) % 1000) / 1000.0)
    return {
        "vlm_score": round(ref_vlm, 4),
        "vla_score": round(ref_vla, 4),
        "execution_score": round(ref_exec, 4),
        "annotated": False,
    }
