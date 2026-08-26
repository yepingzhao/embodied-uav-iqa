"""Utilities for AirCopBench VQA data processing.

Provides functions for:
- Parsing distortion keys from filenames
- Building VQA split lookups from raw train/test JSONs
- Grouping VQA entries by scene+frame
- Extracting subtask types and building sample IDs
- Resolving UAV image paths
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distortion key parsing
# ---------------------------------------------------------------------------


def parse_distortion_key(key: str) -> Tuple[Optional[str], Optional[float]]:
    """Parse a distortion filename into (distortion_name, intensity_level).

    Handles these formats:
        'gaussian_blur_L04'                     -> ('gaussian_blur', 0.4)
        'ref_id__propeller_shadow_L4.png'       -> ('propeller_shadow', 0.4)
        'path/to/scene__color_noise_L0_5.jpeg'  -> ('color_noise', 0.5)

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
# VQA split lookup
# ---------------------------------------------------------------------------


def _ref_id_from_vqa_path(vqa_path: str) -> str:
    """Convert a VQA path to its flat reference identifier.

    'Sim_3_UAVs/Samples/images/scene_001/UAV1/UAV1_frame_001.jpg'
    -> 'Sim_3_UAVs_Samples_images_scene_001_UAV1_UAV1_frame_001'
    """
    parts = vqa_path.replace("\\", "/").strip("/").split("/")
    joined = "_".join(parts)
    return str(Path(joined).stem)


def build_vqa_split_lookup(aircopbench_dir: Path) -> dict[str, str]:
    """Build a mapping from reference image stem to split ('train' or 'test').

    Reads VQA JSON files from ``train/`` and ``test/`` subdirectories under
    ``aircopbench_dir``, collects all image paths referenced in ``uav_paths``,
    and assigns each image to the split it appears in.

    Images referenced in test VQA files are assigned to 'test' (even if they
    also appear in train VQA files).  Images only referenced in train VQA
    files are assigned to 'train'.

    The image stem format matches what ``extract_references()`` produces:
    ``_``.join of the relative path parts (without extension).
    """
    train_dir = aircopbench_dir / "train"
    test_dir = aircopbench_dir / "test"

    train_images: set[str] = set()
    test_images: set[str] = set()

    for split_label, vqa_dir in [("train", train_dir), ("test", test_dir)]:
        if not vqa_dir.is_dir():
            _log.warning("VQA directory not found: %s", vqa_dir)
            continue
        for fpath in sorted(vqa_dir.glob("*.json")):
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            if not isinstance(data, list):
                continue
            for item in data:
                for path_str in item.get("uav_paths", {}).values():
                    ref_id = _ref_id_from_vqa_path(path_str)
                    if ref_id:
                        if split_label == "train":
                            train_images.add(ref_id)
                        else:
                            test_images.add(ref_id)

    lookup: dict[str, str] = {}
    for ref_id in train_images - test_images:
        lookup[ref_id] = "train"
    for ref_id in test_images:
        lookup[ref_id] = "test"

    _log.info(
        "VQA split lookup: %d train, %d test (%d in both -> test)",
        len(train_images - test_images),
        len(test_images),
        len(train_images & test_images),
    )
    return lookup


# ---------------------------------------------------------------------------
# VQA entry grouping
# ---------------------------------------------------------------------------


def group_by_scene_frame(entries: list[dict]) -> dict[str, list[dict]]:
    """Group VQA entries by ``sequence_frame``.

    Each group represents all questions for a single multi-UAV capture moment.
    Entries sharing the same ``sequence_frame`` have identical ``uav_paths``.

    Returns:
        Dict mapping ``sequence_frame`` -> list of entries.
    """
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        seq_frame = entry.get("sequence_frame", "")
        if not seq_frame:
            continue
        groups.setdefault(seq_frame, []).append(entry)
    return groups


# ---------------------------------------------------------------------------
# Subtask type extraction
# ---------------------------------------------------------------------------


SUBTASK_NAMES: dict[str, str] = {
    "1.1": "scene_description",
    "1.2": "scene_comparison",
    "1.3": "observing_posture",
    "2.1": "object_recognition",
    "2.2": "object_counting",
    "2.3": "object_grounding",
    "2.4": "object_matching",
    "3.1": "quality_assessment",
    "3.2": "usability_assessment",
    "3.3": "causal_assessment",
    "4.1": "when_to_collaborate",
    "4.2": "what_to_collaborate",
    "4.3": "who_to_collaborate",
    "4.4": "why_to_collaborate",
}

_SUBTASK_CODES = sorted(SUBTASK_NAMES.keys())
SUBTASK_TO_ID: dict[str, int] = {code: i for i, code in enumerate(_SUBTASK_CODES)}

SUBTASK_NAME_LIST: tuple[str, ...] = tuple(SUBTASK_NAMES[code] for code in _SUBTASK_CODES)

SUBTASK_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SUBTASK_NAME_LIST)}

NUM_SUBTASKS = len(SUBTASK_NAMES)


def extract_subtask_type(question_type: str) -> str:
    """Extract the subtask type identifier from a question_type string.

    '1.1 Scene Description (UAV2)' -> '1.1'
    '4.2 What to Collaborate' -> '4.2'
    'Object Matching' -> falls back to empty string
    """
    if not question_type:
        return ""
    m = re.match(r"(\d+\.\d+)", question_type.strip())
    if m:
        return m.group(1)
    return ""


def extract_subtask_id(question_type: str) -> int:
    """Extract the subtask integer ID from a question_type string.

    Returns 0 for unknown types.
    """
    subtask_type = extract_subtask_type(question_type)
    return SUBTASK_TO_ID.get(subtask_type, 0)


_KEYWORD_MAP = {name.replace("_", " ").title(): name for name in SUBTASK_NAME_LIST}


def normalize_subtask_type(question_type: str) -> str:
    """Convert any question_type string to its snake_case subtask name.

    Returns a non-empty string in all cases.
    """
    if question_type:
        code = extract_subtask_type(question_type)
        if code and code in SUBTASK_NAMES:
            return SUBTASK_NAMES[code]

        for keyword, name in _KEYWORD_MAP.items():
            if keyword.lower() in question_type.lower():
                return name

    return "scene_description"


def extract_uav_id_from_question_id(question_id: str, question_type: str = "") -> str:
    """Extract the UAV identifier from a question_id string.

    Examples:
        question_id="Sim3_what2col_UAV2_1" -> "UAV2"
        question_id="MDMT_OB_UAV2_001" -> "UAV2"
        question_id="Sim3_QA_UAV1_1" -> "UAV1"
        question_id="something_no_uav" -> "UAV1"
    """
    m = re.search(r"_UAV(\d+)_", question_id)
    if m:
        return f"UAV{m.group(1)}"

    m = re.search(r"_UAV(\d+)$", question_id)
    if m:
        return f"UAV{m.group(1)}"

    if question_type:
        m = re.search(r"\(UAV(\d+)\)", question_type)
        if m:
            return f"UAV{m.group(1)}"

    return "UAV1"


# ---------------------------------------------------------------------------
# Sample ID construction
# ---------------------------------------------------------------------------


def build_sample_id(
    dataset: str,
    sequence_frame: str,
    distortion_type: str,
    level: int,
    *,
    split: str = "",
    question_id: str = "",
) -> str:
    """Build a unique sample identifier.

    Backward-compatible with the old 4-arg signature (dataset, sequence_frame,
    distortion_type, level).  For the extended format, pass *split* and
    *question_id* as keyword arguments.

    Short format: ``{dataset}__{safe_frame}__{distortion_type}_L{level:02d}``
    Full format:  ``{dataset}__{split}__{safe_frame}__{question_id}__{distortion_type}_L{level:02d}``

    Example: ``Sim3__train__scene_001_frame_001__Sim3_QA_UAV1_1__gaussian_blur_L04``
    """
    safe_frame = sequence_frame.replace("/", "_").replace("\\", "_")
    if split or question_id:
        return (
            f"{dataset}__{split}__{safe_frame}__{question_id}" f"__{distortion_type}_L{level:02d}"
        )
    return f"{dataset}__{safe_frame}__{distortion_type}_L{level:02d}"


def get_dataset_name(vqa_filename: str) -> str:
    """Extract dataset name from a VQA JSON filename.

    'Sim3_VQA_train.json' -> 'Sim3'
    'Real2_VQA_test.json'  -> 'Real2'
    """
    stem = Path(vqa_filename).stem
    m = re.match(r"(\w+)_VQA_", stem)
    if m:
        return m.group(1)
    return stem.split("_")[0]


# ---------------------------------------------------------------------------
# Deterministic seed for distortion reproducibility
# ---------------------------------------------------------------------------


def seed_for_distortion(
    dataset: str,
    sequence_frame: str,
    distortion_type: str,
    base_seed: int = 42,
) -> int:
    """Derive a deterministic seed for a specific distortion application.

    Ensures that re-running inject with the same base_seed produces
    identical results for any given (dataset, scene_frame, distortion).
    """
    key = f"{dataset}__{sequence_frame}__{distortion_type}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (base_seed + h) % (2**31)


# ---------------------------------------------------------------------------
# Legacy: AirCopBench annotation score parsing (for C2 correlation validation)
# ---------------------------------------------------------------------------


def parse_quality_score(quality_str: str) -> float:
    """Parse a quality string like 'Good (4/5)' to a float score.

    'Excellent (5/5)' -> 1.0
    'Good (4/5)'       -> 0.8
    'Fair (3/5)'       -> 0.6
    'Poor (2/5)'       -> 0.4
    'Very Poor (1/5)'  -> 0.2
    """
    import re

    m = re.search(r"\((\d+)/5\)", quality_str)
    if m:
        return float(m.group(1)) / 5.0
    return 0.5


def parse_usability(usability_str: str) -> float:
    """Parse a usability string like '1 (Available)' to a float score.

    '1 (Available)'           -> 1.0
    '2 (Partially available)' -> 0.5
    '3'                       -> 0.25
    """
    val = usability_str.strip()
    if val.startswith("1"):
        return 1.0
    if val.startswith("2"):
        return 0.5
    if val.startswith("3"):
        return 0.25
    return 0.0


def build_ref_score_lookup(aircopbench_dir: Path) -> dict:
    """Build a mapping: ref_id -> {cognitive_score, annotated}.

    Reads AirCopBench Annotations/*.json files (single-image quality labels),
    extracts Quality/Usibility fields, and computes a combined cognitive_score
    as 0.4*quality + 0.6*usability.

    This is used by C2 correlation validation to compare VLM-predicted
    cognitive scores against human annotations.
    """
    lookup: dict = {}
    annotations_dirs = sorted(aircopbench_dir.rglob("Annotations"))
    if not annotations_dirs:
        _log.warning("No Annotations/ directories found under %s", aircopbench_dir)
        return lookup

    for ann_dir in annotations_dirs:
        for fpath in sorted(ann_dir.glob("*.json")):
            if "VQA" in fpath.name:
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                img_path = entry.get("img1", "")
                if not img_path:
                    continue
                ref_id = str(Path(img_path).stem)
                quality = parse_quality_score(entry.get("Quality", "Fair (3/5)"))
                usability = parse_usability(entry.get("Usibility", "1 (Available)"))
                combined = 0.4 * quality + 0.6 * usability
                lookup[ref_id] = {
                    "cognitive_score": combined,
                    "annotated": True,
                }

    _log.info("Ref score lookup: %d annotated reference images", len(lookup))
    return lookup
