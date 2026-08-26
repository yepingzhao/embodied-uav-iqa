"""Shared domain taxonomy and annotation parsing contracts."""

from .annotations import (
    NUM_SUBTASKS,
    SUBTASK_NAME_LIST,
    SUBTASK_NAME_TO_ID,
    SUBTASK_NAMES,
    SUBTASK_TO_ID,
    build_ref_score_lookup,
    build_sample_id,
    build_vqa_split_lookup,
    extract_subtask_id,
    extract_subtask_type,
    extract_uav_id_from_question_id,
    get_dataset_name,
    group_by_scene_frame,
    normalize_subtask_type,
    parse_distortion_key,
    seed_for_distortion,
)

__all__ = [
    "NUM_SUBTASKS",
    "SUBTASK_NAME_LIST",
    "SUBTASK_NAME_TO_ID",
    "SUBTASK_NAMES",
    "SUBTASK_TO_ID",
    "build_ref_score_lookup",
    "build_sample_id",
    "build_vqa_split_lookup",
    "extract_subtask_id",
    "extract_subtask_type",
    "extract_uav_id_from_question_id",
    "get_dataset_name",
    "group_by_scene_frame",
    "normalize_subtask_type",
    "parse_distortion_key",
    "seed_for_distortion",
]
