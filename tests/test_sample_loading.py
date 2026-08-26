"""Characterization tests for loading flat processed samples."""

import json

from uav_iqa.data.samples import load_benchmark_samples


def test_load_benchmark_samples_projects_current_flat_schema(tmp_path):
    """A processed flat entry maps to the exact manifest-compatible projection."""
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    entry = {
        "sample_id": "sample-17",
        "uav_paths": {
            "uav_b": "raw/mission/b_view.jpg",
            "uav_a": "raw/mission/a_view.jpg",
        },
        "distorted_uav_paths": {
            "uav_b": "processed/mission/b_view.jpg",
            "uav_a": "processed/mission/a_view.jpg",
        },
        "distortion_info": {
            "type": "propeller_vibration_blur",
            "category": "uav",
            "intensity": 0.4,
        },
        "subtask_type": "scene_understanding",
        "cognitive_score": 0.73,
    }
    with (split_dir / "mission_VQA_001.json").open("w") as file:
        json.dump([entry], file)

    assert load_benchmark_samples(tmp_path, "train") == [
        {
            "path": "processed/mission/a_view.jpg",
            "ref_path": "raw/mission/a_view.jpg",
            "uav_paths": {
                "uav_b": "raw/mission/b_view.jpg",
                "uav_a": "raw/mission/a_view.jpg",
            },
            "uav_keys": ["uav_a", "uav_b"],
            "distorted_uav_paths": {
                "uav_b": "processed/mission/b_view.jpg",
                "uav_a": "processed/mission/a_view.jpg",
            },
            "task": "scene_understanding",
            "distortion": "propeller_vibration_blur",
            "category": "uav",
            "intensity_level": 0.4,
            "score": 0.73,
            "ref_id": "a_view",
            "sample_id": "sample-17",
        }
    ]
