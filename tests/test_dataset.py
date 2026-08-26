"""Tests for UAVQualityDataset — multi-image loading, question text output, collation."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_iqa.data import UAVQualityDataset, validate_manifest


class TestValidateManifest:
    def test_valid_flat_manifest(self, tmp_path):
        """validate_manifest should read a flat JSON array."""
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([{"a": 1}, {"b": 2}]))
        result = validate_manifest(manifest)
        assert result["valid"] is True
        assert result["count"] == 2

    def test_invalid_json(self, tmp_path):
        manifest = tmp_path / "bad.json"
        manifest.write_text("not json")
        result = validate_manifest(manifest)
        assert result["valid"] is False

    def test_not_a_list(self, tmp_path):
        manifest = tmp_path / "obj.json"
        manifest.write_text(json.dumps({"key": "val"}))
        result = validate_manifest(manifest)
        assert result["valid"] is False

    def test_missing_file(self, tmp_path):
        result = validate_manifest(tmp_path / "nonexistent.json")
        assert result["valid"] is False


class TestDatasetOutput:
    """Test that __getitem__ returns the correct fields."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create a minimal processed dataset with one entry per split."""
        data_root = tmp_path / "data"
        for split in ("train", "test"):
            split_dir = data_root / split
            split_dir.mkdir(parents=True)
            entries = []
            for i in range(3):
                entries.append({
                    "sample_id": f"Sim3__scene_{i:03d}__blur_L02",
                    "subtask_type": "scene_description",
                    "cognitive_score": 0.5 + i * 0.1,
                    "question": f"What is visible in scene {i}?",
                    "uav_paths": {
                        "UAV1": f"refs/uav_{i:03d}.png",
                        "UAV2": f"refs/uav_{i:03d}_uav2.png",
                    },
                    "distorted_uav_paths": {},
                    "distortion_info": {"type": "blur", "intensity": 0.2},
                })
            with open(split_dir / f"Sim3_VQA_{split}.json", "w") as f:
                json.dump(entries, f)
        return str(data_root)

    def test_returns_question_text(self, data_dir):
        """__getitem__ should include the 'question' field."""
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        sample = ds[0]
        assert "question" in sample, "dataset must return 'question' field"
        assert isinstance(sample["question"], str)
        assert len(sample["question"]) > 0

    def test_question_matches_input(self, data_dir):
        """question text should match what was in the JSON."""
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        sample = ds[1]
        assert sample["question"] == "What is visible in scene 1?"

    def test_returns_images_tensor(self, data_dir):
        """images should be a (N, 3, H, W) tensor."""
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        sample = ds[0]
        assert "images" in sample
        img = sample["images"]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 4  # (N_UAV, 3, H, W) — multi-image stack
        assert img.shape[1] == 3  # 3 channels

    def test_returns_task_id(self, data_dir):
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        sample = ds[0]
        assert "task_id" in sample
        assert isinstance(sample["task_id"], torch.Tensor)

    def test_returns_score(self, data_dir):
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        sample = ds[0]
        assert "score" in sample
        assert isinstance(sample["score"], torch.Tensor)

    def test_returns_sample_id(self, data_dir):
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        sample = ds[0]
        assert "sample_id" in sample
        assert sample["sample_id"].startswith("Sim3")

    def test_len(self, data_dir):
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        assert len(ds) == 3

    def test_multiple_samples(self, data_dir):
        """All samples should have different question text."""
        ds = UAVQualityDataset(data_root=data_dir, split="train")
        questions = [ds[i]["question"] for i in range(len(ds))]
        assert len(set(questions)) == 3  # all unique


class TestCollateFunction:
    def test_collate_preserves_question(self, tmp_path):
        """collate_fn should include question texts in the batch."""
        batch = [
            {
                "images": torch.zeros(2, 3, 256, 256),
                "task_id": torch.tensor(0, dtype=torch.long),
                "score": torch.tensor(0.5),
                "sample_id": "test_0",
                "question": "What is the quality?",
                "distortion": "blur_L02",
                "num_uavs": 2,
            },
            {
                "images": torch.zeros(3, 3, 256, 256),
                "task_id": torch.tensor(1, dtype=torch.long),
                "score": torch.tensor(0.8),
                "sample_id": "test_1",
                "question": "Can you see the target?",
                "distortion": "noise_L04",
                "num_uavs": 3,
            },
        ]
        result = UAVQualityDataset.collate_fn(batch)

        assert "question" in result, "collate_fn must return 'question' field"
        assert result["question"] == [
            "What is the quality?",
            "Can you see the target?",
        ]
        assert result["images"].shape == (2, 3, 3, 256, 256)  # padded to max N=3
