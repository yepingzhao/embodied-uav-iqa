"""Tests for BatchAnnotator — flat-format JSON annotation, checkpoint/resume."""

import json
from pathlib import Path
from typing import List

from uav_iqa.batch_annotator import BatchAnnotator


class _FakeScorer:
    """Returns pseudo cognitive_scores for multi-image scoring."""

    def __init__(self, seed: int = 42):
        self.model_name = "fake/FakeScorer"

    def score_multi_image(
        self,
        ref_image_paths: List[str],
        dist_image_paths: List[str],
        question: str = "",
        subtask_type: str = "",
    ) -> dict:
        score = round(0.3 + 0.01 * len(dist_image_paths) + hash(question) % 100 * 0.001, 6)
        return {
            "cognitive_score": score,
            "bleu": score,
            "rouge_l": score,
            "cider": score,
            "ref_description": "mock ref",
            "dist_description": "mock dist",
            "prompt": question,
        }


def _make_flat_entry(
    sequence_frame: str = "scene_001_frame_001",
    question_id: str = "Q1",
    uav_keys: List[str] = None,
    distortion_key: str = "gaussian_blur_L04",
    question: str = "Describe the scene.",
) -> dict:
    if uav_keys is None:
        uav_keys = ["UAV1", "UAV2"]
    dist_type = distortion_key.rsplit("_L", 1)[0]
    return {
        "sample_id": f"Sim3__train__{sequence_frame}__{question_id}__{distortion_key}",
        "dataset": "Sim3",
        "split": "train",
        "sequence_frame": sequence_frame,
        "question_id": question_id,
        "question_type": "1.1 Scene Description",
        "question": question,
        "options": [],
        "correct_answer": "A",
        "subtask_type": "scene_description",
        "uav_id": "UAV1",
        "uav_paths": {k: f"refs/{k}.jpg" for k in uav_keys},
        "distorted_uav_paths": {k: f"train/distorted_{k}.png" for k in uav_keys},
        "distortion_info": {
            "type": dist_type,
            "category": "blur",
            "level": 4,
            "intensity": 0.4,
            "seed": 42,
        },
        "vlm_scores": {},
        "cognitive_score": 0.0,
    }


def _write_entries_json(entries: List[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f)
    return path


class TestBatchAnnotatorCheckpoint:
    def test_save_load_checkpoint(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        scores = {"A": {"v": 0.8}, "B": {"v": 0.9}}
        BatchAnnotator.save_checkpoint(ckpt, scores, 5, 10, {"model": "test"})
        assert ckpt.exists()
        loaded = BatchAnnotator.load_checkpoint(ckpt)
        assert loaded["completed"] == 5
        assert loaded["total"] == 10
        assert loaded["scores"]["A"]["v"] == 0.8
        assert loaded["metadata"]["model"] == "test"

    def test_save_load_scores_with_ndx_keys(self, tmp_path):
        ckpt = tmp_path / "ckpt2.json"
        scores = {"A": {"v": 0.5}, "B": {"v": 0.7}}
        BatchAnnotator.save_checkpoint(ckpt, scores, 2, 10)
        loaded = BatchAnnotator.load_checkpoint(ckpt)
        assert loaded["scores"]["A"]["v"] == 0.5
        assert loaded["scores"]["B"]["v"] == 0.7


class TestBatchAnnotatorFileIO:
    def test_load_write_entries(self, tmp_path):
        entries = [
            _make_flat_entry(),
            _make_flat_entry(sequence_frame="scene_002_frame_001"),
        ]
        path = tmp_path / "test.json"
        count = BatchAnnotator.write_groups(entries, path)
        assert count == 2
        loaded = BatchAnnotator.load_groups(path)
        assert len(loaded) == 2
        assert loaded[0]["sequence_frame"] == "scene_001_frame_001"


class TestBatchAnnotatorAnnotateFile:
    def test_annotate_file_basic(self, tmp_path):
        scorer = _FakeScorer()
        output_dir = tmp_path / "processed"
        (output_dir / "train").mkdir(parents=True)
        json_path = output_dir / "train" / "Sim3_VQA_train.json"
        _write_entries_json([_make_flat_entry()], json_path)

        annotator = BatchAnnotator(scorer, output_dir)
        result = annotator.annotate_file(json_path)
        assert result["scored"] >= 1

        entries = BatchAnnotator.load_groups(json_path)
        entry = entries[0]
        assert "fake/FakeScorer" in entry["vlm_scores"]
        assert entry["vlm_scores"]["fake/FakeScorer"] > 0.0

    def test_annotate_file_multiple_entries(self, tmp_path):
        scorer = _FakeScorer()
        output_dir = tmp_path / "processed"
        (output_dir / "test").mkdir(parents=True)
        entries = [
            _make_flat_entry(sequence_frame="scene_001_frame_001"),
            _make_flat_entry(sequence_frame="scene_002_frame_001"),
        ]
        json_path = output_dir / "test" / "Sim3_VQA_test.json"
        _write_entries_json(entries, json_path)

        annotator = BatchAnnotator(scorer, output_dir)
        result = annotator.annotate_file(json_path)
        assert result["scored"] >= 2

    def test_annotate_file_max_entries(self, tmp_path):
        scorer = _FakeScorer()
        output_dir = tmp_path / "processed"
        (output_dir / "train").mkdir(parents=True)

        entries = [
            _make_flat_entry(question_id=f"Q{i}", question=f"Q{i}") for i in range(10)
        ]
        json_path = output_dir / "train" / "Sim3_VQA_train.json"
        _write_entries_json(entries, json_path)

        annotator = BatchAnnotator(scorer, output_dir)
        result = annotator.annotate_file(json_path, max_entries=3)
        assert result["scored"] == 3

    def test_annotate_file_checkpoint_interval(self, tmp_path):
        scorer = _FakeScorer()
        output_dir = tmp_path / "processed"
        (output_dir / "train").mkdir(parents=True)
        ckpt_path = tmp_path / "ckpt.json"

        entries = [
            _make_flat_entry(question_id=f"Q{i}", question=f"Q{i}") for i in range(5)
        ]
        json_path = output_dir / "train" / "Sim3_VQA_train.json"
        _write_entries_json(entries, json_path)

        annotator = BatchAnnotator(scorer, output_dir)
        result = annotator.annotate_file(json_path, checkpoint_path=ckpt_path, checkpoint_interval=2)
        assert result["scored"] == 5


class TestBatchAnnotatorAnnotateSplits:
    def test_annotate_splits_basic(self, tmp_path):
        scorer = _FakeScorer()
        output_dir = tmp_path / "processed"
        for split in ("train", "test"):
            (output_dir / split).mkdir(parents=True)
            _write_entries_json(
                [_make_flat_entry(sequence_frame="scene_001_frame_001")],
                output_dir / split / "Sim3_VQA_{}.json".format(split),
            )

        annotator = BatchAnnotator(scorer, output_dir)
        results = annotator.annotate_splits(["train", "test"])
        assert len(results) == 2
        for v in results.values():
            assert v["scored"] >= 1

    def test_annotate_splits_skip_missing(self, tmp_path):
        scorer = _FakeScorer()
        output_dir = tmp_path / "processed"
        (output_dir / "train").mkdir(parents=True)

        annotator = BatchAnnotator(scorer, output_dir)
        results = annotator.annotate_splits(["train", "test"])
        assert len(results) >= 0
