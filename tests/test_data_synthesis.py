"""Tests for uav_iqa.data_synthesis — DatasetFormat registry, formats, and pipeline."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from uav_iqa.data_synthesis import (
    AirCopBenchFormat,
    DatasetFormat,
    DataSynthesisPipeline,
    GenericImageDirFormat,
    create_pipeline,
)


def _make_dummy_image(path: Path, size: int = 32):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_vqa_json(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f)


# ===========================================================================
# DatasetFormat registry
# ===========================================================================


class TestRegistry:
    def test_list_formats_includes_registered(self):
        formats = DatasetFormat.list_formats()
        assert "aircopbench" in formats
        assert "generic" in formats

    def test_get_returns_correct_class(self):
        assert DatasetFormat.get("aircopbench") is AirCopBenchFormat
        assert DatasetFormat.get("generic") is GenericImageDirFormat

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset format"):
            DatasetFormat.get("bogus")

    def test_register_new_format(self):
        @DatasetFormat.register
        class _TestFormat(DatasetFormat):
            name = "_test_fmt"

            def get_exclude_dirs(self):
                return set()

            def find_reference_images(self, input_root):
                return []

        assert "_test_fmt" in DatasetFormat.list_formats()
        del DatasetFormat._registry["_test_fmt"]


# ===========================================================================
# AirCopBenchFormat
# ===========================================================================


class TestAirCopBenchFormat:
    def test_exclude_dirs(self):
        fmt = AirCopBenchFormat()
        dirs = fmt.get_exclude_dirs()
        assert "loss" in dirs
        assert "Annotations" in dirs
        assert "distorted" in dirs

    def test_find_reference_images_excludes_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_image(root / "good.jpg")
            _make_dummy_image(root / "loss" / "bad.jpg")
            _make_dummy_image(root / "noise" / "bad2.jpg")

            fmt = AirCopBenchFormat()
            images = fmt.find_reference_images(root)
            paths = [p.name for p in images]
            assert "good.jpg" in paths
            assert "bad.jpg" not in paths
            assert "bad2.jpg" not in paths


# ===========================================================================
# GenericImageDirFormat
# ===========================================================================


class TestGenericImageDirFormat:
    def test_no_exclude_dirs(self):
        fmt = GenericImageDirFormat()
        assert fmt.get_exclude_dirs() == set()

    def test_find_reference_images_finds_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_image(root / "a.jpg")
            _make_dummy_image(root / "sub" / "b.png")

            fmt = GenericImageDirFormat()
            images = fmt.find_reference_images(root)
            assert len(images) == 2


# ===========================================================================
# create_pipeline factory
# ===========================================================================


class TestCreatePipeline:
    def test_default_is_aircopbench(self):
        p = create_pipeline()
        assert isinstance(p.format, AirCopBenchFormat)

    def test_explicit_generic(self):
        p = create_pipeline("generic")
        assert isinstance(p.format, GenericImageDirFormat)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_pipeline("nonexistent")


# ===========================================================================
# DataSynthesisPipeline — extract step
# ===========================================================================


class TestPipelineExtract:
    def test_extract_copies_files(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            src = Path(src)
            dst = Path(dst)
            _make_dummy_image(src / "img1.jpg")
            _make_dummy_image(src / "subdir" / "img2.png")

            pipeline = DataSynthesisPipeline(GenericImageDirFormat())
            out = pipeline.extract_references(src, dst, copy=True)
            assert out == dst
            extracted = list(dst.rglob("*"))
            assert len([p for p in extracted if p.suffix in (".jpg", ".png")]) == 2

    def test_extract_symlink_mode(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            src = Path(src)
            dst = Path(dst)
            _make_dummy_image(src / "img1.jpg")

            pipeline = DataSynthesisPipeline(GenericImageDirFormat())
            pipeline.extract_references(src, dst, copy=False)
            extracted = list(dst.rglob("*.jpg"))
            assert len(extracted) == 1
            assert extracted[0].is_symlink() or extracted[0].exists()


# ===========================================================================
# DataSynthesisPipeline — inject step
# ===========================================================================


class TestPipelineInject:
    def test_inject_with_mock_vqa_data(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            src = Path(src)
            dst = Path(dst)

            # Create mock UAV images
            for uav in ("UAV1", "UAV2", "UAV3"):
                uav_dir = src / f"Sim_3_UAVs/Samples/images/scene_001/{uav}"
                uav_dir.mkdir(parents=True)
                _make_dummy_image(uav_dir / f"{uav}_frame_001.jpg", size=64)

            # Create mock VQA JSON in train/
            train_dir = dst / "train"
            train_dir.mkdir(parents=True)
            vqa_entry = {
                "sequence_frame": "scene_001_frame_001",
                "question_id": "Q1",
                "question_type": "1.1 Scene Description",
                "question": "Describe the scene.",
                "options": [],
                "correct_answer": "",
                "uav_paths": {
                    "UAV1": "Sim_3_UAVs/Samples/images/scene_001/UAV1/UAV1_frame_001.jpg",
                    "UAV2": "Sim_3_UAVs/Samples/images/scene_001/UAV2/UAV2_frame_001.jpg",
                    "UAV3": "Sim_3_UAVs/Samples/images/scene_001/UAV3/UAV3_frame_001.jpg",
                },
            }
            _make_vqa_json(train_dir / "Sim3_VQA_train.json", [vqa_entry])

            pipeline = DataSynthesisPipeline(GenericImageDirFormat(), seed=42)
            stats = pipeline.inject_distortions(
                input_root=src,
                output_dir=dst,
                dry_run=True,
                fmt="png",
            )

            assert stats["total_groups"] == 1
            assert stats["total_entries"] >= 1
            assert stats["total_distortions"] >= 1

            # Check output JSON was written
            out_vqa = train_dir / "Sim3_VQA_train.json"
            assert out_vqa.exists()
            with open(out_vqa) as f:
                data = json.load(f)
            assert len(data) == 1
            group = data[0]
            assert "distortions" in group
            assert "vqa_entries" in group
            assert len(group["distortions"]) >= 1
            for dk, di in group["distortions"].items():
                assert "sample_id" in di
                assert "type" in di
                assert "category" in di
                assert "distorted_uav_paths" in di
                assert len(di["distorted_uav_paths"]) == 3

            # Check distorted images exist
            distorted_dir = dst / "distorted"
            assert distorted_dir.is_dir()
            distorted_imgs = list(distorted_dir.glob("*.png"))
            assert len(distorted_imgs) >= 3


# ===========================================================================
# DataSynthesisPipeline — steps parsing
# ===========================================================================


class TestPipelineStepsParsing:
    def test_all_expands_to_four(self):
        steps = DataSynthesisPipeline._parse_steps("all")
        assert steps == {"extract", "inject", "annotate", "aggregate"}

    def test_comma_separated(self):
        steps = DataSynthesisPipeline._parse_steps("extract,inject")
        assert steps == {"extract", "inject"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown steps"):
            DataSynthesisPipeline._parse_steps("bogus")

    def test_manifest_is_invalid(self):
        with pytest.raises(ValueError, match="Unknown steps"):
            DataSynthesisPipeline._parse_steps("manifest")
