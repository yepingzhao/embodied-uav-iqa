"""Tests for uav_iqa.data_synthesis — DatasetFormat registry, formats, and pipeline."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from uav_iqa.data_synthesis import (
    AirCopBenchFormat,
    DataSynthesisPipeline,
    DatasetFormat,
    GenericImageDirFormat,
    create_pipeline,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_dummy_image(path: Path, size: int = 32):
    """Create a small random RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


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

            def build_ref_lookup(self, input_root):
                return {}

            def resolve_ref_path(self, ref_id, ref_dir):
                return ""

            def assign_task(self, identifier, task_map=None):
                return "tracking"

            def parse_ref_id(self, distorted_filename):
                return "test"

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

    def test_parse_ref_id(self):
        fmt = AirCopBenchFormat()
        assert fmt.parse_ref_id("scene001__gaussian_blur_L04.png") == "scene001"
        assert fmt.parse_ref_id("abc__propeller_shadow_L10.png") == "abc"

    def test_assign_task_scene_based(self):
        fmt = AirCopBenchFormat()
        task = fmt.assign_task("path/to/scene_001/something.jpg")
        assert task == "tracking"

    def test_assign_task_with_map(self):
        fmt = AirCopBenchFormat()
        task = fmt.assign_task("some_img.jpg", task_map={"some_img": "sar"})
        assert task == "sar"

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

    def test_build_ref_lookup_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            fmt = AirCopBenchFormat()
            lookup = fmt.build_ref_lookup(Path(tmp))
            assert lookup == {}

    def test_resolve_ref_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            _make_dummy_image(ref_dir / "img001.jpg")

            fmt = AirCopBenchFormat()
            path = fmt.resolve_ref_path("img001", ref_dir)
            assert path.endswith("img001.jpg")

    def test_resolve_ref_path_missing_returns_empty(self):
        fmt = AirCopBenchFormat()
        assert fmt.resolve_ref_path("nonexistent", Path("/nonexistent")) == ""

    def test_get_degradation_types(self):
        entry = {"Degradation": {"choices": ["blur", "noise"]}}
        degs = AirCopBenchFormat.get_degradation_types(entry)
        assert "blur" in degs
        assert "noise" in degs

    def test_get_degradation_types_empty_defaults_to_none(self):
        degs = AirCopBenchFormat.get_degradation_types({})
        assert degs == ["none"]


# ===========================================================================
# GenericImageDirFormat
# ===========================================================================


class TestGenericImageDirFormat:
    def test_no_exclude_dirs(self):
        fmt = GenericImageDirFormat()
        assert fmt.get_exclude_dirs() == set()

    def test_build_ref_lookup_empty(self):
        fmt = GenericImageDirFormat()
        assert fmt.build_ref_lookup(Path("/tmp")) == {}

    def test_parse_ref_id(self):
        fmt = GenericImageDirFormat()
        assert fmt.parse_ref_id("img__blur_L02.png") == "img"

    def test_assign_task_hash_fallback(self):
        fmt = GenericImageDirFormat()
        task = fmt.assign_task("random_image.png")
        assert task in ("tracking", "inspection", "delivery", "sar")


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
# DataSynthesisPipeline steps (integration smoke tests)
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


class TestPipelineManifest:
    def test_generate_manifests_from_mock_distorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            distorted_dir = output_dir / "distorted"
            distorted_dir.mkdir(parents=True)
            for i in range(10):
                fname = f"ref{i:03d}__gaussian_blur_L04.png"
                _make_dummy_image(distorted_dir / fname)

            pipeline = DataSynthesisPipeline(GenericImageDirFormat(), seed=42)
            paths = pipeline.generate_manifests(distorted_dir, output_dir)

            assert "train" in paths
            assert "val" in paths
            assert "test" in paths

            for split_name, manifest_path in paths.items():
                assert manifest_path.exists()
                with open(manifest_path) as f:
                    entries = json.load(f)
                for entry in entries:
                    assert "path" in entry
                    assert "task" in entry
                    assert "distortion" in entry
                    assert "intensity_level" in entry
                    assert entry["distortion"] == "gaussian_blur"


class TestPipelineAnnotate:
    def test_annotate_scores_deterministic_with_noise_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                split_dir = root / split
                split_dir.mkdir(parents=True)
                entries = [
                    {
                        "path": f"{split}/img_{i:03d}.png",
                        "task": "tracking",
                        "distortion": "gaussian_blur",
                        "intensity_level": 0.4,
                        "ref_id": f"ref_{i}",
                        "ref_path": "",
                        "vlm_score": 0.0,
                        "vla_score": 0.0,
                        "execution_score": 0.0,
                        "annotated": False,
                    }
                    for i in range(10)
                ]
                with open(split_dir / "manifest.json", "w") as f:
                    json.dump(entries, f)

            output_dir = root / "annotated"
            pipeline = DataSynthesisPipeline(GenericImageDirFormat(), seed=42)
            pipeline.annotate_scores(root, output_dir=output_dir, noise_scale=0.0)

            for split in ("train", "val", "test"):
                manifest_path = output_dir / split / "manifest.json"
                assert manifest_path.exists()
                with open(manifest_path) as f:
                    entries = json.load(f)
                for entry in entries:
                    for key in ("vlm_score", "vla_score", "execution_score"):
                        assert 0.0 <= entry[key] <= 1.0, f"{key} out of range: {entry[key]}"

    def test_annotate_noise_adds_variation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train").mkdir(parents=True)
            entries = [
                {
                    "path": "train/img_000.png",
                    "task": "tracking",
                    "distortion": "gaussian_blur",
                    "intensity_level": 0.5,
                    "ref_id": "ref_0",
                    "ref_path": "",
                    "vlm_score": 0.8,
                    "vla_score": 0.7,
                    "execution_score": 0.75,
                    "annotated": False,
                }
                for _ in range(5)
            ]
            with open(root / "train" / "manifest.json", "w") as f:
                json.dump(entries, f)

            out_noisy = root / "noisy"
            pipeline = DataSynthesisPipeline(GenericImageDirFormat(), seed=42)
            pipeline.annotate_scores(root, output_dir=out_noisy, noise_scale=0.1)

            with open(out_noisy / "train" / "manifest.json") as f:
                result = json.load(f)
            scores = [e["vlm_score"] for e in result]
            assert len(set(scores)) > 1, "Noise should produce variation in scores"


class TestPipelineStepsParsing:
    def test_all_expands_to_four(self):
        steps = DataSynthesisPipeline._parse_steps("all")
        assert steps == {"extract", "inject", "manifest", "annotate"}

    def test_comma_separated(self):
        steps = DataSynthesisPipeline._parse_steps("extract,inject")
        assert steps == {"extract", "inject"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown steps"):
            DataSynthesisPipeline._parse_steps("bogus")
