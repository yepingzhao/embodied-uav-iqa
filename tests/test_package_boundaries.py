"""Architecture-boundary tests for the completed package migration."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "uav_iqa.model",
        "uav_iqa.distortion",
        "uav_iqa.distortion_synthesis",
        "uav_iqa.dataset",
        "uav_iqa.data_module",
        "uav_iqa.lightning_module",
        "uav_iqa.losses",
        "uav_iqa.callbacks",
        "uav_iqa.metrics",
        "uav_iqa.annotations",
        "uav_iqa.batch_annotator",
        "uav_iqa.text_metrics",
        "uav_iqa.evaluation.metrics",
    ],
)
def test_removed_top_level_modules_are_not_compatibility_aliases(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_root_import_is_lightweight():
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "uav_iqa" or name.startswith("uav_iqa.")
    }
    try:
        for name in original_modules:
            del sys.modules[name]

        importlib.import_module("uav_iqa")

        assert "uav_iqa.vlm.scorer" not in sys.modules
        assert "uav_iqa.training.module" not in sys.modules
        assert "uav_iqa.data.synthesis" not in sys.modules
    finally:
        for name in list(sys.modules):
            if name == "uav_iqa" or name.startswith("uav_iqa."):
                del sys.modules[name]
        sys.modules.update(original_modules)
