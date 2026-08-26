"""Active LightningCLI configurations must target the new package contracts."""

from pathlib import Path

import yaml

from uav_iqa.domain import SUBTASK_NAME_LIST

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _configs():
    return [CONFIG_DIR / "default.yaml", *sorted((CONFIG_DIR / "experiments").glob("*.yaml"))]


def test_active_configs_use_renamed_model_switches_and_training_callbacks():
    for path in _configs():
        config = yaml.safe_load(path.read_text())
        model = config["model"]
        assert "use_fab" not in model, path
        assert "use_cbam" not in model, path
        assert "use_frequency_encoder" in model, path
        assert "use_spatial_attention" in model, path
        callback_paths = [entry["class_path"] for entry in config["trainer"]["callbacks"]]
        assert not any("uav_iqa.callbacks" in value for value in callback_paths), path
        assert any("uav_iqa.training" in value for value in callback_paths), path


def test_active_subtask_filters_are_semantic_task_names():
    for path in _configs():
        data = yaml.safe_load(path.read_text())["data"]
        value = data.get("subtask_filter")
        if value is None:
            continue
        assert set(value.split(",")).issubset(SUBTASK_NAME_LIST), path
