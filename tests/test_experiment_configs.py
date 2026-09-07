"""Active LightningCLI configurations must target the new package contracts."""

import inspect
from pathlib import Path

import yaml

from uav_iqa.domain import SUBTASK_NAME_LIST

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _configs():
    paths = sorted((CONFIG_DIR / "experiments").glob("*.yaml"))
    assert paths, "No maintained experiment configurations found"
    return paths


def test_config_run_identity_matches_filename():
    for path in _configs():
        config = yaml.safe_load(path.read_text())
        trainer = config["trainer"]
        assert trainer["default_root_dir"] == f"outputs/{path.stem}", path
        for logger in trainer["logger"]:
            args = logger["init_args"]
            assert args["name"] == path.stem, path
            if logger["class_path"].endswith("CSVLogger"):
                assert args["save_dir"] == "outputs", path
            else:
                assert args["save_dir"] == f"outputs/{path.stem}", path


def test_active_configs_match_model_and_data_constructor_parameters():
    from uav_iqa.data import UAVQualityDataModule
    from uav_iqa.training import UAVQualityTrainingModule

    for section, cls in (
        ("model", UAVQualityTrainingModule),
        ("data", UAVQualityDataModule),
    ):
        parameters = set(inspect.signature(cls.__init__).parameters) - {"self"}
        for path in _configs():
            config = yaml.safe_load(path.read_text())
            assert set(config[section]).issubset(parameters), (path, section)


def test_active_configs_are_not_duplicates_except_for_run_identity():
    seen = {}
    for path in _configs():
        config = yaml.safe_load(path.read_text())
        config["trainer"].pop("default_root_dir", None)
        for logger in config["trainer"]["logger"]:
            logger["init_args"].pop("name", None)
            logger["init_args"].pop("save_dir", None)
        signature = yaml.safe_dump(config, sort_keys=True)
        assert signature not in seen, (path, seen.get(signature))
        seen[signature] = path


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
