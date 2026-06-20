#!/usr/bin/env python3
"""Convert experiment configs from nested init_args format to flat format.

Before (broken in jsonargparse 4.x when class is pre-registered):
    model:
      class_path: uav_iqa.lightning_module.UAVIQALightningModule
      init_args:
        backbone: mobilenetv4_conv_small

After (compatible with pre-registered classes):
    model:
      backbone: mobilenetv4_conv_small

Also fixes CSVLogger requiring save_dir, and removes CurriculumStageCallback
from experiment configs (defined via defaults).
"""

import yaml
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "experiments")
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "default.yaml"
)


def flatten_section(config, section_name):
    """Flatten class_path + init_args to direct params."""
    if section_name not in config:
        return
    section = config[section_name]
    if not isinstance(section, dict):
        return
    if "init_args" in section:
        init_args = section.pop("init_args")
        section.pop("class_path", None)
        for k, v in init_args.items():
            if k not in section:
                section[k] = v


def fix_callbacks(trainer):
    """Ensure CurriculumStageCallback exists and CSVLogger has save_dir."""
    callbacks = trainer.get("callbacks", [])
    if not isinstance(callbacks, list):
        return

    # Ensure CurriculumStageCallback is present (controls 3-stage curriculum)
    has_curriculum = any(
        isinstance(cb, dict) and "CurriculumStageCallback" in cb.get("class_path", "")
        for cb in callbacks
    )
    if not has_curriculum:
        callbacks.insert(
            1,
            {  # after SetupRunCallback
                "class_path": "uav_iqa.callbacks.CurriculumStageCallback",
                "init_args": {
                    "vlm_epochs": 20,
                    "vla_epochs": 20,
                    "execution_epochs": 10,
                },
            },
        )

    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        cp = cb.get("class_path", "")
        if "CSVLogger" in cp:
            ia = cb.setdefault("init_args", {})
            ia.setdefault("save_dir", ".")


def fix_loggers(trainer):
    """Ensure CSVLogger has save_dir."""
    loggers = trainer.get("logger", [])
    if not isinstance(loggers, list):
        return
    for lg in loggers:
        if not isinstance(lg, dict):
            continue
        cp = lg.get("class_path", "")
        if "CSVLogger" in cp:
            ia = lg.setdefault("init_args", {})
            ia.setdefault("save_dir", ".")


def process_config(filepath):
    with open(filepath) as f:
        config = yaml.safe_load(f)

    flatten_section(config, "model")
    flatten_section(config, "data")

    trainer = config.get("trainer", {})
    if isinstance(trainer, dict):
        fix_callbacks(trainer)
        fix_loggers(trainer)

    with open(filepath, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    return True


def main():
    for fname in sorted(os.listdir(CONFIG_DIR)):
        if not fname.endswith(".yaml"):
            continue
        fpath = os.path.join(CONFIG_DIR, fname)
        try:
            process_config(fpath)
            print(f"  OK  {fname}")
        except Exception as e:
            print(f"  FAIL {fname}: {e}")

    if os.path.exists(DEFAULT_CONFIG):
        try:
            process_config(DEFAULT_CONFIG)
            print("  OK  configs/default.yaml")
        except Exception as e:
            print(f"  FAIL configs/default.yaml: {e}")


if __name__ == "__main__":
    main()
