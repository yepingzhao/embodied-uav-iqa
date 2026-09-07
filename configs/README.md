# Training configurations

Use [experiments/full_model.yaml](experiments/full_model.yaml) as the reference
configuration. Each experiment is a self-contained LightningCLI configuration; there is
no separate default template. Filenames describe the model or data variant rather than
historical run IDs; configuration availability does not indicate completed experiments.

| Configuration | Difference from the reference |
| --- | --- |
| [full_model.yaml](experiments/full_model.yaml) | Reference: all components enabled |
| [no_task_conditioning.yaml](experiments/no_task_conditioning.yaml) | Disable task conditioning |
| [no_frequency_encoder.yaml](experiments/no_frequency_encoder.yaml) | Disable the frequency encoder |
| [no_spatial_attention.yaml](experiments/no_spatial_attention.yaml) | Disable spatial attention |
| [backbone_mobilevit_s.yaml](experiments/backbone_mobilevit_s.yaml) | Use the MobileViT-S backbone |
| [backbone_efficientvit_b0.yaml](experiments/backbone_efficientvit_b0.yaml) | Use the EfficientViT-B0 backbone |
| [generic_only.yaml](experiments/generic_only.yaml) | Keep generic distortions only |
| [uav_only.yaml](experiments/uav_only.yaml) | Keep UAV-specific distortions only |

## Usage

Run from the repository root after installing the project:

```bash
# These configurations explicitly include WandbLogger.
uv sync --group dev --extra wandb

python scripts/train.py fit --config configs/experiments/full_model.yaml

# Model/data parameters are flat, not nested under init_args.
python scripts/train.py fit --config configs/experiments/full_model.yaml \
  --data.batch_size 32 --trainer.max_epochs 3
```

Configure W&B authentication/offline mode before training, or replace `trainer.logger`
with a CSV-only logger configuration for local-only runs. Missing W&B credentials do
not trigger an automatic CSV-only fallback.

## Scope and prerequisites

- Model, data, and optimization parameters are unchanged by cleanup and renaming. Verify `data.data_root` contains
  the intended aggregated labels and resolvable image paths before training; a valid
  configuration does not certify dataset readiness.
- `subtask_filter` selects semantic AirCopBench subtasks and applies to every split.
  It does not implement independent training/held-out task selections. No maintained
  leave-one-task-out configuration is provided until that protocol is implemented.
- The current DataModule uses the test split for both validation and testing. Establish
  an independent validation protocol before reporting final test performance.
- Changing `trainer.default_root_dir` alone does not change the explicit logger paths.
  Set logger names/save directories as well when isolating multi-seed runs.
