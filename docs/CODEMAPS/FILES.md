# File Tree Codemap

**Last Updated:** 2026-06-20

## Root Layout

```
embodied-uav-iqa/
├── src/uav_iqa/           # Core library (12 modules, ~3,559 LOC total)
├── scripts/               # 5 executable experiment scripts
├── configs/               # YAML configuration (LightningCLI) + 21 experiment configs
├── tests/                 # pytest test suite (3 files, 35+ tests)
├── refine-logs/           # 16 research refinement artifacts
├── docs/                  # Literature reviews, research roadmap, codemaps
├── main.py                # Unified training entry point (LightningCLI)
├── pyproject.toml         # Project metadata, deps, tool config
├── CLAUDE.md              # Agent guidance (architecture, commands)
├── AGENTS.md              # Quick-reference for development agents
├── README.md              # Project overview & usage
├── .gitignore             # Git ignore rules
├── .python-version        # Python version pinning
├── uv.lock                # uv dependency lockfile
├── config.yaml            # Root config template (optional)
├── checkpoints/           # Model checkpoints (gitignored)
├── data/                  # Datasets: raw/ and processed/ (gitignored)
├── outputs/               # Experiment outputs (gitignored)
└── .venv/                 # Virtual environment (gitignored)
```

---

## `src/uav_iqa/` — Core Library

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 83 | Public API: exports **35 symbols** (was 22) |
| `distortion.py` | 685 | 24 distortion models + UAVDistortionPipeline + `UAV_DISTORTION_NAMES` constant |
| `model.py` | 395 | UAVIQANet: backbone → FPN → CBAM → FAB → task heads, `forward_features()` feature sharing |
| `dataset.py` | 260 | UAVIQADataset + `validate_manifest()` + constants (TASK_NAMES, TASK_TO_ID) |
| `losses.py` | 30 | ListMLELoss + CrossTaskRegularization (replaces former `trainer.py`) |
| `metrics.py` | 150 | SRCC, PLCC, RMSE, Kendall τ + per-task / per-distortion / **per-category** metrics |
| `annotations.py` | 367 | AirCopBench annotation parsing, degradation factors, score synthesis, **parse_distortion_key**, **compute_synthetic_score** |
| `data_synthesis.py` | **769** | **NEW** — dataset-agnostic pipeline: DatasetFormat (ABC + registry), AirCopBenchFormat, GenericImageDirFormat, DataSynthesisPipeline, create_pipeline |
| `lightning_module.py` | 280 | LightningModule: UAVIQANet + MSE/ListMLE/cross-task, feature sharing, per-distortion ranking |
| `data_module.py` | 157 | LightningDataModule: manifest filtering (task/distortion/leave-out), train/val/test splits |
| `callbacks.py` | 256 | SetupRunCallback + CurriculumStageCallback + MetricsHistoryCallback + ResultsSavingCallback |
| `utils.py` | **127** | **Expanded** — `count_parameters()`, `find_images()`, `load_image_tensor()`, `load_manifest()`, `write_manifest()`, `split_samples()`, `load_task_map()`, `setup_logging()` |

### Per-File Dependencies

```
__init__.py
  → distortion.py, model.py, dataset.py, metrics.py,
    losses.py, annotations.py, utils.py, data_synthesis.py,
    lightning_module.py, data_module.py
  (10 internal deps)

distortion.py
  → cv2, numpy, scipy.signal, albumentations

model.py
  → torch, timm

dataset.py
  → torch, numpy

losses.py
  → torch

metrics.py
  → numpy, scipy.stats, dataset.py (TASK_NAMES)

annotations.py
  → json, hashlib, re, logging, pathlib, dataset.py (TASK_NAMES)

data_synthesis.py (NEW)
  → numpy, json, shutil, re, logging, pathlib, abc, typing
  → uav_iqa.distortion (UAVDistortionPipeline)
  → uav_iqa.annotations (build_ref_score_lookup, assign_task_label, ...)
  → uav_iqa.utils (find_images, split_samples, write_manifest)

lightning_module.py
  → lightning, torch, numpy
  → metrics.py (evaluate_iqa, per_task_metrics, per_distortion_metrics)
  → model.py (UAVIQANet), losses.py (ListMLELoss, CrossTaskRegularization)
  → distortion.py (UAV_DISTORTION_NAMES)

data_module.py
  → lightning, dataset.py (UAVIQADataset), distortion.py (UAV_DISTORTION_NAMES)

callbacks.py
  → lightning, yaml, json, hashlib, subprocess, pathlib
  → utils.py (count_parameters)

utils.py
  → torch, numpy, PIL.Image, json, logging, pathlib
  → dataset.py (validate_manifest) [via write_manifest]
```

---

## `scripts/` — Experiment Entrypoints

| File | Lines | Purpose | Pipeline Stage |
|------|-------|---------|----------------|
| `data_synthesis.py` | 196 | Unified data synthesis CLI (extract/inject/manifest/annotate/all). Uses `DataSynthesisPipeline` from `src/uav_iqa/data_synthesis.py` | M1 |
| `benchmark_iqa_methods.py` | 331 | Benchmark 15+ IQA methods (pyiqa) on test set | M2 |
| `overfit_sanity_check.py` | 105 | Overfit correctness test: train on 100 images, verify loss → 0 | Validation |
| `visualize_distortions.py` | 108 | Visual sanity check: grid of all 24 distortions × 5 intensities | Validation |
| `validate_synth_real_correlation.py` | 132 | C2 correlation validation: synthetic vs real scores | Validation |

### Script Execution Order (Standard Pipeline)

```
1. data_synthesis.py all            # Extract → inject → manifest → annotate (full M1 pipeline)
2. main.py                         # Train UAVIQANet via LightningCLI + experiment config
3. benchmark_iqa_methods.py         # Benchmark (optional, after training)
```

### Script CLI Patterns

All scripts support `--help` (argparse). Common conventions:

```bash
# Data paths: --manifest-dir, --aircopbench-dir, --data-root, --image-dir
# Training via main.py: python main.py fit --config configs/experiments/<name>.yaml
# Parallelism via --workers (default 4, 8, or 16)
# Scripts use `sys.path.insert(0, "src")` to resolve imports
```

---

## `configs/` — Configuration

| File | Purpose |
|------|---------|
| `default.yaml` | Reference template: model arch, data params, trainer settings, callbacks |
| `experiments/` | **21 self-contained experiment configs** (r013–r024c) — each is a complete LightningCLI YAML |

### Experiment Configs (21 total)

All 21 configs follow the same structure as `default.yaml` (67 lines) with per-experiment overrides:

| Config Range | Count | Experiment | Key Difference from Baseline |
|-------------|-------|------------|------------------------------|
| r013 | 1 | Task-conditioned (full baseline) | use_task_conditioning=true |
| r014 | 1 | Task-agnostic | use_task_conditioning=false |
| r016 | 1 | No FAB | use_fab=false |
| r017 | 1 | No task conditioning | use_task_conditioning=false |
| r018 | 1 | No CBAM | use_cbam=false |
| r019 | 1 | MobileViT-S backbone | backbone=mobilevit_s |
| r020 | 1 | EfficientViT-B0 backbone | backbone=efficientvit_b0 |
| r021 | 1 | Generic distortions only | distortion_filter=generic |
| r021b | 1 | UAV distortions only | distortion_filter=uav_only |
| r022 | 1 | VLM annotation only | annotator_stage=vlm |
| r022b | 1 | VLA annotation only | annotator_stage=vla |
| r023 | 1 | No execution scores | total_epochs=40 (skip execution stage) |
| r024a | 4 | Single-task | task=tracking/inspection/delivery/sar |
| r024b | 4 | Leave-one-task-out | leave_out_task=tracking/inspection/delivery/sar |
| r024c | 1 | Multi-task (all 4 tasks) | task=null (no filter) |

### Config Structure (default.yaml)

```yaml
seed_everything: 42
model:
  class_path: uav_iqa.lightning_module.UAVIQALightningModule
  init_args:
    backbone: mobilenetv4_conv_small
    num_tasks: 4
    use_fab: true
    use_cbam: true
    use_task_conditioning: true
    freeze_backbone_stage: 2
    lambda_rank: 0.3
    lambda_cross_task: 0.1
    lr: 1.2e-3
    weight_decay: 1.0e-4
    warmup_epochs: 5
    total_epochs: 50
    annotator_stage: vla
data:
  class_path: uav_iqa.data_module.UAVIQDataModule
  init_args:
    data_root: data/processed
    batch_size: 256
    num_workers: 16
    image_size: 256
    annotator_stage: vla
    task: null
    val_task: null
    distortion_filter: null
    leave_out_task: null
    dry_run: false
trainer:
  accelerator: auto
  devices: 1
  precision: 16-mixed
  max_epochs: 50
  gradient_clip_val: 1.0
  log_every_n_steps: 10
  logger:
    - class_path: swanlab.integration.pytorch_lightning.SwanLabLogger
      init_args:
        project: uav-iqa
    - class_path: lightning.pytorch.loggers.CSVLogger
      init_args: {}
  callbacks:
    - class_path: uav_iqa.callbacks.SetupRunCallback
    - class_path: uav_iqa.callbacks.CurriculumStageCallback
      init_args: {vlm_epochs: 20, vla_epochs: 20, execution_epochs: 10}
    - class_path: uav_iqa.callbacks.MetricsHistoryCallback
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args: {monitor: val/srcc, mode: max, save_top_k: 1, filename: best_model}
    - class_path: uav_iqa.callbacks.ResultsSavingCallback
```

---

## `tests/` — Test Suite

| File | Tests | Coverage |
|------|-------|----------|
| `test_distortion.py` | 10 | All 6 UAV distortions + pipeline + intensity range + generate_all + determinism |
| `test_lightning.py` | 94 | LightningModule init, ablations, optimizer config, training step; DataModule setup with mock data |
| `test_data_synthesis.py` | **336** (NEW) | DatasetFormat registry (4), AirCopBenchFormat (10), GenericImageDirFormat (4), create_pipeline (3), PipelineExtract (2), PipelineManifest (1), PipelineAnnotate (2), PipelineStepsParsing (3) |

### Test Dependencies

```
test_distortion.py        → uav_iqa.distortion
test_lightning.py         → uav_iqa.lightning_module, uav_iqa.data_module
test_data_synthesis.py    → uav_iqa.data_synthesis (DatasetFormat, DataSynthesisPipeline,
                             AirCopBenchFormat, GenericImageDirFormat, create_pipeline)
```

---

## `refine-logs/` — Research Refinement Artifacts

| File | Purpose |
|------|---------|
| `FINAL_PROPOSAL.md` | Refined method thesis (score 9.0/10) |
| `EXPERIMENT_PLAN.md` | 33 experiments across 6 milestones, 4 claims |
| `EXPERIMENT_TRACKER.md` | Run-by-run status tracker |
| `REVIEW_SUMMARY.md` | External review resolutions |
| `REFINEMENT_REPORT.md` | Refinement process report |
| `REFINE_STATE.json` | Refinement state machine |
| `score-history.md` | Score progression across rounds |
| `round-0-initial-proposal.md` | Original raw proposal |
| `round-1-refinement.md` | First refinement pass |
| `round-1-review.md` | First external review |
| `round-2-refinement.md` | Second refinement pass |
| `round-2-review.md` | Second external review |
| `round-3-review.md` | Third external review |
| `codex_round_1_review_bundle.md` | Codex MCP review bundle 1 |
| `codex_round_2_review_bundle.md` | Codex MCP review bundle 2 |
| `codex_round_3_review_bundle.md` | Codex MCP review bundle 3 |

---

## `docs/` — Documentation

| File | Purpose |
|------|---------|
| `CODEMAPS/` | (This directory) — architectural maps |
| `DELETION_LOG.md` | Log of deleted/refactored files |
| `低空无人机具身智能的图像质量评估-文献综述.md` | Literature review (Chinese) |
| `图像质量评估 for Embodied AI.md` | Topic overview (Chinese) |
| `Embodied Image Quality Assessment for Robotic Intelligence.md` | Topic overview (English) |
| `AirCopBench A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning.md` | AirCopBench summary |
| `低空无人机具身智能的图像质量评估-研究路线图.md` | Research roadmap (Chinese) |

---

## Root Entry Points and Config Files

| File | Purpose |
|------|---------|
| `main.py` | **Unified training entry point** — vanilla LightningCLI (fit/test/predict). Usage: `python main.py fit --config configs/experiments/<name>.yaml` |
| `pyproject.toml` | Package metadata, dependencies, ruff config, pytest config |
| `.python-version` | Python version pinning (3.10+) |
| `uv.lock` | Reproducible dependency lock file |
| `config.yaml` | Optional root-level config template |
| `.gitignore` | Ignores: data/, outputs/, checkpoints/, __pycache__/, .venv/, lightning_logs/ |
| `CLAUDE.md` | Detailed agent guidance (architecture, commands, gotchas) |
| `AGENTS.md` | Quick-reference for AI coding agents |
| `README.md` | Project overview, setup, usage, and documentation index |
