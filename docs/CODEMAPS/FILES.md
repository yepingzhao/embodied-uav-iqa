# File Tree Codemap

**Last Updated:** 2026-06-20

## Root Layout

```
embodied-uav-iqa/
├── src/uav_iqa/           # Core library (~2.2K LOC total)
├── scripts/               # 8 executable experiment scripts
├── configs/               # YAML configuration (LightningCLI) + 21 experiment configs
├── tests/                 # pytest test suite (2 files, 17 tests)
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
| `__init__.py` | 49 | Public API: exports 22 symbols |
| `distortion.py` | 664 | 24 distortion models + UAVDistortionPipeline |
| `model.py` | 385 | UAVIQANet: backbone → FPN → CBAM → FAB → task heads, `forward_features()` feature sharing |
| `dataset.py` | 159 | UAVIQADataset: manifest.json loader, annotation scores, augmentation |
| `losses.py` | 32 | ListMLELoss + CrossTaskRegularization (replaces former `trainer.py`) |
| `evaluate.py` | 109 | SRCC, PLCC, RMSE, Kendall τ + per-task/per-distortion metrics |
| `annotation_utils.py` | 155 | AirCopBench annotation parsing, degradation factors, score synthesis |
| `lightning_model.py` | 272 | LightningModule: UAVIQANet + MSE/ListMLE/cross-task, feature sharing, per-distortion ranking |
| `lightning_data.py` | 155 | LightningDataModule: manifest filtering (task/distortion/leave-out), train/val/test splits |
| `callbacks.py` | 234 | SetupRunCallback + CurriculumStageCallback + MetricsHistoryCallback + ResultsSavingCallback |
| `utils.py` | 5 | count_parameters() |

### Per-File Dependencies

```
__init__.py
  → distortion.py, model.py, dataset.py, evaluate.py,
    losses.py, annotation_utils.py,
    lightning_model.py, lightning_data.py

distortion.py
  → cv2, numpy, scipy.signal, albumentations

model.py
  → torch, timm

dataset.py
  → torch, PIL, numpy

losses.py
  → torch

evaluate.py
  → numpy, scipy.stats

annotation_utils.py
  → json, hashlib, re, logging, pathlib

lightning_model.py
  → lightning, torch, evaluate.py, model.py, losses.py

lightning_data.py
  → lightning, dataset.py, distortion.py

callbacks.py
  → lightning, yaml, utils.py

utils.py
  → (none)
```

---

## `scripts/` — Experiment Entrypoints

| File | Lines | Purpose | Pipeline Stage |
|------|-------|---------|----------------|
| `synthesize_data.py` | — | Unified data synthesis CLI (extract/inject/manifest/annotate/all) | M1 |
| `run_m2_benchmark.py` | 311 | Benchmark 15+ IQA methods (pyiqa) on test set | M2 |
| `run_overfit.py` | 107 | Overfit correctness test: train on 100 images, verify loss → 0 | Validation |
| `run_distortion.py` | 101 | Visual sanity check: grid of all 24 distortions × 5 intensities | Validation |
| `run_c2_correlation.py` | — | C2 correlation validation: synthetic vs real scores | Validation |

### Script Execution Order (Standard Pipeline)

```
1. synthesize_data.py all          # Extract → inject → manifest → annotate (full M1 pipeline)
2. main.py                         # Train UAVIQANet via LightningCLI + experiment config
3. run_m2_benchmark.py             # Benchmark (optional, after training)
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

### Experiment Configs

All 21 configs follow the same structure as `default.yaml` with per-experiment overrides:

| Config Range | Experiment | Key Difference from Baseline |
|-------------|------------|------------------------------|
| r013 | Task-conditioned (full baseline) | use_task_conditioning=true |
| r014 | Task-agnostic | use_task_conditioning=false |
| r016 | No FAB | use_fab=false |
| r017 | No task conditioning | use_task_conditioning=false |
| r018 | No CBAM | use_cbam=false |
| r019 | MobileViT-S backbone | backbone=mobilevit_s |
| r020 | EfficientViT-B0 backbone | backbone=efficientvit_b0 |
| r021 | Generic distortions only | distortion_filter=generic |
| r021b | UAV distortions only | distortion_filter=uav_only |
| r022 | VLM annotation only | annotator_stage=vlm |
| r022b | VLA annotation only | annotator_stage=vla |
| r023 | No execution scores | total_epochs=40 (skip execution stage) |
| r024a | Single-task (4 variants) | task=tracking/inspection/delivery/sar |
| r024b | Leave-one-task-out (4 variants) | leave_out_task=tracking/inspection/delivery/sar |
| r024c | Multi-task (all 4 tasks) | task=null (no filter) |

### Config Structure (default.yaml)

```yaml
seed_everything: 42
model:
  class_path: uav_iqa.lightning_model.UAVIQALightningModule
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
  class_path: uav_iqa.lightning_data.UAVIQDataModule
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
| `test_lightning.py` | 5+ | LightningModule init, ablations, optimizer config, training step; DataModule setup with mock data |

### Test Dependencies

```
test_distortion.py → uav_iqa.distortion
test_lightning.py  → uav_iqa.lightning_model, uav_iqa.lightning_data
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
