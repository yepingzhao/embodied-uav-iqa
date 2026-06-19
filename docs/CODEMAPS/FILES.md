# File Tree Codemap

**Last Updated:** 2026-06-19

## Root Layout

```
embodied-uav-iqa/
├── src/uav_iqa/           # Core library (~1.9K LOC total)
├── scripts/               # 8 executable experiment scripts
├── configs/               # YAML configuration (LightningCLI)
├── tests/                 # pytest test suite (2 files, 17 tests)
├── refine-logs/           # 16 research refinement artifacts
├── docs/                  # Literature reviews & research roadmap
├── pyproject.toml         # Project metadata, deps, tool config
├── CLAUDE.md              # Agent guidance (architecture, commands)
├── AGENTS.md              # Quick-reference for development agents
├── README.md              # Project overview & usage
├── .gitignore             # Git ignore rules
├── .python-version        # Python version pinning
├── uv.lock                # uv dependency lockfile
├── checkpoints/           # Model checkpoints (gitignored)
├── data/                  # Datasets (gitignored)
├── outputs/               # Experiment outputs (gitignored)
└── .venv/                 # Virtual environment (gitignored)
```

---

## `src/uav_iqa/` — Core Library

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 33 | Public API: exports 17 symbols |
| `distortion.py` | 665 | 24 distortion models + UAVDistortionPipeline |
| `model.py` | 379 | UAVIQANet architecture: backbone → FPN → CBAM → FAB → task heads |
| `dataset.py` | 134 | UAVIQADataset: manifest.json loader |
| `trainer.py` | 32 | ListMLELoss + CrossTaskRegularization |
| `evaluate.py` | 109 | SRCC, PLCC, RMSE, Kendall τ metrics |
| `lightning_model.py` | 224 | LightningModule: UAVIQANet + MSE/ListMLE/cross-task loss |
| `lightning_data.py` | 152 | LightningDataModule: manifest filtering, train/val/test splits |
| `callbacks.py` | 75 | CurriculumStageCallback + MetricsHistoryCallback |
| `cli.py` | 97 | UAVIQACLI: multi-seed LightningCLI |
| `utils.py` | 4 | count_parameters() |

### Per-File Dependencies

```
__init__.py
  → distortion.py, model.py, dataset.py, evaluate.py,
    lightning_model.py, lightning_data.py, cli.py

distortion.py
  → cv2, numpy, scipy.signal, albumentations

model.py
  → torch, timm

dataset.py
  → torch, PIL, numpy

trainer.py
  → torch

evaluate.py
  → numpy, scipy.stats

lightning_model.py
  → lightning, torch, evaluate.py, model.py, trainer.py

lightning_data.py
  → lightning, dataset.py

callbacks.py
  → lightning

cli.py
  → lightning, callbacks.py, utils.py

utils.py
  → (none)
```

---

## `scripts/` — Experiment Entrypoints

| File | Lines | Purpose | Pipeline Stage |
|------|-------|---------|----------------|
| `extract_aircopbench_refs.py` | ~150 | Extract clean reference frames from AirCopBench directory tree | M1.1 |
| `run_m1_inject.py` | ~100 | Batch distortion injection: 24 types × 5 levels per ref image | M1.2 |
| `run_m1_manifest.py` | ~120 | Scan distorted dir → train/val/test manifest.json with placeholder scores | M1.3 |
| `run_m1_aircopbench.py` | ~200 | Full M1: scan AirCopBench → extract → index → inject → build manifest with real annotations | M1 Alternative |
| `run_m2_benchmark.py` | ~250 | Benchmark 15+ IQA methods (pyiqa) on test set | M2 |
| `run_m3_train.py` | ~200 | Main training script: multi-seed, ablations, cross-task, leave-one-out, dry-run | M3 |
| `run_overfit.py` | ~80 | Overfit correctness test: train on 100 images, verify loss → 0 | Validation |
| `run_distortion.py` | ~60 | Visual sanity check: grid of all 24 distortions × 5 intensities | Validation |

### Script Execution Order (Standard Pipeline)

```
1. extract_aircopbench_refs.py     # Extract clean frames
2. run_m1_inject.py                # Inject distortions
3. run_m1_manifest.py              # Generate manifest.json
4. run_m3_train.py                 # Train model
5. run_m2_benchmark.py             # Benchmark (optional, after training)
```

### Script CLI Patterns

All scripts support `--help` (argparse). Common conventions:

```bash
# I/O paths via --data-dir, --output-dir, --image-dir, --distorted-dir
# Parallelism via --workers (default 4 or 8)
# Scripts use `sys.path.insert(0, "src")` to resolve imports
```

---

## `configs/` — Configuration

| File | Purpose |
|------|---------|
| `default.yaml` | LightningCLI format: model arch, data params, trainer settings |
| `experiment/` | Placeholder directory (empty) for experiment-specific configs |

### Config Structure (default.yaml)

```yaml
seed_everything: 42
model:
  class_path: uav_iqa.lightning_model.UAVIQALightningModule
  init_args:
    backbone: mobilenetv4_conv_small
    use_fab: true
    use_cbam: true
    use_task_conditioning: true
    freeze_backbone_stage: 2
    lambda_rank: 0.3
    lambda_cross_task: 0.1
    lr: 3.0e-4
    warmup_epochs: 5
    total_epochs: 50
data:
  class_path: uav_iqa.lightning_data.UAVIQDataModule
  init_args:
    data_root: data/database
    batch_size: 64
    image_size: 256
trainer:
  accelerator: auto
  precision: 16-mixed
  max_epochs: 50
  gradient_clip_val: 1.0
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

## Root Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, ruff config, pytest config |
| `.python-version` | Python version pinning (3.10+) |
| `uv.lock` | Reproducible dependency lock file |
| `.gitignore` | Ignores: data/, outputs/, checkpoints/, __pycache__/, .venv/, wandb/ |
| `CLAUDE.md` | Detailed agent guidance (architecture, commands, gotchas) |
| `AGENTS.md` | Quick-reference for AI coding agents |
| `README.md` | Project overview, setup, usage, and documentation index |
