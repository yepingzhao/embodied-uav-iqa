# File Tree Codemap

**Last Updated:** 2026-06-27

## Root Layout

```
embodied-uav-iqa/
├── src/uav_iqa/           # Core library (16 modules, 5,769 LOC total)
├── scripts/               # 10 executable experiment scripts
├── configs/               # YAML configuration (LightningCLI) + 21 experiment configs
├── tests/                 # pytest test suite (8 files, 154 tests)
├── refine-logs/           # 17 research refinement artifacts
├── docs/                  # Literature reviews, research roadmap, codemaps
├── scripts/train.py        # Unified training entry point (LightningCLI)
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

## `src/uav_iqa/` — Core Library (16 modules, 5,769 LOC)

*(Note: `vlm_vla_scorer.py` backward-compat shim was removed; its contents are now split into `vlm/` subpackage, `vla_scorer.py`, `text_metrics.py`, and `batch_annotator.py`.)*

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 84 | Public API: exports **35 symbols** |
| `distortion.py` | **787** | 36 distortion models (6 UAV + 30 generic) + UAVDistortionPipeline + `UAV_DISTORTION_NAMES` constant. Supports JPEG output format. |
| `model.py` | **426** | UAVIQANet: backbone → FPN → CBAM → FAB → task heads, `forward_features()` feature sharing. **Dynamic stage probing** + **regex-based freeze** for backbone-agnostic compatibility |
| `dataset.py` | 250 | UAVIQADataset + `validate_manifest()` + constants (TASK_NAMES, TASK_TO_ID) |
| `text_metrics.py` | **318** | BLEU, ROUGE-L, CIDEr text similarity metrics for VLM comparison-based scoring. Pure Python (no nltk). |
| `vla_scorer.py` | 79 | Base scoring: `BaseScorer` ABC + `extract_ref_id()` + task type constants |
| `vlm/` | **1472** | VLM scoring subpackage: `VLMConfig` + `MODEL_REGISTRY` (config.py, 175 lines), `VLMScorer` with vLLM/transformers backends + `VQAIndex` (scorer.py 1232 + vqa_index.py 58 + __init__.py 7) |
| `batch_annotator.py` | **365** | Batch annotation engine: `BatchAnnotator` with manifest I/O, filtering, checkpoint/resume, multi-split orchestration |
| `losses.py` | 30 | ListMLELoss + CrossTaskRegularization |
| `metrics.py` | 145 | SRCC, PLCC, RMSE, Kendall τ + per-task / per-distortion / **per-category** metrics |
| `annotations.py` | 162 | AirCopBench annotation parsing: `parse_distortion_key`, `parse_quality_score`, `parse_usability`, `build_ref_score_lookup`, `assign_task_label`. |
| `data_synthesis.py` | **822** | Dataset-agnostic pipeline: DatasetFormat (ABC + registry), AirCopBenchFormat, GenericImageDirFormat, DataSynthesisPipeline, create_pipeline. Supports `fmt` (png/jpeg) and `max_refs_per_source`. |
| `lightning_module.py` | 289 | LightningModule: UAVIQANet + MSE/ListMLE/cross-task, feature sharing, per-distortion ranking, DDP gathering |
| `data_module.py` | 176 | LightningDataModule: manifest filtering (task/distortion/leave-out), stage-aware setup, DDP-compatible |
| `callbacks.py` | **215** | SetupRunCallback + CurriculumStageCallback + **MetricsHistoryCallback** + **ResultsSavingCallback** (DDP-safe) |
| `utils.py` | **149** | `count_parameters()`, `find_images()`, `load_image_tensor()`, `load_manifest()`, `write_manifest()`, `split_samples()`, `load_task_map()`, `setup_logging()` |

### Per-File Dependencies

```
__init__.py
  → distortion.py, model.py, dataset.py, metrics.py,
    losses.py, annotations.py, utils.py, data_synthesis.py,
    lightning_module.py, data_module.py,
    vla_scorer.py, vlm/ (config, scorer), batch_annotator.py
  (13 internal deps)

text_metrics.py
  → math, collections, numpy (stdlib + numpy)

distortion.py
  → cv2, numpy, tqdm, albumentations
  → basicsr + realesrgan (optional, for LowResSuperResolution)

model.py
  → torch, timm, math, re

dataset.py
  → torch, numpy

vla_scorer.py
  → hashlib, abc, pathlib (stdlib)

vlm/config.py
  → logging, dataclasses (stdlib)

vlm/scorer.py
  → vla_scorer.py (BaseScorer, extract_ref_id)
  → vlm/config.py (VLMConfig, MODEL_REGISTRY)
  → vlm/vqa_index.py (VQAIndex)
  → vllm (optional), transformers (optional), PIL, torch, tqdm

vlm/vqa_index.py
  → json, logging, pathlib (stdlib)

batch_annotator.py
  → vla_scorer.py (BaseScorer)
  → vlm/scorer.py (VLMScorer, for type hints)
  → json, tempfile, os, time, pathlib (stdlib)
  → tqdm (optional)

losses.py
  → torch

metrics.py
  → numpy, scipy.stats, dataset.py (TASK_NAMES)

annotations.py
  → json, hashlib, re, logging, pathlib, dataset.py (TASK_NAMES)

data_synthesis.py
  → numpy, json, shutil, logging, pathlib, abc, typing
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
  → lightning, hashlib, pathlib, json, logging, subprocess
  → utils.py (count_parameters)
  → yaml (for ResultsSavingCallback._read_seed_from_config)

utils.py
  → torch, numpy, PIL.Image, json, logging, pathlib
  → dataset.py (validate_manifest) [via write_manifest]
```

---

## `scripts/` — Experiment Entrypoints

| File | Lines | Purpose | Pipeline Stage |
|------|-------|---------|----------------|
| `data_synthesis.py` | 306 | Unified data synthesis CLI (extract/inject/manifest/annotate/all). Uses `DataSynthesisPipeline` from `src/uav_iqa/data_synthesis.py`. Supports `--max-refs` and `--format` (png/jpeg). | M1 |
| `download_models.py` | **277** | Download VLM model weights from HuggingFace Hub for offline use. Supports all 15 registered models, auth token, validation loading. | M1.5 |
| `vlm_annotate.py` | **261** | Batch VLM annotation CLI — score manifest entries with real VLM models via VLMScorer + BatchAnnotator. Supports single-model, multi-model ensemble, all 15 models, distortion/task filtering, checkpoint/resume | M1.5 |
| `benchmark_iqa_methods.py` | 594 | Benchmark 15+ IQA methods (pyiqa) on test set | M2 |
| `finetune_baselines.py` | **420** | Fine-tune DL-based IQA baselines (brisque/niqe/clipiqa/maniqa/topiq_nr) on UAV training data. Uses standalone LightningModule + LightningDataModule, no LightningCLI | M2 |
| `overfit_sanity_check.py` | 105 | Overfit correctness test: train on 100 images, verify loss → 0 | Validation |
| `visualize_distortions.py` | 108 | Visual sanity check: grid of all 36 distortions × 1 random intensity | Validation |
| `validate_synth_real_correlation.py` | 188 | C2 correlation validation: synthetic vs real scores | Validation |

| `fix_configs.py` | 130 | Convert experiment YAML configs from nested `class_path+init_args` to flat format; ensures CurriculumStageCallback and CSVLogger `save_dir` are present | Utility |
| `train.py` | 35 | **Unified training entry point** — vanilla LightningCLI (fit/test/predict). Usage: `python scripts/train.py fit --config configs/experiments/<name>.yaml` | M3 |

### Script Execution Order (Standard Pipeline)

```
1. data_synthesis.py all            # Extract → inject → manifest → annotate (full M1 pipeline)
2. download_models.py --all         # Download VLM model weights (optional, M1.5)
3. vlm_annotate.py                  # Batch VLM annotation (optional, M1.5)
4. scripts/train.py                 # Train UAVIQANet via LightningCLI + experiment config
5. benchmark_iqa_methods.py         # Benchmark zero-shot IQA methods (optional, after training)
6. finetune_baselines.py            # Fine-tune DL baselines on UAV data (optional, M2-R009a)

Utility:

- fix_configs.py                   # Batch-convert experiment configs between formats
- visualize_distortions.py         # Visual sanity check for all 36 distortions
- overfit_sanity_check.py          # Model correctness verification (100-image overfit)
- validate_synth_real_correlation.py  # C2 synthetic↔real score correlation
```

### Script CLI Patterns

All scripts support `--help` (argparse). Common conventions:

```bash
# Data paths: --manifest-dir, --aircopbench-dir, --data-root, --image-dir
# Training via scripts/train.py: python scripts/train.py fit --config configs/experiments/<name>.yaml
# Parallelism via --workers (default 0 = auto-detect up to 16, or --workers 8)
# Most scripts import from installed package (`pip install -e .` or `uv sync`)
# scripts/download_models.py uses importlib lazy-import of MODEL_REGISTRY from
#   src/uav_iqa/vlm/config.py (stdlib-only import path, avoids lightning/torch chain)
# Only test_lightning.py still uses sys.path.insert for src/ resolution
```

---

## `configs/` — Configuration

| File | Purpose |
|------|---------|
| `default.yaml` | Reference template: model arch, data params, trainer settings, callbacks |
| `experiments/` | **21 self-contained experiment configs** (r013–r024c) — each is a complete LightningCLI YAML |

### Experiment Configs (21 total)

All 21 configs follow the same structure as `default.yaml` (70 lines) with per-experiment overrides:

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

YAML configs use LightningCLI's **flat format** — `model:`/`data:` keys map directly to `__init__` parameters of the registered classes. No `class_path`/`init_args` nesting needed for model and data sections (LightningCLI resolves them via the `LightningCLI(class, datamodule)` call in `train.py`). Logger/callback sections still use `class_path`/`init_args` for class resolution.

```yaml
seed_everything: 42
model:
  backbone: mobilenetv4_conv_small
  num_tasks: 4
  use_fab: true
  use_cbam: true
  use_task_conditioning: true
  freeze_backbone_stage: 2
  lambda_rank: 0.3
  lambda_cross_task: 0.1
  lr: 0.0012
  weight_decay: 0.0001
  warmup_epochs: 5
  total_epochs: 50
  annotator_stage: vla
data:
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
  enable_progress_bar: true
  default_root_dir: outputs/default
  logger:
    - class_path: lightning.pytorch.loggers.CSVLogger
      init_args:
        save_dir: outputs
        name: default
    - class_path: lightning.pytorch.loggers.WandbLogger
      init_args:
        project: uav-iqa
        name: default
        save_dir: outputs/default
  callbacks:
    - class_path: uav_iqa.callbacks.SetupRunCallback
    - class_path: uav_iqa.callbacks.CurriculumStageCallback
      init_args:
        vlm_epochs: 20
        vla_epochs: 20
        execution_epochs: 10
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: null
        filename: best_{val/srcc:.4f}
        monitor: val/srcc
        mode: max
        save_top_k: 1
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: null
        filename: epoch_{epoch:03d}
        every_n_epochs: 5
        save_top_k: -1
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: null
        filename: last
        save_last: true
```

> **Note**: Configs were converted from `class_path`/`init_args` nesting to flat format in 2026-06 (`scripts/fix_configs.py`). Before that, each section had explicit `class_path` and `init_args` keys. Logger/callback sections still require `class_path`/`init_args` because those are dynamically loaded by LightningCLI.

---

## `tests/` — Test Suite (8 files, 154 tests)

| File | Lines | Tests | Coverage |
|------|-------|-------|----------|
| `test_distortion.py` | 187 | **19** | All 6 UAV distortions + pipeline + 7 generic distortion categories + intensity range + determinism |
| `test_lightning.py` | 94 | **6** | LightningModule init, ablations, optimizer config, training step; DataModule setup with mock data |
| `test_data_synthesis.py` | 265 | **27** | DatasetFormat registry (4), AirCopBenchFormat (10), GenericImageDirFormat (4), create_pipeline (3), PipelineExtract (2), PipelineManifest (1), PipelineStepsParsing (3) |
| `test_text_metrics.py` | 79 | **10** | BLEU identical/different/completely different, brevity penalty, smoothing; ROUGE-L identical/different/partial/precision/recall; CIDEr identical/different/multi-ref; cognitive score defaults/custom weights, empty inputs, length mismatch |
| `test_vlm_config.py` | 197 | **21** | VLMConfig creation/validation (4), MODEL_REGISTRY (15 models, 6 families, required fields, chat templates, trust_remote_code), VLMScorer model name resolution (2) |
| `test_vlm_scorer.py` | 924 | **45** | BaseScorer ABC (abstract, return type, batch format), VLMScorer init/backend config, prompt building (single, batch, all tasks), scoring pipeline (offline, error handling, mock), text metrics integration, chat template formatting |
| `test_vlm_smoke.py` | 61 | **15** (1 function × 15 parametrized) | GPU smoke test: load each VLM model and score one image (requires GPU + vlm extras). Skip with `pytest -m "not smoke"`. |
| `test_batch_annotator.py` | 286 | **11** | BatchAnnotator: filtering (unannotated, task, distortion, none, combined, max), checkpoint save/load, manifest write, resume skip, resume partial, end-to-end synthetic |

### Test Dependencies

```
test_distortion.py        → uav_iqa.distortion
test_lightning.py         → uav_iqa.lightning_module, uav_iqa.data_module
test_data_synthesis.py    → uav_iqa.data_synthesis (DatasetFormat, DataSynthesisPipeline,
                             AirCopBenchFormat, GenericImageDirFormat, create_pipeline)
test_text_metrics.py      → uav_iqa.text_metrics (compute_bleu, compute_rouge_l, compute_cider, compute_cognitive_score)
test_vlm_config.py        → uav_iqa.vlm (VLMConfig, MODEL_REGISTRY, VLMScorer)
test_batch_annotator.py   → uav_iqa.batch_annotator, uav_iqa.vla_scorer
test_vlm_scorer.py        → uav_iqa.vla_scorer, uav_iqa.vlm (scorer, config)
test_vlm_smoke.py         → uav_iqa.vlm (GPU required)
```

---

## `refine-logs/` — Research Refinement Artifacts

| File | Purpose |
|------|---------|
| `FINAL_PROPOSAL.md` | Refined method thesis (score 9.0/10) |
| `EXPERIMENT_PLAN.md` | 33 experiments across 6 milestones, 4 claims |
| `EXPERIMENT_RESULTS.md` | Initial experiment results (M0 sanity checks) |
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
| `EXPERIMENTS.md` | Executable experiment documentation (Chinese) — per-experiment training commands |
| `低空无人机具身智能的图像质量评估-文献综述.md` | Literature review (Chinese) |
| `图像质量评估 for Embodied AI.md` | Topic overview (Chinese) |
| `Embodied Image Quality Assessment for Robotic Intelligence.md` | Topic overview (English) |
| `AirCopBench A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning.md` | AirCopBench summary |
| `低空无人机具身智能的图像质量评估-研究路线图.md` | Research roadmap (Chinese) |

---

## Root Entry Points and Config Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | **Unified training entry point** — vanilla LightningCLI (fit/test/predict). Usage: `python scripts/train.py fit --config configs/experiments/<name>.yaml` |
| `pyproject.toml` | Package metadata, dependencies, ruff config, pytest config |
| `.python-version` | Python version pinning (3.10+) |
| `uv.lock` | Reproducible dependency lock file |
| `config.yaml` | Optional root-level config template |
| `.gitignore` | Ignores: data/, outputs/, checkpoints/, __pycache__/, .venv/, lightning_logs/ |
| `CLAUDE.md` | Detailed agent guidance (architecture, commands, gotchas) |
| `AGENTS.md` | Quick-reference for AI coding agents |
| `README.md` | Project overview, setup, usage, and documentation index |
