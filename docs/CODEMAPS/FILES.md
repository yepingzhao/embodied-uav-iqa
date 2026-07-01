# File Tree Codemap

**Last Updated:** 2026-07-01

## Root Layout

```
embodied-uav-iqa/
├── src/uav_iqa/           # Core library (31 .py files, ~8K LOC total)
│   ├── 16 top-level modules
│   └── 15 inference/ modules
├── scripts/               # 12 executable experiment scripts
├── configs/               # YAML configuration (LightningCLI) + 21 experiment configs
├── tests/                 # pytest test suite (13+ files)
├── refine-logs/           # 17 research refinement artifacts
├── docs/                  # Literature reviews, research roadmap, codemaps, inference spec
├── pyproject.toml         # Project metadata, deps, tool config
├── CLAUDE.md              # Agent guidance (architecture, commands)
├── AGENTS.md              # Quick-reference for development agents
├── README.md              # Project overview & usage
├── .gitignore             # Git ignore rules
├── .python-version        # Python version pinning
├── uv.lock                # uv dependency lockfile
├── checkpoints/           # Model checkpoints (gitignored)
├── data/                  # Datasets: raw/ and processed/ (gitignored)
├── outputs/               # Experiment outputs (gitignored)
└── .venv/                 # Virtual environment (gitignored)
```

---

## `src/uav_iqa/` — Core Library (31 .py files, ~8K LOC)

*(Note: `vlm_vla_scorer.py` backward-compat shim was removed; its contents are now split into `vlm/` subpackage, `vla_scorer.py`, `text_metrics.py`, and `batch_annotator.py`.)*

### Top-Level Modules (16 files)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 100 | Public API: exports **45 symbols** |
| `distortion.py` | **735** | 36 distortion models (6 UAV + 30 generic) + UAVDistortionPipeline + `UAV_DISTORTION_NAMES` constant. Supports JPEG output format. Parallel batch injection via `ProcessPoolExecutor`. |
| `model.py` | **438** | UAVIQANet: backbone → FPN → CBAM → FAB → task heads, `forward_features()` feature sharing. **Dynamic stage probing** + **regex-based freeze** for backbone-agnostic compatibility |
| `dataset.py` | 169 | UAVIQADataset: flat JSON loader with multi-UAV image support + `validate_manifest()` |
| `text_metrics.py` | **326** | BLEU, ROUGE-L, CIDEr text similarity metrics for VLM comparison-based scoring. Pure Python (no nltk). |
| `vla_scorer.py` | 60 | Base scoring: `BaseScorer` ABC + `extract_ref_id()` + task type constants |
| `vlm/` | **1678** | VLM scoring subpackage: `VLMConfig` + `MODEL_REGISTRY` (config.py, 175 lines), `VLMScorer` with vLLM/transformers backends + `VQAIndex` (scorer.py 1419 + vqa_index.py 77 + __init__.py 7) |
| `batch_annotator.py` | 228 | Batch annotation engine: `BatchAnnotator` with manifest I/O, filtering, checkpoint/resume, multi-split orchestration |
| `losses.py` | 30 | ListMLELoss + CrossTaskRegularization |
| `metrics.py` | 145 | SRCC, PLCC, RMSE, Kendall τ + per-task / per-distortion / **per-category** metrics |
| `annotations.py` | **402** | AirCopBench annotation parsing: `parse_distortion_key`, `parse_quality_score`, `parse_usability`, `build_ref_score_lookup`, `assign_task_label`. Also: `SUBTASK_TO_ID`, `SUBTASK_NAME_LIST`, `SUBTASK_NAME_TO_ID`, `NUM_SUBTASKS` constants + `seed_for_distortion`. |
| `data_synthesis.py` | 769 | Dataset-agnostic pipeline: DatasetFormat (ABC + registry), AirCopBenchFormat, GenericImageDirFormat, DataSynthesisPipeline, create_pipeline. Supports `fmt` (png/jpeg). |
| `lightning_module.py` | 282 | LightningModule: UAVIQANet + MSE/ListMLE/cross-task, feature sharing, per-distortion ranking, DDP gathering |
| `data_module.py` | 124 | LightningDataModule: manifest filtering (subtask/distortion), multi-UAV support (max_uavs), DDP-compatible |
| `callbacks.py` | 206 | CurriculumStageCallback (deprecated), SetupRunCallback + **MetricsHistoryCallback** + **ResultsSavingCallback** (DDP-safe) |
| `inference/` | — | Multi-GPU offline inference framework (15 modules; see below) |
| `utils.py` | 179 | `count_parameters()`, `find_images()`, `load_image_tensor()`, `load_flat_samples()`, `split_samples()`, `load_task_map()`, `setup_logging()` |

### Inference Subpackage (`inference/` — 15 modules)

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: exports **25 symbols** (BaseExecutor, BaseQueue, BaseStorage, CheckpointStore, Collector, DummyExecutor, FileRecord, InferenceConfig, InferenceEngine, JsonStorage, Metrics, MpQueue, Result, Scheduler, Task, TaskRepository, TaskStatus, VLMExecutor, Validator, Writer, make_result_queue, make_task_queue, worker_main) |
| `types.py` | Core data types: `Task`, `Result`, `FileRecord`, `TaskStatus` (dataclasses) |
| `config.py` | `InferenceConfig`: model, paths, chunk/batch sizes, GPU devices |
| `queue.py` | `BaseQueue` ABC + `MpQueue` (multiprocessing.Queue), factory functions |
| `storage.py` | `BaseStorage` ABC + `JsonStorage` (file scan, chunk load, output write) |
| `checkpoint.py` | `CheckpointStore` — SQLite-backed task-level checkpoint/resume |
| `repository.py` | `TaskRepository` — create, query, and update task states |
| `executor.py` | `BaseExecutor` ABC + `DummyExecutor` (synthetic) + `VLMExecutor` (real inference) |
| `worker.py` | `worker_main` — multiprocessing worker: load data → execute → return result |
| `scheduler.py` | `Scheduler` — scan input files, generate tasks, enqueue pending/resume tasks |
| `collector.py` | `Collector` — track completed chunks, detect file completion, notify Writer |
| `writer.py` | `Writer` — merge chunks, restore order, atomic file write |
| `validator.py` | `Validator` — verify file count, record count, chunk completeness, order |
| `metrics.py` | `Metrics` — runtime stats: queue size, samples/sec, latency, ETA |
| `engine.py` | `InferenceEngine` — wire all modules, manage lifecycle, launch workers |

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
  → torch, numpy, annotations.py (SUBTASK_NAME_TO_ID)

vla_scorer.py
  → hashlib, abc, pathlib, annotations.py (SUBTASK_NAMES)

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

inference/__init__.py
  → .checkpoint, .collector, .config, .engine, .executor,
    .metrics, .queue, .repository, .scheduler, .storage,
    .types, .validator, .worker, .writer
  → all 14 inference submodules

inference/types.py
  → dataclasses, pathlib (stdlib)

inference/config.py
  → dataclasses, pathlib, typing (stdlib)

inference/queue.py
  → abc, multiprocessing (stdlib)

inference/storage.py
  → abc, json, pathlib, typing (stdlib)

inference/checkpoint.py
  → sqlite3, json, logging, pathlib, time, typing (stdlib)

inference/repository.py
  → checkpoint.py (CheckpointStore), types.py (Task, TaskStatus)
  → pathlib, typing (stdlib)

inference/executor.py
  → abc, typing (stdlib)
  → optional: torch, transformers, vllm (VLMExecutor)

inference/worker.py
  → executor.py, storage.py, queue.py, types.py
  → logging, os, traceback (stdlib)

inference/scheduler.py
  → repository.py, queue.py, types.py, storage.py
  → pathlib, logging (stdlib)

inference/collector.py
  → queue.py (ResultQueue), types.py (Result)
  → pathlib, logging, collections (stdlib)

inference/writer.py
  → types.py (Result)
  → json, pathlib, logging (stdlib)

inference/validator.py
  → storage.py, types.py
  → pathlib, logging (stdlib)

inference/metrics.py
  → time, logging (stdlib)

inference/engine.py
  → all inference modules (wiring)
  → multiprocessing, logging, typing (stdlib)

losses.py
  → torch

metrics.py
  → numpy, scipy.stats, annotations.py (SUBTASK_NAME_LIST as TASK_NAMES)

annotations.py
  → json, hashlib, re, logging, pathlib (stdlib only, no internal deps)

data_synthesis.py
  → numpy, json, shutil, logging, pathlib, abc, typing
  → uav_iqa.distortion (UAVDistortionPipeline)
  → uav_iqa.annotations (build_ref_score_lookup, assign_task_label, ...)
  → uav_iqa.utils (find_images, split_samples)

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
   (independent, no internal deps)
```

---

## `scripts/` — Experiment Entrypoints

| File | Lines | Purpose | Pipeline Stage |
|------|-------|---------|----------------|
| `data_synthesis.py` | 252 | Unified data synthesis CLI (extract/inject/annotate/aggregate/all). Uses `DataSynthesisPipeline` from `src/uav_iqa/data_synthesis.py`. Supports `--format` (png/jpeg). | M1 |
| `download_models.py` | **303** | Download VLM model weights from HuggingFace Hub for offline use. Supports all 15 registered models, auth token, validation loading. | M1.5 |
| `vlm_annotate.py` | 166 | Batch VLM annotation CLI — score manifest entries with real VLM models via VLMScorer + BatchAnnotator. Supports single-model, multi-model ensemble, all 15 models, distortion/task filtering, checkpoint/resume | M1.5 |
| `vlm_cognitive_score_vqa.py` | **283** | VQA-paradigm cognitive scoring: scores distorted vs. clean image pairs using VLM + BLEU/ROUGE-L/CIDEr. Two scoring modes: GT-normalized (指标1) and Embodied-IQA-style direct comparison (指标2). | M1.5 |
| `benchmark_iqa_methods.py` | 591 | Benchmark 15+ IQA methods (pyiqa) on test set | M2 |
| `finetune_baselines.py` | **399** | Fine-tune DL-based IQA baselines (brisque/niqe/clipiqa/maniqa/topiq_nr) on UAV training data. Uses standalone LightningModule + LightningDataModule, no LightningCLI | M2 |
| `overfit_sanity_check.py` | 105 | Overfit correctness test: train on 100 images, verify loss → 0 | Validation |
| `visualize_distortions.py` | 108 | Visual sanity check: grid of all 36 distortions × 1 random intensity | Validation |
| `validate_synth_real_correlation.py` | 188 | C2 correlation validation: synthetic vs real scores | Validation |
| `fix_configs.py` | 130 | Convert experiment YAML configs from nested `class_path+init_args` to flat format; ensures SetupRunCallback and CSVLogger `save_dir` are present | Utility |
| `train.py` | 35 | **Unified training entry point** — vanilla LightningCLI (fit/test/predict). Usage: `python scripts/train.py fit --config configs/experiments/<name>.yaml` | M3 |
| `inference.py` | 125 | **Multi-GPU offline inference CLI** — entry point for the inference framework. Usage: `python scripts/inference.py --input-dir data/processed --output-dir data/output --checkpoint-path data/checkpoint.db --num-gpus 4 --model Qwen2.5-VL --backend vllm` | M3+ |

### Script Execution Order (Standard Pipeline)

```
1. data_synthesis.py all            # Extract → inject → annotate → aggregate (full M1 pipeline)
2. download_models.py --all         # Download VLM model weights (optional, M1.5)
3. vlm_annotate.py                  # Batch VLM annotation (optional, M1.5)
4. vlm_cognitive_score_vqa.py       # VQA-paradigm cognitive scoring (optional, M1.5)
5. scripts/train.py                 # Train UAVIQANet via LightningCLI + experiment config
6. inference.py                     # Multi-GPU offline inference (optional, M3+)
7. benchmark_iqa_methods.py         # Benchmark zero-shot IQA methods (optional, after training)
8. finetune_baselines.py            # Fine-tune DL baselines on UAV data (optional, M2-R009a)

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

All 21 configs follow the same structure as `default.yaml` (68 lines) with per-experiment overrides:

| Config Range | Count | Experiment | Key Difference from Baseline |
|-------------|-------|------------|------------------------------|
| r013 | 1 | Subtask-conditioned (full baseline) | use_task_conditioning=true |
| r014 | 1 | Task-agnostic | use_task_conditioning=false |
| r016 | 1 | No FAB | use_fab=false |
| r017 | 1 | No task conditioning | use_task_conditioning=false |
| r018 | 1 | No CBAM | use_cbam=false |
| r019 | 1 | MobileViT-S backbone | backbone=mobilevit_s |
| r020 | 1 | EfficientViT-B0 backbone | backbone=efficientvit_b0 |
| r021 | 1 | Generic distortions only | distortion_filter=generic |
| r021b | 1 | UAV distortions only | distortion_filter=uav_only |
| r022 | 1 | VLM-only score | annotation_src=vlm |
| r022b | 1 | VLA-only score | annotation_src=vla |
| r023 | 1 | No execution score | total_epochs=50 (flat config) |
| r024a | 4 | Single-dimension | subtask_filter=scene_understanding/object_understanding/planning/collaboration |
| r024b | 4 | Leave-one-dimension-out | leave_out_subtask=scene_understanding/object_understanding/planning/collaboration |
| r024c | 1 | Multi-task (all 14 subtasks) | task=null (no filter) |

### Config Structure (default.yaml)

YAML configs use LightningCLI's **flat format** for `model:`/`data:` sections (keys map directly to `__init__` parameters). Logger and callback sections use `class_path`/`init_args` for dynamic class resolution.

```yaml
seed_everything: 42
model:
  backbone: mobilenetv4_conv_small
  num_tasks: 14
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
data:
  data_root: data/processed
  batch_size: 256
  num_workers: 16
  image_size: 256
  subtask_filter: null
  distortion_filter: null
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
        filename: epoch{epoch:03d}
        every_n_epochs: 5
        save_top_k: -1
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: null
        filename: best_epoch{epoch:03d}_srcc{val_srcc:.4f}
        monitor: val/srcc
        mode: max
        save_top_k: 1
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: null
        filename: last
        save_last: true
```

> **Note**: `model:`/`data:` use flat format (no `class_path`/`init_args` nesting) because `LightningCLI(class, datamodule)` in `train.py` resolves them directly. Logger/callback sections still require `class_path`/`init_args`. Deprecated `CurriculumStageCallback` is present in all configs but is a no-op (the 3-stage curriculum system was removed in favor of single cognitive_score).

---

## `tests/` — Test Suite (13+ files)

| File | Coverage |
|------|----------|
| `test_distortion.py` | All 6 UAV distortions + pipeline + 7 generic distortion categories + intensity range + determinism |
| `test_lightning.py` | LightningModule init, ablations, optimizer config, training step; DataModule setup with mock data |
| `test_data_synthesis.py` | DatasetFormat registry, AirCopBenchFormat, GenericImageDirFormat, create_pipeline, PipelineExtract, PipelineManifest, PipelineStepsParsing |
| `test_text_metrics.py` | BLEU identical/different/completely different, brevity penalty, smoothing; ROUGE-L identical/different/partial/precision/recall; CIDEr identical/different/multi-ref; cognitive score defaults/custom weights, empty inputs, length mismatch |
| `test_vlm_config.py` | VLMConfig creation/validation (4), MODEL_REGISTRY (15 models, 6 families, required fields, chat templates, trust_remote_code), VLMScorer model name resolution |
| `test_vlm_scorer.py` | BaseScorer ABC (abstract, return type, batch format), VLMScorer init/backend config, prompt building (single, batch, all tasks), scoring pipeline (offline, error handling, mock), text metrics integration, chat template formatting |
| `test_vlm_smoke.py` | GPU smoke test: load each VLM model and score one image (requires GPU + vlm extras). Skip with `pytest -m "not smoke"`. |
| `test_batch_annotator.py` | BatchAnnotator: filtering (unannotated, task, distortion, none, combined, max), checkpoint save/load, manifest write, resume skip, resume partial, end-to-end synthetic |
| `test_annotations.py` | Annotation parsing utilities: distortion keys, score extraction, subtask mapping |
| `test_dataset.py` | UAVIQADataset collation, multi-UAV image handling, task ID resolution |
| `test_model_text.py` | Model text-related utilities and export helpers |
| `test_inference_phase1.py` | Inference framework Phase 1: types, config, queue, storage, checkpoint, repository |
| `test_inference_phase2.py` | Phase 2: executor, worker — single task execution |
| `test_inference_phase3.py` | Phase 3: scheduler, collector, writer — dynamic multi-GPU output |
| `test_inference_phase4.py` | Phase 4: engine, validator, metrics — lifecycle management + resume |
| `test_inference_phase5.py` | Phase 5: CLI integration — end-to-end pipeline via scripts/inference.py |

### Test Dependencies

```
test_distortion.py            → uav_iqa.distortion
test_lightning.py             → uav_iqa.lightning_module, uav_iqa.data_module
test_data_synthesis.py        → uav_iqa.data_synthesis
test_text_metrics.py          → uav_iqa.text_metrics
test_vlm_config.py            → uav_iqa.vlm (VLMConfig, MODEL_REGISTRY, VLMScorer)
test_batch_annotator.py       → uav_iqa.batch_annotator, uav_iqa.vla_scorer
test_vlm_scorer.py            → uav_iqa.vla_scorer, uav_iqa.vlm
test_vlm_smoke.py             → uav_iqa.vlm (GPU required)
test_annotations.py           → uav_iqa.annotations
test_dataset.py               → uav_iqa.dataset
test_model_text.py            → uav_iqa.model, uav_iqa.text_metrics
test_inference_phase1.py      → uav_iqa.inference (types, config, queue, storage, checkpoint, repository)
test_inference_phase2.py      → uav_iqa.inference (executor, worker)
test_inference_phase3.py      → uav_iqa.inference (scheduler, collector, writer)
test_inference_phase4.py      → uav_iqa.inference (engine, validator, metrics)
test_inference_phase5.py      → scripts.inference, uav_iqa.inference (end-to-end)
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
| `INFERENCE_FRAMEWORK.md` | Multi-GPU offline inference framework architecture specification |
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
