# File Tree Codemap

<!-- Generated: 2026-07-21 | Files scanned: ~90 source files | Token estimate: ~1100 -->

## Root Layout

```
embodied-uav-iqa/
├── src/uav_iqa/           # Core library (34 .py files total)
│   ├── 16 top-level modules
│   ├── 5 vlm/ subpackage modules
│   └── 15 inference/ modules
├── scripts/               # 13 executable experiment scripts
├── configs/               # YAML configuration (LightningCLI) + 21 experiment configs
├── tests/                 # pytest test suite (19 files)
├── figures/               # Paper figures + generation scripts
├── refine-logs/           # 17 research refinement artifacts
├── docs/                  # Literature reviews, research roadmap, codemaps
├── PAPER_PLAN.md          # Paper outline and writing plan
├── MANIFEST.md            # Output manifest for paper assets
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

## `src/uav_iqa/` — Core Library (34 .py files)

> **Note (2026-07):** `data_synthesis.py` was renamed to `distortion_synthesis.py` and its VLM annotate/aggregate steps were removed (now handled by `batch_annotator.py` + `update_vlm_scores.py`). `vlm_vla_scorer.py` compat shim was previously removed; contents are in `vlm/`, `vla_scorer.py`, `text_metrics.py`, and `batch_annotator.py`.

### Top-Level Modules (16 files)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 99 | Public API: exports **44 symbols** |
| `distortion.py` | **735** | 36 distortion models (6 UAV + 30 generic) + UAVDistortionPipeline + `UAV_DISTORTION_NAMES`. Supports JPEG format. Parallel batch injection via `ProcessPoolExecutor`. |
| `model.py` | **438** | UAVIQANet: backbone → FPN → CBAM → FAB → task heads, `forward_features()` feature sharing. Dynamic stage probing + regex-based freeze for backbone-agnostic compatibility. |
| `distortion_synthesis.py` | 466 | Dataset-agnostic distortion injection: DatasetFormat (ABC + registry), AirCopBenchFormat, GenericImageDirFormat, DistortionSynthesisPipeline (extract → inject), create_pipeline. *(Renamed from data_synthesis.py; VLM annotate/aggregate removed.)* |
| `annotations.py` | **402** | AirCopBench annotation parsing: `parse_distortion_key`, `parse_quality_score`, `parse_usability`, `build_ref_score_lookup`, `assign_task_label`. Constants: `SUBTASK_TO_ID`, `SUBTASK_NAME_LIST`, `SUBTASK_NAME_TO_ID`, `NUM_SUBTASKS` + `seed_for_distortion`. |
| `text_metrics.py` | **326** | BLEU, ROUGE-L, CIDEr text similarity (pure Python, no nltk). `compute_cognitive_score()`: weighted combo (1:1:0.1) for VLM comparison scoring. |
| `lightning_module.py` | 282 | LightningModule: UAVIQANet + MSE/ListMLE/cross-task, feature sharing, per-distortion ranking, DDP gathering. |
| `batch_annotator.py` | 228 | Batch annotation engine: `BatchAnnotator` with manifest I/O, filtering, checkpoint/resume, multi-split orchestration. |
| `callbacks.py` | 206 | CurriculumStageCallback (deprecated no-op), SetupRunCallback + **MetricsHistoryCallback** + **ResultsSavingCallback** (DDP-safe). |
| `utils.py` | 179 | `count_parameters()`, `find_images()`, `load_image_tensor()`, `load_flat_samples()`, `split_samples()`, `load_task_map()`, `setup_logging()`. |
| `dataset.py` | 169 | UAVIQADataset: flat JSON loader with multi-UAV image support + `validate_manifest()`. |
| `metrics.py` | 145 | SRCC, PLCC, RMSE, Kendall τ + per-task / per-distortion / **per-category** metrics. |
| `data_module.py` | 124 | LightningDataModule: manifest filtering (subtask/distortion), multi-UAV support (max_uavs), DDP-compatible. |
| `losses.py` | 30 | ListMLELoss + CrossTaskRegularization. |
| `vla_scorer.py` | 60 | Base scoring: `BaseScorer` ABC + `extract_ref_id()` + task type constants. |
| `vlm/` | ~1678 | VLM scoring subpackage (5 files: __init__, config, backends, scorer, vqa_index). |

### VLM Subpackage (`vlm/` — 5 files)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 7 | Subpackage init, re-exports |
| `config.py` | 175 | `VLMConfig` + `MODEL_REGISTRY` (15 models, 6 families, chat templates, trust_remote_code) |
| `backends.py` | ~100 | ChatTemplate processor + vLLM batch inference helpers + image placeholder embedding |
| `scorer.py` | 1419+ | `VLMScorer`: `score_vqa_with_gt()` (指标1+2), `score_multi_image()` (multi-UAV), `score_batch_comparison()`, `score_batch_multi_image()`. Supports vLLM (continuous batching) + transformers backends. |
| `vqa_index.py` | 77 | `VQAIndex` for question-answer indexing and retrieval |

### Inference Subpackage (`inference/` — 15 modules)

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: exports **25 symbols** |
| `types.py` | Core data types: `Task`, `Result`, `FileRecord`, `TaskStatus` (dataclasses) |
| `config.py` | `InferenceConfig`: model, paths, chunk/batch sizes, GPU devices, file glob filter |
| `queue.py` | `BaseQueue` ABC + `MpQueue` (multiprocessing.Queue), factory functions |
| `storage.py` | `BaseStorage` ABC + `JsonStorage` (file scan, chunk load, output write) |
| `checkpoint.py` | `CheckpointStore` — SQLite-backed task-level checkpoint/resume |
| `repository.py` | `TaskRepository` — create, query, and update task states |
| `executor.py` | `BaseExecutor` ABC + `DummyExecutor` + `VLMExecutor` (batch inference with vLLM) |
| `worker.py` | `worker_main` — multiprocessing worker: load data → execute → return result |
| `scheduler.py` | `Scheduler` — scan input files, generate tasks, enqueue pending/resume tasks |
| `collector.py` | `Collector` — track completed chunks, detect file completion, notify Writer |
| `writer.py` | `Writer` — merge chunks, restore order, atomic file write |
| `validator.py` | `Validator` — verify file count, record count, chunk completeness, order |
| `metrics.py` | `Metrics` — runtime stats: queue size, samples/sec, latency, ETA |
| `engine.py` | `InferenceEngine` — wire all modules, manage lifecycle, launch workers |

---

## `scripts/` — Experiment Entrypoints (13 scripts)

| File | Lines | Purpose | Stage |
|------|-------|---------|-------|
| `distortion_synthesis.py` | 81 | Distortion injection CLI (extract/inject). Uses `DistortionSynthesisPipeline`. *(Renamed from data_synthesis.py.)* | M1 |
| `download_models.py` | **303** | Download VLM model weights from HuggingFace Hub. Supports all 15 registered models, auth token, validation loading. | M1.5 |
| `vlm_cognitive_score_vqa.py` | **283** | VQA-paradigm cognitive scoring: distorted vs. clean image pairs using VLM + BLEU/ROUGE-L/CIDEr. Two modes: GT-normalized (指标1) and Embodied-IQA direct (指标2). | M1.5 |
| `update_vlm_scores.py` | 107 | Compute per-model cognitive_score from VLM annotations, update target files with multi-model average. | M1.5 |
| `benchmark_iqa_methods.py` | 591 | Benchmark 15+ IQA methods (pyiqa) on test set. | M2 |
| `finetune_baselines.py` | **399** | Fine-tune DL-based IQA baselines (brisque/niqe/clipiqa/maniqa/topiq_nr) on UAV data. | M2 |
| `overfit_sanity_check.py` | 105 | Overfit correctness test: train on 100 images, verify loss → 0. | Validation |
| `visualize_distortions.py` | 108 | Visual sanity check: grid of all 36 distortions × random intensity. | Validation |
| `validate_synth_real_correlation.py` | 188 | C2 correlation validation: synthetic vs real scores. | Validation |
| `fix_configs.py` | 130 | Config migration: nested → flat format, ensure SetupRunCallback/CSVLogger. | Utility |
| `train.py` | 35 | **Unified training entry point** — LightningCLI (fit/test/predict). | M3 |
| `inference.py` | 125 | **Multi-GPU offline inference CLI** — batch scoring via inference framework. | M3+ |
| `reset_empty_tasks.py` | 111 | Reset hung or zero-output tasks in inference checkpoint DB. | Utility |

### Script Execution Order

```
1. distortion_synthesis.py extract/inject  # Extract VQA JSONs, inject distortions
2. download_models.py --all                # Download VLM weights (optional)
3. BatchAnnotator (via API)                # VLM multi-image scoring
4. update_vlm_scores.py                    # Aggregate per-model scores → cognitive_score
5. scripts/train.py                        # Train UAVIQANet via LightningCLI
6. inference.py                            # Multi-GPU offline inference (optional)
7. benchmark_iqa_methods.py                # Zero-shot IQA benchmarking (optional)
8. finetune_baselines.py                   # Fine-tune DL baselines (optional)

Utility:
  - fix_configs.py                         # Config migration
  - visualize_distortions.py               # Visual sanity check
  - overfit_sanity_check.py                # Model correctness verification
  - validate_synth_real_correlation.py     # C2 correlation validation
  - reset_empty_tasks.py                   # Reset hung inference tasks
```

---

## `figures/` — Paper Figures

| File | Purpose |
|------|---------|
| `gen_all_figures.py` (515 lines) | Reproducible generation of all 6 paper figures |
| `paper_plot_style.py` (165 lines) | Shared matplotlib style, color palettes, helper functions |
| `latex_includes.tex` (83 lines) | LaTeX snippet for figure inclusion in paper |
| `fig1_distribution.pdf` | Score distribution across tasks and datasets |
| `fig2_uav_vs_generic.pdf` | UAV vs. generic distortion score comparison |
| `fig3_three_dimensions.pdf` | Three cognitive dimensions visualization |
| `fig4_jnd_sensitivity.pdf` | JND sensitivity analysis across intensities |
| `fig5_intensity_curves.pdf` | Distortion intensity vs. quality degradation curves |
| `fig6_source_comparison.pdf` | Annotation source comparison (VLM vs. VLA) |

---

## `configs/` — Configuration

| File | Purpose |
|------|---------|
| `default.yaml` | Reference template: model arch, data params, trainer, callbacks |
| `experiments/` | **21 self-contained experiment configs** (r013–r024c) |

### Experiment Configs (21 total)

| Config Range | Count | Experiment | Key Difference |
|-------------|-------|------------|----------------|
| r013 | 1 | Subtask-conditioned (baseline) | use_task_conditioning=true |
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
| r023 | 1 | No execution score | annotation_src≈no_exec |
| r024a | 4 | Single-task | subtask_filter={tracking,inspection,delivery,sar} |
| r024b | 4 | Leave-one-task-out | leave_out={tracking,inspection,delivery,sar} |
| r024c | 1 | Multi-task (all 14) | no filter |

---

## `tests/` — Test Suite (19 files)

| File | Coverage |
|------|----------|
| `test_distortion.py` | 6 UAV + 7 generic distortion categories + pipeline + intensity + determinism |
| `test_distortion_synthesis.py` | DatasetFormat registry, AirCopBenchFormat, GenericImageDirFormat, create_pipeline, extract/inject pipeline steps |
| `test_lightning.py` | LightningModule init, ablations, optimizer, training step; DataModule with mock data |
| `test_text_metrics.py` | BLEU, ROUGE-L, CIDEr edge cases; cognitive_score computation |
| `test_vlm_config.py` | VLMConfig, MODEL_REGISTRY (15 models, 6 families), VLMScorer name resolution |
| `test_vlm_scorer.py` | BaseScorer ABC, VLMScorer init/backend, prompt building, scoring pipeline, text metrics integration |
| `test_vlm_smoke.py` | GPU smoke test: load VLM, score one image (GPU required, `-m "not smoke"`) |
| `test_vlm_backends.py` | Chat template processing, batch inference formatting |
| `test_vlm_load_and_failfast.py` | Model loading config tests, fail-fast error detection |
| `test_batch_annotator.py` | BatchAnnotator: filtering, checkpoint save/load, manifest write, resume, E2E |
| `test_annotations.py` | Annotation parsing: distortion keys, score extraction, subtask mapping |
| `test_dataset.py` | UAVIQADataset collation, multi-UAV image handling, task ID resolution |
| `test_model_text.py` | Model text utilities and export helpers |
| `test_inference_phase1.py` | types, config, queue, storage, checkpoint, repository |
| `test_inference_phase2.py` | executor, worker — single task execution |
| `test_inference_phase3.py` | scheduler, collector, writer — dynamic multi-GPU output |
| `test_inference_phase4.py` | engine, validator, metrics — lifecycle management + resume |
| `test_inference_phase5.py` | CLI integration — E2E via scripts/inference.py |

---

## `docs/` — Documentation

| File | Purpose |
|------|---------|
| `CODEMAPS/` | (This directory) — architectural maps |
| `INFERENCE_FRAMEWORK.md` | Multi-GPU offline inference framework architecture specification |
| `DELETION_LOG.md` | Log of deleted/refactored files |
| `EXPERIMENTS.md` | Executable experiment documentation — per-experiment training commands |
| `低空无人机具身智能的图像质量评估-文献综述.md` | Literature review (Chinese) |
| `图像质量评估 for Embodied AI.md` | Topic overview (Chinese) |
| `Embodied Image Quality Assessment for Robotic Intelligence.md` | Topic overview (English) |
| `AirCopBench A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning.md` | AirCopBench summary |
| `低空无人机具身智能的图像质量评估-研究路线图.md` | Research roadmap (Chinese) |

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
