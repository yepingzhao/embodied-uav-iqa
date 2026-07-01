# UAV-Embodied-IQA — Codemap Index

**Last Updated:** 2026-07-01
**Project:** Visual Quality Assessment for Aerial Embodied Intelligence
**arXiv:** 2511.11025

## Overview

This directory contains architectural codemaps for the UAV-Embodied-IQA research codebase. Each document covers a specific aspect of the system.

## Codemap Inventory

| Map | Description |
|-----|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system architecture, component relationships, data flow |
| [MODULES.md](MODULES.md) | Core library modules: public APIs, dependencies, exports |
| [FILES.md](FILES.md) | Complete file tree with per-file purpose annotations |
| [INFERENCE_FRAMEWORK.md](../INFERENCE_FRAMEWORK.md) | Multi-GPU offline inference framework architecture |

## Quick Navigation

```
src/uav_iqa/           →  Core library (see MODULES.md)
scripts/               →  Executable experiment entrypoints (see FILES.md)
configs/               →  LightningCLI YAML configurations
tests/                 →  pytest test suite
refine-logs/           →  Research refinement artifacts (proposal, plan, reviews)
docs/                  →  Literature reviews, research roadmap, inference framework spec
docs/INFERENCE_FRAMEWORK.md →  Multi-GPU offline inference architecture spec
```

## 4 Research Claims

| Claim | Statement | Primary Codemap |
|-------|-----------|-----------------|
| **C1** | UAV-specific distortions are **distinct** from generic distortions | [MODULES.md](MODULES.md) § Distortion |
| **C2** | Synthetic distortions **correlate with real** UAV-degraded quality | [MODULES.md](MODULES.md) § Data Pipeline |
| **C3** | Existing IQA methods **fail** on UAV-specific distortions | [FILES.md](FILES.md) § Benchmark |
| **C4** | Task-conditioned model **generalizes across** embodied tasks | [MODULES.md](MODULES.md) § Model |

## Key Entry Points

| Entry Point | Purpose |
|-------------|---------|
| `scripts/train.py` | **Unified training entry point** — LightningCLI with YAML configs |
| `scripts/data_synthesis.py` | Full M1 data synthesis pipeline (extract/inject/manifest/annotate) |
| `scripts/download_models.py` | Download VLM model weights from HuggingFace Hub for offline annotation scoring |
| `scripts/vlm_annotate.py` | Batch VLM annotation CLI — score manifest entries with real VLMs via VLMScorer + BatchAnnotator |
| `scripts/benchmark_iqa_methods.py` | Benchmark 15+ existing IQA methods |
| `scripts/finetune_baselines.py` | Fine-tune DL-based IQA baselines (brisque/niqe/clipiqa/maniqa/topiq_nr) on UAV data |
| `scripts/vlm_cognitive_score_vqa.py` | VQA-paradigm cognitive scoring with VLM + BLEU/ROUGE-L/CIDEr |
| `scripts/inference.py` | **Multi-GPU offline inference CLI** — batch scoring via inference framework |
| `scripts/overfit_sanity_check.py` | Model correctness overfit test |
| `scripts/validate_synth_real_correlation.py` | C2 correlation validation |
| `scripts/fix_configs.py` | Convert experiment configs from nested to flat `init_args` format |
| `scripts/visualize_distortions.py` | Visual sanity check: grid of all 36 distortions × 1 random intensity |


## Experiment Configs (`configs/experiments/`)

| Config | Purpose |
|--------|---------|
| `r013_task_cond.yaml` | Full model: task-conditioned, CBAM, FAB (baseline) |
| `r014_task_agnostic.yaml` | Task-agnostic head (shared MLP) |
| `r016_no_fab.yaml` | Ablate FrequencyAwareBranch |
| `r017_no_task_cond.yaml` | Ablate task conditioning |
| `r018_no_cbam.yaml` | Ablate CBAM |
| `r019_mobilevit_s.yaml` | MobileViT-S backbone |
| `r020_efficientvit_b0.yaml` | EfficientViT-B0 backbone |
| `r021_generic_only.yaml` | Train on generic distortions only |
| `r021b_uav_only.yaml` | Train on UAV distortions only |
| `r022_vlm_only.yaml` | VLM annotation stage only |
| `r022b_vla_only.yaml` | VLA annotation stage only |
| `r023_no_exec.yaml` | Exclude execution scores |
| `r024a_{tracking,inspection,delivery,sar}.yaml` | Single-task training |
| `r024b_leave_{tracking,inspection,delivery,sar}.yaml` | Leave-one-task-out |
| `r024c_multitask.yaml` | Full multi-task training |

## External Data Dependencies

| Resource | Source | Used By |
|----------|--------|---------|
| AirCopBench dataset | arXiv 2511.11025 | `data_synthesis.py` (extract/manifest/annotate steps) |
| Real-ESRGAN (optional) | GitHub (xinntao/Real-ESRGAN) | `LowResSuperResolution` distortion |
| pyiqa (optional) | PyPI | `benchmark_iqa_methods.py`, `finetune_baselines.py` |
| openVLA (manual install) | GitHub | VLA annotation stage |
| VLM libs (optional) | vllm, transformers, accelerate | VLM annotation stage |

## Key Public API (`__init__.py`)

The package exports **45 symbols** (from `uav_iqa/__init__.py`). The `inference/` subpackage exports **25 additional symbols** (from `uav_iqa/inference/__init__.py`).

### Top-Level (uav_iqa)

| Category | Count | Symbols |
|----------|-------|---------|
| Distortion models | 7 | `UAVDistortionPipeline`, 6 UAV-specific distortion classes |
| Model | 1 | `UAVIQANet` |
| Dataset | 2 | `UAVIQADataset`, `validate_manifest` |
| Metrics | 4 | `compute_srcc`, `compute_plcc`, `evaluate_iqa`, `per_distortion_category_metrics` |
| Lightning wrappers | 2 | `UAVIQALightningModule`, `UAVIQADataModule` |
| Losses | 2 | `ListMLELoss`, `CrossTaskRegularization` |
| Annotation utilities | 11 | `parse_distortion_key`, `build_ref_score_lookup`, `build_vqa_split_lookup`, `group_by_scene_frame`, `extract_subtask_type`, `extract_subtask_id`, `extract_uav_id_from_question_id`, `normalize_subtask_type`, `build_sample_id`, `get_dataset_name`, `seed_for_distortion` |
| Constants | 5 | `SUBTASK_NAMES`, `SUBTASK_TO_ID`, `SUBTASK_NAME_LIST`, `SUBTASK_NAME_TO_ID`, `NUM_SUBTASKS` |
| Utility functions | 5 | `setup_logging`, `load_task_map`, `load_flat_samples`, `split_samples`, `find_images` |
| Data synthesis | 3 | `DatasetFormat`, `DataSynthesisPipeline`, `create_pipeline` |
| VLM/VLA scoring | 3 | `BaseScorer`, `VLMScorer`, `BatchAnnotator` |

### Inference Subpackage (uav_iqa.inference)

Exports **25 symbols**: BaseExecutor, BaseQueue, BaseStorage, CheckpointStore, Collector, DummyExecutor, FileRecord, InferenceConfig, InferenceEngine, JsonStorage, Metrics, MpQueue, Result, Scheduler, Task, TaskRepository, TaskStatus, VLMExecutor, Validator, Writer, make_result_queue, make_task_queue, worker_main.
