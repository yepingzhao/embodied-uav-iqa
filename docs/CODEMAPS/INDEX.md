# UAV-Embodied-IQA — Codemap Index

**Last Updated:** 2026-06-20
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

## Quick Navigation

```
src/uav_iqa/           →  Core library (see MODULES.md)
scripts/               →  Executable experiment entrypoints (see FILES.md)
configs/               →  LightningCLI YAML configurations
tests/                 →  pytest test suite
refine-logs/           →  Research refinement artifacts (proposal, plan, reviews)
docs/                  →  Literature reviews and research roadmap
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
| `main.py` | **Unified training entry point** — LightningCLI with YAML configs |
| `scripts/synthesize_data.py` | Full M1 data synthesis pipeline (extract/inject/manifest/annotate) |
| `scripts/run_m2_benchmark.py` | Benchmark 15+ existing IQA methods |
| `scripts/run_overfit.py` | Model correctness overfit test |
| `scripts/run_c2_correlation.py` | C2 correlation validation |

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
| AirCopBench dataset | arXiv 2511.11025 | `synthesize_data.py` (extract step) |
| Real-ESRGAN (optional) | GitHub (xinntao/Real-ESRGAN) | `LowResSuperResolution` distortion |
| pyiqa (optional) | PyPI | `run_m2_benchmark.py` |
| openVLA (manual install) | GitHub | VLA annotation stage |
| VLM libs (optional) | vllm, transformers, accelerate | VLM annotation stage |
