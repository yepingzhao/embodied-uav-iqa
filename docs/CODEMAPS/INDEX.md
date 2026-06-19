# UAV-Embodied-IQA — Codemap Index

**Last Updated:** 2026-06-19
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
| `scripts/run_m3_train.py` | Training (multi-seed, ablations, cross-task) |
| `scripts/run_m2_benchmark.py` | Benchmark 15+ existing IQA methods |
| `scripts/run_m1_aircopbench.py` | Full M1 data pipeline |
| `scripts/run_overfit.py` | Model correctness overfit test |

## External Data Dependencies

| Resource | Source | Used By |
|----------|--------|---------|
| AirCopBench dataset | arXiv 2511.11025 | `extract_aircopbench_refs.py` |
| Real-ESRGAN (optional) | GitHub (xinntao/Real-ESRGAN) | `LowResSuperResolution` distortion |
| pyiqa (optional) | PyPI | `run_m2_benchmark.py` |
| openVLA (manual install) | GitHub | VLA annotation stage |
| VLM libs (optional) | vllm, transformers, accelerate | VLM annotation stage |
