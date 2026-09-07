# UAV-Embodied-IQA Codemap Index

This directory documents the current layered package structure.

| Map | Use it for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Data flow, package ownership, and dependency direction |
| [MODULES.md](MODULES.md) | Public APIs, internal modules, and removed-path boundaries |
| [FILES.md](FILES.md) | Repository tree, entrypoints, and test ownership |
| [INFERENCE_FRAMEWORK.md](INFERENCE_FRAMEWORK.md) | Multi-GPU offline inference details |

## Quick navigation

```text
domain       → shared task taxonomy and AirCopBench definitions
models       → UAVIQANet and neural components
distortions  → visual degradation pipeline
data         → processed samples and data loading
training     → Lightning optimization
evaluation   → numeric IQA + text similarity metrics
vlm          → VLM scoring and batch annotation
inference    → multi-GPU offline execution
```

The primary commands are `scripts/distortion_synthesis.py`, `scripts/train.py`,
`scripts/benchmark_iqa_methods.py`, and `scripts/inference.py`.  See
`AGENTS.md` for development commands and `configs/experiments/` for experiment
configurations.
