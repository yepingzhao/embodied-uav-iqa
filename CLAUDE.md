# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

UAV-Embodied-IQA: visual quality assessment for aerial embodied intelligence (research codebase). It provides:
- **6 UAV-specific distortion models** (propeller vibration, atmospheric scattering, 6DoF viewpoint blur, packet-loss blocks, low-res+SR artifacts, propeller shadow)
- **18 generic distortion types** via Albumentations
- **UAV-IQANet**: a lightweight NR-IQA model (~5.4M params, MobileNetV4-S backbone + PANet FPN + frequency-aware branch + task-conditioned FiLM heads)
- **Benchmarking** against 15+ existing IQA methods via pyiqa

## Project structure

```
src/uav_iqa/           # Core library
  distortion.py        #   Distortion models (UAVDistortionPipeline + 6 UAV + 18 generic)
  model.py             #   UAVIQANet (backbone → PANet FPN → CBAM → FAB → task heads)
  dataset.py           #   UAVIQADataset — loads manifest.json, image/scores/task_id
  trainer.py           #   ListMLELoss + CrossTaskRegularization (used by LightningModule)
  evaluate.py          #   SRCC, PLCC, RMSE, Kendall tau metrics
  utils.py             #   count_parameters
configs/default.yaml   # Training/model/distortion/benchmark configuration
scripts/               # Executable experiment scripts
  run_m3_train.py      #   Training (multi-seed, ablations, cross-task, leave-one-out)
  run_m2_benchmark.py  #   Benchmark existing IQA methods on the UAV dataset
  run_m1_aircopbench.py #  Full M1 pipeline: scan AirCopBench → build index → inject → manifest
  run_m1_inject.py     #   Batch distortion injection from reference images
  run_m1_manifest.py   #   Generate train/val/test manifest splits from distorted directory
  run_distortion.py    #   Verify all 24 distortions produce visually plausible outputs
  run_overfit.py       #   Overfit test: train on 100 random images, verify loss → 0
  extract_aircopbench_refs.py # Extract clean reference frames from AirCopBench directory tree
data/                  # Datasets (raw = external inputs, processed = generated artifacts)
tests/                 # pytest tests (test_distortion.py, test_lightning.py)
refine-logs/           # Research-refine artifacts (FINAL_PROPOSAL, EXPERIMENT_PLAN, etc.)
docs/                  # Literature reviews and research roadmap (Chinese + English)
```

## Data pipeline (scripts in execution order)

1. **`extract_aircopbench_refs.py`** — Extract clean reference frames from AirCopBench (`data/raw/AirCopBench/`) → `data/processed/ref_images/`
2. **`run_m1_inject.py`** — Apply all 24 distortions × 5 intensity levels to reference images → `data/processed/distorted/`
3. **`run_m1_manifest.py`** — Scan distorted directory, generate train/val/test `manifest.json` with placeholder scores → `data/processed/{train,val,test}/`
4. **`run_m1_aircopbench.py`** — Alternative M1 pipeline that uses real AirCopBench annotations (Quality/Usability scores) as labels
5. **`run_m3_train.py`** — Train UAVIQANet on the dataset (reads `data/processed/`, writes `outputs/training/`)
6. **`run_m2_benchmark.py`** — Evaluate existing IQA methods (PSNR, SSIM, LPIPS, BRISQUE, CLIP-IQA, MANIQA, etc.) on the test set

## Setup and development commands

```bash
# Install with all extras
uv sync --group dev

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/uav_iqa --cov-report=term-missing

# Lint/format
ruff check src/ tests/ scripts/
black src/ tests/ scripts/

# Single test
pytest tests/test_distortion.py::test_pipeline_has_all_distortions -v
```

## Model architecture (UAVIQANet)

```
Input (3×256×256)
  ├─ MobileNetV4-S backbone (frozen first 2 stages) → multi-scale features
  ├─ PANet FPN → spatial pyramid fusion
  ├─ CBAM (channel + spatial attention) → f_s ∈ R^256
  ├─ FrequencyAwareBranch (patch FFT + log-polar transform + tiny CNN) → f_f ∈ R^64
  ├─ CrossAttentionGate: α = σ(W·[f_s, f_f]) → α ⊙ f_f
  └─ Concat[f_s, α⊙f_f] → TaskConditionedHead (FiLM: task_embed → γ,β modulate hidden)
       → 1 score per task
```

Task types: `tracking=0`, `inspection=1`, `delivery=2`, `sar=3`

## Key concepts

- **Manifest format**: JSON list of `{path, task, distortion, intensity_level, ref_id, vlm_score, vla_score, execution_score, annotated}`
- **3-stage curriculum**: VLM annotations (epochs 1-20) → VLA (21-40) → Execution (41-50)
- **Loss**: MSE + λ_rank * ListMLE (per-distortion ranking) + λ_cross_task * (- task score variance)
- **Ablation toggles** on UAVIQANet: `--no-fab`, `--no-cbam`, `--no-task-cond`
- **Distortion naming**: `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`)

## Common training invocations

```bash
# Standard training
python scripts/run_m3_train.py --data-dir data/database --output-dir outputs/training

# Multi-seed
python scripts/run_m3_train.py --seeds 42 100 200

# Ablations
python scripts/run_m3_train.py --no-fab
python scripts/run_m3_train.py --no-task-cond

# Per-task / cross-task / leave-one-out
python scripts/run_m3_train.py --task tracking
python scripts/run_m3_train.py --leave-out sar

# Dry run (100 samples, fast verification)
python scripts/run_m3_train.py --dry-run
```

## Research state

The proposal is **READY** (score 9.0/10). Read these for context:
- `refine-logs/FINAL_PROPOSAL.md` — refined method thesis
- `refine-logs/EXPERIMENT_PLAN.md` — 33 runs, 6 milestones, 4 claims
- `refine-logs/EXPERIMENT_TRACKER.md` — per-run status tracker

4 claims to validate: C1 (UAV distortion distinctiveness), C2 (synthetic↔real correlation), C3 (existing IQA failure), C4 (cross-task generalization).

## Notes

- `data/` and `outputs/` are gitignored — datasets must be downloaded separately (AirCopBench from arXiv 2511.11025)
- Real-ESRGAN dependency is optional (for `LowResSuperResolution` distortion); falls back to bicubic + sharpen if not installed
- The scripts use `sys.path.insert` to import from `src/` — install with `uv sync` when possible
- openVLA and CARLA require manual installation (not on PyPI); not needed for basic training/inference
