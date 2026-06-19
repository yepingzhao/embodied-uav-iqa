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
src/uav_iqa/           # Core library (~2.2K LOC total)
  __init__.py          #   Public API: exports 22 symbols
  distortion.py        #   24 distortion models (UAVDistortionPipeline + 6 UAV + 18 generic)
  model.py             #   UAVIQANet (backbone → PANet FPN → CBAM → FAB → task heads)
  dataset.py           #   UAVIQADataset — loads manifest.json, image/scores/task_id
  losses.py            #   ListMLELoss + CrossTaskRegularization
  annotation_utils.py  #   AirCopBench annotation parsing, degradation factors, score synthesis
  data_synthesis.py   #   Dataset-agnostic data pipeline (DatasetFormat + DataSynthesisPipeline)
  lightning_model.py   #   UAVIQALightningModule (training_step, validation_step, etc.)
  lightning_data.py    #   UAVIQDataModule (train/val/test dataloaders, manifest filtering)
  evaluate.py          #   SRCC, PLCC, RMSE, Kendall tau metrics
  callbacks.py         #   SetupRunCallback, CurriculumStageCallback, MetricsHistoryCallback, ResultsSavingCallback
  utils.py             #   count_parameters
main.py                # Unified training entry point (LightningCLI, reads experiment YAMLs)
configs/               # YAML-driven configuration
  default.yaml         #   Default training/model/distortion config template
  experiments/         #   21 per-experiment configs (r013–r024c)
scripts/               # Data pipeline + benchmark scripts
  synthesize_data.py   #   Unified data synthesis CLI (extract/inject/manifest/annotate/all)
  run_m2_benchmark.py  #   Benchmark 15+ IQA methods via pyiqa
  run_distortion.py    #   Verify all 24 distortions produce visually plausible outputs
  run_overfit.py       #   Overfit test: train on 100 random images, verify loss → 0
  run_c2_correlation.py #  C2 correlation validation: synthetic vs real scores
data/                  # Datasets (raw = external inputs, processed = generated artifacts)
tests/                 # pytest tests (test_distortion.py, test_lightning.py)
refine-logs/           # Research-refine artifacts (FINAL_PROPOSAL, EXPERIMENT_PLAN, etc.)
docs/                  # CODEMAPS, literature reviews, and research roadmap
```

## Data pipeline

Data synthesis is handled by `src/uav_iqa/data_synthesis.py` (library) and `scripts/synthesize_data.py` (CLI). The pipeline has 4 steps — run individually or end-to-end:

```bash
# Full pipeline (all 4 steps)
python scripts/synthesize_data.py all \
  --dataset aircopbench \
  --input-root data/raw/AirCopBench \
  --output-dir data/processed

# Or run steps individually:
python scripts/synthesize_data.py extract --dataset aircopbench --input-root ... --output-dir ...
python scripts/synthesize_data.py inject --image-dir ... --output-dir ...
python scripts/synthesize_data.py manifest --dataset aircopbench --distorted-dir ... --output-dir ...
python scripts/synthesize_data.py annotate --dataset aircopbench --manifest-dir ...
```

1. **`extract`** — Extract clean reference frames from dataset → `data/processed/ref_images/`
2. **`inject`** — Apply all 24 distortions × 5 intensity levels → `data/processed/distorted/`
3. **`manifest`** — Scan distorted dir, generate train/val/test `manifest.json` → `data/processed/{train,val,test}/`
4. **`annotate`** — Annotate manifest entries using annotations (if available) and degradation model: `score = ref_score × degradation_factor(distortion, intensity)` + noise
5. **`main.py`** — Train UAVIQANet via LightningCLI + experiment config (reads `data/processed/`, writes `outputs/<experiment>_seed<N>/`)
6. **`run_m2_benchmark.py`** — Evaluate existing IQA methods (PSNR, SSIM, LPIPS, BRISQUE, CLIP-IQA, MANIQA, etc.) on the test set

## Setup and development commands

```bash
# Install with all extras
uv sync --group dev

# Install with SwanLab experiment tracking (optional)
uv sync --extra swanlab

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
- **Loss**: MSE + λ_rank * ListMLE (per-distortion ranking) + λ_cross_task * CrossTaskRegularization (negative pairwise score variance)
- **Ablation toggles**: configured via `model.init_args.use_fab/cbam/task_conditioning` in experiment YAML (e.g., `r016_no_fab.yaml`)
- **Distortion naming**: `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`)

## Common training invocations

All hyperparameters live in self-contained `configs/experiments/<name>.yaml`. Training uses `main.py` (LightningCLI) as the single entry point.

```bash
# Standard training (task-conditioned, all components enabled)
python main.py fit --config configs/experiments/r013_task_cond.yaml

# Multi-seed (shell loop — root dir via --trainer.default_root_dir)
for seed in 42 100 200; do
  python main.py fit --config configs/experiments/r013_task_cond.yaml \
    --seed_everything $seed \
    --trainer.default_root_dir "outputs/r013_seed${seed}"
done

# Ablations (each is a self-contained config)
python main.py fit --config configs/experiments/r016_no_fab.yaml         # w/o FAB
python main.py fit --config configs/experiments/r017_no_task_cond.yaml   # w/o task conditioning
python main.py fit --config configs/experiments/r018_no_cbam.yaml        # w/o CBAM
python main.py fit --config configs/experiments/r019_mobilevit_s.yaml    # MobileViT-S backbone
python main.py fit --config configs/experiments/r020_efficientvit_b0.yaml # EfficientViT-B0 backbone

# Distortion filtering
python main.py fit --config configs/experiments/r021_generic_only.yaml   # only generic distortions
python main.py fit --config configs/experiments/r021b_uav_only.yaml      # only UAV distortions

# Curriculum ablation (annotation source only)
python main.py fit --config configs/experiments/r022_vlm_only.yaml       # VLM only
python main.py fit --config configs/experiments/r022b_vla_only.yaml      # VLA only
python main.py fit --config configs/experiments/r023_no_exec.yaml        # VLM+VLA, no execution

# Per-task training
python main.py fit --config configs/experiments/r024a_tracking.yaml
python main.py fit --config configs/experiments/r024a_delivery.yaml
python main.py fit --config configs/experiments/r024a_inspection.yaml
python main.py fit --config configs/experiments/r024a_sar.yaml

# Leave-one-task-out
python main.py fit --config configs/experiments/r024b_leave_tracking.yaml
python main.py fit --config configs/experiments/r024b_leave_sar.yaml

# Full multi-task (all tasks)
python main.py fit --config configs/experiments/r024c_multitask.yaml

# Override any config key from CLI
python main.py fit --config configs/experiments/r013_task_cond.yaml \
  --trainer.max_epochs 100 \
  --data.init_args.batch_size 32
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
- Package is installed via `uv sync` — `main.py` and scripts use `from uav_iqa.xxx` imports
- openVLA and CARLA require manual installation (not on PyPI); not needed for basic training/inference
- **SwanLab**: Set `SWANLAB_API_KEY` env var (or use `.env` file) to enable cloud experiment tracking. Without it, training falls back to local CSVLogger + history.json logging. Config at `trainer.logger` in `configs/default.yaml`.
- **`UAVIQACLI` removed (2026-06)**: Training now uses vanilla `lightning.pytorch.cli.LightningCLI` via `main.py`. Multi-seed loops and experiment configs are managed externally (shell scripts). Callbacks (`ResultsSavingCallback`, `MetricsHistoryCallback`) handle test evaluation and result saving.
