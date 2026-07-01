# AGENTS.md

Research codebase: UAV-Embodied-IQA — visual quality assessment for aerial embodied intelligence.
See `CLAUDE.md` for detailed architecture and design rationale.

## Commands

```bash
# Install (required before running scripts)
uv sync --group dev

# Run tests
pytest tests/ -v

# Lint then format (order matters)
ruff check src/ tests/ scripts/
black src/ tests/ scripts/

# Single test
pytest tests/test_distortion.py::test_pipeline_has_all_distortions -v

# Overfit test (fast correctness check of the model)
python scripts/overfit_sanity_check.py

# Download VLM/VLA model weights for annotation scoring (optional)
python scripts/download_models.py --all

# Validate downloaded models load correctly
python scripts/download_models.py --all --validate
```

## Architecture at a glance

```
src/uav_iqa/
  distortion.py       # 6 UAV + 30 generic distortion models
  model.py            # UAVIQANet (MobileNetV4-S + PANet FPN + CBAM + FAB + task-conditioned heads)
  dataset.py          # UAVIQADataset: grouped JSON → image/cognitive_score/subtask
  losses.py           # ListMLELoss + CrossTaskRegularization (ranking & cross-task losses)
  annotations.py      # AirCopBench annotation parsing, degradation factors, score synthesis
  data_synthesis.py   # Dataset-agnostic data pipeline (DatasetFormat + DataSynthesisPipeline)
  metrics.py          # SRCC, PLCC, RMSE, Kendall tau
  lightning_module.py # UAVIQALightningModule (MSE + ListMLE + cross-task loss)
  data_module.py      # UAVIQADataModule (train/test dataloaders, task/distortion/LOO filters)
  callbacks.py        # SetupRunCallback, MetricsHistoryCallback, ResultsSavingCallback; CurriculumStageCallback (deprecated/no-op)
  text_metrics.py     # BLEU, ROUGE-L, CIDEr text similarity for VLM comparison scoring
  vlm/                # VLM scoring subpackage: config, scorer, VQA index
  vla_scorer.py       # BaseScorer: VLA/execution score interface
  batch_annotator.py  # BatchAnnotator: multi-GPU batch annotation across splits
  inference/          # Multi-GPU offline inference framework (15 modules)
  utils.py            # count_parameters, find_images, load_image_tensor, logging

scripts/
  data_synthesis.py              # Unified data synthesis CLI (extract/inject/annotate/aggregate/all)
  download_models.py             # Download VLM model weights from HuggingFace Hub
  vlm_annotate.py                # Batch VLM annotation CLI with checkpoint/resume
  benchmark_iqa_methods.py       # Benchmark existing IQA methods (pyiqa)
  visualize_distortions.py       # Visual sanity check of all distortions
  overfit_sanity_check.py        # 100-image overfit test
  validate_synth_real_correlation.py  # C2 correlation validation
  inference.py                   # Multi-GPU offline inference CLI

configs/default.yaml          # Model/data/training config template
configs/experiments/          # 21 per-experiment configs (r013–r024c)
scripts/train.py              # Unified training entry point (LightningCLI)
```

## Gotchas

- **Scripts use `sys.path.insert`** to import from `src/`. Either `pip install -e .` first or run from repo root.
- **`data/` and `outputs/` are gitignored.** Datasets must be downloaded separately (AirCopBench: arXiv 2511.11025).
- **Real-ESRGAN is optional** (`LowResSuperResolution` distortion); falls back to bicubic+sharpen if not installed.
- **openVLA and CARLA are manual installs** (not on PyPI); not needed for basic training/inference.
- **VLM extras** (`vllm`, `transformers`, `accelerate`) for annotation scoring: `uv sync --group dev --extra vlm`.
- **Data pipeline order matters**: `extract → inject → annotate → aggregate → train → benchmark`.
- **Distortion naming**: `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`).
- **14 subtask types** across 4 dimensions: scene_understanding (1.x), object_understanding (2.x), planning (3.x), collaboration (4.x).
- **Ablation toggles**: use per-experiment config in `configs/experiments/` (e.g., `r016_no_fab.yaml`), or override via CLI: `--model.init_args.use_fab false`.
- **Test coverage is sparse** (only `test_distortion.py`, `test_lightning.py`, `test_data_synthesis.py`, `test_text_metrics.py`, `test_vlm_config.py`, `test_vlm_scorer.py`, `test_vlm_smoke.py`, `test_batch_annotator.py`, `test_annotations.py`, `test_dataset.py`, `test_model_text.py`, and `test_inference_phase[1-5].py` exist). Add tests to `tests/` when implementing new functionality.
- **`scipy` removed as a direct dependency** for distortion models (`distortion.py` uses `cv2.filter2D` with manual wrap padding instead of `scipy.signal.convolve2d`). The runtime `scipy` dep is retained for metric computation.
- **Training entry point**: `scripts/train.py` (vanilla LightningCLI). `main.py`, `run_m3_train.py` and `UAVIQACLI` were removed in 2026-06 refactor.

## Research context

The proposal is **READY** (score 9.0/10, 3 rounds of external review). Key docs:

```
refine-logs/FINAL_PROPOSAL.md      # Method thesis
refine-logs/EXPERIMENT_PLAN.md     # 33 runs, 6 milestones, 4 claims
refine-logs/EXPERIMENT_TRACKER.md  # Run-by-run status (all TODO)
refine-logs/REVIEW_SUMMARY.md      # Review resolution log
```

4 claims to validate: C1 (UAV distortion distinctiveness), C2 (synthetic↔real correlation), C3 (existing IQA failure), C4 (cross-task generalization).

## Conventions

- Python 3.10+, torch 2.1+
- ruff with `line-length=100`, black with matching limit
- pytest config in `pyproject.toml`: `pythonpath = ["src"]`
- Technical writing, code identifiers, commit messages: English
- User-facing communication: Chinese
