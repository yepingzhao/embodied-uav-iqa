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
python scripts/run_overfit.py
```

## Architecture at a glance

```
src/uav_iqa/
  distortion.py       # 6 UAV + 18 generic distortion models
  model.py            # UAVIQANet (MobileNetV4-S + PANet FPN + CBAM + FAB + task-conditioned heads)
  dataset.py          # UAVIQADataset: manifest.json → image/scores/task_id
  losses.py           # ListMLELoss + CrossTaskRegularization (ranking & cross-task losses)
  annotation_utils.py # AirCopBench annotation parsing, degradation factors, score synthesis
  data_synthesis.py  # Dataset-agnostic data pipeline (DatasetFormat + DataSynthesisPipeline)
  evaluate.py         # SRCC, PLCC, RMSE, Kendall tau
  lightning_model.py  # UAVIQALightningModule (MSE + ListMLE + cross-task loss, curriculum)
  lightning_data.py   # UAVIQDataModule (manifest filtering, task/distortion/LOO filters)
  callbacks.py        # CurriculumStageCallback, MetricsHistoryCallback, SetupRunCallback, ResultsSavingCallback

scripts/
  synthesize_data.py          # Unified data synthesis CLI (extract/inject/manifest/annotate/all)
  run_m2_benchmark.py         # Benchmark existing IQA methods (pyiqa)
  run_distortion.py           # Visual sanity check of all distortions
  run_overfit.py              # 100-image overfit test
  run_c2_correlation.py       # C2 correlation validation

configs/default.yaml          # Model/data/training config template
configs/experiments/          # 21 per-experiment configs (r013–r024c)
main.py                       # Unified training entry point (LightningCLI)
```

## Gotchas

- **Scripts use `sys.path.insert`** to import from `src/`. Either `pip install -e .` first or run from repo root.
- **`data/` and `outputs/` are gitignored.** Datasets must be downloaded separately (AirCopBench: arXiv 2511.11025).
- **Real-ESRGAN is optional** (`LowResSuperResolution` distortion); falls back to bicubic+sharpen if not installed.
- **openVLA and CARLA are manual installs** (not on PyPI); not needed for basic training/inference.
- **VLM extras** (`vllm`, `transformers`, `accelerate`) for annotation scoring: `uv sync --group dev --extra vlm`.
- **Data pipeline order matters**: `extract_refs → inject distortions → generate manifest → annotate scores → train → benchmark`.
- **3-stage curriculum**: VLM annotations (epochs 1-20) → VLA (21-40) → Execution (41-50). Annotation source gated by kwargs.
- **Distortion naming**: `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`).
- **4 task types**: `tracking=0`, `inspection=1`, `delivery=2`, `sar=3`.
- **Ablation toggles**: use per-experiment config in `configs/experiments/` (e.g., `r016_no_fab.yaml`), or override via CLI: `--model.init_args.use_fab false`.
- **Test coverage is sparse** (only `test_distortion.py` and `test_lightning.py` exist). Add tests to `tests/` when implementing new functionality.
- **Training entry point**: `main.py` (vanilla LightningCLI). `run_m3_train.py` and `UAVIQACLI` were removed in 2026-06 refactor.

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
