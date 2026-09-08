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

# Download VLM model weights for annotation scoring (optional)
python scripts/download_models.py --all

# Validate downloaded models load correctly
python scripts/download_models.py --all --validate
```

## Architecture at a glance

```
src/uav_iqa/
  domain/             # Shared AirCopBench taxonomy, parsing, degradation factors, score synthesis
  models/             # UAVIQANet composition, spatial/frequency/text components and heads
  distortions/        # 6 UAV + 30 generic distortions and pipeline
  data/               # Dataset adapters, synthesis, samples, dataset, and datamodule
  training/           # Lightning module, losses, and callbacks
  evaluation/         # IQA metrics plus BLEU, ROUGE-L, CIDEr, cognitive score
  vlm/                # VLM scoring, BatchAnnotator, config, scorer, VQA index
  inference/          # Multi-GPU offline inference framework (15 modules)
  baselines/          # Existing IQA evaluator and fine-tuning
  utils.py            # count_parameters, find_images, load_image_tensor, logging

scripts/
  distortion_synthesis.py              # Unified data synthesis CLI (extract/inject/annotate/aggregate/all)
  download_models.py             # Download VLM model weights from HuggingFace Hub
  vlm_annotate.py                # Batch VLM annotation CLI with checkpoint/resume
  benchmark_iqa_methods.py       # Benchmark existing IQA methods (pyiqa)
  inference.py                   # Multi-GPU offline inference CLI

configs/README.md             # Maintained configuration catalog and prerequisites
configs/experiments/          # 8 configs; full_model.yaml is the reference
scripts/train.py              # Unified training entry point (LightningCLI)
```

## Gotchas

- **Scripts use `sys.path.insert`** to import from `src/`. Either `pip install -e .` first or run from repo root.
- **`data/` and `outputs/` are gitignored.** Datasets must be downloaded separately (AirCopBench: arXiv 2511.11025).
- **Real-ESRGAN is optional** (`LowResSuperResolution` distortion); falls back to bicubic+sharpen if not installed.
- **VLM extras** (`vllm`, `transformers`, `accelerate`) for annotation scoring: `uv sync --group dev --extra vlm`.
- **Data pipeline order matters**: `extract → inject → annotate → aggregate → train → benchmark`.
- **Distortion naming**: `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`).
- **14 subtask types** across 4 dimensions: scene_understanding (1.x), object_understanding (2.x), planning (3.x), collaboration (4.x).
- **Ablation toggles**: use per-experiment config in `configs/experiments/` (e.g., `no_frequency_encoder.yaml`), or override via CLI: `--model.use_frequency_encoder false`.
- **Test coverage is sparse** (only `test_distortion.py`, `test_lightning.py`, `test_distortion_synthesis.py`, `test_text_metrics.py`, `test_vlm_config.py`, `test_vlm_scorer.py`, `test_vlm_smoke.py`, `test_batch_annotator.py`, `test_annotations.py`, `test_dataset.py`, `test_model_text.py`, and `test_inference_phase[1-5].py` exist). Add tests to `tests/` when implementing new functionality.
- **`scipy` removed as a direct dependency** for distortion models (`distortion.py` uses `cv2.filter2D` with manual wrap padding instead of `scipy.signal.convolve2d`). The runtime `scipy` dep is retained for metric computation.
- **Training entry point**: `scripts/train.py` (vanilla LightningCLI). `main.py`, `run_m3_train.py` and `UAVIQACLI` were removed in 2026-06 refactor.

## Research plan

`docs/research/RESEARCH_PLAN.md` is the single maintained research plan, with the
current data/implementation blockers, claim map, experiment protocol, and milestone
tracker. Do not treat proposed experiments or cited paper results as project evidence.

## Conventions

- Python 3.10+, torch 2.1+
- ruff with `line-length=100`, black with matching limit
- pytest config in `pyproject.toml`: `pythonpath = ["src"]`
- Technical writing, code identifiers, commit messages: English
- User-facing communication: Chinese
