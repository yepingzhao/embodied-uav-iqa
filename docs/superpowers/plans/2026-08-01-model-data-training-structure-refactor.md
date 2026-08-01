# Model, Data, and Training Structure Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize UAV-Embodied-IQA's model, distortion, data-synthesis, training, and evaluation code into responsibility-oriented packages with descriptive names while preserving observable numerical and data behavior.

**Architecture:** Split the current top-level implementation modules into `models`, `distortions`, `data`, `training`, and `evaluation` packages with one-way dependencies. Preserve algorithms and schemas, update all active composition roots atomically, and deliberately remove old import/config/checkpoint compatibility.

**Tech Stack:** Python 3.10+, PyTorch, timm, Lightning, OpenCV, Albumentations, NumPy/SciPy, pytest, pytest-cov, Ruff, LightningCLI.

---

## Execution constraints

- Work from the approved design: `docs/superpowers/specs/2026-08-01-model-data-training-structure-refactor-design.md`.
- Do not modify these unrelated local files:
  - `PAPER_PLAN.md`
  - `refine-logs/EXPERIMENT_PLAN.md`
  - `refine-logs/EXPERIMENT_TRACKER.md`
  - `refine-logs/EXPERIMENT_PLAN_20260727.md`
  - `refine-logs/MANIFEST.md`
- Preserve model calculations, distortion algorithms, processed JSON fields, training batch fields, loss formulas, metric formulas, and logged metric keys.
- Do not add compatibility aliases for removed modules, class names, config keys, CLI names, or checkpoints.
- Do not change `UAVIQANet` or the paper name UAV-IQANet.
- Do not run formal GPU experiments.
- Do not create commits unless the user explicitly requests commits. At each checkpoint, inspect the focused diff and record verification output instead.
- The remote runtime repository is `c171_vm:/usr/storage/xjp/projects/embodied-uav-iqa/`. Never overwrite a dirty remote worktree.

## Locked file map

### Create

- `src/uav_iqa/models/__init__.py` — model package facade.
- `src/uav_iqa/models/spatial.py` — `PANFeaturePyramid`, `ConvolutionalBlockAttention`.
- `src/uav_iqa/models/frequency.py` — `LogPolarFrequencyEncoder`, `FrequencyFeatureGate`.
- `src/uav_iqa/models/text.py` — `QuestionTextEncoder`.
- `src/uav_iqa/models/heads.py` — `TaskConditionedRegressor`, shared regressor factory/module.
- `src/uav_iqa/models/uav_iqa_net.py` — `UAVIQANet` composition.
- `src/uav_iqa/distortions/__init__.py` — intentional distortion API.
- `src/uav_iqa/distortions/base.py` — common image helpers and `BaseDistortion`.
- `src/uav_iqa/distortions/uav.py` — six UAV-specific distortion classes.
- `src/uav_iqa/distortions/generic.py` — `GenericDistortions`.
- `src/uav_iqa/distortions/pipeline.py` — `UAVDistortionPipeline`.
- `src/uav_iqa/data/__init__.py` — data package facade.
- `src/uav_iqa/data/samples.py` — processed entry loading, benchmark projection, manifest validation.
- `src/uav_iqa/data/adapters.py` — dataset adapter registry and implementations.
- `src/uav_iqa/data/synthesis.py` — `DistortionSynthesisPipeline`, `create_pipeline`.
- `src/uav_iqa/data/dataset.py` — `UAVQualityDataset`.
- `src/uav_iqa/data/datamodule.py` — `UAVQualityDataModule`.
- `src/uav_iqa/training/__init__.py` — training API.
- `src/uav_iqa/training/losses.py` — existing losses.
- `src/uav_iqa/training/callbacks.py` — existing callbacks.
- `src/uav_iqa/training/module.py` — `UAVQualityTrainingModule`.
- `src/uav_iqa/evaluation/__init__.py` — evaluation API.
- `src/uav_iqa/evaluation/metrics.py` — IQA metrics.
- `tests/test_model_structure.py` — model component and package-boundary tests.
- `tests/test_sample_loading.py` — processed entry and benchmark projection tests.
- `tests/test_package_boundaries.py` — removed paths and lightweight root import tests.
- `tests/test_experiment_configs.py` — LightningCLI/config identifier checks.

### Modify

- `src/uav_iqa/__init__.py` — reduce to lightweight package metadata/facade.
- `src/uav_iqa/utils.py` — remove processed sample projection after callers migrate.
- `src/uav_iqa/baselines/evaluator.py` — new distortion/evaluation imports if present.
- `src/uav_iqa/baselines/finetune.py` — new data/model/evaluation imports.
- `src/uav_iqa/batch_annotator.py` — new data contract imports only if needed.
- `src/uav_iqa/inference/executor.py` and other matching inference files — update moved imports only.
- `scripts/train.py` — new training/data classes.
- `scripts/distortion_synthesis.py` — new adapter/synthesis imports and names.
- `scripts/visualize_distortions.py` — new distortion imports.
- `scripts/benchmark_iqa_methods.py` — `load_benchmark_samples` and evaluation imports.
- `scripts/validate_synth_real_correlation.py` — `load_benchmark_samples` and evaluation imports.
- `scripts/finetune_baselines.py` — new package imports if required.
- `configs/experiments/*.yaml` — `use_frequency_encoder`, new callback paths, semantic subtask filters.
- `configs/default.yaml` — align with current constructors or remove stale unsupported keys.
- Existing tests — import and class-name migration without weakening assertions.
- `README.md`, `CLAUDE.md`, `AGENTS.md`, `docs/CODEMAPS/ARCHITECTURE.md`, `docs/CODEMAPS/FILES.md`, `docs/EXPERIMENTS.md` — current paths and commands only.

### Remove after migration

- `src/uav_iqa/model.py`
- `src/uav_iqa/distortion.py`
- `src/uav_iqa/distortion_synthesis.py`
- `src/uav_iqa/dataset.py`
- `src/uav_iqa/data_module.py`
- `src/uav_iqa/lightning_module.py`
- `src/uav_iqa/losses.py`
- `src/uav_iqa/callbacks.py`
- `src/uav_iqa/metrics.py`

---

### Task 1: Capture the pre-refactor behavior baseline

**Files:**
- Create: `tests/test_model_structure.py`
- Create: `tests/test_sample_loading.py`
- Modify: `tests/test_distortion.py`

- [ ] **Step 1: Record the untouched working-tree baseline**

Run:

```bash
git status --short
git diff -- src tests scripts configs README.md CLAUDE.md AGENTS.md docs/CODEMAPS docs/EXPERIMENTS.md
```

Expected: only pre-existing paper/experiment-plan changes plus approved spec/plan; no source changes.

- [ ] **Step 2: Add current model characterization tests**

Create `tests/test_model_structure.py` importing current symbols and checking exact component output shapes, single/multi-image model output shapes, text/no-text paths, task-conditioned/shared heads, and `[0, 1]` score range. Disable timm weight downloads in tests by monkeypatching `timm.create_model(..., pretrained=False)`.

Use these exact current component cases:

```python
pyramid = PanetFPN([16, 32, 64], out_channels=32).eval()
features = [
    torch.randn(2, 16, 32, 32),
    torch.randn(2, 32, 16, 16),
    torch.randn(2, 64, 8, 8),
]
assert [tuple(item.shape) for item in pyramid(features)] == [
    (2, 32, 32, 32),
    (2, 32, 16, 16),
    (2, 32, 8, 8),
]
assert CBAM(32).eval()(torch.randn(2, 32, 16, 16)).shape == (2, 32, 16, 16)
assert FrequencyAwareBranch().eval()(torch.randn(2, 3, 64, 64)).shape == (2, 64)
assert CrossAttentionGate().eval()(torch.randn(2, 256), torch.randn(2, 64)).shape == (2, 64)
assert TaskConditionedHead().eval()(torch.randn(2, 320), torch.tensor([0, 1])).shape == (2,)
assert UAVIQATextEncoder(text_dim=128).eval()(["scene?", "target?"]).shape == (2, 128)
```

- [ ] **Step 3: Add processed-to-benchmark projection coverage**

Create `tests/test_sample_loading.py` using one synthetic processed entry and assert the exact current `load_flat_samples()` output keys: `path`, `ref_path`, `uav_paths`, `uav_keys`, `distorted_uav_paths`, `task`, `distortion`, `category`, `intensity_level`, `score`, `ref_id`, `sample_id`.

- [ ] **Step 4: Add stable distortion registry/determinism coverage**

Append a test to `tests/test_distortion.py` that creates two `UAVDistortionPipeline(seed=17)` instances, asserts identical UAV/all-name lists, and asserts identical outputs for all six UAV distortions at intensity `0.4` on the same deterministic image.

- [ ] **Step 5: Run the characterization suite**

Run:

```bash
pytest tests/test_model_structure.py tests/test_model_text.py tests/test_distortion.py tests/test_distortion_synthesis.py tests/test_sample_loading.py tests/test_dataset.py tests/test_lightning.py -v
```

Expected: PASS except existing explicit GPU skips. Correct inaccurate assertions rather than changing implementation.

---

### Task 2: Split and rename model components

**Files:**
- Create: `src/uav_iqa/models/__init__.py`
- Create: `src/uav_iqa/models/spatial.py`
- Create: `src/uav_iqa/models/frequency.py`
- Create: `src/uav_iqa/models/text.py`
- Create: `src/uav_iqa/models/heads.py`
- Modify: `tests/test_model_structure.py`
- Modify: `tests/test_model_text.py`

- [ ] **Step 1: Change tests to approved imports and names**

Use:

```python
from uav_iqa.models.frequency import FrequencyFeatureGate, LogPolarFrequencyEncoder
from uav_iqa.models.heads import TaskConditionedRegressor
from uav_iqa.models.spatial import ConvolutionalBlockAttention, PANFeaturePyramid
from uav_iqa.models.text import QuestionTextEncoder
```

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_model_structure.py tests/test_model_text.py -v`.

Expected: collection FAIL with `ModuleNotFoundError: No module named 'uav_iqa.models'`.

- [ ] **Step 3: Move spatial components**

Move the complete `PanetFPN` and `CBAM` bodies from `model.py` to `models/spatial.py`, renaming only to `PANFeaturePyramid` and `ConvolutionalBlockAttention`. Preserve all layers, interpolation, ordering, and operations.

- [ ] **Step 4: Move frequency components**

Move complete `FrequencyAwareBranch` and `CrossAttentionGate` bodies to `models/frequency.py`, renamed to `LogPolarFrequencyEncoder` and `FrequencyFeatureGate`. Preserve `_lp_flat_idx`, all defaults, FFT/log-polar operations, and gating math.

- [ ] **Step 5: Move text encoder**

Move the complete `UAVIQATextEncoder` body to `models/text.py` as `QuestionTextEncoder`, preserving vocabulary, indexes, tokenization, kernels, pooling, projection, and device behavior.

- [ ] **Step 6: Move regression heads**

Create `models/heads.py` with `TaskConditionedRegressor`, preserving the current embedding, gamma/beta projections, hidden layer, sigmoid, and defaults. Remove the unused checkpoint migration method. Add `create_shared_regressor(in_features, hidden_dim=128)` returning the existing four-layer sequential head.

- [ ] **Step 7: Define component exports**

Create `models/__init__.py` exporting only the five renamed components and `TaskConditionedRegressor`; do not export old names.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
pytest tests/test_model_structure.py tests/test_model_text.py -v
ruff check src/uav_iqa/models tests/test_model_structure.py tests/test_model_text.py
```

Expected: component tests PASS.

---

### Task 3: Move `UAVIQANet` composition

**Files:**
- Create: `src/uav_iqa/models/uav_iqa_net.py`
- Modify: `src/uav_iqa/models/__init__.py`
- Modify: `tests/test_model_structure.py`
- Modify: `tests/test_model_text.py`

- [ ] **Step 1: Point tests to `from uav_iqa.models import UAVIQANet`**

Rename constructor keys in tests: `use_fab` → `use_frequency_encoder`; `use_cbam` → `use_spatial_attention`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_model_structure.py tests/test_model_text.py -v`.

Expected: import FAIL because `UAVIQANet` is not exported.

- [ ] **Step 3: Move full model composition**

Create `models/uav_iqa_net.py` from the existing class body. Use this constructor:

```python
def __init__(
    self,
    backbone: str = "mobilenetv4_conv_small",
    num_tasks: int | None = None,
    freeze_backbone_stage: int = 2,
    use_frequency_encoder: bool = True,
    use_spatial_attention: bool = True,
    use_task_conditioning: bool = True,
    use_text_encoder: bool = True,
    text_dim: int = 128,
):
```

Wire renamed attributes `feature_pyramid`, `pyramid_projection`, `spatial_attention`, `frequency_encoder`, `frequency_gate`, `question_encoder`, `regressor`, and `shared_regressor`. Preserve timm probing/fallback, tensor calculations, dimensions, multi-UAV averaging, null text embedding, task default, all public methods, and output shapes.

- [ ] **Step 4: Export `UAVIQANet`**

Add `from .uav_iqa_net import UAVIQANet` and `"UAVIQANet"` to `models.__all__`.

- [ ] **Step 5: Prove transition equivalence before deleting the old module**

Add temporary old/new component comparisons: instantiate old/new with identical args, load `old.state_dict()` into new, run identical fixed tensors, and use `torch.testing.assert_close`. Cover spatial pyramid, spatial attention, frequency encoder, frequency gate, text encoder, and task-conditioned regressor.

- [ ] **Step 6: Verify model package**

Run:

```bash
pytest tests/test_model_structure.py tests/test_model_text.py tests/test_lightning.py -v
ruff check src/uav_iqa/models tests/test_model_structure.py tests/test_model_text.py
```

Expected: PASS except explicit GPU skip.

---

### Task 4: Split the distortion package

**Files:**
- Create: `src/uav_iqa/distortions/{__init__,base,uav,generic,pipeline}.py`
- Modify: `tests/test_distortion.py`
- Modify: `scripts/visualize_distortions.py`

- [ ] **Step 1: Switch tests to `uav_iqa.distortions`**

Import the six UAV distortion classes, `GenericDistortions`, and `UAVDistortionPipeline` from the new package.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_distortion.py -v`.

Expected: missing package failure.

- [ ] **Step 3: Move base helpers and `BaseDistortion`**

Move `_load_image`, `_to_uint8`, `_motion_blur_kernel`, `_filter2d_wrap`, and `BaseDistortion` unchanged to `distortions/base.py`.

- [ ] **Step 4: Move the six UAV algorithms**

Move complete bodies unchanged to `distortions/uav.py`. Preserve published class names and metadata names.

- [ ] **Step 5: Move generic algorithms**

Move complete `GenericDistortions` implementation to `distortions/generic.py`. Preserve `CATEGORIES`, validation, intensity mapping, randomness, and dtype.

- [ ] **Step 6: Move orchestration**

Move complete `UAVDistortionPipeline` to `distortions/pipeline.py`, importing sibling implementations. Preserve registries, ordering, `INTENSITY_LEVELS`, naming, categories, `apply_distortion`, and `generate_all`.

- [ ] **Step 7: Export intentional API and update callers**

Export the eight public concrete/pipeline symbols. Update direct live imports, including `scripts/visualize_distortions.py`; do not create old-path aliases.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
pytest tests/test_distortion.py -v
ruff check src/uav_iqa/distortions tests/test_distortion.py scripts/visualize_distortions.py
```

Expected: PASS and unchanged deterministic outputs.

---

### Task 5: Centralize processed samples and adapters

**Files:**
- Create: `src/uav_iqa/data/{__init__,samples,adapters}.py`
- Modify: `tests/test_sample_loading.py`
- Modify: `tests/test_distortion_synthesis.py`

- [ ] **Step 1: Change tests to new APIs**

Use `load_processed_entries`, `load_benchmark_samples`, and `validate_manifest` from `uav_iqa.data.samples`; use `DatasetAdapter`, `AirCopBenchAdapter`, and `ImageDirectoryAdapter` from `uav_iqa.data.adapters`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_sample_loading.py tests/test_distortion_synthesis.py -v`.

Expected: missing modules/symbols.

- [ ] **Step 3: Implement `data/samples.py`**

Move manifest validation from `dataset.py`. Implement `load_processed_entries(data_root, split)` to load sorted `*_VQA_*.json` arrays and raise a path-specific `ValueError` for non-list payloads. Move the exact `utils.load_flat_samples` projection to `load_benchmark_samples`, preserving all output keys and score-default behavior.

- [ ] **Step 4: Implement adapters**

Move the existing registry and concrete classes, renamed to `DatasetAdapter`, `AirCopBenchAdapter`, and `ImageDirectoryAdapter`. Keep keys `aircopbench` and `generic`; rename `list_formats()` to `list_adapters()` and error text to `Unknown dataset adapter`.

- [ ] **Step 5: Export initial data API**

Export adapters and the three sample functions in `data/__init__.py`.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
pytest tests/test_sample_loading.py tests/test_distortion_synthesis.py -v
ruff check src/uav_iqa/data tests/test_sample_loading.py tests/test_distortion_synthesis.py
```

Expected: sample/adapter tests PASS; old synthesis may remain until Task 6.

---

### Task 6: Move synthesis, dataset, and data module

**Files:**
- Create: `src/uav_iqa/data/{synthesis,dataset,datamodule}.py`
- Modify: `src/uav_iqa/data/__init__.py`
- Modify: `tests/test_distortion_synthesis.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_lightning.py`
- Modify: `scripts/distortion_synthesis.py`

- [ ] **Step 1: Rename tests to `DistortionSynthesisPipeline`, `UAVQualityDataset`, and `UAVQualityDataModule`**

Keep all expected JSON and batch keys unchanged.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_distortion_synthesis.py tests/test_dataset.py tests/test_lightning.py -v`.

Expected: missing renamed classes.

- [ ] **Step 3: Move synthesis orchestration**

Move complete pipeline behavior to `data/synthesis.py`, rename class and constructor parameter/attribute from format to adapter, and keep `create_pipeline(dataset="aircopbench", seed=42)`. Preserve output paths, fields, statistics, multiprocessing, atomic writes, deterministic seeds, and dry-run behavior.

- [ ] **Step 4: Move and rename dataset**

Move complete `UAVIQADataset` body to `UAVQualityDataset`. Use `data.samples.validate_manifest`. Preserve constructor, transforms, loading, fallback, sample keys, task/score conversion, padding, and collate shapes.

- [ ] **Step 5: Move and rename data module**

Move complete `UAVIQADataModule` body to `UAVQualityDataModule`; update types and collate references. Preserve split behavior, filters, augmentation, dry-run limits, and DataLoader options.

- [ ] **Step 6: Export full data API and update CLI**

Export the three classes and `create_pipeline`. Update `scripts/distortion_synthesis.py` to new imports and `DatasetAdapter.list_adapters()` while preserving command behavior.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
pytest tests/test_distortion_synthesis.py tests/test_dataset.py tests/test_lightning.py::TestLightningDataModule -v
ruff check src/uav_iqa/data scripts/distortion_synthesis.py tests/test_distortion_synthesis.py tests/test_dataset.py
```

Expected: PASS with unchanged generated fields/images and batch shapes.

---

### Task 7: Move evaluation and training infrastructure

**Files:**
- Create: `src/uav_iqa/evaluation/{__init__,metrics}.py`
- Create: `src/uav_iqa/training/{__init__,losses,callbacks,module}.py`
- Modify: `tests/test_lightning.py`

- [ ] **Step 1: Rename training tests**

Use `UAVQualityTrainingModule`, `use_frequency_encoder`, and `use_spatial_attention`. Add a test that `use_text_encoder=False` reaches `module.model.use_text_encoder`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_lightning.py -v`.

Expected: missing training package.

- [ ] **Step 3: Move metrics unchanged**

Move all functions to `evaluation/metrics.py`; export `compute_srcc`, `compute_plcc`, `compute_rmse`, `compute_kendall_tau`, `evaluate_iqa`, `per_task_metrics`, `per_distortion_metrics`, and `per_distortion_category_metrics`.

- [ ] **Step 4: Move losses and callbacks unchanged**

Move complete bodies to training package and update only imports.

- [ ] **Step 5: Move and rename Lightning module**

Create `UAVQualityTrainingModule` with renamed model switches plus explicit `use_text_encoder` and `text_dim`. Pass every model option to `UAVIQANet`. Preserve losses, weights, DDP gathering, step logic, metric/log keys, optimizer, and scheduler.

- [ ] **Step 6: Export training API**

Export `UAVQualityTrainingModule`, both losses, and three callback classes.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
pytest tests/test_lightning.py tests/test_dataset.py -v
ruff check src/uav_iqa/training src/uav_iqa/evaluation tests/test_lightning.py
```

Expected: PASS except explicit GPU skip.

---

### Task 8: Migrate scripts, configs, and untouched subpackages

**Files:**
- Modify: `scripts/train.py`, benchmark/validation/fine-tune scripts
- Modify: relevant `src/uav_iqa/baselines/*.py`, `src/uav_iqa/inference/*.py`, `src/uav_iqa/batch_annotator.py`
- Modify: `configs/default.yaml`, `configs/experiments/*.yaml`
- Create: `tests/test_experiment_configs.py`

- [ ] **Step 1: Add failing config tests**

Assert every active YAML lacks `use_fab`/`use_cbam`, contains `use_frequency_encoder`/`use_spatial_attention`, has no `uav_iqa.callbacks.*` path, and uses only names from `SUBTASK_NAME_LIST` in non-null `subtask_filter`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_experiment_configs.py -v`.

Expected: old keys, callback paths, and numeric filters fail.

- [ ] **Step 3: Update `scripts/train.py`**

Import `UAVQualityDataModule` from `uav_iqa.data` and `UAVQualityTrainingModule` from `uav_iqa.training`; pass both to LightningCLI. Keep dotenv ordering.

- [ ] **Step 4: Update all live import callers**

Replace `load_flat_samples` with `load_benchmark_samples`; import metrics from `uav_iqa.evaluation`; import model/data/training/distortion symbols from owning packages. Do not reorganize VLM/inference/baseline bodies.

- [ ] **Step 5: Update every active YAML**

Rename model keys, update callback paths to `uav_iqa.training.*`, and map numeric task filters exactly through `annotations.SUBTASK_NAMES` without changing task inclusion. Align `configs/default.yaml` strictly with actual constructors.

- [ ] **Step 6: Search for stale live identifiers**

Run:

```bash
rg -n "uav_iqa\.(model|distortion|distortion_synthesis|dataset|data_module|lightning_module|losses|metrics|callbacks)|UAVIQADataset|UAVIQADataModule|UAVIQALightningModule|DataSynthesisPipeline|DatasetFormat|use_fab|use_cbam" src scripts tests configs
```

Expected: only temporary old modules/equivalence test remain until Task 9.

- [ ] **Step 7: Verify composition roots**

Run:

```bash
pytest tests/test_experiment_configs.py -v
python scripts/train.py --help
python scripts/distortion_synthesis.py --help
python scripts/benchmark_iqa_methods.py --help
ruff check src/ tests/ scripts/
```

Expected: PASS/exit 0.

---

### Task 9: Remove old modules and enforce boundaries

**Files:**
- Modify: `src/uav_iqa/__init__.py`, `src/uav_iqa/utils.py`, `tests/test_model_structure.py`
- Create: `tests/test_package_boundaries.py`
- Remove: nine superseded top-level modules

- [ ] **Step 1: Add failing boundary tests**

Parametrize old module names and assert `importlib.import_module()` raises `ModuleNotFoundError`. Clear loaded `uav_iqa*` modules, import root `uav_iqa`, and assert it does not load `uav_iqa.vlm.scorer`, `uav_iqa.training.module`, or `uav_iqa.data.synthesis`.

- [ ] **Step 2: Verify RED**

Run `pytest tests/test_package_boundaries.py -v`.

Expected: old-module cases fail because paths exist.

- [ ] **Step 3: Minimize root facade**

Replace root contents with:

```python
"""UAV-Embodied-IQA research package."""

__all__: list[str] = []
```

- [ ] **Step 4: Remove moved utility and transition tests**

Delete `utils.load_flat_samples` after all callers migrate. Remove temporary old/new equivalence tests after they passed; retain new behavioral tests.

- [ ] **Step 5: Delete superseded modules**

Delete exactly `model.py`, `distortion.py`, `distortion_synthesis.py`, `dataset.py`, `data_module.py`, `lightning_module.py`, `losses.py`, `callbacks.py`, and `metrics.py` under `src/uav_iqa/`.

- [ ] **Step 6: Verify boundaries and focused regressions**

Run:

```bash
pytest tests/test_package_boundaries.py tests/test_model_structure.py tests/test_model_text.py tests/test_distortion.py tests/test_distortion_synthesis.py tests/test_sample_loading.py tests/test_dataset.py tests/test_lightning.py tests/test_experiment_configs.py -v
ruff check src/ tests/ scripts/
```

Expected: PASS except explicit GPU skip.

---

### Task 10: Update current docs and complete local verification

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `AGENTS.md`, `docs/CODEMAPS/ARCHITECTURE.md`, `docs/CODEMAPS/FILES.md`, `docs/EXPERIMENTS.md`

- [ ] **Step 1: Update current paths, class names, and verified commands**

Document five package boundaries and actual script commands. Replace stale deleted `data_synthesis.py`/`vlm_annotate.py`, old imports, and callback paths. Do not rewrite historical research records.

- [ ] **Step 2: Search for stale references**

Run repository search across current docs/source/tests/configs for old modules/classes/keys and stale script names. Expected: zero live references; historical records may remain and must be reported explicitly.

- [ ] **Step 3: Run full tests**

Run `pytest tests/ -v`.

Expected: PASS except existing marker-driven skips.

- [ ] **Step 4: Measure changed-package coverage**

Run focused tests with coverage over `models`, `distortions`, `data`, `training`, and `evaluation`, `--cov-report=term-missing`.

Expected: aggregate changed-package coverage ≥80%. Add behavioral tests for uncovered real branches; do not exclude code to inflate coverage.

- [ ] **Step 5: Run lint/import/CLI smoke**

Run:

```bash
ruff check src/ tests/ scripts/
python -c "import uav_iqa; from uav_iqa.models import UAVIQANet; from uav_iqa.data import UAVQualityDataset, UAVQualityDataModule; from uav_iqa.training import UAVQualityTrainingModule; from uav_iqa.distortions import UAVDistortionPipeline"
python scripts/train.py --help
python scripts/distortion_synthesis.py --help
python scripts/benchmark_iqa_methods.py --help
```

Expected: all exit 0.

- [ ] **Step 6: Inspect scope**

Run `git status --short`, `git diff --check`, `git diff --stat`, and explicit diffs for all protected paper/refine-log paths. Expected: protected files contain only pre-existing changes.

- [ ] **Step 7: Run mandatory parallel reviews**

Dispatch `ecc:code-reviewer`, `ecc:python-reviewer`, and `ecc:mle-reviewer`. Fix confirmed CRITICAL/HIGH findings and rerun proving commands; avoid unrelated cleanup.

---

### Task 11: Synchronize safely and verify remote CPU behavior

**Surface:** `c171_vm:/usr/storage/xjp/projects/embodied-uav-iqa/`

- [ ] **Step 1: Re-check remote safety**

Run:

```bash
ssh c171_vm 'cd /usr/storage/xjp/projects/embodied-uav-iqa && git branch --show-current && git rev-parse HEAD && git status --short'
```

Expected: `main`, expected base/authorized commit, clean status. Stop on dirtiness or unexpected drift; never reset/clean/stash remote work.

- [ ] **Step 2: Build a complete refactor-only binary patch**

Create `/tmp/uav_iqa_structure_refactor.patch` in two parts:

1. append `git diff --binary 5496117 -- <explicit tracked refactor paths>` for modified and deleted tracked files;
2. for every new untracked refactor file reported by `git status --short`, append `git diff --no-index --binary /dev/null "$path"` and accept exit code `1` as the expected “files differ” result.

The explicit path set may contain only `src/uav_iqa/`, `tests/`, `scripts/`, `configs/`, the current architecture/command docs, and this refactor's spec/plan. Exclude `PAPER_PLAN.md` and all `refine-logs/` paths. Inspect patch headers with `rg '^diff --git '` and require every intended new package file to appear exactly once. Verify the completed patch against a clean temporary worktree at commit `5496117` with `git apply --check`; remove that temporary worktree afterward without affecting the current worktree.

- [ ] **Step 3: Dry-check remotely**

Pipe patch to remote `git apply --check -` from the repository. Expected: exit 0; stop rather than force on failure.

- [ ] **Step 4: Apply verified patch**

Pipe the same patch to remote `git apply -`. Do not commit, pull, reset, or start GPU work.

- [ ] **Step 5: Run remote focused CPU tests**

Run focused tests for model structure, distortions, synthesis, samples, dataset, Lightning, configs, and package boundaries. Expected: PASS except explicit GPU skips.

- [ ] **Step 6: Run remote full verification and CPU forward**

Run:

```bash
ssh c171_vm 'cd /usr/storage/xjp/projects/embodied-uav-iqa && pytest tests/ -v && ruff check src/ tests/ scripts/ && python -c "import torch; from uav_iqa.models import UAVIQANet; model = UAVIQANet(use_frequency_encoder=False, use_spatial_attention=False, use_text_encoder=False); model.eval(); output = model(torch.randn(1, 3, 256, 256), torch.zeros(1, dtype=torch.long)); assert output.shape == (1,)"'
```

Expected: PASS. No formal training or benchmark.

- [ ] **Step 7: Report exact state**

Report changed/verified locally and remotely, exact commands, no commit, no push, and no GPU experiment. Do not claim completion with unresolved tests, lint, smoke, patch safety, or review findings.
