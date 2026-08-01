# Model, Data, and Training Structure Refactor Design

## Status

Approved for implementation planning.

## Scope

This design covers the first refactor phase of UAV-Embodied-IQA:

- core `UAVIQANet` implementation;
- distortion definitions and orchestration;
- processed-sample loading and validation;
- distortion dataset synthesis;
- PyTorch datasets and Lightning data modules;
- losses, callbacks, and the Lightning training module;
- IQA evaluation metrics;
- affected scripts, experiment configurations, tests, and command documentation.

This phase does not reorganize `vlm/`, `inference/`, or `baselines/`. Their imports may change only where required by moved symbols. The model's numerical behavior, distortion algorithms, processed JSON fields, training batch fields, loss composition, and metric formulas must remain unchanged.

The paper-level method name **UAV-IQANet** and implementation class `UAVIQANet` remain stable. Compatibility with old import paths, YAML files, CLI commands, or checkpoints is not required.

The authoritative execution repository is `c171_vm:/usr/storage/xjp/projects/embodied-uav-iqa/`. At design time, its clean `main` branch and local `main` both point to commit `5496117`. Local uncommitted paper and experiment-planning files are unrelated and must not be modified.

## Goals

1. Give each package one clear responsibility.
2. Replace misleading or ambiguous implementation names with names that describe actual behavior.
3. Remove the root package's eager, cross-subsystem import graph.
4. Establish one owner for the processed-sample contract used by synthesis and training.
5. Split large implementation files into cohesive units without changing research behavior.
6. Make future model-architecture experiments local to `models/` and future data-format work local to `data/`.
7. Preserve deterministic and numerical behavior through regression tests rather than compatibility shims.

## Non-goals

- changing the UAV-IQANet architecture or introducing new research components;
- changing tensor shapes, forward semantics, losses, score formulas, or JSON fields;
- redesigning the VLM annotation or offline inference frameworks;
- reorganizing baseline evaluation or fine-tuning;
- loading old checkpoints or old experiment YAML files after the refactor;
- providing deprecated aliases, re-export shims, or migration utilities;
- running formal GPU experiments as part of structural verification;
- modifying research plans, paper claims, or experiment results.

## Target package structure

```text
src/uav_iqa/
  __init__.py
  annotations.py
  utils.py

  models/
    __init__.py
    uav_iqa_net.py
    spatial.py
    frequency.py
    text.py
    heads.py

  distortions/
    __init__.py
    base.py
    uav.py
    generic.py
    pipeline.py

  data/
    __init__.py
    samples.py
    dataset.py
    datamodule.py
    adapters.py
    synthesis.py

  training/
    __init__.py
    module.py
    losses.py
    callbacks.py

  evaluation/
    __init__.py
    metrics.py

  baselines/
  inference/
  vlm/
```

The following top-level implementation files are removed after their callers migrate:

- `model.py`;
- `distortion.py`;
- `distortion_synthesis.py`;
- `dataset.py`;
- `data_module.py`;
- `lightning_module.py`;
- `losses.py`;
- `callbacks.py`;
- `metrics.py`.

`annotations.py` remains at the package root because its task taxonomy and AirCopBench parsing are shared domain definitions, not training-data infrastructure. `utils.py` remains temporarily, but processed-sample loading and validation move out of it.

## Dependency rules

- `models` may depend on PyTorch, `timm`, and domain constants from `annotations` only.
- `distortions` may depend on image-processing libraries but not on datasets, Lightning, VLM, or training.
- `data` may depend on `annotations`, `distortions`, PyTorch data APIs, and Lightning data APIs.
- `training` may depend on `models`, `evaluation`, and Lightning. It must not own dataset parsing.
- `evaluation` may depend on numerical/statistical libraries only. It must not depend on models, data modules, or Lightning.
- scripts and experiment configuration are composition roots and may import across these packages.
- `baselines`, `inference`, and `vlm` may consume the new packages but must not become dependencies of them.

These rules prevent circular imports and keep optional VLM dependencies out of normal model and data imports.

## Naming standard

Paper and domain abbreviations that are established terms remain unchanged: `UAVIQANet`, UAV-IQANet, IQA, VLM, SRCC, PLCC, RMSE, and PAN.

Implementation names change as follows:

| Current name | New name | Reason |
|---|---|---|
| `PanetFPN` | `PANFeaturePyramid` | Correct acronym casing and describe the component. |
| `CBAM` | `ConvolutionalBlockAttention` | Avoid an unexplained implementation abbreviation. |
| `FrequencyAwareBranch` | `LogPolarFrequencyEncoder` | Name the actual transformation and role. |
| `CrossAttentionGate` | `FrequencyFeatureGate` | The current operation is gating, not cross-attention. |
| `TaskConditionedHead` | `TaskConditionedRegressor` | State the output role. |
| `UAVIQATextEncoder` | `QuestionTextEncoder` | The input is question text, not generic UAV-IQA text. |
| `UAVIQADataset` | `UAVQualityDataset` | Improve readability without changing the paper brand. |
| `UAVIQADataModule` | `UAVQualityDataModule` | Match the dataset name and responsibility. |
| `UAVIQALightningModule` | `UAVQualityTrainingModule` | Identify the training concern instead of the framework wrapper only. |
| `DatasetFormat` | `DatasetAdapter` | Implementations adapt dataset discovery and layout. |
| `AirCopBenchFormat` | `AirCopBenchAdapter` | Match the interface role. |
| `GenericImageDirFormat` | `ImageDirectoryAdapter` | Remove vague “generic” naming. |
| `DataSynthesisPipeline` | `DistortionSynthesisPipeline` | Distinguish it from VLM annotation or other synthesis. |

Local variables use domain names such as `images`, `spatial_features`, `frequency_features`, `fused_features`, and `task_ids`. Short mathematical names are acceptable only in compact formulas or established metric code.

Configuration keys mirror the new component vocabulary. For example, `use_fab` becomes `use_frequency_encoder`. Existing `use_task_conditioning` remains meaningful. All active experiment YAML files and script overrides must be updated atomically; no old-key aliases are added.

## Model package design

### `models/spatial.py`

Owns `PANFeaturePyramid` and `ConvolutionalBlockAttention`. These classes accept and return tensors or feature lists only. They do not select backbones or know about tasks, text, losses, or Lightning.

### `models/frequency.py`

Owns `LogPolarFrequencyEncoder` and `FrequencyFeatureGate`. Existing FFT, log-polar binning, CNN layers, dimensions, and gating calculations remain identical. Registered buffer names should remain unchanged unless the equivalence test explicitly maps them before loading state.

### `models/text.py`

Owns `QuestionTextEncoder`. Character vocabulary, token IDs, padding, convolution kernels, pooling, and projection behavior remain identical.

### `models/heads.py`

Owns `TaskConditionedRegressor` and the shared-regression alternative. The task-conditioned implementation keeps the same embedding, FiLM projections, hidden layer, sigmoid output, and default dimensions.

The currently unused checkpoint migration method is removed. Old checkpoints are outside scope, and keeping migration code would contradict the explicit break in compatibility.

### `models/uav_iqa_net.py`

Owns `UAVIQANet` and model composition only:

1. validate and create the `timm` backbone;
2. select the last three feature stages;
3. compose spatial, frequency, text, and regression components;
4. freeze requested backbone stages;
5. expose feature extraction, normal forward, and all-task forward methods.

The class keeps its public name and forward semantics. It must preserve:

- single-image and multi-UAV inputs;
- mean pooling across the UAV dimension;
- optional frequency, attention, task-conditioning, and text branches;
- null text embedding behavior;
- pretrained-backbone warning and random-initialization fallback;
- task-ID default behavior;
- score range and output shapes.

## Distortion package design

### `distortions/base.py`

Defines the smallest shared distortion protocol or abstract interface and shared result/type definitions. It must not contain dataset orchestration.

### `distortions/uav.py`

Contains the six UAV-specific distortion implementations. Existing algorithm names visible in generated metadata remain unchanged so processed samples and result grouping do not change.

### `distortions/generic.py`

Contains Albumentations-backed generic distortion application and any generic image helpers needed only by that implementation.

### `distortions/pipeline.py`

Owns distortion registry, category metadata, severity naming, lookup, and `UAVDistortionPipeline`. Registry contents, distortion order where observable, deterministic seeding behavior, and generated names must remain identical.

## Data package design

### `data/samples.py`

Becomes the single owner of the processed-sample contract. It contains:

- `load_processed_entries` for reading and validating the flat JSON entries written by synthesis;
- `load_benchmark_samples` replacing `utils.load_flat_samples` for the current baselines and synthetic-to-real validation scripts;
- manifest/sample validation currently associated with the training dataset.

The JSON schema does not change in this phase. Synthesis writes it and training reads it through the same module-level contract. `load_benchmark_samples` remains a projection of that current schema into the fields expected by baseline code; it is not an old-path compatibility shim. Its existing field mapping and score-default behavior remain unchanged in this structural phase.

### `data/adapters.py`

Contains `DatasetAdapter`, its registry, `AirCopBenchAdapter`, and `ImageDirectoryAdapter`. Adapter responsibilities remain limited to dataset-specific image discovery and excluded-directory policy.

### `data/synthesis.py`

Contains `DistortionSynthesisPipeline` and its factory. It orchestrates adapters, annotation helpers, distortions, output paths, multiprocessing, and atomic JSON writes. It does not define distortion algorithms or training datasets.

### `data/dataset.py`

Contains `UAVQualityDataset` and collate behavior. It consumes processed samples, resolves image paths, applies transforms, and returns the existing batch fields. Field names and tensor shapes remain unchanged.

### `data/datamodule.py`

Contains `UAVQualityDataModule`, split loading, filters, and dataloaders. Task and distortion filtering behavior remains unchanged. Active YAML configurations must use semantic task names if that is the current runtime contract; invalid numeric or stale filter values must be corrected rather than accepted through aliases.

## Training and evaluation design

### `training/module.py`

Contains `UAVQualityTrainingModule`. It composes `UAVIQANet`, losses, metrics, and optimizer/scheduler behavior. Constructor parameter names change only where required by the approved naming standard. All active configurations move to the new class path and keys.

Training behavior remains identical:

- MSE plus weighted ListMLE and cross-task regularization;
- shared feature extraction for prediction and all-task scoring;
- DDP gathering behavior;
- per-task and per-distortion evaluation;
- optimizer and scheduler parameters;
- logging metric names unless a verified inconsistency requires correction.

Model text-encoder options must be represented consistently between `UAVIQANet` and the training module. The refactor must not silently rely on the model default when the training configuration is expected to control an option.

### `training/losses.py` and `training/callbacks.py`

Move existing behavior with descriptive imports and no algorithm changes.

### `evaluation/metrics.py`

Moves IQA metric calculations and grouped metric helpers. Metric names, formulas, edge-case behavior, and output dictionaries remain unchanged.

## Public API and composition roots

The package-root `uav_iqa/__init__.py` becomes a minimal facade. It must not eagerly import VLM, Lightning, OpenCV-heavy synthesis, or optional model backends. Public imports should come from their owning subpackages, for example:

```python
from uav_iqa.models import UAVIQANet
from uav_iqa.data import UAVQualityDataset, UAVQualityDataModule
from uav_iqa.training import UAVQualityTrainingModule
from uav_iqa.distortions import UAVDistortionPipeline
```

Subpackage `__init__.py` files export only their intentional public types. Internal components may be imported from their module when tests or experiments genuinely need them.

`scripts/train.py` remains the training composition root using LightningCLI, with the new training and data class paths. Distortion synthesis and visualization scripts import the new data/distortion packages. All active experiment YAML files use the new class names, class paths, and configuration keys.

Because compatibility is intentionally broken, removed root imports and old module paths must fail rather than being preserved through aliases.

## Error handling

The refactor preserves fail-fast boundary behavior:

- unknown dataset adapter: explicit `ValueError` listing available adapters;
- missing or invalid processed-sample field: explicit validation error with sample context;
- unknown distortion: explicit error rather than silent fallback;
- unavailable `timm` backbone: explicit error;
- missing batch inputs: explicit `KeyError` naming the required field.

No broad exception swallowing or default data substitution is added. The existing pretrained-weight load failure path remains a logged warning followed by random initialization because changing it would alter established runtime behavior.

## Migration sequence

1. Add characterization tests for public numerical and data behavior before moving code.
2. Create `models/`, move and rename components, then update model tests and direct callers.
3. Create `distortions/`, move algorithms and orchestration, then run deterministic distortion tests.
4. Create `data/`, centralize the sample contract, and migrate synthesis, dataset, and data module.
5. Create `evaluation/` and `training/`, migrate metrics, losses, callbacks, and the Lightning module.
6. Update scripts, active experiment YAML files, package exports, and direct imports in untouched subpackages.
7. Remove superseded top-level files only after repository-wide searches show no remaining imports.
8. Update current command and architecture documentation. Historical experiment records are not rewritten.
9. Run local verification, synchronize through the normal Git workflow only when explicitly requested, and run remote CPU verification in the authoritative repository.

Each stage must leave imports resolvable and its focused tests passing before the next stage starts.

## Verification strategy

### Characterization tests

Before moving implementation, add tests that capture the behavior being preserved:

- fixed-seed `UAVIQANet` output shapes and forward variants;
- old/new component state transfer and output equivalence during the transition;
- single-image and multi-UAV feature pooling;
- text-present and null-text paths;
- task-conditioned and shared-head paths;
- deterministic distortion outputs and stable distortion metadata;
- synthesis output sample fields and deterministic IDs;
- dataset output keys and tensor shapes;
- data-module filter behavior;
- one training and validation batch smoke test;
- metric output keys and values for fixed arrays.

Once old modules are removed, transition-only comparisons may be replaced by golden fixtures or direct formula assertions so tests do not retain obsolete architecture.

### Commands

Focused tests run after each package migration. Final local verification must include:

```bash
pytest tests/ -v
ruff check src/ tests/ scripts/
```

Coverage should be measured for changed packages. New or moved behavior must retain at least the repository's 80% target where practical; uncovered compatibility code is not acceptable because compatibility code is out of scope.

Remote verification on `c171_vm:/usr/storage/xjp/projects/embodied-uav-iqa/` must be read/write only when implementation is authorized and synchronized. The proving run is limited to import checks, focused CPU tests, and a CPU smoke forward/data batch. Formal GPU training and benchmark experiments require a separate request.

## Acceptance criteria

The refactor is complete when all of the following are true:

1. The target packages exist and superseded top-level implementation modules are removed.
2. Repository-wide search finds no imports from removed paths and no old renamed class or configuration identifiers except historical documentation explicitly marked as historical.
3. `UAVIQANet` retains its paper brand and observable forward behavior.
4. Distortion algorithms, names, deterministic seeding, and processed JSON fields are unchanged.
5. The processed-sample contract is owned by `data/samples.py` and shared by synthesis and training readers.
6. Active experiment YAML files resolve through LightningCLI with the new class paths and keys.
7. Package-root import does not eagerly load VLM or Lightning subsystems.
8. Focused and full tests pass, `ruff` passes, and changed-package coverage is reported.
9. Remote CPU smoke verification passes in the authoritative repository.
10. No local paper-plan or experiment-result file unrelated to this refactor is modified.
