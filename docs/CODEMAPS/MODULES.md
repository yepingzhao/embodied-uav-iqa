# Module Codemap

## Package boundaries

`uav_iqa` has a deliberately lightweight root package.  Import concrete
subpackages instead of importing application components from `uav_iqa`.

```text
domain ───────┬──────► models
              ├──────► data ─────► distortions
              ├──────► evaluation
              └──────► training ─► models + evaluation

evaluation ───────────► vlm
vlm ──────────────────► inference
scripts/configs ──────► composition roots; may combine all packages
```

The arrows point from a consumer to the package it is allowed to use.  In
particular, `models`, `distortions`, and `evaluation` do not depend on
Lightning or dataset parsing.

## Core packages

| Package | Responsibility | Main API |
|---|---|---|
| `domain/` | AirCopBench task taxonomy, annotation parsing, sample IDs, real-score lookup | `SUBTASK_*`, `normalize_subtask_type()`, `build_ref_score_lookup()` |
| `models/` | UAVIQANet composition and independent model components | `UAVIQANet`, `PANFeaturePyramid`, `ConvolutionalBlockAttention`, `LogPolarFrequencyEncoder`, `QuestionTextEncoder` |
| `distortions/` | Six UAV-specific and 30 generic image distortions | `UAVDistortionPipeline`, `UAV_DISTORTION_NAMES` |
| `data/` | Processed-sample contract, adapters, synthesis, dataset, and datamodule | `DatasetAdapter`, `DistortionSynthesisPipeline`, `UAVQualityDataset`, `UAVQualityDataModule` |
| `training/` | Lightning training module, ranking/cross-task losses, callbacks | `UAVQualityTrainingModule`, `ListMLELoss`, `CrossTaskRegularization` |
| `evaluation/` | Numeric IQA metrics and text-comparison metrics | `evaluate_iqa()`, `compute_cognitive_score()` |
| `vlm/` | VLM configuration, backend routing, scoring, VQA prompt index, batch annotation | `VLMScorer`, `BatchAnnotator`, `VQAIndex` |
| `baselines/` | Existing IQA method evaluation and fine-tuning | baseline evaluator and data module |
| `inference/` | Multi-GPU offline VLM inference framework | `InferenceEngine`, `VLMExecutor`, `JsonStorage` |

## Internal organization

```text
models/
  uav_iqa_net.py     # composes backbone, spatial/frequency/text modules and heads
  spatial.py         # PAN feature pyramid and spatial attention
  frequency.py       # log-polar frequency encoder and feature gate
  text.py            # character-level question encoder
  heads.py           # task-conditioned and shared regressors

distortions/
  base.py            # common image helpers and BaseDistortion
  uav.py             # UAV-specific distortions
  generic.py         # Albumentations-backed generic distortions
  pipeline.py        # deterministic group-level distortion pipeline

data/
  samples.py         # processed JSON loading, validation, baseline projection
  adapters.py        # adapter registry, AirCopBench and image-directory adapters
  synthesis.py       # extract/inject orchestration
  dataset.py         # flat processed dataset and collate function
  datamodule.py      # Lightning data loading and filtering

training/
  module.py          # optimization, loss composition, evaluation logging
  losses.py          # ListMLE and cross-task regularization
  callbacks.py       # setup, history, and results callbacks

evaluation/
  iqa.py             # SRCC, PLCC, RMSE, Kendall tau, grouped IQA metrics
  text.py            # BLEU, ROUGE-L, CIDEr, cognitive-score aggregation

vlm/
  config.py          # model registry and family-specific configuration
  backends.py        # vLLM and Transformers execution helpers
  scorer.py          # description/VQA comparison scoring
  annotator.py       # BatchAnnotator checkpoint/resume service
  vqa_index.py       # filesystem-backed VQA prompt lookup
```

## Removed paths

The following modules intentionally have no compatibility aliases:

```text
uav_iqa.model                    uav_iqa.distortion
uav_iqa.distortion_synthesis     uav_iqa.dataset
uav_iqa.data_module              uav_iqa.lightning_module
uav_iqa.losses                   uav_iqa.callbacks
uav_iqa.metrics                  uav_iqa.annotations
uav_iqa.batch_annotator          uav_iqa.text_metrics
uav_iqa.evaluation.metrics
```

`tests/test_package_boundaries.py` enforces this rule and also verifies that
importing `uav_iqa` itself does not pull in VLM, training, or synthesis code.
