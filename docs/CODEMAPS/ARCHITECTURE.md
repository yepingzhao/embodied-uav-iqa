# Architecture Codemap

## End-to-end flow

```text
AirCopBench / image directory
        │
        ▼
data.adapters + data.synthesis ───────► processed flat JSON entries
        │                                      │
        │                                      ├────► data.dataset/datamodule
        │                                      │            │
        │                                      │            ▼
        │                                      │     training.module + models
        │                                      │            │
        │                                      │            ▼
        │                                      │       evaluation.iqa
        │                                      │
        │                                      └────► vlm.annotator + vlm.scorer
        │                                                   │
        │                                                   ▼
        │                                            evaluation.text
        │                                                   │
        └──────────────────────────────────────────── enriched VLM scores
```

## Layer ownership

- `domain` owns task names, taxonomy normalization, dataset-specific parsing,
  and real-score lookup.  It contains no Torch, Lightning, or VLM dependency.
- `distortions` owns image degradation algorithms.  Synthesis uses the
  distortion pipeline but distortions never read datasets.
- `data` owns the flat processed-entry schema.  Adapters and synthesis write
  it; datasets, baseline projections, and datamodules read it.
- `models` owns only neural-network components.  Its sole domain dependency is
  the number of subtasks.
- `training` composes model outputs with MSE, ListMLE, and cross-task losses.
- `evaluation` is framework-independent: `iqa.py` evaluates numeric quality
  predictions and `text.py` evaluates VLM-generated descriptions.
- `vlm` owns model registry/backends, score generation, VQA indexing, and
  resumable batch annotation.  It consumes `evaluation.text` rather than
  implementing metrics itself.

## Composition roots

`scripts/train.py`, `scripts/distortion_synthesis.py`, benchmark scripts, and
the YAML experiment configs are composition roots.  They may combine packages
that library modules otherwise keep independent.

The standard workflow is:

```text
extract → inject → annotate → aggregate → train → benchmark
```

The annotation step may use `BatchAnnotator` directly or the multi-GPU
`inference/` framework.  Both invoke `VLMScorer`; neither owns text metrics or
dataset parsing.
