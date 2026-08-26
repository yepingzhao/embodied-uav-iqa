# File Tree Codemap

## Repository layout

```text
embodied-uav-iqa/
├── src/uav_iqa/
│   ├── domain/          # task taxonomy and AirCopBench domain parsing
│   ├── models/          # UAVIQANet and independent neural-network components
│   ├── distortions/     # UAV-specific and generic degradation models
│   ├── data/            # sample contract, adapters, synthesis, dataset, datamodule
│   ├── training/        # Lightning module, losses, callbacks
│   ├── evaluation/      # IQA and text-similarity metrics
│   ├── vlm/             # VLM scorer, backends, VQA index, batch annotator
│   ├── baselines/       # existing IQA baselines
│   ├── inference/       # multi-GPU offline inference framework
│   └── utils.py         # small shared logging/image/parameter helpers
├── scripts/             # CLI composition roots
├── configs/             # default template and experiments r013–r024c
├── tests/               # unit, package-boundary, and inference tests
└── docs/                # research context, executable experiments, codemaps
```

## Main entrypoints

| Entry point | Purpose |
|---|---|
| `scripts/train.py` | LightningCLI training, test, and prediction entry point |
| `scripts/distortion_synthesis.py` | Extract and inject distortions through `data.synthesis` |
| `scripts/benchmark_iqa_methods.py` | Benchmark existing full-/no-reference IQA methods |
| `scripts/validate_synth_real_correlation.py` | C2 synthetic-to-real correlation evaluation |
| `scripts/inference.py` | Multi-GPU VLM inference framework CLI |
| `scripts/download_models.py` | Download/validate optional local VLM weights |

## Important tests

| Test | Contract covered |
|---|---|
| `test_model_structure.py`, `test_model_text.py` | Componentized model construction and text input |
| `test_distortion.py` | Six UAV distortions, generic categories, determinism |
| `test_distortion_synthesis.py`, `test_sample_loading.py` | Adapter registry and processed-entry schema |
| `test_dataset.py`, `test_lightning.py` | Dataset/data module and training module behavior |
| `test_annotations.py` | Domain task parsing and sample identifiers |
| `test_text_metrics.py` | Evaluation text metrics and cognitive score |
| `test_batch_annotator.py` | VLM batch annotation/checkpoint behavior |
| `test_package_boundaries.py` | Removed paths and lightweight root import |
| `test_inference_phase[1-5].py` | Inference framework lifecycle and CLI |

`data/`, `outputs/`, checkpoints, and downloaded model weights are external or
generated artifacts and are intentionally not part of the source tree.
