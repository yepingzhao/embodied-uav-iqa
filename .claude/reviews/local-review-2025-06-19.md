# Code Review: Local Uncommitted Changes

**Reviewed**: 2026-06-19
**Branch**: feat/dev (first commit)
**Decision**: APPROVE with comments

## Summary

Initial commit of a well-structured UAV-Embodied-IQA research codebase (~1,582 lines of core source, 16 tests all passing). Architecture is clean: distortion pipeline → dataset → model → Lightning training. No security issues, no lint errors. All 16 tests pass. Some medium-level code quality items noted below.

## Findings

### CRITICAL
None

### HIGH
None

### MEDIUM

| # | File | Line | Issue |
|---|------|------|-------|
| M1 | `src/uav_iqa/model.py` | 129-135 | `FrequencyAwareBranch._log_polar` uses nested Python for-loops over B and C dimensions. For typical batch sizes this is slow but functional. Consider vectorized `scatter_add` for production use. |
| M2 | `src/uav_iqa/model.py` | 272-279 | `UAVIQANet.__init__` silently falls back to random init when pretrained weights fail — only prints a warning. For reproducibility, consider making this a hard error or logging more prominently. |
| M3 | `src/uav_iqa/lightning_model.py` | 84-89 | `training_step` creates a `torch.tensor(...)` for each unique distortion per batch via list comprehension. Many small tensor allocations per step. Consider computing the mask differently. |
| M4 | `src/uav_iqa/lightning_data.py` | 14-21 | `UAV_DISTORTIONS` set is duplicated from `distortion.py`. Single source of truth preferred — import from distortion module or define in a shared constants location. |
| M5 | `src/uav_iqa/distortion.py` | 344 | Comment says "18 generic distortion types" but `CATEGORIES` dict has 27 entries. Comment is outdated/misleading. |
| M6 | `src/uav_iqa/dataset.py` | 92-95 | `_augment` creates a new `transforms.Compose` on every call. Should be created once in `__init__`. |
| M7 | `tests/test_distortion.py` | — | No tests for `GenericDistortions` class (27 distortion types). Only UAV-specific distortions have test coverage. |
| M8 | `scripts/run_m2_benchmark.py` | 206-207 | `ahiq` and `topiq_fr` categorized as "FR" in `AVAILABLE_METHODS` but their methods live on `NRMethods` class — namespace confusion. |

### LOW

| # | File | Line | Issue |
|---|------|------|-------|
| L1 | `src/uav_iqa/distortion.py` | 521-522 | `_inject_one_image_mp` re-imports `cv2` and `pathlib` already imported at module level. |
| L2 | `src/uav_iqa/model.py` | 352-379 | `forward_all_tasks` duplicates ~20 lines from `forward`. Could call a shared helper. |
| L3 | `src/uav_iqa/model.py` | 320-323 | `_freeze_backbone_stages` assumes `stages.X` naming pattern — specific to MobileNetV4. No validation. |
| L4 | `src/uav_iqa/evaluate.py` | 83-91 | `compute_metrics` is a thin alias for `evaluate_iqa` with lowercase keys — redundant API surface. |
| L5 | `src/uav_iqa/utils.py` | 1-4 | Missing type annotations. |
| L6 | `src/uav_iqa/__init__.py` | 1-17 | `__all__` defined before imports — works but unconventional order. |
| L7 | `src/uav_iqa/lightning_model.py` | 118-132 | `on_validation_epoch_end` accumulates all predictions in memory — could OOM on large val sets. |
| L8 | `scripts/run_m2_benchmark.py` | 267-284 | FR methods silently return 0.0 on ImportError or missing ref — masks failures that should be surfaced. |

## Validation Results

| Check | Result |
|---|---|
| Ruff lint | ✅ Pass (0 issues) |
| Bandit security | ✅ Pass (0 issues) |
| Pytest (16 tests) | ✅ Pass (16/16 in 6.79s) |
| Type check | ⏭️ Skipped (no mypy/pyright configured) |

## Files Reviewed

| File | Lines | Type |
|------|-------|------|
| `src/uav_iqa/__init__.py` | 33 | Source — Public API |
| `src/uav_iqa/callbacks.py` | 75 | Source — Training callbacks |
| `src/uav_iqa/cli.py` | 97 | Source — CLI entry point |
| `src/uav_iqa/dataset.py` | 134 | Source — Data loading |
| `src/uav_iqa/distortion.py` | 665 | Source — Distortion models |
| `src/uav_iqa/evaluate.py` | 109 | Source — Metrics |
| `src/uav_iqa/lightning_data.py` | 152 | Source — Lightning DataModule |
| `src/uav_iqa/lightning_model.py` | 224 | Source — Lightning Module |
| `src/uav_iqa/model.py` | 379 | Source — Model architecture |
| `src/uav_iqa/trainer.py` | 32 | Source — Loss functions |
| `src/uav_iqa/utils.py` | 4 | Source — Utilities |
| `tests/test_distortion.py` | 104 | Test |
| `tests/test_lightning.py` | 94 | Test |
| `scripts/run_m3_train.py` | 98 | Script |
| `scripts/run_m2_benchmark.py` | 319 | Script |
| `pyproject.toml` | 63 | Config |
| `configs/default.yaml` | 44 | Config |

## Notes

- This is a research codebase initial commit. The code quality is above average for research code — clean structure, consistent naming, good docstrings.
- All 6 UAV-specific distortion models + pipeline are well-tested with shape/dtype/determinism checks.
- The Lightning integration follows best practices (save_hyperparameters, configure_optimizers with schedulers, proper callback usage).
- No security vulnerabilities found — no hardcoded secrets, no network-facing endpoints, no subprocess calls.
- The `FrequencyAwareBranch._log_polar` loop (M1) is the most impactful medium issue — it limits batch size for the FAB component. For research-scale training this is acceptable.
