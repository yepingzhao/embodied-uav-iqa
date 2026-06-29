# Code Deletion Log

## [2026-06-19] Dead Code Cleanup Session

### Unused Imports Removed (Type Annotations)

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/evaluate.py` | `Tuple` from `typing` import | Never used in file |
| `src/uav_iqa/dataset.py` | `Tuple, List` from `typing` import | Only `Optional` was used |
| `src/uav_iqa/distortion.py` | `Tuple` from `typing` import | Only `Optional` was used |
| `src/uav_iqa/model.py` | `Tuple` from `typing` import | Only `Optional` was used |
| `scripts/run_m1_manifest.py` | `Dict` from `typing` import | Only `List` was used |

### Unused Imports Removed (Standard Library)

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/trainer.py` | `import math` | No `math.*` usage in file |
| `src/uav_iqa/utils.py` | `import math` | No `math.*` usage in file |
| `src/uav_iqa/utils.py` | `import random` inside `set_seed()` | Redundant: `random` already imported at module level |
| `scripts/run_m3_train.py` | `import os` | No `os.*` usage (uses `Path` instead) |
| `scripts/run_m1_manifest.py` | `import os` | No `os.*` usage |

### Unused Function Import Removed

| File | Removed | Reason |
|------|---------|--------|
| `scripts/run_m3_train.py` | `load_yaml_config` from import | Imported but never called |

### Unused Function Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/utils.py` | `load_yaml_config()` | No callers anywhere in codebase |

### Unused Export Removed from Public API

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/__init__.py` | `GenericDistortions` | Not directly imported by any external consumer; only used internally in `distortion.py` via `UAVDistortionPipeline` |

### Unused Variable Removed

| File | Removed | Reason |
|------|---------|--------|
| `scripts/run_m1_manifest.py` | `DISTORTION_NAMES` LHS assignment | Assigned but never referenced |

### Dead Code Removed

| File | Change | Reason |
|------|--------|--------|
| `scripts/run_m1_aircopbench.py` | `hasattr(os.path, 'basename')` guard simplified | `os.path.basename` always exists — unnecessary conditional |
| `src/uav_iqa/distortion.py` | Redundant inner `from concurrent.futures import ProcessPoolExecutor` in `inject_directory` else-branch | Already imported at top of method |
| `src/uav_iqa/distortion.py` | Redundant `import os, cv2` in `inject_directory` | Not used in method body |

### Unused Dependencies Removed from `pyproject.toml`

| Package | Reason |
|---------|--------|
| `huggingface-hub>=0.20.0` | Not imported anywhere in codebase |
| `einops>=0.7.0` | Not imported anywhere in codebase |
| `matplotlib>=3.7.0` | Not imported anywhere in codebase |
| `seaborn>=0.12.0` | Not imported anywhere in codebase |
| `scikit-learn>=1.3.0` | Not imported anywhere in codebase |

## [2026-06-19] Dead Code Cleanup Session (Batch 2)

### Unused Function Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/utils.py` | `set_seed()` | No callers anywhere in codebase |

### Deprecated Class Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/trainer.py` | `UAVIQATrainer` class (~146 lines) | Deprecated since `UAVIQALightningModule` was introduced; no external imports |

### Unused Variable Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/distortion.py` | `var_limit` in `GenericDistortions.apply` | Assigned but never used |
| `src/uav_iqa/model.py` | `patches_padded` in `FrequencyAwareBranch.forward` | No-op `F.pad` result never used |
| `src/uav_iqa/trainer.py` | `device` in `ListMLELoss.forward` | Assigned but never used |
| `scripts/run_m2_benchmark.py` | `e` in `except Exception as e` | Unused exception variable |

### Unused Import Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/trainer.py` | `from typing import Optional` | Only `Dict` was used (also removed with UAVIQATrainer) |
| `src/uav_iqa/trainer.py` | `from .evaluate import compute_srcc, compute_plcc` | Only used by removed `UAVIQATrainer` class |
| `src/uav_iqa/trainer.py` | `import warnings` | Only used by removed `UAVIQATrainer` class |
| `scripts/run_m2_benchmark.py` | `per_distortion_metrics` from import | Imported but never called |

### F-String Without Placeholders Fixed

| File | Change |
|------|--------|
| `scripts/run_m1_inject.py` | `f"24 distortions × 5 levels = 120 variants per image"` → regular string |
| `scripts/run_m2_benchmark.py` | `f"Benchmark Complete"` → regular string |

### Public API Cleanup

| File | Change |
|------|--------|
| `src/uav_iqa/__init__.py` | Added `__all__` list for explicit re-export (silences F401) |

### Documentation Updated

| File | Change |
|------|--------|
| `CLAUDE.md` | Removed `set_seed` reference; updated `trainer.py` and `utils.py` descriptions |
| `AGENTS.md` | Updated `trainer.py` description to remove `UAVIQATrainer` |

### Impact

- Files modified: 9 (+2 doc files)
- Unused classes removed: 1 (~146 lines)
- Unused functions removed: 1
- Unused variables removed: 4
- Unused imports removed: 5
- Total lines of code removed: ~175
- F-strings fixed: 2
- All ruff checks passing
- All 16 non-slow tests passing

### Testing

- Ruff lint: All checks passed (was 24 errors, now 0)
- Python syntax: Verified on all modified files
- pytest: 16/16 fast tests passing, 1 slow test skipped
- No functional changes — only dead code/import cleanup

## [2026-06-20] Dead Code Cleanup Session (Batch 3)

### Unused Import Removed

| File | Removed | Reason |
|------|---------|--------|
| `scripts/benchmark_iqa_methods.py` | `from scipy.special import gamma` inside `brisque_dct()` | Imported but never used; function uses only numpy stats |

### Unused Variables Renamed to `_`

| File | Variable | Context |
|------|----------|---------|
| `scripts/benchmark_iqa_methods.py` | `intens` | Unpacked from `load_image_and_score()` but never used in loop |
| `scripts/benchmark_iqa_methods.py` | `can_finetune` | Unpacked from `AVAILABLE_METHODS` tuple but never used in zero-shot loop body |

### Lightning Step Unused Parameters Renamed to `_`

| File | Method | Parameter |
|------|--------|-----------|
| `src/uav_iqa/lightning_module.py` | `training_step` | `batch_idx` → `_` |
| `src/uav_iqa/lightning_module.py` | `validation_step` | `batch_idx` → `_` |
| `src/uav_iqa/lightning_module.py` | `test_step` | `batch_idx` → `_` |
| `scripts/finetune_baselines.py` | `training_step` | `batch_idx` → `_` |
| `scripts/finetune_baselines.py` | `validation_step` | `batch_idx` → `_` |

### E402 Import Moved to Top of File

| File | Import | Reason |
|------|--------|--------|
| `scripts/finetune_baselines.py` | `from uav_iqa.dataset import UAVIQADataset` | Moved from line 140 (after class def) to top of file to fix E402 |

### F-String Without Placeholders Fixed

| File | Change |
|------|--------|
| `scripts/fix_configs.py` | `f"  OK  configs/default.yaml"` → regular string `"  OK  configs/default.yaml"` |

### Impact

- Files modified: 5
- Unused imports removed: 1
- Unused variables renamed to `_`: 2
- Lightning step params renamed to `_`: 5
- E402 violations fixed: 1
- F-strings fixed: 1
- All ruff checks passing (was 2 errors, now 0)
- All 54 tests passing

### Testing

- Ruff lint: All checks passed
- Black format: All files reformatted
- pytest: 54/54 tests passing
- No functional changes — only cleanup of unused code

## [2026-06-20] Batch 4: Code Cleanup

### Unused Dependency Removed from `pyproject.toml`

| Package | Reason |
|---------|--------|
| `tqdm>=4.65.0` | Not imported anywhere in codebase (0 occurrences) |

### Unused Parameters Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/data_module.py` | `train_split`, `val_split` in `UAVIQDataModule.__init__` | Stored via `save_hyperparameters()` but never directly accessed |

### Dead Methods Removed

| File | Method | Reason |
|------|--------|--------|
| `src/uav_iqa/data_synthesis.py` | `AirCopBenchFormat.build_image_index()` (46 lines) | Defined but never called anywhere |
| `src/uav_iqa/data_synthesis.py` | `AirCopBenchFormat._extract_scene_and_frame()` (12 lines) | Only called by `build_image_index` |
| `src/uav_iqa/data_synthesis.py` | `AirCopBenchFormat.build_annotation_summary()` (24 lines) | Defined but never called anywhere |

### Unused Import Removed

| File | Removed | Reason |
|------|---------|--------|
| `src/uav_iqa/data_synthesis.py` | `import os` | No longer used after method removals |
| `src/uav_iqa/data_synthesis.py` | `import re` | No longer used after method removals |

### Impact

- Files modified: 3 (`pyproject.toml`, `data_module.py`, `data_synthesis.py`)
- Dependencies removed: 1
- Methods removed: 3 (82 lines)
- Imports removed: 2
- Total lines of code removed: ~85
- All ruff checks passing
- All 54 tests passing

### Testing

- Ruff lint: All checks passed
- pytest: 54/54 tests passing
- No functional changes — only dead code/import/dependency cleanup

## [2026-06-20] Batch 5: Code Cleanup

### Import Sorting Fixed

| File | Change |
|------|--------|
| `scripts/benchmark_iqa_methods.py` | Imports reorganized to standard order |
| `scripts/finetune_baselines.py` | Imports reorganized to standard order |
| `scripts/fix_configs.py` | Imports reorganized to standard order |
| `scripts/validate_synth_real_correlation.py` | Imports reorganized to standard order |
| `src/uav_iqa/__init__.py` | Imports reorganized to standard order |
| `src/uav_iqa/distortion.py` | Inline imports in method reorganized |
| `src/uav_iqa/lightning_module.py` | Imports reorganized to standard order |
| `tests/test_data_synthesis.py` | Imports reorganized to standard order |
| `tests/test_distortion.py` | Imports reorganized to standard order |

### Line Too Long Fixed

| File | Change |
|------|--------|
| `src/uav_iqa/data_module.py:62` | Split long warning string (105→99 chars) |

### Shebang Executability Fixed

| File | Change |
|------|--------|
| 8 script files under `scripts/` | `chmod +x` for files with `#!/usr/bin/env python3` |

### Impact

- Files modified: 10 (9 import sort + 1 line length)
- Files made executable: 8
- All ruff checks passing
- All 54 tests passing

### Testing

- Ruff lint: All checks passed
- pytest: 54/54 tests passing
- No functional changes — only cleanup of imports, formatting, and permissions

## [2026-06-25] Dead Code Cleanup Session (Batch 6)

### Unused Import Removed

| File | Removed | Reason |
|------|---------|--------|
| `scripts/smoke_test_vlm.py` | `import traceback` | Imported but never used anywhere in file |

### Multiple Imports Fixed on One Line

| File | Change | Reason |
|------|--------|--------|
| `src/uav_iqa/vlm_vla_scorer.py:988` | `import sys as _sys, importlib as _il, os as _os` → split into 3 separate imports | E401: Multiple imports on one line |

### Unused Methods Detected (Left in Place for Future Use)

| File | Method | Reason |
|------|--------|--------|
| `src/uav_iqa/batch_annotator.py:353` | `BatchAnnotator.annotate_all_splits()` | Defined but never called anywhere in codebase. Left in place as future-proof utility for batch annotation across splits. |

### Impact

- Files modified: 2
- Imports removed: 1
- Multiple imports split: 1
- Ruff lint: 0 violations (was 1 E401 + 1 F401)
- All 65 tests passing

### Clarifications on False Positives

The following vulture/ruff findings were reviewed and determined to be **not dead code**:

| Item | Reason |
|------|--------|
| `CurriculumStageCallback`, `SetupRunCallback`, `MetricsHistoryCallback`, `ResultsSavingCallback` | Lightning framework hooks — called dynamically |
| `UAVIQALightningModule` methods (`training_step`, `validation_step`, `test_step`, etc.) | Lightning framework hooks — called dynamically |
| `UAVIQDataModule` methods (`setup`, `train_dataloader`, `val_dataloader`, etc.) | Lightning framework hooks — called dynamically |
| `ALL_TASKS` and task constants in `vlm_vla_scorer.py` | Used in `tests/test_vlm_scorer.py` |
| `# noqa: F401` directives in `vlm_vla_scorer.py` | Defensive — prevent F401 in editors with different configs |
| Lambda parameters (`seq_length`, `self_m`, `infer_mode`, `num_logits_to_keep`) | Intentionally match patched method interface signatures |
| Test mock parameters (`is_quantized`, `allow_all_kernels`, `return_tensors`, etc.) | Mock interface matching — parameters must exist to match patched API even if unused in body |
| `ERA001` commented-code findings (losses.py:20, vlm_vla_scorer.py:564, tests:1257/1259) | Documentation comments, not commented-out code |
| `annotate_all_splits` in `batch_annotator.py` | Defined as new utility method for future use |

### Testing

- Ruff lint: All checks passed (0 violations)
- pytest: 65/65 tests passing (distortion + data_synthesis + lightning + batch_annotator)
- No functional changes — only dead code import cleanup and import formatting

## [2026-06-27] Batch 7: Dead Re-export Shim Removal

### Dead File Removed

| File | Reason |
|------|--------|
| `src/uav_iqa/vlm_scorer.py` (5 lines) | Backward-compatible re-export shim with **zero importers** in codebase. All consumers import directly from `uav_iqa.vlm` subpackage. Was left over from the refactor that split `vlm_vla_scorer.py` into the `vlm/` subpackage + `vla_scorer.py`. |

### Docstring Updated

| File | Change |
|------|--------|
| `src/uav_iqa/annotations.py` | `uav_iqa.vlm_scorer.VLMScorer` → `uav_iqa.vlm.scorer.VLMScorer` |

### Documentation Updated

| File | Change |
|------|--------|
| `AGENTS.md` | `vlm_scorer.py` → `vlm/` |
| `CLAUDE.md` | `vlm_scorer.py` → `vlm/` |
| `README.md` | `vlm_scorer.py` → `vlm/` |
| `docs/CODEMAPS/ARCHITECTURE.md` | `vlm_scorer.py` → `vlm/scorer.py` (2 refs) |
| `docs/CODEMAPS/FILES.md` | Replaced `vlm_scorer.py` entry with `vlm/` subpackage; updated deps |
| `docs/CODEMAPS/MODULES.md` | Replaced `vlm_scorer.py` with `vlm/` subpackage details |

### Impact

- Files deleted: 1 (5 lines)
- Documentation files updated: 6
- Total lines of code removed: 5 (plus redundant docs references)
- All ruff checks passing
- All 88 tests passing

### Testing

- Ruff lint: All checks passed (0 violations)
- pytest: 88/88 tests passing (distortion + text_metrics + data_synthesis + batch_annotator + vlm_config)
- No functional changes — only dead file removal and doc updates
