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
