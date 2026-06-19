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

### Documentation Updated

| File | Change |
|------|--------|
| `CLAUDE.md` | Removed `load_yaml_config` from project structure listing |

### Impact

- Files modified: 13
- Dependencies removed: 5
- Lines of code removed: ~20
- Dependency count reduction: 5 packages from install requirements

### Testing

- All 10 existing tests passing
- Python syntax verified on all modified files
- No functional changes — only dead code/import cleanup
