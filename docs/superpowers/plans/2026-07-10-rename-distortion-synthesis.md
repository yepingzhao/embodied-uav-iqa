# Rename data_synthesis → distortion_synthesis & Remove Annotate/Aggregate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `data_synthesis.py` 重命名为 `distortion_synthesis.py`，移除不属于失真合成的 VLM 标注和分数聚合逻辑。

**Architecture:** 库文件 (`src/uav_iqa/distortion_synthesis.py`) 只保留失真注入；CLI (`scripts/distortion_synthesis.py`) 只保留 `inject` 子命令；VLM 打分由 `scripts/inference.py` 独立处理。

**Tech Stack:** Python, no new dependencies

**Importers/callers:** `src/uav_iqa/__init__.py:22`, `scripts/data_synthesis.py:34`, `tests/test_data_synthesis.py:11`

**Affected API:**
- `DatasetFormat`, `create_pipeline` — 不变
- 类重命名: `DataSynthesisPipeline` → 不变（保持兼容）
- 删除: `annotate_scores`, `aggregate_scores`, `merge_sidecar_scores`, `run_full`, `_parse_steps`

## Global Constraints

- 现有 `inject` CLI 接口不变
- 不改动 `DatasetFormat` ABC 接口
- 不改动 `inject_distortions` 方法签名
- 删除 `scripts/vlm_annotate.py`（已标记 DEPRECATED）

---

### Task 1: 从库中删除 annotate/aggregate/merge 方法

**Files:**
- Modify: `src/uav_iqa/data_synthesis.py`

- [ ] **Step 1: 删除 L460-853（annotate → _parse_steps）并更新 docstring**

删除 `annotate_scores`（L460）到 `_parse_steps` 结束（L853）。

同时更新文件头 docstring（L3-6）从：
```python
Provides:
- DatasetFormat: minimal abstract base for extract step
- AirCopBenchFormat / GenericImageDirFormat: concrete format classes
- DataSynthesisPipeline: orchestrates inject/annotate/aggregate steps
- create_pipeline: convenience factory function
```
改为：
```python
Provides:
- DatasetFormat: minimal abstract base for extract step
- AirCopBenchFormat / GenericImageDirFormat: concrete format classes
- DataSynthesisPipeline: multi-UAV distortion injection
- create_pipeline: convenience factory function
```

- [ ] **Step 2: 运行测试确认 inject 功能未被破坏**

```bash
uv run pytest tests/test_data_synthesis.py -v 2>&1
```
预期：inject/format/registry 测试通过，`TestPipelineStepsParsing` 会失败（`_parse_steps` 已删除）。

### Task 2: 更新 CLI 脚本

**Files:**
- Modify: `scripts/data_synthesis.py`

- [ ] **Step 1: 删除 annotate/aggregate/merge/all 相关代码**

1. 替换 L1-37 为精简版（删除 annotate/aggregate 相关描述）
2. 删除 `_add_scorer_args`（L54-78）、`_make_scorer`（L84-94）
3. 删除 `cmd_annotate`、`cmd_aggregate`、`cmd_merge`、`cmd_all`
4. 删除 annotate/aggregate/merge/all 子命令定义
5. `main()` 中只保留 inject 子命令

### Task 3: 重命名文件并删除废弃脚本

**Files:**
- Rename: `src/uav_iqa/data_synthesis.py` → `src/uav_iqa/distortion_synthesis.py`
- Rename: `scripts/data_synthesis.py` → `scripts/distortion_synthesis.py`
- Rename: `tests/test_data_synthesis.py` → `tests/test_distortion_synthesis.py`
- Delete: `scripts/vlm_annotate.py`

- [ ] **Step 1: 执行 git mv 和 git rm**

```bash
cd /usr/storage/xjp/projects/embodied-uav-iqa
git mv src/uav_iqa/data_synthesis.py src/uav_iqa/distortion_synthesis.py
git mv scripts/data_synthesis.py scripts/distortion_synthesis.py
git mv tests/test_data_synthesis.py tests/test_distortion_synthesis.py
git rm scripts/vlm_annotate.py
```

### Task 4: 更新 __init__.py imports

**Files:**
- Modify: `src/uav_iqa/__init__.py:22-26`

- [ ] **Step 1: 更新 import 路径**

```python
# before
from .data_synthesis import (
    DatasetFormat,
    DataSynthesisPipeline,
    create_pipeline,
)
# after
from .distortion_synthesis import (
    DatasetFormat,
    DataSynthesisPipeline,
    create_pipeline,
)
```

### Task 5: 更新测试文件

**Files:**
- Modify: `tests/test_distortion_synthesis.py`

- [ ] **Step 1: 更新 import 和删除废弃测试**

1. 更新 import: `from uav_iqa.distortion_synthesis import`
2. 删除 `TestPipelineStepsParsing` 类（L206-221）— `_parse_steps` 已删除

### Task 6: 验证和提交

- [ ] **Step 1: 运行全部测试**

```bash
uv run pytest tests/test_distortion_synthesis.py tests/test_inference_phase*.py -v 2>&1
```
预期：全部通过

- [ ] **Step 2: 检查残留引用**

```bash
grep -r "from uav_iqa.data_synthesis\|import data_synthesis" src/ scripts/ tests/ --include="*.py" | grep -v __pycache__
```
预期：无输出

- [ ] **Step 3: 检查 CLI 正常工作**

```bash
uv run python scripts/distortion_synthesis.py --help
uv run python scripts/distortion_synthesis.py inject --help
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename data_synthesis to distortion_synthesis, remove annotate/aggregate

Co-Authored-By: Claude <noreply@anthropic.com>"
```
