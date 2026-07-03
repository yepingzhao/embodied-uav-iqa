"""Tests for Phase 4: engine, validator, metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from uav_iqa.inference import (
    DummyExecutor,
    InferenceConfig,
    InferenceEngine,
    JsonStorage,
    Metrics,
    Validator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """Two files: 10 and 25 records."""
    d = tmp_path / "input"
    d.mkdir()
    (d / "Sim3_VQA_train.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": f"a{i}",
                    "question": f"q{i}",
                    "uav_paths": {"UAV1": "r.jpg"},
                    "distorted_uav_paths": {"UAV1": "d.png"},
                    "vlm_scores": {},
                    "cognitive_score": None,
                }
                for i in range(10)
            ]
        )
    )
    (d / "Real2_VQA_train.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": f"b{i}",
                    "question": f"q{i}",
                    "uav_paths": {"UAV1": "r.jpg"},
                    "distorted_uav_paths": {"UAV1": "d.png"},
                    "vlm_scores": {},
                    "cognitive_score": None,
                }
                for i in range(25)
            ]
        )
    )
    return d


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    return tmp_path / "checkpoint.db"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_initial_state(self) -> None:
        m = Metrics()
        assert m.completed_tasks == 0
        assert m.total_tasks == 0
        assert m.progress_pct() == 0.0
        snap = m.snapshot()
        assert snap["completed_tasks"] == 0

    def test_record_task(self) -> None:
        m = Metrics(total_tasks=10)
        m.record_task(num_outputs=256, failed=0)
        assert m.completed_tasks == 1
        assert m.completed_samples == 256
        assert m.progress_pct() == 10.0

    def test_record_failed(self) -> None:
        m = Metrics(total_tasks=10)
        m.record_task(num_outputs=10, failed=2)
        assert m.failed_tasks == 2

    def test_throughput(self) -> None:
        m = Metrics(total_tasks=10)
        m.start_time = time.time() - 10  # 10 seconds ago
        m.record_task(100, 0)
        m.record_task(100, 0)
        assert m.tasks_per_sec() == pytest.approx(0.2, abs=0.01)
        assert m.samples_per_sec() == pytest.approx(20.0, abs=1.0)

    def test_eta(self) -> None:
        m = Metrics(total_tasks=100)
        m.start_time = time.time() - 10
        # Complete 10 tasks in 10 seconds → 1 task/sec → 90 sec remaining
        for _ in range(10):
            m.record_task(10, 0)
        eta = m.eta_seconds()
        assert 80 < eta < 100

    def test_eta_infinite_when_no_progress(self) -> None:
        m = Metrics(total_tasks=10)
        assert m.eta_seconds() == float("inf")

    def test_snapshot_keys(self) -> None:
        m = Metrics(total_tasks=5)
        m.record_task(100, 0)
        snap = m.snapshot()
        expected_keys = {
            "progress_pct", "completed_tasks", "total_tasks",
            "completed_samples", "total_samples",
            "tasks_per_sec", "samples_per_sec",
            "elapsed_sec", "eta_sec", "failed_tasks",
        }
        assert set(snap.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidator:
    def test_valid_output(
        self, input_dir: Path, output_dir: Path
    ) -> None:
        # Copy input to output (simulating perfect run)
        output_dir.mkdir()
        for name in ["Sim3_VQA_train.json", "Real2_VQA_train.json"]:
            in_data = json.loads((input_dir / name).read_text())
            (output_dir / name).write_text(json.dumps(in_data))

        storage = JsonStorage(input_dir)
        v = Validator(storage)
        result = v.validate(input_dir, output_dir)
        assert result["valid"] is True
        assert result["file_count"] == 2
        assert result["total_records"] == 35

    def test_missing_file(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir()
        # Only write one file
        in_data = json.loads((input_dir / "Sim3_VQA_train.json").read_text())
        (output_dir / "Sim3_VQA_train.json").write_text(json.dumps(in_data))

        storage = JsonStorage(input_dir)
        v = Validator(storage)
        result = v.validate(input_dir, output_dir)
        assert result["valid"] is False
        assert any("Real2" in e for e in result["errors"])

    def test_count_mismatch(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir()
        (output_dir / "Sim3_VQA_train.json").write_text(json.dumps([{"x": 1}]))
        (output_dir / "Real2_VQA_train.json").write_text(json.dumps([{"x": 1}]))

        storage = JsonStorage(input_dir)
        v = Validator(storage)
        result = v.validate(input_dir, output_dir)
        assert result["valid"] is False
        assert any("count mismatch" in e for e in result["errors"])

    def test_order_mismatch(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir()
        # Reverse the order
        in_data = json.loads((input_dir / "Sim3_VQA_train.json").read_text())
        reversed_data = list(reversed(in_data))
        (output_dir / "Sim3_VQA_train.json").write_text(json.dumps(reversed_data))
        in_data2 = json.loads((input_dir / "Real2_VQA_train.json").read_text())
        (output_dir / "Real2_VQA_train.json").write_text(json.dumps(in_data2))

        storage = JsonStorage(input_dir)
        v = Validator(storage)
        result = v.validate(input_dir, output_dir)
        assert result["valid"] is False
        assert any("order mismatch" in e for e in result["errors"])

    def test_empty_dirs(self, tmp_path: Path) -> None:
        empty_in = tmp_path / "in"
        empty_out = tmp_path / "out"
        empty_in.mkdir()
        empty_out.mkdir()
        storage = JsonStorage(empty_in)
        v = Validator(storage)
        result = v.validate(empty_in, empty_out)
        assert result["valid"] is True
        assert result["file_count"] == 0


# ---------------------------------------------------------------------------
# Engine tests (single-GPU, DummyExecutor)
# ---------------------------------------------------------------------------


class TestInferenceEngine:
    def test_full_run(
        self,
        input_dir: Path,
        output_dir: Path,
        checkpoint_path: Path,
    ) -> None:
        """End-to-end engine run with DummyExecutor."""
        cfg = InferenceConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            chunk_size=10,
            batch_size=4,
            num_gpus=1,
            model_name="dummy",
        )

        def factory(gpu_id: int) -> tuple[DummyExecutor, int]:
            return DummyExecutor(model_name="test", score=0.5), gpu_id

        engine = InferenceEngine(cfg, executor_factory=factory)
        result = engine.run()

        assert result["validation"]["valid"] is True
        assert result["validation"]["file_count"] == 2
        assert result["validation"]["total_records"] == 35
        assert "Sim3_VQA_train.json" in result["files_written"]
        assert "Real2_VQA_train.json" in result["files_written"]

        # Verify output content
        out_a = json.loads((output_dir / "Sim3_VQA_train.json").read_text())
        assert len(out_a) == 10
        for entry in out_a:
            assert entry["vlm_scores"]["test"] == 0.5
            assert entry["cognitive_score"] == 0.5

    def test_empty_input(
        self,
        tmp_path: Path,
        checkpoint_path: Path,
    ) -> None:
        empty_in = tmp_path / "empty_in"
        empty_in.mkdir()
        empty_out = tmp_path / "empty_out"

        cfg = InferenceConfig(
            input_dir=empty_in,
            output_dir=empty_out,
            checkpoint_path=checkpoint_path,
            num_gpus=1,
        )
        engine = InferenceEngine(cfg)
        result = engine.run()
        assert result["validation"]["valid"] is True
        assert result["validation"]["file_count"] == 0
        assert result["files_written"] == {}

    def test_record_order_preserved(
        self,
        input_dir: Path,
        output_dir: Path,
        checkpoint_path: Path,
    ) -> None:
        """Output records must match input order."""
        cfg = InferenceConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            chunk_size=5,  # force multiple chunks
            batch_size=2,
            num_gpus=1,
        )

        def factory(gpu_id: int) -> tuple[DummyExecutor, int]:
            return DummyExecutor(model_name="order", score=0.1), gpu_id

        engine = InferenceEngine(cfg, executor_factory=factory)
        engine.run()

        out_a = json.loads((output_dir / "Sim3_VQA_train.json").read_text())
        for i, entry in enumerate(out_a):
            assert entry["sample_id"] == f"a{i}"

        out_b = json.loads((output_dir / "Real2_VQA_train.json").read_text())
        for i, entry in enumerate(out_b):
            assert entry["sample_id"] == f"b{i}"

    def test_resume(
        self,
        input_dir: Path,
        output_dir: Path,
        checkpoint_path: Path,
    ) -> None:
        """First run completes, second run with resume=True should be a no-op."""
        cfg = InferenceConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            chunk_size=10,
            num_gpus=1,
        )

        def factory(gpu_id: int) -> tuple[DummyExecutor, int]:
            return DummyExecutor(model_name="r1", score=0.3), gpu_id

        engine1 = InferenceEngine(cfg, executor_factory=factory)
        result1 = engine1.run()
        assert result1["validation"]["valid"] is True

        # Second run with resume=True — all tasks already SUCCESS
        def factory2(gpu_id: int) -> tuple[DummyExecutor, int]:
            return DummyExecutor(model_name="r2", score=0.9), gpu_id

        engine2 = InferenceEngine(cfg, executor_factory=factory2)
        result2 = engine2.run(resume=True)
        # Should have 0 tasks to process
        assert len(result2["files_written"]) == 0  # nothing new to write

    def test_gpu_id_passthrough_from_factory(
        self,
        input_dir: Path,
        output_dir: Path,
        checkpoint_path: Path,
    ) -> None:
        """Factory returns gpu_id != loop index — engine must use factory's GPU ID.

        Regression test: before the fix, the engine passed the loop index ``i``
        as ``gpu_id`` to ``worker_main``, ignoring the factory's GPU mapping.
        When ``--gpu-ids 6,7`` was used with ``--num-gpus 2``, workers ended
        up setting ``CUDA_VISIBLE_DEVICES=0`` or ``1`` instead of ``6`` or ``7``,
        causing vLLM to load on the wrong (occupied) GPU.
        """
        from unittest.mock import patch

        cfg = InferenceConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            chunk_size=10,
            num_gpus=2,
            model_name="dummy",
        )

        # Simulate --gpu-ids 6,7: factory maps index→physical GPU
        gpu_ids = [6, 7]
        gpu_ids_received: list[int] = []

        def factory(gpu_idx: int) -> tuple[DummyExecutor, int]:
            gpu_ids_received.append(gpu_idx)
            gpu_id = gpu_ids[gpu_idx] if gpu_idx < len(gpu_ids) else gpu_idx
            return DummyExecutor(model_name="test", score=0.5), gpu_id

        # Intercept mp.Process to capture the args passed to worker_main
        with patch("uav_iqa.inference.engine.mp.Process") as mock_process:
            engine = InferenceEngine(cfg, executor_factory=factory)
            engine.run()

        # Should have launched 2 workers
        assert mock_process.call_count == 2

        # Extract gpu_id (args[1]) from each Process call
        gpu_ids_passed: list[int] = []
        for call_args in mock_process.call_args_list:
            _, kwargs = call_args
            args = kwargs.get("args", [])
            # args = (worker_id, gpu_id, task_queue, result_queue, storage, executor, batch_size)
            gpu_ids_passed.append(args[1])

        # Factory received indices 0, 1
        assert gpu_ids_received == [0, 1]
        # Engine passed factory-returned GPU IDs (6,7), NOT loop indices (0,1)
        assert gpu_ids_passed == [6, 7], (
            f"Expected GPU IDs [6, 7] from factory, got {gpu_ids_passed}"
        )
