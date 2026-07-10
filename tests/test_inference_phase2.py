"""Tests for Phase 2: executor and worker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uav_iqa.inference import (
    DummyExecutor,
    JsonStorage,
    Result,
    Task,
    make_result_queue,
    make_task_queue,
)
from uav_iqa.inference.worker import worker_main, _process_task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """Create a temp dir with one JSON file of 20 records."""
    d = tmp_path / "input"
    d.mkdir()
    records = [
        {
            "sample_id": f"s{i}",
            "question": f"What is in image {i}?",
            "subtask_type": "scene_description",
            "uav_paths": {"UAV1": f"ref/uav1_{i}.jpg", "UAV2": f"ref/uav2_{i}.jpg"},
            "distorted_uav_paths": {
                "UAV1": f"dist/uav1_{i}.png",
                "UAV2": f"dist/uav2_{i}.png",
            },
            "vlm_scores": {},
            "cognitive_score": None,
        }
        for i in range(20)
    ]
    (d / "Sim3_VQA_train.json").write_text(json.dumps(records))
    return d


@pytest.fixture
def storage(input_dir: Path) -> JsonStorage:
    return JsonStorage(input_dir)


@pytest.fixture
def task(input_dir: Path) -> Task:
    return Task(
        task_id="Sim3_VQA_train__0000",
        file_path=(input_dir / "Sim3_VQA_train.json").resolve(),
        chunk_id=0,
        start=0,
        end=10,
    )


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------


class TestDummyExecutor:
    def test_lifecycle(self) -> None:
        ex = DummyExecutor(model_name="test", score=0.7)
        ex.prepare()
        batch = [{"question": "q1", "uav_paths": {}, "distorted_uav_paths": {}}]
        results = ex.infer(batch)
        assert len(results) == 1
        assert results[0]["vlm_scores"]["test"] == 0.7
        assert results[0]["cognitive_score"] == 0.7
        ex.finalize()

    def test_infer_before_prepare_raises(self) -> None:
        ex = DummyExecutor()
        with pytest.raises(RuntimeError, match="before prepare"):
            ex.infer([{"question": "q"}])

    def test_preserves_existing_vlm_scores(self) -> None:
        ex = DummyExecutor(model_name="model_b", score=0.3)
        ex.prepare()
        batch = [{"question": "q", "vlm_scores": {"model_a": 0.9}}]
        results = ex.infer(batch)
        assert results[0]["vlm_scores"]["model_a"] == 0.9
        assert results[0]["vlm_scores"]["model_b"] == 0.3

    def test_empty_question_returns_zero(self) -> None:
        ex = DummyExecutor(model_name="d", score=0.5)
        ex.prepare()
        # DummyExecutor doesn't have the question-check logic, but VLMExecutor does.
        # Here we just test that DummyExecutor processes any entry.
        batch = [{"question": "", "uav_paths": {}, "distorted_uav_paths": {}}]
        results = ex.infer(batch)
        assert len(results) == 1
        assert results[0]["cognitive_score"] == 0.5

    def test_multiple_entries(self) -> None:
        ex = DummyExecutor(model_name="m", score=0.1)
        ex.prepare()
        batch = [{"question": f"q{i}"} for i in range(5)]
        results = ex.infer(batch)
        assert len(results) == 5
        for r in results:
            assert r["cognitive_score"] == 0.1


# ---------------------------------------------------------------------------
# _process_task tests
# ---------------------------------------------------------------------------


class TestProcessTask:
    def test_basic(self, storage: JsonStorage, task: Task) -> None:
        ex = DummyExecutor(model_name="test", score=0.5)
        ex.prepare()
        result = _process_task(task, storage, ex, batch_size=4)
        assert isinstance(result, Result)
        assert result.task_id == "Sim3_VQA_train__0000"
        assert result.chunk_id == 0
        assert len(result.outputs) == 10
        assert result.failed == []
        for out in result.outputs:
            assert out["vlm_scores"]["test"] == 0.5
            assert out["cognitive_score"] == 0.5

    def test_preserves_entry_fields(self, storage: JsonStorage, task: Task) -> None:
        ex = DummyExecutor(model_name="m", score=0.8)
        ex.prepare()
        result = _process_task(task, storage, ex, batch_size=16)
        # Original fields should be preserved
        assert result.outputs[0]["sample_id"] == "s0"
        assert result.outputs[0]["question"] == "What is in image 0?"
        assert result.outputs[0]["subtask_type"] == "scene_description"
        assert "UAV1" in result.outputs[0]["uav_paths"]

    def test_batch_size_smaller_than_chunk(self, storage: JsonStorage, task: Task) -> None:
        """With batch_size=3 and 10 records, we need 4 batches (3+3+3+1)."""
        ex = DummyExecutor(model_name="m", score=0.2)
        ex.prepare()
        result = _process_task(task, storage, ex, batch_size=3)
        assert len(result.outputs) == 10

    def test_batch_size_larger_than_chunk(self, storage: JsonStorage, task: Task) -> None:
        """With batch_size=100 and 10 records, we need 1 batch."""
        ex = DummyExecutor(model_name="m", score=0.2)
        ex.prepare()
        result = _process_task(task, storage, ex, batch_size=100)
        assert len(result.outputs) == 10

    def test_executor_not_prepared_marks_failed(self, storage: JsonStorage, task: Task) -> None:
        """If executor.infer raises, the batch is marked as failed but not crashed."""
        ex = DummyExecutor()
        # _process_task catches the error and marks entries as failed
        result = _process_task(task, storage, ex, batch_size=4)
        assert len(result.outputs) == 10
        assert len(result.failed) == 10  # all entries failed


# ---------------------------------------------------------------------------
# worker_main tests (single-process, no GPU)
# ---------------------------------------------------------------------------


class TestWorkerMain:
    def test_processes_tasks_and_exits_on_sentinel(
        self, storage: JsonStorage, task: Task
    ) -> None:
        """End-to-end worker test using in-process queues."""
        tq = make_task_queue()
        rq = make_result_queue()
        ex = DummyExecutor(model_name="wtest", score=0.42)

        # Enqueue 2 tasks + sentinel
        tq.put(task)
        tq.put(
            Task(
                task_id="Sim3_VQA_train__0001",
                file_path=task.file_path,
                chunk_id=1,
                start=10,
                end=20,
            )
        )
        tq.put_sentinel()  # signal worker to exit

        n = worker_main(
            worker_id=0,
            gpu_id=0,
            task_queue=tq,
            result_queue=rq,
            storage=storage,
            executor=ex,
            batch_size=8,
        )
        assert n == 2

        # Collect results
        r1 = rq.get()
        r2 = rq.get()
        assert r1 is not None
        assert r2 is not None
        assert r1.task_id == "Sim3_VQA_train__0000"
        assert r2.task_id == "Sim3_VQA_train__0001"
        assert len(r1.outputs) == 10
        assert len(r2.outputs) == 10
        assert r1.outputs[0]["cognitive_score"] == 0.42

    def test_empty_queue_immediate_exit(self, storage: JsonStorage) -> None:
        """Worker should exit immediately if queue is already closed."""
        tq = make_task_queue()
        rq = make_result_queue()
        ex = DummyExecutor(model_name="e", score=0.0)
        tq.put_sentinel()  # immediately signal → only sentinel, no tasks

        n = worker_main(0, 0, tq, rq, storage, ex, batch_size=4)
        assert n == 0

    def test_result_order_preserved(self, storage: JsonStorage, task: Task) -> None:
        """Outputs within a result must preserve the chunk's original order."""
        tq = make_task_queue()
        rq = make_result_queue()
        ex = DummyExecutor(model_name="order", score=0.1)
        tq.put(task)
        tq.put_sentinel()

        worker_main(0, 0, tq, rq, storage, ex, batch_size=3)

        result = rq.get()
        assert result is not None
        for i, out in enumerate(result.outputs):
            assert out["sample_id"] == f"s{i}"
