"""Tests for Phase 3: scheduler, collector, writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uav_iqa.inference import (
    CheckpointStore,
    Collector,
    DummyExecutor,
    JsonStorage,
    Scheduler,
    TaskRepository,
    Writer,
    make_result_queue,
    make_task_queue,
)
from uav_iqa.inference.worker import worker_main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """Two files: 10 and 25 records."""
    d = tmp_path / "input"
    d.mkdir()
    (d / "Sim3_VQA_train.json").write_text(
        json.dumps([{"sample_id": f"a{i}", "question": f"q{i}"} for i in range(10)])
    )
    (d / "Real2_VQA_train.json").write_text(
        json.dumps([{"sample_id": f"b{i}", "question": f"q{i}"} for i in range(25)])
    )
    return d


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def storage(input_dir: Path) -> JsonStorage:
    return JsonStorage(input_dir)


@pytest.fixture
def checkpoint(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "ckpt.db")


@pytest.fixture
def repo(checkpoint: CheckpointStore) -> TaskRepository:
    return TaskRepository(checkpoint)


@pytest.fixture
def writer(storage: JsonStorage, output_dir: Path) -> Writer:
    return Writer(storage, output_dir)


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_fresh_run(self, storage: JsonStorage, repo: TaskRepository) -> None:
        tq = make_task_queue()
        sched = Scheduler(storage, repo, tq, chunk_size=10)
        n = sched.run(resume=False)
        # 10/10 = 1 chunk + 25/10 = 3 chunks = 4 total
        assert n == 4
        # Signal sentinel then drain
        tq.put_sentinel()
        tasks = []
        while True:
            t = tq.get()
            if t is None:
                break
            tasks.append(t)
        assert len(tasks) == 4

    def test_resume_run(self, storage: JsonStorage, repo: TaskRepository) -> None:
        # First run
        tq1 = make_task_queue()
        Scheduler(storage, repo, tq1, chunk_size=10).run(resume=False)
        # Complete one task
        first_task = repo.pending()[0]
        repo.mark_success(first_task.task_id)
        # Resume — should enqueue remaining 3
        tq2 = make_task_queue()
        n = Scheduler(storage, repo, tq2, chunk_size=10).run(resume=True)
        assert n == 3

    def test_max_entries(self, storage: JsonStorage, repo: TaskRepository) -> None:
        tq = make_task_queue()
        sched = Scheduler(storage, repo, tq, chunk_size=10, max_entries=15)
        n = sched.run(resume=False)
        # 10 records → 1 chunk, 5 records → 1 chunk = 2
        assert n == 2

    def test_empty_dir(self, tmp_path: Path, repo: TaskRepository) -> None:
        s = JsonStorage(tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        tq = make_task_queue()
        n = Scheduler(s, repo, tq, chunk_size=10).run(resume=False)
        assert n == 0

    def test_task_ids_unique(self, storage: JsonStorage, repo: TaskRepository) -> None:
        tq = make_task_queue()
        Scheduler(storage, repo, tq, chunk_size=10).run(resume=False)
        tq.put_sentinel()
        tasks = []
        while True:
            t = tq.get()
            if t is None:
                break
            tasks.append(t)
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestWriter:
    def test_write_single_file(self, writer: Writer, input_dir: Path) -> None:
        source = input_dir / "Sim3_VQA_train.json"
        chunks = [(0, [{"sample_id": f"a{i}", "score": 0.1 * i} for i in range(10)])]
        out_path = writer.write_file(source, chunks, total_records=10)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) == 10
        assert data[0]["sample_id"] == "a0"
        assert data[9]["score"] == 0.9

    def test_write_multiple_chunks_order(
        self, writer: Writer, input_dir: Path
    ) -> None:
        """Chunks must be assembled in chunk_id order, not insertion order."""
        source = input_dir / "Real2_VQA_train.json"
        # Insert in reverse order
        chunks = [
            (2, [{"sample_id": f"b{i}"} for i in range(20, 25)]),
            (1, [{"sample_id": f"b{i}"} for i in range(10, 20)]),
            (0, [{"sample_id": f"b{i}"} for i in range(0, 10)]),
        ]
        out_path = writer.write_file(source, chunks, total_records=25)
        data = json.loads(out_path.read_text())
        # Order should be b0..b24
        for i, entry in enumerate(data):
            assert entry["sample_id"] == f"b{i}"

    def test_write_count_mismatch_raises(
        self, writer: Writer, input_dir: Path
    ) -> None:
        source = input_dir / "Sim3_VQA_train.json"
        chunks = [(0, [{"x": 1}, {"x": 2}])]  # only 2, expected 10
        with pytest.raises(ValueError, match="mismatch"):
            writer.write_file(source, chunks, total_records=10)

    def test_write_atomic_overwrite(self, writer: Writer, input_dir: Path) -> None:
        source = input_dir / "Sim3_VQA_train.json"
        chunks = [(0, [{"sample_id": "a0", "v": 1}])]
        writer.write_file(source, chunks, total_records=1)
        chunks2 = [(0, [{"sample_id": "a0", "v": 2}])]
        writer.write_file(source, chunks2, total_records=1)
        data = json.loads((writer.output_dir / source.name).read_text())
        assert data[0]["v"] == 2


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------


class TestCollector:
    def test_collect_and_write(
        self,
        storage: JsonStorage,
        repo: TaskRepository,
        writer: Writer,
        input_dir: Path,
    ) -> None:
        """End-to-end: scheduler → worker → collector → writer."""
        tq = make_task_queue()
        rq = make_result_queue()

        # Schedule
        Scheduler(storage, repo, tq, chunk_size=10).run(resume=False)

        # Run worker (DummyExecutor) — put sentinel to signal exit
        tq.put_sentinel()
        ex = DummyExecutor(model_name="test", score=0.5)
        worker_main(0, 0, tq, rq, storage, ex, batch_size=4)

        # Close result queue with 1 sentinel (1 worker)
        rq.put_sentinel()

        # Run collector
        collector = Collector(rq, repo, storage, writer, num_workers=1)
        files_written = collector.run()

        assert len(files_written) == 2
        assert "Sim3_VQA_train.json" in files_written
        assert "Real2_VQA_train.json" in files_written
        assert files_written["Sim3_VQA_train.json"] == 10
        assert files_written["Real2_VQA_train.json"] == 25

        # Verify output files
        out_a = json.loads((writer.output_dir / "Sim3_VQA_train.json").read_text())
        out_b = json.loads((writer.output_dir / "Real2_VQA_train.json").read_text())
        assert len(out_a) == 10
        assert len(out_b) == 25
        for entry in out_a + out_b:
            assert entry["vlm_scores"]["test"] == 0.5
            assert entry["cognitive_score"] == 0.5

    def test_preserves_record_order(
        self,
        storage: JsonStorage,
        repo: TaskRepository,
        writer: Writer,
    ) -> None:
        """Records in output must match input order."""
        tq = make_task_queue()
        rq = make_result_queue()

        # Small chunk size to force multiple chunks
        Scheduler(storage, repo, tq, chunk_size=5).run(resume=False)

        tq.put_sentinel()
        ex = DummyExecutor(model_name="order_test", score=0.1)
        worker_main(0, 0, tq, rq, storage, ex, batch_size=2)
        rq.put_sentinel()

        collector = Collector(rq, repo, storage, writer, num_workers=1)
        collector.run()

        out_a = json.loads((writer.output_dir / "Sim3_VQA_train.json").read_text())
        for i, entry in enumerate(out_a):
            assert entry["sample_id"] == f"a{i}"

        out_b = json.loads((writer.output_dir / "Real2_VQA_train.json").read_text())
        for i, entry in enumerate(out_b):
            assert entry["sample_id"] == f"b{i}"

    def test_marks_tasks_success(
        self,
        storage: JsonStorage,
        repo: TaskRepository,
        writer: Writer,
    ) -> None:
        tq = make_task_queue()
        rq = make_result_queue()
        Scheduler(storage, repo, tq, chunk_size=10).run(resume=False)
        tq.put_sentinel()
        worker_main(0, 0, tq, rq, storage, DummyExecutor(), batch_size=4)
        rq.put_sentinel()
        Collector(rq, repo, storage, writer, num_workers=1).run()

        assert repo.count_success() == 4
        assert repo.count_failed() == 0

    def test_empty_input(
        self,
        tmp_path: Path,
        repo: TaskRepository,
        writer: Writer,
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        s = JsonStorage(empty_dir)
        tq = make_task_queue()
        rq = make_result_queue()
        Scheduler(s, repo, tq, chunk_size=10).run(resume=False)
        rq.put_sentinel()
        collector = Collector(rq, repo, s, writer, num_workers=1)
        result = collector.run()
        assert result == {}

    def test_resume_after_partial_completion(
        self,
        storage: JsonStorage,
        repo: TaskRepository,
        writer: Writer,
    ) -> None:
        """Simulate partial completion: some tasks done, resume the rest."""
        tq = make_task_queue()
        rq = make_result_queue()
        Scheduler(storage, repo, tq, chunk_size=10).run(resume=False)

        # Complete first 2 tasks (file A's 1 chunk + file B's first chunk)
        tq.put_sentinel()
        worker_main(0, 0, tq, rq, storage, DummyExecutor(), batch_size=4)
        rq.put_sentinel()
        Collector(rq, repo, storage, writer, num_workers=1).run()

        # Now simulate a fresh run with remaining tasks
        tq2 = make_task_queue()
        rq2 = make_result_queue()
        Scheduler(storage, repo, tq2, chunk_size=10).run(resume=True)
        tq2.put_sentinel()
        worker_main(0, 0, tq2, rq2, storage, DummyExecutor(model_name="m2", score=0.2), batch_size=4)
        rq2.put_sentinel()
        Collector(rq2, repo, storage, writer, num_workers=1).run()

        # All tasks should be success now
        assert repo.count_success() == 4
