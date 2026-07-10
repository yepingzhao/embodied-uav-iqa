"""Tests for Phase 1 foundation modules: types, config, queue, storage, checkpoint, repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uav_iqa.inference import (
    CheckpointStore,
    InferenceConfig,
    JsonStorage,
    MpQueue,
    Task,
    TaskRepository,
    TaskStatus,
    make_result_queue,
    make_task_queue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_input_dir(tmp_path: Path) -> Path:
    """Create a temp dir with two JSON files of different sizes."""
    d = tmp_path / "input"
    d.mkdir()
    # File A: 10 records (VQA naming convention)
    records_a = [{"id": i, "value": f"a{i}"} for i in range(10)]
    (d / "Sim3_VQA_train.json").write_text(json.dumps(records_a))
    # File B: 25 records
    records_b = [{"id": i, "value": f"b{i}"} for i in range(25)]
    (d / "Real2_VQA_train.json").write_text(json.dumps(records_b))
    return d


@pytest.fixture
def store(tmp_input_dir: Path) -> JsonStorage:
    return JsonStorage(tmp_input_dir)


@pytest.fixture
def checkpoint(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "checkpoint.db")


@pytest.fixture
def repo(checkpoint: CheckpointStore) -> TaskRepository:
    return TaskRepository(checkpoint)


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------


class TestTypes:
    def test_task_size(self) -> None:
        t = Task(task_id="x__0000", file_path=Path("/a.json"), chunk_id=0, start=0, end=256)
        assert t.size == 256

    def test_task_size_zero(self) -> None:
        t = Task(task_id="x__0000", file_path=Path("/a.json"), chunk_id=0, start=10, end=10)
        assert t.size == 0

    def test_task_status_values(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = InferenceConfig(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            checkpoint_path=tmp_path / "ckpt.db",
        )
        assert cfg.chunk_size == 256
        assert cfg.batch_size == 16
        assert cfg.num_gpus == 1
        assert cfg.seed == 42


# ---------------------------------------------------------------------------
# queue.py
# ---------------------------------------------------------------------------


class TestMpQueue:
    def test_put_get(self) -> None:
        q = MpQueue[int]()
        q.put(1)
        q.put(2)
        assert q.get() == 1
        assert q.get() == 2

    def test_close_returns_none_after_drain(self) -> None:
        q = MpQueue[int]()
        q.put(42)
        q.close()
        q.put_sentinel()
        assert q.get() == 42
        assert q.get() is None

    def test_qsize(self) -> None:
        q = MpQueue[int]()
        q.put(1)
        q.put(2)
        assert q.qsize() >= 1  # mp.Queue.qsize is approximate

    def test_factory_functions(self) -> None:
        tq = make_task_queue()
        rq = make_result_queue()
        tq.put("hello")
        assert tq.get() == "hello"
        rq.put(123)
        assert rq.get() == 123


# ---------------------------------------------------------------------------
# storage.py
# ---------------------------------------------------------------------------


class TestJsonStorage:
    def test_scan_files(self, store: JsonStorage) -> None:
        files = store.scan_files()
        assert len(files) == 2
        names = {f.name for f in files}
        assert names == {"Sim3_VQA_train.json", "Real2_VQA_train.json"}
        counts = {f.name: f.record_count for f in files}
        assert counts["Sim3_VQA_train.json"] == 10
        assert counts["Real2_VQA_train.json"] == 25

    def test_count_records(self, store: JsonStorage, tmp_input_dir: Path) -> None:
        assert store.count_records(tmp_input_dir / "Sim3_VQA_train.json") == 10
        assert store.count_records(tmp_input_dir / "Real2_VQA_train.json") == 25

    def test_load_chunk_full(self, store: JsonStorage, tmp_input_dir: Path) -> None:
        chunk = store.load_chunk(tmp_input_dir / "Sim3_VQA_train.json", 0, 10)
        assert len(chunk) == 10
        assert chunk[0]["id"] == 0
        assert chunk[9]["id"] == 9

    def test_load_chunk_partial(self, store: JsonStorage, tmp_input_dir: Path) -> None:
        chunk = store.load_chunk(tmp_input_dir / "Real2_VQA_train.json", 5, 10)
        assert len(chunk) == 5
        assert chunk[0]["id"] == 5
        assert chunk[4]["id"] == 9

    def test_load_chunk_out_of_range(self, store: JsonStorage, tmp_input_dir: Path) -> None:
        with pytest.raises(IndexError):
            store.load_chunk(tmp_input_dir / "Sim3_VQA_train.json", 0, 100)

    def test_load_chunk_negative_start(self, store: JsonStorage, tmp_input_dir: Path) -> None:
        with pytest.raises(IndexError):
            store.load_chunk(tmp_input_dir / "Sim3_VQA_train.json", -1, 5)

    def test_write_output_atomic(self, store: JsonStorage, tmp_path: Path) -> None:
        out = tmp_path / "output" / "result.json"
        records = [{"id": i, "score": i * 0.1} for i in range(5)]
        store.write_output(out, records)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == records

    def test_write_output_overwrite(self, store: JsonStorage, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        store.write_output(out, [{"a": 1}])
        store.write_output(out, [{"b": 2}])
        loaded = json.loads(out.read_text())
        assert loaded == [{"b": 2}]

    def test_scan_empty_dir(self, tmp_path: Path) -> None:
        s = JsonStorage(tmp_path)
        assert s.scan_files() == []

    def test_scan_non_json_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("hello")
        (tmp_path / "data.json").write_text("[]")
        s = JsonStorage(tmp_path, glob_pattern="*.json")
        files = s.scan_files()
        assert len(files) == 1
        assert files[0].name == "data.json"

    def test_invalid_json_array(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.json"
        fpath.write_text(json.dumps({"not": "a list"}))
        s = JsonStorage(tmp_path)
        with pytest.raises(ValueError, match="JSON array"):
            s.count_records(fpath)


# ---------------------------------------------------------------------------
# checkpoint.py
# ---------------------------------------------------------------------------


class TestCheckpointStore:
    def test_insert_and_pending(self, checkpoint: CheckpointStore) -> None:
        t = Task(task_id="f__0000", file_path=Path("/a.json"), chunk_id=0, start=0, end=10)
        checkpoint.insert_task(t)
        pending = checkpoint.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].task_id == "f__0000"

    def test_set_status(self, checkpoint: CheckpointStore) -> None:
        t = Task(task_id="f__0001", file_path=Path("/a.json"), chunk_id=1, start=10, end=20)
        checkpoint.insert_task(t)
        checkpoint.set_status("f__0001", TaskStatus.RUNNING)
        assert checkpoint.count_by_status(TaskStatus.RUNNING) == 1
        assert checkpoint.count_by_status(TaskStatus.PENDING) == 0

    def test_mark_success(self, checkpoint: CheckpointStore) -> None:
        t = Task(task_id="f__0002", file_path=Path("/a.json"), chunk_id=2, start=20, end=30)
        checkpoint.insert_task(t)
        checkpoint.set_status("f__0002", TaskStatus.SUCCESS)
        assert checkpoint.count_by_status(TaskStatus.SUCCESS) == 1
        # Should no longer appear in pending
        pending = checkpoint.get_pending_tasks()
        assert all(p.task_id != "f__0002" for p in pending)

    def test_mark_failed_with_retry(self, checkpoint: CheckpointStore) -> None:
        t = Task(task_id="f__0003", file_path=Path("/a.json"), chunk_id=3, start=30, end=40)
        checkpoint.insert_task(t)
        checkpoint.set_status("f__0003", TaskStatus.FAILED, increment_retry=True)
        assert checkpoint.count_by_status(TaskStatus.FAILED) == 1
        # Failed tasks should appear in pending (resumable)
        pending = checkpoint.get_pending_tasks()
        assert any(p.task_id == "f__0003" for p in pending)

    def test_pending_or_running_includes_running(self, checkpoint: CheckpointStore) -> None:
        t = Task(task_id="f__0004", file_path=Path("/a.json"), chunk_id=4, start=40, end=50)
        checkpoint.insert_task(t)
        checkpoint.set_status("f__0004", TaskStatus.RUNNING)
        tasks = checkpoint.get_pending_or_running_tasks()
        assert any(p.task_id == "f__0004" for p in tasks)

    def test_get_tasks_for_file(self, checkpoint: CheckpointStore) -> None:
        for i in range(3):
            t = Task(
                task_id=f"f__{i:04d}",
                file_path=Path("/f.json"),
                chunk_id=i,
                start=i * 10,
                end=(i + 1) * 10,
            )
            checkpoint.insert_task(t)
        chunks = checkpoint.get_tasks_for_file("/f.json")
        assert len(chunks) == 3
        assert chunks[0].chunk_id == 0
        assert chunks[2].chunk_id == 2

    def test_total_tasks(self, checkpoint: CheckpointStore) -> None:
        assert checkpoint.total_tasks() == 0
        t = Task(task_id="x", file_path=Path("/x.json"), chunk_id=0, start=0, end=1)
        checkpoint.insert_task(t)
        assert checkpoint.total_tasks() == 1

    def test_clear(self, checkpoint: CheckpointStore) -> None:
        t = Task(task_id="x", file_path=Path("/x.json"), chunk_id=0, start=0, end=1)
        checkpoint.insert_task(t)
        checkpoint.clear()
        assert checkpoint.total_tasks() == 0

    def test_idempotent_insert(self, checkpoint: CheckpointStore) -> None:
        """INSERT OR REPLACE should not duplicate."""
        t = Task(task_id="dup", file_path=Path("/x.json"), chunk_id=0, start=0, end=1)
        checkpoint.insert_task(t)
        checkpoint.insert_task(t)
        assert checkpoint.total_tasks() == 1


# ---------------------------------------------------------------------------
# repository.py
# ---------------------------------------------------------------------------


class TestTaskRepository:
    def test_create_tasks(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        tasks = repo.create_tasks(files, chunk_size=10)
        # fileA: 10 records / 10 = 1 chunk
        # fileB: 25 records / 10 = 3 chunks
        assert len(tasks) == 4

    def test_create_tasks_small_chunk(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        tasks = repo.create_tasks(files, chunk_size=5)
        # fileA: 10 / 5 = 2, fileB: 25 / 5 = 5
        assert len(tasks) == 7

    def test_create_tasks_max_entries(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        tasks = repo.create_tasks(files, chunk_size=10, max_entries=15)
        # fileA: 10 records → 1 chunk (budget left: 5)
        # fileB: min(25, 5) = 5 → 1 chunk
        assert len(tasks) == 2

    def test_pending_returns_new_tasks(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        repo.create_tasks(files, chunk_size=10)
        pending = repo.pending()
        assert len(pending) == 4
        assert all(isinstance(t, Task) for t in pending)

    def test_mark_running_success(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        repo.create_tasks(files, chunk_size=10)
        pending = repo.pending()
        first = pending[0]
        repo.mark_running(first.task_id)
        # Should no longer be in pending (PENDING or FAILED)
        pending_after = repo.pending()
        assert all(t.task_id != first.task_id for t in pending_after)
        repo.mark_success(first.task_id)
        assert repo.count_success() == 1

    def test_mark_failed(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        repo.create_tasks(files, chunk_size=10)
        pending = repo.pending()
        first = pending[0]
        repo.mark_failed(first.task_id)
        # Failed should still be pending (resumable)
        pending_after = repo.pending()
        assert any(t.task_id == first.task_id for t in pending_after)
        assert repo.count_failed() == 1

    def test_file_complete(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        repo.create_tasks(files, chunk_size=10)
        file_a = str(files[0].path)
        # Not complete yet
        assert not repo.file_complete(file_a)
        # Mark all chunks of fileA as success
        chunks = repo.file_chunks(file_a)
        for chunk in chunks:
            repo.mark_success(chunk.task_id)
        assert repo.file_complete(file_a)

    def test_file_complete_empty(self, repo: TaskRepository) -> None:
        assert not repo.file_complete("/nonexistent.json")

    def test_resume_after_crash(self, repo: TaskRepository, store: JsonStorage) -> None:
        """Simulate a crash: tasks left in RUNNING state."""
        files = store.scan_files()
        repo.create_tasks(files, chunk_size=10)
        pending = repo.pending()
        # Mark one as RUNNING (crash before completion)
        first = pending[0]
        repo.mark_running(first.task_id)
        # On resume, pending_or_running should include it
        resumable = repo.pending_or_running()
        assert any(t.task_id == first.task_id for t in resumable)
        # But plain pending() should NOT include RUNNING
        plain_pending = repo.pending()
        assert all(t.task_id != first.task_id for t in plain_pending)

    def test_total_and_counts(self, repo: TaskRepository, store: JsonStorage) -> None:
        files = store.scan_files()
        repo.create_tasks(files, chunk_size=10)
        assert repo.total() == 4
        assert repo.count_success() == 0
        assert repo.count_failed() == 0
        pending = repo.pending()
        repo.mark_success(pending[0].task_id)
        assert repo.count_success() == 1
