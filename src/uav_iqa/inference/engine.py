"""Engine: orchestrates the full inference pipeline lifecycle.

The engine is the **only** module that knows about all components and how
to wire them together. It:

1. Creates queues, storage, repository, writer, validator, metrics.
2. Runs the scheduler (in the main process).
3. Launches workers (as :class:`multiprocessing.Process`).
4. Runs the collector (in the main process).
5. Waits for workers to finish.
6. Validates output.
7. Reports metrics.

The engine does **not** contain business logic — it only wires modules
and manages processes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

from .collector import Collector
from .config import InferenceConfig
from .executor import BaseExecutor, DummyExecutor
from .metrics import Metrics
from .queue import make_result_queue, make_task_queue
from .repository import TaskRepository
from .scheduler import Scheduler
from .storage import JsonStorage
from .validator import Validator
from .worker import worker_main
from .writer import Writer

_log = logging.getLogger(__name__)


class InferenceEngine:
    """Top-level orchestrator for multi-GPU offline inference.

    Example::

        cfg = InferenceConfig(
            input_dir=Path("data/processed/train"),
            output_dir=Path("data/output/train"),
            checkpoint_path=Path("data/checkpoint.db"),
            num_gpus=4,
            model_name="Qwen2.5-VL",
        )
        engine = InferenceEngine(cfg)
        result = engine.run()
        assert result["valid"]
    """

    def __init__(
        self,
        config: InferenceConfig,
        *,
        executor_factory=None,
    ) -> None:
        self.config = config
        self._executor_factory = executor_factory or self._default_executor_factory
        self.metrics = Metrics()

    def _default_executor_factory(self, gpu_id: int) -> tuple[BaseExecutor, int]:
        """Create an executor and return it with the actual GPU device index."""
        return DummyExecutor(model_name=self.config.model_name), gpu_id

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        """Execute the full pipeline.

        Args:
            resume: If True, resume from the checkpoint database. Pending,
                failed, and running tasks are re-enqueued.

        Returns:
            Dict with keys: ``files_written``, ``validation``, ``metrics``.
        """
        from .checkpoint import CheckpointStore

        # -- 1. Initialize modules -------------------------------------------
        storage = JsonStorage(self.config.input_dir, self.config.glob_pattern)
        checkpoint = CheckpointStore(self.config.checkpoint_path)
        repo = TaskRepository(checkpoint)
        task_queue = make_task_queue()
        result_queue = make_result_queue()
        writer = Writer(storage, self.config.output_dir)

        # -- 2. Schedule (main process) --------------------------------------
        scheduler = Scheduler(
            storage, repo, task_queue,
            chunk_size=self.config.chunk_size,
            max_entries=self.config.max_entries,
        )
        if not resume:
            writer.clean_chunks()
        n_tasks = scheduler.run(resume=resume)
        self.metrics.set_totals(total_tasks=n_tasks, total_samples=0)

        if n_tasks == 0:
            _log.info("No tasks to process, exiting")
            return {
                "files_written": {},
                "validation": {"valid": True, "errors": [], "file_count": 0, "total_records": 0},
                "metrics": self.metrics.snapshot(),
            }

        # -- 3. Launch workers (child processes) -----------------------------
        workers: list[mp.Process] = []
        for i in range(self.config.num_gpus):
            executor, actual_gpu_id = self._executor_factory(i)
            p = mp.Process(
                target=worker_main,
                args=(
                    i,
                    actual_gpu_id,
                    task_queue,
                    result_queue,
                    storage,
                    executor,
                    self.config.batch_size,
                ),
                daemon=False,
            )
            p.start()
            workers.append(p)
            _log.info("Launched worker %d (pid=%d, gpu=%d)", i, p.pid, actual_gpu_id)

        # -- 4. Collector + worker-supervisor (concurrent) -------------------
        import threading

        def _supervisor() -> None:
            for p in workers:
                p.join()
            for _ in workers:
                result_queue.put_sentinel()

        sup = threading.Thread(target=_supervisor, daemon=True)
        sup.start()

        try:
            collector = Collector(
                result_queue, repo, storage, writer,
                num_workers=self.config.num_gpus,
            )
            files_written = collector.run()
        finally:
            # Ensure workers are terminated even if collector crashes
            for p in workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)

        # -- 5. Wait for supervisor to finish --------------------------------
        sup.join(timeout=10)

        # -- 6. Validate ----------------------------------------------------
        validator = Validator(storage)
        validation = validator.validate(self.config.input_dir, self.config.output_dir)

        # -- 7. Report ------------------------------------------------------
        _log.info("Engine complete: %s", self.metrics.snapshot())
        return {
            "files_written": files_written,
            "validation": validation,
            "metrics": self.metrics.snapshot(),
        }
