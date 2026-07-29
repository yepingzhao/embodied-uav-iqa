"""Worker process: owns one GPU, executes tasks from the queue.

The worker is **completely stateless** with respect to file structure.
It receives a :class:`Task`, loads the chunk data via :class:`BaseStorage`,
runs the executor in batches, and puts a :class:`Result` on the result
queue. It does not write output files, track file completion, or parse
file paths beyond what the Task provides.

Lifecycle::

    TaskQueue.get()  ──→  Storage.load_chunk()
                                      │
                                 Executor.infer()
                                      │
                               ResultQueue.put()  ──→  loop

The worker exits when it receives ``None`` from the task queue
(sentinel from the scheduler closing the queue).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .executor import BaseExecutor
from .queue import BaseQueue
from .storage import BaseStorage
from .types import Result, Task

_log = logging.getLogger(__name__)


def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue: BaseQueue[Task | None],
    result_queue: BaseQueue[Result | None],
    storage: BaseStorage,
    executor: BaseExecutor,
    batch_size: int = 16,
) -> int:
    """Entry point for a worker process.

    Args:
        worker_id: Logical worker index (for logging).
        gpu_id: GPU device index (set via ``CUDA_VISIBLE_DEVICES`` before
            calling this function).
        task_queue: Queue of :class:`Task` items. A ``None`` item signals
            shutdown.
        result_queue: Queue for :class:`Result` items.
        storage: Storage backend for loading chunks.
        executor: Inference executor (must be prepared before calling).
        batch_size: Number of samples per GPU forward pass.

    Returns:
        Number of tasks processed.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # After CUDA_VISIBLE_DEVICES remapping, the target GPU is always
    # device 0 from the worker's perspective.  Update the executor's
    # device attribute so that model.to(device) does not reference a
    # now-invalid ordinal (e.g. "cuda:1" → "cuda:0").
    if hasattr(executor, "device") and executor.device:
        executor.device = "cuda:0"
    _log.info("Worker %d started on GPU %d", worker_id, gpu_id)

    executor.prepare()
    tasks_done = 0

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            result = _process_task(task, storage, executor, batch_size)
            result_queue.put(result)
            tasks_done += 1
            _log.debug(
                "Worker %d completed task %s (%d outputs)",
                worker_id, task.task_id, len(result.outputs),
            )
    except Exception:
        _log.error("Worker %d fatal error", worker_id, exc_info=True)
        raise
    finally:
        executor.finalize()
        _log.info("Worker %d exiting, processed %d tasks", worker_id, tasks_done)

    return tasks_done


def _process_task(
    task: Task,
    storage: BaseStorage,
    executor: BaseExecutor,
    batch_size: int,
) -> Result:
    """Load a chunk, run inference in batches, and assemble a Result."""
    records = storage.load_chunk(task.file_path, task.start, task.end)
    outputs: list[dict[str, Any]] = []
    failed: list[int] = []

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        try:
            batch_outputs = executor.infer(batch)
        except Exception:
            _log.error(
                "Inference failed for task %s batch %d", task.task_id, batch_start,
                exc_info=True,
            )
            # Mark entire batch as failed, keep originals
            batch_outputs = batch
            for i in range(len(batch)):
                failed.append(batch_start + i)

        if len(batch_outputs) != len(batch):
            _log.warning(
                "Executor returned %d outputs for %d inputs in task %s",
                len(batch_outputs), len(batch), task.task_id,
            )
            # Pad or truncate to match
            if len(batch_outputs) < len(batch):
                batch_outputs = list(batch_outputs) + batch[len(batch_outputs):]
            else:
                batch_outputs = batch_outputs[: len(batch)]

        outputs.extend(batch_outputs)

    return Result(
        task_id=task.task_id,
        file_path=task.file_path,
        chunk_id=task.chunk_id,
        outputs=outputs,
        failed=failed,
    )
