"""Multi-GPU offline inference framework.

A lightweight, testable, resumable inference engine for single-machine
multi-GPU batch scoring. See ``docs/INFERENCE_FRAMEWORK.md`` for the full
architecture specification.
"""

from .checkpoint import CheckpointStore
from .collector import Collector
from .config import InferenceConfig
from .engine import InferenceEngine
from .executor import BaseExecutor, DummyExecutor, VLMExecutor
from .metrics import Metrics
from .queue import BaseQueue, MpQueue, make_result_queue, make_task_queue
from .repository import TaskRepository
from .scheduler import Scheduler
from .storage import BaseStorage, JsonStorage
from .types import FileRecord, Result, Task, TaskStatus
from .validator import Validator
from .worker import worker_main
from .writer import Writer

__all__ = [
    "BaseExecutor",
    "BaseQueue",
    "BaseStorage",
    "CheckpointStore",
    "Collector",
    "DummyExecutor",
    "FileRecord",
    "InferenceConfig",
    "InferenceEngine",
    "JsonStorage",
    "Metrics",
    "MpQueue",
    "Result",
    "Scheduler",
    "Task",
    "TaskRepository",
    "TaskStatus",
    "VLMExecutor",
    "Validator",
    "Writer",
    "make_result_queue",
    "make_task_queue",
    "worker_main",
]
