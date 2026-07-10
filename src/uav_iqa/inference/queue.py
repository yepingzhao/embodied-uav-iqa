"""Abstract queue interface for task and result distribution.

Phase 1 uses :class:`multiprocessing.Queue`. The abstraction allows future
backends (Ray, Redis) to replace the implementation without touching workers
or the scheduler.
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

# A unique string sentinel that survives pickling across processes.
_SENTINEL = "__INFERENCE_QUEUE_SENTINEL__"


class BaseQueue(ABC, Generic[T]):
    """Abstract queue with ``put``, ``get``, and ``close`` semantics."""

    @abstractmethod
    def put(self, item: T) -> None:
        """Enqueue *item*. Blocks if the queue is bounded and full."""

    @abstractmethod
    def get(self) -> T | None:
        """Dequeue and return the next item, or ``None`` if the queue is
        closed and empty."""

    @abstractmethod
    def close(self) -> None:
        """Signal that no more items will be added. Subsequent ``get``
        calls return ``None`` once the queue is drained."""

    @abstractmethod
    def qsize(self) -> int:
        """Approximate number of items currently in the queue."""

    def put_sentinel(self) -> None:
        """Insert a sentinel value to mark the end of the stream."""
        self.put(_SENTINEL)  # type: ignore[arg-type]

    def _is_sentinel(self, item: object) -> bool:
        """Check if *item* is the close sentinel."""
        return item == _SENTINEL


class MpQueue(BaseQueue[T]):
    """A thin wrapper around :class:`multiprocessing.Queue`.

    A sentinel value is used to signal closure because ``mp.Queue`` has no
    native "close" that unblocks pending ``get`` calls.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._q: mp.Queue[T] = mp.Queue(maxsize=maxsize)
        self._closed = False

    def put(self, item: T) -> None:
        self._q.put(item)

    def get(self) -> T | None:
        item = self._q.get()
        if self._is_sentinel(item):
            return None
        return item  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def qsize(self) -> int:
        return self._q.qsize()


def make_task_queue(maxsize: int = 0) -> BaseQueue:
    """Factory for the default task queue."""
    return MpQueue(maxsize=maxsize)


def make_result_queue(maxsize: int = 0) -> BaseQueue:
    """Factory for the default result queue."""
    return MpQueue(maxsize=maxsize)
