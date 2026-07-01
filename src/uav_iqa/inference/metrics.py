"""Runtime metrics for the inference engine.

Tracks throughput, ETA, and queue depth. Designed to be polled by the
engine for logging or progress reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Lightweight runtime statistics.

    Call :meth:`record_task` each time a result is collected, and
    :meth:`snapshot` to get a point-in-time summary.
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_samples: int = 0
    completed_samples: int = 0
    start_time: float = field(default_factory=time.time)

    def record_task(self, num_outputs: int, failed: int = 0) -> None:
        """Record completion of one task."""
        self.completed_tasks += 1
        self.completed_samples += num_outputs
        self.failed_tasks += failed

    def set_totals(self, total_tasks: int, total_samples: int) -> None:
        """Set the total expected counts (from the scheduler)."""
        self.total_tasks = total_tasks
        self.total_samples = total_samples

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def tasks_per_sec(self) -> float:
        elapsed = self.elapsed()
        if elapsed <= 0:
            return 0.0
        return self.completed_tasks / elapsed

    def samples_per_sec(self) -> float:
        elapsed = self.elapsed()
        if elapsed <= 0:
            return 0.0
        return self.completed_samples / elapsed

    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        rate = self.tasks_per_sec()
        if rate <= 0:
            return float("inf")
        remaining = self.total_tasks - self.completed_tasks
        return remaining / rate

    def progress_pct(self) -> float:
        """Completion percentage (0–100)."""
        if self.total_tasks <= 0:
            return 0.0
        return 100.0 * self.completed_tasks / self.total_tasks

    def snapshot(self) -> dict[str, float | int]:
        """Return a dict suitable for logging."""
        return {
            "progress_pct": round(self.progress_pct(), 1),
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "completed_samples": self.completed_samples,
            "total_samples": self.total_samples,
            "tasks_per_sec": round(self.tasks_per_sec(), 2),
            "samples_per_sec": round(self.samples_per_sec(), 2),
            "elapsed_sec": round(self.elapsed(), 1),
            "eta_sec": round(self.eta_seconds(), 1) if self.eta_seconds() != float("inf") else -1,
            "failed_tasks": self.failed_tasks,
        }
