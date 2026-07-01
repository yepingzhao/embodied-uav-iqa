"""Configuration for the inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InferenceConfig:
    """All knobs needed to run the inference engine.

    Attributes:
        input_dir: Directory containing input ``*_VQA_*.json`` files.
        output_dir: Directory where output JSONs are written.
        checkpoint_path: Path to the SQLite checkpoint database.
        chunk_size: Number of records per Task (scheduling unit).
        batch_size: Number of samples per GPU forward pass.
        num_gpus: Number of GPUs (workers) to launch.
        model_name: VLM model short name (e.g. ``"Qwen2.5-VL"``).
        backend: VLM backend (``"auto"``, ``"vllm"``, ``"transformers"``, ``"none"``).
        device: Device override (e.g. ``"cuda:0"``). If empty, auto-assign per worker.
        seed: Random seed for reproducibility.
        glob_pattern: Glob pattern for scanning input files.
        max_entries: Optional cap on total entries (for testing). ``0`` = no limit.
        scorer_kwargs: Extra keyword arguments forwarded to the VLM scorer.
    """

    input_dir: Path
    output_dir: Path
    checkpoint_path: Path
    chunk_size: int = 256
    batch_size: int = 16
    num_gpus: int = 1
    model_name: str = "Qwen2.5-VL"
    backend: str = "auto"
    device: str = ""
    seed: int = 42
    glob_pattern: str = "*_VQA_*.json"
    max_entries: int = 0
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)
