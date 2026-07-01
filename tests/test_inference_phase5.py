"""Tests for Phase 5: CLI (scripts/inference.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_input(tmp_path: Path) -> Path:
    """Create a minimal input dir with one VQA file."""
    d = tmp_path / "input"
    d.mkdir()
    records = [
        {
            "sample_id": f"s{i}",
            "question": f"q{i}",
            "uav_paths": {"UAV1": "r.jpg"},
            "distorted_uav_paths": {"UAV1": "d.png"},
            "vlm_scores": {},
            "cognitive_score": None,
        }
        for i in range(10)
    ]
    (d / "Sim3_VQA_train.json").write_text(json.dumps(records))
    return d


class TestCLI:
    def test_dry_run(self, tmp_path: Path) -> None:
        """Run the CLI in dry-run mode and verify output."""
        input_dir = _make_input(tmp_path)
        output_dir = tmp_path / "output"
        checkpoint = tmp_path / "ckpt.db"
        repo_root = Path(__file__).resolve().parent.parent

        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "inference.py"),
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--checkpoint-path", str(checkpoint),
                "--num-gpus", "1",
                "--chunk-size", "10",
                "--batch-size", "4",
                "--dry-run",
                "--log-level", "WARNING",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(repo_root),
        )

        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert output_dir.exists()
        out_file = output_dir / "Sim3_VQA_train.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 10
        for entry in data:
            assert entry["vlm_scores"]["dry_run"] == 0.5
            assert entry["cognitive_score"] == 0.5

    def test_help(self) -> None:
        """--help should exit 0 and show usage."""
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "inference.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
        assert result.returncode == 0
        assert "inference" in result.stdout.lower()
        assert "--input-dir" in result.stdout

    def test_missing_required_arg(self) -> None:
        """Missing --input-dir should exit non-zero."""
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "inference.py"),
             "--output-dir", "/tmp/out",
             "--checkpoint-path", "/tmp/ckpt.db"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
        assert result.returncode != 0

    def test_resume_flag(self, tmp_path: Path) -> None:
        """Run once, then run with --resume (should be a no-op)."""
        input_dir = _make_input(tmp_path)
        output_dir = tmp_path / "output"
        checkpoint = tmp_path / "ckpt.db"
        repo_root = Path(__file__).resolve().parent.parent
        script = str(repo_root / "scripts" / "inference.py")
        common_args = [
            sys.executable, script,
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--checkpoint-path", str(checkpoint),
            "--num-gpus", "1",
            "--dry-run",
            "--log-level", "WARNING",
        ]

        # First run
        r1 = subprocess.run(common_args, capture_output=True, text=True, timeout=60, cwd=str(repo_root))
        assert r1.returncode == 0

        # Resume run — should complete with no new work
        r2 = subprocess.run(common_args + ["--resume"], capture_output=True, text=True, timeout=60, cwd=str(repo_root))
        assert r2.returncode == 0
