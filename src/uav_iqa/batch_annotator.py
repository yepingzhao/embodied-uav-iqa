"""Batch annotation engine for multi-UAV grouped JSON scoring.

Orchestrates loading grouped processed JSON files, scoring scene+frame groups
with ``VLMScorer.score_multi_image()``, checkpointing progress, and writing
updated JSONs back to disk.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


class BatchAnnotator:
    """Orchestrate batch VLM annotation of grouped processed JSONs.

    Parameters
    ----------
    scorer:
        A scorer instance (e.g. ``VLMScorer``).
    output_dir:
        Root processed output directory (contains train/ and test/ subdirs).
    """

    def __init__(self, scorer, output_dir: Path):
        self.scorer = scorer
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_groups(json_path: Path) -> list[dict]:
        with open(json_path) as f:
            return json.load(f)

    @staticmethod
    def write_groups(groups: list[dict], output_path: Path) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(output_path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(groups, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, output_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return len(groups)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    @staticmethod
    def save_checkpoint(
        checkpoint_path: Path,
        scores: Dict[str, dict],
        completed: int,
        total: int,
        metadata: Optional[dict] = None,
    ):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        scores_list = [[k, v] for k, v in scores.items()]
        payload = {
            "completed": completed,
            "total": total,
            "scores": scores_list,
            "metadata": metadata or {},
        }
        fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(checkpoint_path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> dict:
        with open(checkpoint_path) as f:
            payload = json.load(f)
        payload["scores"] = dict(payload["scores"])
        return payload

    # ------------------------------------------------------------------
    # Core annotation
    # ------------------------------------------------------------------

    def annotate_file(
        self,
        json_path: Path,
        *,
        max_entries: Optional[int] = None,
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: int = 100,
    ) -> dict:
        """Annotate one grouped JSON file.

        Parameters
        ----------
        json_path:
            Path to the processed JSON (e.g. ``train/Sim3_VQA_train.json``).
        max_entries:
            Cap on how many VQA entries to score (None = unlimited).
        checkpoint_path:
            Where to save/load checkpoint.
        checkpoint_interval:
            Save checkpoint every N entries.

        Returns
        -------
        dict with ``scored``, ``total``.
        """
        groups = self.load_groups(json_path)

        scored = 0
        total = 0

        model_name = getattr(self.scorer, "model_name", "unknown")

        for gi, group in enumerate(groups):
            uav_paths = group.get("uav_paths", {})
            uav_keys = group.get("uav_keys", [])
            distortions = group.get("distortions", {})

            if not uav_paths or not uav_keys:
                continue

            ref_image_paths = [
                str(self.output_dir.parent / uav_paths.get(k, "").lstrip("/"))
                for k in uav_keys
            ]

            for dist_key, dist_info in distortions.items():
                dist_image_paths = []
                for k in uav_keys:
                    dp = dist_info.get("distorted_uav_paths", {}).get(k, "")
                    dist_image_paths.append(
                        str(self.output_dir.parent / dp.lstrip("/"))
                    )

                for entry in group.get("vqa_entries", []):
                    total += 1

                    if max_entries and scored >= max_entries:
                        break

                    cognitive = self.scorer.score_multi_image(
                        ref_image_paths=ref_image_paths,
                        dist_image_paths=dist_image_paths,
                        question=entry.get("question", ""),
                        subtask_type=entry.get("subtask_type", ""),
                    )

                    entry.setdefault("vlm_scores", {})[model_name] = cognitive
                    scored += 1

                    if checkpoint_path and scored % checkpoint_interval == 0:
                        self.write_groups(groups, json_path)
                        self.save_checkpoint(
                            checkpoint_path,
                            scores={"groups_processed": gi, "entries_scored": scored},
                            completed=scored,
                            total=total,
                            metadata={
                                "json_path": str(json_path),
                                "model": model_name,
                                "timestamp": time.time(),
                            },
                        )

                if max_entries and scored >= max_entries:
                    break
            if max_entries and scored >= max_entries:
                break

        self.write_groups(groups, json_path)

        return {"scored": scored, "total": total}

    def annotate_splits(
        self,
        splits: List[str],
        *,
        max_entries: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 100,
        resume: bool = False,
    ) -> Dict[str, dict]:
        """Annotate all VQA JSONs in the given splits.

        Returns
        -------
        Dict mapping ``split/json_name`` -> ``{scored, total}``.
        """
        results: Dict[str, dict] = {}

        for split in splits:
            split_dir = self.output_dir / split
            if not split_dir.is_dir():
                _log.warning("Split directory not found: %s", split_dir)
                continue

            for fpath in sorted(split_dir.glob("*_VQA_*.json")):
                key = f"{split}/{fpath.name}"
                ckpt_path = None
                if checkpoint_dir:
                    safe_name = getattr(self.scorer, "model_name", "unknown").replace(
                        "/", "_"
                    )
                    ckpt_path = checkpoint_dir / f"{safe_name}_{split}_{fpath.stem}.json"
                    if resume and ckpt_path.exists():
                        _log.info("Resuming from checkpoint: %s", ckpt_path)
                    elif not resume and ckpt_path.exists():
                        ckpt_path.unlink()

                results[key] = self.annotate_file(
                    json_path=fpath,
                    max_entries=max_entries,
                    checkpoint_path=ckpt_path,
                    checkpoint_interval=checkpoint_interval,
                )
                _log.info("  %s: scored=%d", key, results[key]["scored"])

        return results
