"""VQA question index for AirCopBench images.

Reads VQA JSON files from ``train/`` (and optionally ``test/``) subdirectories.
Filters to single-UAV questions and indexes by UAV image stem.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

_log = logging.getLogger(__name__)


class VQAIndex:
    """Load AirCopBench VQA questions and index by UAV image stem.

    Reads ``Sim*_VQA*train.json`` and ``Sim*_VQA*test.json`` from the
    provided ``vqa_dir``.  Filters to single-UAV questions (options that
    don't reference other UAVs).  Each image maps to a list of formatted
    question prompts for the VLM to answer.
    """

    def __init__(self, vqa_dir: str):
        self._index: Dict[str, List[str]] = {}

        vqa_path = Path(vqa_dir)
        if not vqa_path.is_dir():
            _log.warning("VQA directory not found: %s", vqa_dir)
            return

        glob_patterns = ["Sim*_VQA*train.json", "Sim*_VQA*test.json"]
        for pattern in glob_patterns:
            for fpath in sorted(vqa_path.glob(pattern)):
                self._index_file(fpath)

        # Also check train/ and test/ subdirectories
        for subdir in ("train", "test"):
            sub_path = vqa_path / subdir
            if sub_path.is_dir():
                for pattern in glob_patterns:
                    for fpath in sorted(sub_path.glob(pattern)):
                        self._index_file(fpath)

        total_questions = sum(len(v) for v in self._index.values())
        _log.info(
            "VQAIndex loaded: %d stems, %d questions",
            len(self._index),
            total_questions,
        )

    def _index_file(self, fpath: Path) -> None:
        data = json.loads(fpath.read_text())
        for item in data:
            uav_id = item.get("uav_id", "")
            img = item.get("uav_paths", {}).get(uav_id, "")
            if not img:
                continue
            options = item.get("options", {})
            is_multi = any(
                "UAV" in v and uav_id not in str(v) for v in options.values()
            )
            if is_multi:
                continue
            stem = Path(img).stem
            prompt = f"{item['question']}\nAnswer concisely in 2-4 sentences."
            if stem not in self._index:
                self._index[stem] = []
            self._index[stem].append(prompt)

    def get_prompts(self, image_path: str) -> List[str]:
        """Return VQA question prompts for an image, or empty list if no match."""
        stem = Path(image_path).stem
        for vqa_stem, prompts in self._index.items():
            if stem.endswith(vqa_stem):
                return list(prompts)
        return []
