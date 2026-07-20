"""Scan final VLM JSON output and reset checkpoint tasks whose records
contain empty (prompt/ref_answer/dist_answer all blank) entries back to
FAILED, so a subsequent --resume re-runs only those chunks.

Usage:
    python scripts/reset_empty_tasks.py \
        --input-dir data/annotated/vlm/Qwen2.5-VL \
        --checkpoint data/annotated/checkpoints/qwen2.5-vl.db \
        --chunk-size 256
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

EMPTY = {"", None}


def is_empty_record(rec: dict) -> bool:
    md = rec.get("_model_details", {})
    det = md.get("Qwen2.5-VL") if isinstance(md, dict) else None
    if not isinstance(det, dict):
        return False
    return (
        det.get("prompt") in EMPTY
        and det.get("ref_answer") in EMPTY
        and det.get("dist_answer") in EMPTY
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Map file_stem -> ordered list of record offsets, find empty record indices
    empty_by_file: dict[str, set[int]] = {}

    json_files = sorted(args.input_dir.rglob("*.json"))
    # exclude chunk temp files
    json_files = [p for p in json_files if ".chunks" not in p.parts]

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {jf}: {e}")
            continue
        if not isinstance(data, list) or not data:
            continue
        # determine file_stem relative to input-dir structure
        stem = jf.stem
        empties = {i for i, rec in enumerate(data) if is_empty_record(rec)}
        if empties:
            empty_by_file[stem] = empties

    print(f"Files with empty records: {len(empty_by_file)}")
    for stem, empties in sorted(empty_by_file.items()):
        print(f"  {stem}: {len(empties)} empty records")

    # Build task_id -> empty? from chunk ranges
    affected_tasks: list[str] = []
    for stem, empties in empty_by_file.items():
        # chunk boundaries over record indices 0..max(empties)
        max_idx = max(empties)
        for chunk_id, start in enumerate(range(0, max_idx + 1, args.chunk_size)):
            end = min(start + args.chunk_size, max_idx + 1)
            # does this chunk contain any empty record?
            if any(start <= i < end for i in empties):
                affected_tasks.append(f"{stem}__{chunk_id:04d}")

    affected_tasks = sorted(set(affected_tasks))
    print(f"Total affected tasks to reset: {len(affected_tasks)}")

    if args.dry_run:
        for t in affected_tasks:
            print("  WOULD RESET", t)
        return

    conn = sqlite3.connect(args.checkpoint)
    cur = conn.cursor()
    # verify tasks exist
    placeholders = ",".join("?" * len(affected_tasks))
    rows = cur.execute(
        f"SELECT task_id, status FROM tasks WHERE task_id IN ({placeholders})",
        affected_tasks,
    ).fetchall()
    existing = {r[0] for r in rows}
    print(f"Tasks found in DB: {len(existing)}")
    missing = set(affected_tasks) - existing
    if missing:
        print(f"WARNING: {len(missing)} tasks not in DB (skipped)")
    to_reset = [r[0] for r in rows]
    if to_reset:
        cur.executemany(
            "UPDATE tasks SET status='failed', updated_at=datetime('now') WHERE task_id=?",
            [(t,) for t in to_reset],
        )
        conn.commit()
        print(f"Reset {len(to_reset)} tasks to FAILED")
    conn.close()


if __name__ == "__main__":
    main()
