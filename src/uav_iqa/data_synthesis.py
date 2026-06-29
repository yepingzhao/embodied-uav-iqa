"""Data synthesis pipeline for UAV-IQA with multi-UAV distortion injection.

Provides:
- DatasetFormat: minimal abstract base for extract step
- AirCopBenchFormat / GenericImageDirFormat: concrete format classes
- DataSynthesisPipeline: orchestrates extract/inject/annotate/aggregate steps
- create_pipeline: convenience factory function

The pipeline produces a grouped JSON format where each scene+frame is one
group containing all VQA entries and all distortion variants.  Training
samples are (group, distortion) pairs with multi-image input.
"""

import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from uav_iqa.annotations import (
    build_sample_id,
    extract_subtask_type,
    get_dataset_name,
    group_by_scene_frame,
    seed_for_distortion,
    SUBTASK_NAMES,
)
from uav_iqa.distortion import UAVDistortionPipeline
from uav_iqa.utils import find_images

_log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ===========================================================================
# DatasetFormat — minimal ABC for extract step
# ===========================================================================


class DatasetFormat(ABC):
    """Minimal interface for dataset-specific reference image discovery.

    Subclasses register via ``@DatasetFormat.register`` and must set
    ``name`` as a class-level attribute.
    """

    _registry: dict[str, type["DatasetFormat"]] = {}
    name: str

    @classmethod
    def register(cls, format_cls: type["DatasetFormat"]) -> type["DatasetFormat"]:
        cls._registry[format_cls.name] = format_cls
        return format_cls

    @classmethod
    def get(cls, name: str) -> type["DatasetFormat"]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown dataset format: '{name}'. Available: {cls.list_formats()}"
            )
        return cls._registry[name]

    @classmethod
    def list_formats(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abstractmethod
    def get_exclude_dirs(self) -> set[str]:
        ...

    @abstractmethod
    def find_reference_images(self, input_root: Path) -> list[Path]:
        ...


# ===========================================================================
# Concrete formats
# ===========================================================================


@DatasetFormat.register
class AirCopBenchFormat(DatasetFormat):
    name = "aircopbench"

    EXCLUDE_DIRS: set[str] = {
        "loss",
        "noise",
        "point_clouds",
        "Annotations",
        "original_json",
        "original_xml",
        "train",
        "test",
        ".git",
        "distorted",
    }

    def get_exclude_dirs(self) -> set[str]:
        return self.EXCLUDE_DIRS

    def find_reference_images(self, input_root: Path) -> list[Path]:
        return find_images(input_root, exclude_dirs=self.EXCLUDE_DIRS)


@DatasetFormat.register
class GenericImageDirFormat(DatasetFormat):
    name = "generic"

    def get_exclude_dirs(self) -> set[str]:
        return set()

    def find_reference_images(self, input_root: Path) -> list[Path]:
        return find_images(input_root)


# ===========================================================================
# DataSynthesisPipeline
# ===========================================================================


class DataSynthesisPipeline:
    """Orchestrates the data synthesis pipeline for UAV-IQA.

    Steps:
    1. extract  — symlink reference images and copy VQA JSONs
    2. inject   — group by scene+frame, apply 36 distortions to all UAVs
    3. annotate — VLM multi-image inference for cognitive scores
    4. aggregate — merge model scores, compute final cognitive_score

    ``run_full()`` composes them end-to-end.
    """

    def __init__(self, dataset_format: DatasetFormat, seed: int = 42):
        self.format = dataset_format
        self.seed = seed

    # ---- Step 1: Extract reference frames ----

    def extract_references(
        self,
        input_root: str | Path,
        output_dir: str | Path,
        copy: bool = False,
        max_refs_per_source: dict[str, int] | None = None,
    ) -> Path:
        """Extract clean reference frames into a flat directory.

        Flattens nested directory structure: path parts joined with ``_``.
        Default is symlink to save disk space; use ``copy=True`` for copies.
        Also copies VQA JSON files from ``input_root/{train,test}/`` to
        ``output_dir/../{train,test}/``.

        Args:
            input_root: Dataset root directory.
            output_dir: Where to place extracted reference images (typically
                        ``processed/ref_images``).
            copy: If True, copy files instead of symlinking.
            max_refs_per_source: Per-source image limit keyed by first path
                component relative to ``input_root`` (e.g. ``{"Sim_3_UAVs": 500}``).

        Returns:
            The output_dir Path for chaining.
        """
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = self.format.find_reference_images(input_root)
        if not images:
            raise RuntimeError(f"No reference images found in {input_root}")
        _log.info("Found %d clean reference images in %s", len(images), input_root)

        if max_refs_per_source:
            import random

            rng = random.Random(self.seed)
            by_source: dict[str, list[Path]] = {}
            for img in images:
                source = img.relative_to(input_root).parts[0]
                by_source.setdefault(source, []).append(img)
            images = []
            for source, src_images in sorted(by_source.items()):
                limit = max_refs_per_source.get(source, len(src_images))
                rng.shuffle(src_images)
                sampled = src_images[:limit]
                images.extend(sampled)
                _log.info(
                    "  %s: %d -> %d (limit=%d)",
                    source,
                    len(by_source[source]),
                    len(sampled),
                    limit,
                )
            _log.info("Sampled %d total images", len(images))

        created = 0
        skipped = 0

        def _copy_one(img: Path) -> tuple[int, int]:
            rel = img.relative_to(input_root)
            stem = "_".join(rel.parts).replace("/", "_").replace("\\", "_")
            dest = output_dir / stem
            if not dest.exists():
                if copy:
                    shutil.copy2(img, dest)
                else:
                    try:
                        dest.symlink_to(img.resolve())
                    except OSError:
                        _log.debug("Symlink failed for %s, falling back to copy", img)
                        shutil.copy2(img, dest)
                return 1, 0
            return 0, 1

        max_w = min(16, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            futures = [ex.submit(_copy_one, img) for img in images]
            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting references",
                unit="img",
            ):
                c, s = f.result()
                created += c
                skipped += s

        op = "Copied" if copy else "Symlinked"
        _log.info(
            "%s %d images to %s (%d skipped, already exist)",
            op,
            created,
            output_dir,
            skipped,
        )
        return output_dir

    # ---- Step 2: Inject distortions ----

    def inject_distortions(
        self,
        input_root: str | Path,
        output_dir: str | Path,
        workers: int = 0,
        compress: bool = True,
        dry_run: bool = False,
        fmt: str = "png",
    ) -> dict:
        """Apply all 36 distortion types to all UAV images within each scene+frame.

        Reads VQA JSON files from ``output_dir/{train,test}/``, groups entries
        by ``sequence_frame``, and applies the SAME distortion type+intensity
        to all UAV images in a group.

        Produces:
        - ``distorted/{sample_id}_{uav_idx}.{fmt}`` image files
        - Enriched JSON files in ``output_dir/{train,test}/`` with
          ``reference_uav_paths``, ``distorted_uav_paths``, ``distortions``,
          and per-entry ``vlm_scores`` / ``cognitive_score`` fields.

        Args:
            input_root: Raw dataset root (for resolving VQA UAV paths).
            output_dir: Processed output directory (contains VQA JSONs).
            workers: Parallel worker processes (0 = auto).
            compress: Save images with compression.
            dry_run: Process only first 2 groups, first 3 distortions.
            fmt: Output format, ``"png"`` or ``"jpeg"``.

        Returns:
            Dict with ``total_groups``, ``total_distortions``, ``total_entries``.
        """
        if workers <= 0:
            workers = min(os.cpu_count() or 4, 16)
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        distorted_dir = output_dir / "distorted"
        distorted_dir.mkdir(parents=True, exist_ok=True)

        vqa_files: list[tuple[str, Path]] = []
        for split_name in ("train", "test"):
            split_dir = output_dir / split_name
            if not split_dir.is_dir():
                _log.warning("Split directory not found: %s", split_dir)
                continue
            for fpath in sorted(split_dir.glob("*_VQA_*.json")):
                vqa_files.append((split_name, fpath))

        if not vqa_files:
            raise RuntimeError(
                f"No VQA JSON files found in {output_dir}/{{train,test}}/. "
                f"Run the extract step first to copy VQA files."
            )

        _log.info("Found %d VQA files to process", len(vqa_files))

        all_distortions = UAVDistortionPipeline.get_all_distortion_names()
        _log.info(
            "%d distortion types × 5 intensity levels = %d variants per group",
            len(all_distortions),
            len(all_distortions),
        )

        stats = {"total_groups": 0, "total_distortions": 0, "total_entries": 0}

        for split_name, fpath in vqa_files:
            _log.info("Processing %s/%s", split_name, fpath.name)
            dataset_name = get_dataset_name(fpath.name)

            with open(fpath) as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                _log.warning("Skipping non-list JSON: %s", fpath)
                continue

            groups = group_by_scene_frame(entries)
            _log.info("  %d groups from %d entries", len(groups), len(entries))

            if dry_run:
                groups = dict(list(groups.items())[:2])
                distortions_subset = all_distortions[:3]
                _log.info("  [DRY RUN] %d groups, %d distortion types", len(groups), len(distortions_subset))
            else:
                distortions_subset = all_distortions

            output_entries = self._inject_groups(
                groups=groups,
                dataset_name=dataset_name,
                split_name=split_name,
                input_root=input_root,
                distorted_dir=distorted_dir,
                distortions_subset=distortions_subset,
                compress=compress,
                fmt=fmt,
                workers=workers,
                base_seed=self.seed,
            )

            out_path = output_dir / split_name / fpath.name
            with open(out_path, "w") as f:
                json.dump(output_entries, f, indent=2, ensure_ascii=False)
            _log.info("  Wrote %d entries to %s", len(output_entries), out_path)

            stats["total_groups"] += len(groups)
            stats["total_distortions"] += sum(
                len(g.get("distortions", {}))
                for g in output_entries
                if isinstance(g, dict)
            )
            stats["total_entries"] += len(output_entries)

        _log.info(
            "Inject complete: %d groups, %d entries, %d total distortions",
            stats["total_groups"],
            stats["total_entries"],
            stats["total_distortions"],
        )
        return stats

    def _inject_groups(
        self,
        groups: dict[str, list[dict]],
        dataset_name: str,
        split_name: str,
        input_root: Path,
        distorted_dir: Path,
        distortions_subset: list[str],
        compress: bool,
        fmt: str,
        workers: int,
        base_seed: int,
    ) -> list[dict]:
        """Apply distortions to all groups and return enriched entries."""
        output: list[dict] = []
        ext = ".jpg" if fmt == "jpeg" else ".png"

        for seq_frame, entries in tqdm(
            sorted(groups.items()),
            desc=f"  {dataset_name}/{split_name}",
            unit="group",
        ):
            uav_paths = entries[0].get("uav_paths", {})
            if not uav_paths:
                continue

            uav_keys = sorted(uav_paths.keys())
            num_uavs = len(uav_keys)

            distortions: dict[str, dict] = {}
            for dist_name in distortions_subset:
                intensity = self._pick_intensity(dist_name, dataset_name, seq_frame)
                level = int(intensity * 10)
                seed = seed_for_distortion(dataset_name, seq_frame, dist_name, base_seed)
                sample_id = build_sample_id(dataset_name, seq_frame, dist_name, level)

                dist_pipeline = UAVDistortionPipeline(seed=seed)
                distorted_uav_paths: dict[str, str] = {}

                skip_group = False
                for idx, uav_key in enumerate(uav_keys):
                    uav_rel = uav_paths.get(uav_key, "")
                    if not uav_rel:
                        skip_group = True
                        break
                    uav_abs = input_root / uav_rel.lstrip("/").replace("\\", "/")
                    if not uav_abs.exists():
                        _log.warning("UAV image not found: %s", uav_abs)
                        skip_group = True
                        break

                    out_name = f"{sample_id}_{idx}{ext}"
                    out_path = distorted_dir / out_name

                    if not out_path.exists():
                        img = cv2.imread(str(uav_abs))
                        if img is None:
                            skip_group = True
                            break
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        dist_img = dist_pipeline.apply_distortion(img_rgb, dist_name, intensity)
                        dist_bgr = cv2.cvtColor(dist_img, cv2.COLOR_RGB2BGR)
                        if fmt == "jpeg":
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
                        elif compress:
                            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                        else:
                            encode_params = []
                        success, buf = cv2.imencode(ext, dist_bgr, encode_params)
                        if not success:
                            _log.warning("Encoding failed: %s", out_path)
                            skip_group = True
                            break
                        out_path.write_bytes(buf.tobytes())

                    distorted_uav_paths[uav_key] = str(
                        out_path.relative_to(distorted_dir.parent)
                    )

                if skip_group:
                    continue

                dist_key = f"{dist_name}_L{level:02d}"
                distortions[dist_key] = {
                    "sample_id": sample_id,
                    "type": dist_name,
                    "category": UAVDistortionPipeline.get_distortion_categories().get(
                        dist_name, "unknown"
                    ),
                    "intensity": intensity,
                    "level": level,
                    "seed": seed,
                    "distorted_uav_paths": distorted_uav_paths,
                }

            if not distortions:
                continue

            group_entry = {
                "dataset": dataset_name,
                "split": split_name,
                "sequence_frame": seq_frame,
                "uav_paths": uav_paths,
                "uav_keys": uav_keys,
                "num_uavs": num_uavs,
                "distortions": distortions,
                "vqa_entries": [
                    {
                        "question_id": e.get("question_id", ""),
                        "question_type": e.get("question_type", ""),
                        "question": e.get("question", ""),
                        "options": e.get("options", []),
                        "correct_answer": e.get("correct_answer", ""),
                        "subtask_type": extract_subtask_type(e.get("question_type", "")),
                        "subtask_name": SUBTASK_NAMES.get(
                            extract_subtask_type(e.get("question_type", "")), "unknown"
                        ),
                        "uav_id": e.get("uav_id", ""),
                        "vlm_scores": {},
                        "cognitive_score": 0.0,
                    }
                    for e in entries
                ],
            }
            output.append(group_entry)

        return output

    @staticmethod
    def _pick_intensity(
        distortion_name: str,
        dataset_name: str,
        sequence_frame: str,
    ) -> float:
        """Pick a deterministic random intensity level for a distortion.

        Uses the distortion name + dataset + sequence_frame as seed input
        so re-running produces the same level for the same input.
        """
        key = f"{dataset_name}__{sequence_frame}__{distortion_name}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        rng = np.random.RandomState(h % (2**31))
        return float(rng.choice(UAVDistortionPipeline.INTENSITY_LEVELS))

    # ---- Step 3: Annotate scores ----

    def annotate_scores(
        self,
        output_dir: str | Path,
        scorer,
        scorer_batch_size: int = 8,
        max_entries: int | None = None,
    ) -> None:
        """Annotate processed groups with VLM cognitive scores.

        For each group+distortion pair, generates reference and distorted
        descriptions via multi-image VLM inference, then computes text
        similarity to derive a ``cognitive_score``.

        Writes detailed per-model records to ``vlm/<model>/{split}/*.json``
        and updates the main JSON ``vlm_scores`` / ``cognitive_score`` fields.

        Args:
            output_dir: Processed output directory with train/test JSONs.
            scorer: VLM scorer instance (``VLMScorer``).
            scorer_batch_size: Batch size for VLM scoring.
            max_entries: Limit number of entries to annotate per split (for debugging).
        """
        output_dir = Path(output_dir)
        model_name = getattr(scorer, "model_name", "unknown")

        for split_name in ("train", "test"):
            split_dir = output_dir / split_name
            if not split_dir.is_dir():
                continue

            for fpath in sorted(split_dir.glob("*_VQA_*.json")):
                self._annotate_one_file(
                    fpath=fpath,
                    output_dir=output_dir,
                    split_name=split_name,
                    model_name=model_name,
                    scorer=scorer,
                    scorer_batch_size=scorer_batch_size,
                    max_entries=max_entries,
                )

    def _annotate_one_file(
        self,
        fpath: Path,
        output_dir: Path,
        split_name: str,
        model_name: str,
        scorer,
        scorer_batch_size: int,
        max_entries: int | None,
    ) -> None:
        """Annotate one processed JSON file with a single VLM model."""
        with open(fpath) as f:
            groups = json.load(f)

        if not isinstance(groups, list) or not groups:
            return

        _log.info("Annotating %s [%s/%s]", fpath.name, split_name, model_name)

        processed = 0
        for group in tqdm(groups, desc=f"  {model_name}/{split_name}/{fpath.stem}", unit="grp"):
            distortions = group.get("distortions", {})
            uav_paths = group.get("uav_paths", {})
            uav_keys = group.get("uav_keys", [])

            if not uav_paths or not uav_keys:
                continue

            ref_image_paths = [str(output_dir.parent / uav_paths.get(k, "").lstrip("/"))
                for k in uav_keys]

            for dist_key, dist_info in distortions.items():
                dist_image_paths = []
                for k in uav_keys:
                    dp = dist_info["distorted_uav_paths"].get(k, "")
                    dist_image_paths.append(str(output_dir.parent / dp.lstrip("/")))

                for entry in group.get("vqa_entries", []):
                    question = entry.get("question", "")
                    if not question:
                        continue

                    cognitive = scorer.score_multi_image(
                        ref_image_paths=ref_image_paths,
                        dist_image_paths=dist_image_paths,
                        question=question,
                        subtask_type=entry.get("subtask_type", ""),
                    )

                    entry.setdefault("vlm_scores", {})[model_name] = cognitive
                    entry["cognitive_score"] = round(
                        float(np.mean(list(entry["vlm_scores"].values()))), 6
                    )

                    processed += 1
                    if max_entries and processed >= max_entries:
                        break

                if max_entries and processed >= max_entries:
                    break
            if max_entries and processed >= max_entries:
                break

        with open(fpath, "w") as f:
            json.dump(groups, f, indent=2, ensure_ascii=False)
        _log.info("  Annotated %d entries -> %s", processed, fpath)

    # ---- Step 4: Aggregate scores ----

    def aggregate_scores(
        self,
        output_dir: str | Path,
        strategy: str = "mean",
    ) -> None:
        """Compute final cognitive_score by aggregating per-model vlm_scores.

        Args:
            output_dir: Processed output directory.
            strategy: Aggregation strategy: ``"mean"`` (default).
        """
        output_dir = Path(output_dir)

        for split_name in ("train", "test"):
            split_dir = output_dir / split_name
            if not split_dir.is_dir():
                continue

            for fpath in sorted(split_dir.glob("*_VQA_*.json")):
                with open(fpath) as f:
                    groups = json.load(f)

                updated = 0
                for group in groups:
                    for entry in group.get("vqa_entries", []):
                        scores = entry.get("vlm_scores", {})
                        if scores:
                            if strategy == "mean":
                                entry["cognitive_score"] = round(
                                    float(np.mean(list(scores.values()))), 6
                                )
                            updated += 1

                if updated:
                    with open(fpath, "w") as f:
                        json.dump(groups, f, indent=2, ensure_ascii=False)
                _log.info("Aggregated %d entries in %s", updated, fpath.name)

    # ---- Full pipeline ----

    @staticmethod
    def _copy_vqa_files(input_root: Path, output_dir: Path) -> None:
        """Copy VQA JSON files from raw input to processed output directory."""
        for split_name in ("train", "test"):
            src_dir = input_root / split_name
            if not src_dir.is_dir():
                _log.info("VQA source not found: %s", src_dir)
                continue
            dst_dir = output_dir / split_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fpath in sorted(src_dir.glob("*_VQA_*.json")):
                dst = dst_dir / fpath.name
                if not dst.exists():
                    shutil.copy2(fpath, dst)
            copied = sum(1 for _ in dst_dir.glob("*_VQA_*.json"))
            _log.info("Copied %d VQA files to %s", copied, dst_dir)

    def run_full(
        self,
        input_root: str | Path,
        output_dir: str | Path,
        steps: str = "all",
        copy: bool = False,
        workers: int = 0,
        compress: bool = True,
        split: tuple = (0.8, 0.2),
        dry_run: bool = False,
        fmt: str = "png",
        max_refs_per_source: dict[str, int] | None = None,
        scorer=None,
        scorer_batch_size: int = 8,
        max_annotate_entries: int | None = None,
    ) -> None:
        """Run the full data synthesis pipeline end-to-end.

        Args:
            input_root: Raw dataset root directory.
            output_dir: Root output directory (creates subdirs inside).
            steps: Comma-separated step names: ``"extract"``, ``"inject"``,
                   ``"annotate"``, ``"aggregate"``, or ``"all"``.
            copy: If True, copy reference images instead of symlinking.
            workers: Parallel workers for distortion injection (0 = auto).
            compress: Save distorted images with compression.
            dry_run: Process only 2 groups with 3 distortions for testing.
            fmt: Output format, ``"png"`` or ``"jpeg"``.
            max_refs_per_source: Per-source image limit (e.g. ``{"Sim_3_UAVs": 500}``).
            scorer: VLM scorer for annotation step.
            scorer_batch_size: Batch size for VLM scoring.
            max_annotate_entries: Limit entries during annotation (debugging).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        step_set = self._parse_steps(steps)

        if "extract" in step_set:
            _log.info("=== Step 1: Extract reference frames ===")
            self.extract_references(
                input_root=input_root,
                output_dir=output_dir / "ref_images",
                copy=copy,
                max_refs_per_source=max_refs_per_source,
            )
            self._copy_vqa_files(Path(input_root), output_dir)

        if "inject" in step_set:
            _log.info("=== Step 2: Inject distortions ===")
            self.inject_distortions(
                input_root=input_root,
                output_dir=output_dir,
                workers=workers,
                compress=compress,
                dry_run=dry_run,
                fmt=fmt,
            )

        if "annotate" in step_set:
            _log.info("=== Step 3: Annotate scores ===")
            if scorer is None:
                _log.warning("No scorer provided — skipping annotation step")
            else:
                self.annotate_scores(
                    output_dir=output_dir,
                    scorer=scorer,
                    scorer_batch_size=scorer_batch_size,
                    max_entries=max_annotate_entries,
                )

        if "aggregate" in step_set:
            _log.info("=== Step 4: Aggregate scores ===")
            self.aggregate_scores(output_dir=output_dir)

        _log.info("Pipeline complete. Output: %s", output_dir)

    @staticmethod
    def _parse_steps(steps: str) -> set[str]:
        if steps == "all":
            return {"extract", "inject", "annotate", "aggregate"}
        valid = {"extract", "inject", "annotate", "aggregate"}
        selected = {s.strip() for s in steps.split(",")}
        invalid = selected - valid
        if invalid:
            raise ValueError(f"Unknown steps: {invalid}. Valid: {sorted(valid)}")
        return selected


# ===========================================================================
# Convenience factory
# ===========================================================================


def create_pipeline(
    dataset: str = "aircopbench", seed: int = 42
) -> DataSynthesisPipeline:
    """Create a ``DataSynthesisPipeline`` for a named dataset format.

    Example:
        >>> pipeline = create_pipeline("aircopbench")
        >>> pipeline.run_full("data/raw/AirCopBench", "data/processed")
    """
    fmt_cls = DatasetFormat.get(dataset)
    return DataSynthesisPipeline(fmt_cls(), seed=seed)
