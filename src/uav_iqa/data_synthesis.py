"""Dataset-agnostic data synthesis pipeline for UAV-IQA.

Provides:
- DatasetFormat: abstract base with registry for dataset-specific logic
- AirCopBenchFormat: handles AirCopBench directory structure + annotations
- GenericImageDirFormat: flat directory of images, no annotations
- DataSynthesisPipeline: orchestrates extract/inject/manifest/annotate steps
- create_pipeline: convenience factory function
"""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from uav_iqa.annotations import (
    assign_task_label,
    build_ref_score_lookup,
    degradation_factor,
    parse_distortion_key,
    synthetic_ref_scores,
)
from uav_iqa.distortion import UAVDistortionPipeline
from uav_iqa.utils import find_images, split_samples, write_manifest

_log = logging.getLogger(__name__)


# ===========================================================================
# DatasetFormat — ABC with explicit registry
# ===========================================================================


class DatasetFormat(ABC):
    """Abstract interface for dataset-specific directory structure and annotations.

    Subclasses register via ``@DatasetFormat.register`` and must set ``name``
    as a class-level attribute.  The registry lets callers look up a format by
    string name (e.g. ``DatasetFormat.get("aircopbench")``).
    """

    _registry: dict[str, type["DatasetFormat"]] = {}
    name: str  # set by subclasses

    # ---- Registry ----

    @classmethod
    def register(cls, format_cls: type["DatasetFormat"]) -> type["DatasetFormat"]:
        cls._registry[format_cls.name] = format_cls
        return format_cls

    @classmethod
    def get(cls, name: str) -> type["DatasetFormat"]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown dataset format: '{name}'. " f"Available: {cls.list_formats()}"
            )
        return cls._registry[name]

    @classmethod
    def list_formats(cls) -> list[str]:
        return sorted(cls._registry.keys())

    # ---- Abstract interface ----

    @abstractmethod
    def get_exclude_dirs(self) -> set[str]:
        """Subdirectory names to skip when scanning for reference images."""
        ...

    @abstractmethod
    def find_reference_images(self, input_root: Path) -> list[Path]:
        """Discover clean reference images in the dataset root."""
        ...

    @abstractmethod
    def build_ref_lookup(self, input_root: Path) -> dict[str, dict]:
        """Build a ``ref_id -> {vlm_score, vla_score, execution_score, ...}`` map.

        Returns an empty dict for datasets without annotations.
        """
        ...

    @abstractmethod
    def resolve_ref_path(self, ref_id: str, ref_dir: Optional[Path]) -> str:
        """Resolve a reference image path given a ref_id and optional ref_dir."""
        ...

    @abstractmethod
    def assign_task(self, identifier: str, task_map: Optional[dict] = None) -> str:
        """Determine the task label for a given image identifier."""
        ...

    @abstractmethod
    def parse_ref_id(self, distorted_filename: str) -> str:
        """Extract the reference image id from a distorted image filename."""
        ...


# ===========================================================================
# Concrete formats
# ===========================================================================


@DatasetFormat.register
class AirCopBenchFormat(DatasetFormat):
    """AirCopBench dataset: nested scene/UAV directories with Annotations/*.json."""

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

    # ---- DatasetFormat interface ----

    def get_exclude_dirs(self) -> set[str]:
        return self.EXCLUDE_DIRS

    def find_reference_images(self, input_root: Path) -> list[Path]:
        return find_images(input_root, exclude_dirs=self.EXCLUDE_DIRS)

    def build_ref_lookup(self, input_root: Path) -> dict[str, dict]:
        return build_ref_score_lookup(input_root)

    def resolve_ref_path(self, ref_id: str, ref_dir: Optional[Path]) -> str:
        if ref_dir is None or not ref_dir.exists():
            return ""
        for ext in ("", ".jpg", ".png", ".jpeg"):
            candidate = ref_dir / f"{ref_id}{ext}"
            if candidate.exists():
                return str(candidate)
        return ""

    def assign_task(self, identifier: str, task_map: Optional[dict] = None) -> str:
        return assign_task_label(identifier, task_map=task_map)

    def parse_ref_id(self, distorted_filename: str) -> str:
        return distorted_filename.rsplit("__", 1)[0]

    # ---- AirCopBench-specific helpers ----

    @staticmethod
    def get_degradation_types(entry: dict) -> list[str]:
        """Extract degradation category strings from an annotation entry."""
        degs: list[str] = []
        deg_field = entry.get("Degradation", "")
        if isinstance(deg_field, dict):
            degs.extend(deg_field.get("choices", []))
        elif isinstance(deg_field, str) and deg_field.strip():
            degs.append(deg_field)

        for issue in entry.get("PerceptionIssues", []):
            labels = issue.get("rectanglelabels", [])
            degs.extend(labels)

        other = entry.get("Other Degradation", "")
        if other and other.strip():
            degs.append(other)

        if not degs:
            degs.append("none")
        return degs


@DatasetFormat.register
class GenericImageDirFormat(DatasetFormat):
    """Generic flat image directory: no annotations, hash-based task assignment."""

    name = "generic"

    # ---- DatasetFormat interface ----

    def get_exclude_dirs(self) -> set[str]:
        return set()

    def find_reference_images(self, input_root: Path) -> list[Path]:
        return find_images(input_root)

    def build_ref_lookup(self, input_root: Path) -> dict[str, dict]:
        return {}

    def resolve_ref_path(self, ref_id: str, ref_dir: Optional[Path]) -> str:
        if ref_dir is None or not ref_dir.exists():
            return ""
        for ext in ("", ".jpg", ".png", ".jpeg"):
            candidate = ref_dir / f"{ref_id}{ext}"
            if candidate.exists():
                return str(candidate)
        return ""

    def assign_task(self, identifier: str, task_map: Optional[dict] = None) -> str:
        return assign_task_label(identifier, task_map=task_map)

    def parse_ref_id(self, distorted_filename: str) -> str:
        return distorted_filename.rsplit("__", 1)[0]


# ===========================================================================
# DataSynthesisPipeline
# ===========================================================================


class DataSynthesisPipeline:
    """Orchestrates the 4-step data synthesis pipeline for a dataset format.

    Each step is a public method that can be called independently.
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

        Args:
            input_root: Dataset root directory.
            output_dir: Where to place extracted reference images.
            copy: If True, copy files; default is symlink to save disk space.
            max_refs_per_source: Per-source image limit, keyed by the first
                path component relative to ``input_root`` (e.g.
                ``{"Sim_3_UAVs": 500}``). Sources not listed are unlimited.

        Returns:
            The output_dir Path for chaining.
        """
        input_root = Path(input_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = self.format.find_reference_images(input_root)
        _log.info("Found %d clean reference images in %s", len(images), input_root)

        if max_refs_per_source:
            rng = __import__("random").Random(self.seed)
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
        for img in images:
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
                created += 1
            else:
                skipped += 1

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
        image_dir: str | Path,
        output_dir: str | Path,
        workers: int = 4,
        compress: bool = True,
        dry_run: bool = False,
        fmt: str = "png",
    ) -> dict:
        """Apply all 36 distortion types at 5 intensity levels.

        Args:
            image_dir: Directory of reference images.
            output_dir: Where to save distorted images.
            workers: Parallel worker processes for distortion injection.
            compress: Save distorted images with compression (PNG level 3, JPEG quality 92).
            dry_run: Process only the first 5 images.
            fmt: Output format, ``"png"`` or ``"jpeg"``.

        Returns:
            The results dict from ``UAVDistortionPipeline.inject_directory()``
            (keys: ``total_distorted``, ``failed``, ``errors``).
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = find_images(image_dir)
        if dry_run:
            images = images[:5]
            _log.info("[DRY RUN] Processing %d images", len(images))

        n_distortions = len(UAVDistortionPipeline.get_all_distortion_names())
        n_variants = n_distortions * len(UAVDistortionPipeline.INTENSITY_LEVELS)
        _log.info("Found %d reference images", len(images))
        _log.info(
            "%d distortions x %d levels = %d variants per image",
            n_distortions,
            len(UAVDistortionPipeline.INTENSITY_LEVELS),
            n_variants,
        )
        _log.info("Total expected pairs: %d", len(images) * n_variants)
        _log.info("Output: %s (format: %s)", output_dir, fmt)

        pipeline = UAVDistortionPipeline(seed=self.seed)
        results = pipeline.inject_directory(
            image_paths=[str(p) for p in images],
            output_dir=str(output_dir),
            compress=compress,
            max_workers=workers,
            fmt=fmt,
        )

        _log.info(
            "Done. %d generated, %d failed.",
            results.get("total_distorted", 0),
            results.get("failed", 0),
        )
        return results

    # ---- Step 3: Generate manifests ----

    def generate_manifests(
        self,
        distorted_dir: str | Path,
        output_dir: str | Path,
        ref_dir: str | Path | None = None,
        split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        task_map: dict | None = None,
    ) -> dict[str, Path]:
        """Scan distorted images, build manifest entries, split into train/val/test.

        Args:
            distorted_dir: Directory containing distorted PNG images.
            output_dir: Where to write per-split ``manifest.json`` files.
            ref_dir: Optional directory of reference images for ``ref_path`` lookup.
            split: Train/val/test ratios (must sum to 1.0).
            task_map: Optional dict mapping scene/identifier patterns to tasks.

        Returns:
            Dict mapping split name to manifest path,
            e.g. ``{"train": Path(...), "val": Path(...), "test": Path(...)}``.
        """
        distorted_dir = Path(distorted_dir)
        output_dir = Path(output_dir)
        ref_dir = Path(ref_dir) if ref_dir else None

        _log.info("Scanning %s for distorted images...", distorted_dir)
        samples = self._gather_samples(distorted_dir, output_dir, ref_dir, task_map)
        _log.info("Found %d samples", len(samples))

        split_data = split_samples(samples, ratios=list(split), seed=self.seed)

        manifest_paths: dict[str, Path] = {}
        for split_name, entries in split_data.items():
            manifest_path = output_dir / split_name / "manifest.json"
            n = write_manifest(entries, manifest_path, _log=_log)
            n_dist = len({s["distortion"] for s in entries})
            n_ref = len({s["ref_id"] for s in entries})
            _log.info(
                "  %s: %d samples, %d distortions, %d references -> %s",
                split_name,
                n,
                n_dist,
                n_ref,
                manifest_path,
            )
            manifest_paths[split_name] = manifest_path

        stats = {
            "total": len(samples),
            "train": len(split_data["train"]),
            "val": len(split_data["val"]),
            "test": len(split_data["test"]),
            "split_ratios": list(split),
            "seed": self.seed,
        }
        stats_path = output_dir / "split_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        _log.info("Split stats saved to %s", stats_path)

        return manifest_paths

    def _gather_samples(
        self,
        distorted_dir: Path,
        output_dir: Path,
        ref_dir: Optional[Path],
        task_map: Optional[dict],
    ) -> list[dict]:
        """Build manifest entries from distorted image files.

        The ``path`` field in each entry is relative to ``output_dir``
        (the directory that will be used as ``data_root`` at training time).
        """
        png_files = find_images(distorted_dir)

        # Pre-build ref_path lookup if ref_dir given
        ref_by_stem: dict[str, str] = {}
        if ref_dir and ref_dir.exists():
            for p in find_images(ref_dir):
                ref_by_stem[p.stem] = str(p)

        samples: list[dict] = []
        for img_path in sorted(png_files):
            dist_name, intensity = parse_distortion_key(img_path.stem)
            if dist_name is None:
                continue

            # Path relative to output_dir (data_root for training)
            try:
                rel_path = str(img_path.relative_to(output_dir))
            except ValueError:
                rel_path = str(img_path)
            ref_id = self.format.parse_ref_id(img_path.name)

            task = self.format.assign_task(str(img_path), task_map=task_map)
            ref_path = self.format.resolve_ref_path(ref_id, ref_dir)
            if not ref_path and ref_id in ref_by_stem:
                ref_path = ref_by_stem[ref_id]

            samples.append(
                {
                    "path": rel_path,
                    "task": task,
                    "distortion": dist_name,
                    "intensity_level": round(intensity, 2),
                    "ref_id": ref_id,
                    "ref_path": ref_path,
                    "vlm_score": 0.0,
                    "vla_score": 0.0,
                    "execution_score": 0.0,
                    "annotated": False,
                }
            )
        return samples

    # ---- Step 4: Annotate scores ----

    def annotate_scores(
        self,
        manifest_dir: str | Path,
        input_root: str | Path | None = None,
        output_dir: str | Path | None = None,
        noise_scale: float = 0.02,
    ) -> None:
        """Annotate manifest entries with degradation-model scores.

        Reads AirCopBench human annotations (if available) and applies the
        degradation model: ``score = ref_score * (1 - alpha * intensity) + noise``.

        Entries without real annotations receive synthetic reference scores.

        Args:
            manifest_dir: Directory containing train/val/test/manifest.json files.
            input_root: Dataset root for building annotation lookup.
            output_dir: Where to write annotated manifests (default: manifest_dir).
            noise_scale: Gaussian noise std multiplier (0 disables noise).
        """
        manifest_dir = Path(manifest_dir)
        output_dir = Path(output_dir) if output_dir else manifest_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(self.seed)
        ref_lookup: dict[str, dict] = {}
        if input_root:
            ref_lookup = self.format.build_ref_lookup(Path(input_root))

        for split_name in ("train", "val", "test"):
            manifest_path = manifest_dir / split_name / "manifest.json"
            if not manifest_path.exists():
                _log.info(
                    "Skip %s: manifest not found at %s", split_name, manifest_path
                )
                continue

            entries = self._annotate_one_manifest(
                manifest_path, ref_lookup, rng, noise_scale
            )

            out_path = output_dir / split_name / "manifest.json"
            write_manifest(entries, out_path, _log=_log)

        _log.info("Score distribution:")
        for split_name in ("train", "val", "test"):
            manifest_path = output_dir / split_name / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                entries = json.load(f)
            for key in ("vlm_score", "vla_score", "execution_score"):
                scores = [e[key] for e in entries]
                _log.info(
                    "  %s/%s: mean=%.3f std=%.3f min=%.3f max=%.3f",
                    split_name,
                    key,
                    float(np.mean(scores)),
                    float(np.std(scores)),
                    float(np.min(scores)),
                    float(np.max(scores)),
                )

    def _annotate_one_manifest(
        self,
        manifest_path: Path,
        ref_lookup: dict[str, dict],
        rng: np.random.RandomState,
        noise_scale: float,
    ) -> list[dict]:
        with open(manifest_path) as f:
            entries = json.load(f)

        annotated = 0
        synthetic = 0

        for entry in entries:
            ref_id = entry.get("ref_id", "")
            distortion = entry.get("distortion", "none")
            task = entry.get("task", "tracking")
            intensity = entry.get("intensity_level", 0.5)

            # Assign reference scores
            if ref_id in ref_lookup:
                ref_scores = ref_lookup[ref_id]
                entry["vlm_score"] = ref_scores["vlm_score"]
                entry["vla_score"] = ref_scores["vla_score"]
                entry["execution_score"] = ref_scores["execution_score"]
                entry["annotated"] = True
                annotated += 1
            else:
                synth = synthetic_ref_scores(ref_id)
                entry.update(synth)
                synthetic += 1

            # Apply degradation model
            deg = degradation_factor(distortion, task)
            alpha = 1.0 - deg
            degrade = alpha * intensity

            for key in ("vlm_score", "vla_score", "execution_score"):
                ref_score = entry[key]
                noise = (
                    rng.normal(0, noise_scale * intensity) if noise_scale > 0 else 0.0
                )
                degraded = ref_score * (1.0 - degrade) + noise
                entry[key] = round(max(0.0, min(1.0, degraded)), 4)

        _log.info(
            "%s: %d annotated, %d synthetic (total %d)",
            manifest_path.name,
            annotated,
            synthetic,
            len(entries),
        )
        return entries

    # ---- Full pipeline ----

    def run_full(
        self,
        input_root: str | Path,
        output_dir: str | Path,
        steps: str = "all",
        copy: bool = False,
        workers: int = 4,
        compress: bool = True,
        split: tuple = (0.8, 0.1, 0.1),
        task_map: dict | None = None,
        noise_scale: float = 0.02,
        dry_run: bool = False,
        fmt: str = "png",
        max_refs_per_source: dict[str, int] | None = None,
    ) -> None:
        """Run the full data synthesis pipeline end-to-end.

        Args:
            input_root: Dataset root directory.
            output_dir: Root output directory (creates ref_images/, distorted/
                        subdirs inside).
            steps: Comma-separated step names: ``"extract"``, ``"inject"``,
                   ``"manifest"``, ``"annotate"``, or ``"all"``.
            copy: If True, copy reference images instead of symlinking.
            workers: Parallel workers for distortion injection.
            compress: Save distorted images with compression.
            split: Train/val/test ratios.
            task_map: Optional dict for task assignment.
            noise_scale: Gaussian noise std for score degradation.
            dry_run: Inject only 5 images.
            fmt: Output format, ``"png"`` or ``"jpeg"``.
            max_refs_per_source: Per-source image limit for extraction
                (e.g. ``{"Sim_3_UAVs": 500}``).
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

        if "inject" in step_set:
            _log.info("=== Step 2: Inject distortions ===")
            self.inject_distortions(
                image_dir=output_dir / "ref_images",
                output_dir=output_dir / "distorted",
                workers=workers,
                compress=compress,
                dry_run=dry_run,
                fmt=fmt,
            )

        if "manifest" in step_set:
            _log.info("=== Step 3: Generate manifests ===")
            ref_dir = output_dir / "ref_images"
            self.generate_manifests(
                distorted_dir=output_dir / "distorted",
                output_dir=output_dir,
                ref_dir=ref_dir if ref_dir.exists() else None,
                split=split,
                task_map=task_map,
            )

        if "annotate" in step_set:
            _log.info("=== Step 4: Annotate scores ===")
            self.annotate_scores(
                manifest_dir=output_dir,
                input_root=input_root,
                output_dir=output_dir,
                noise_scale=noise_scale,
            )

        _log.info("Pipeline complete. Output: %s", output_dir)

    @staticmethod
    def _parse_steps(steps: str) -> set[str]:
        if steps == "all":
            return {"extract", "inject", "manifest", "annotate"}
        valid = {"extract", "inject", "manifest", "annotate"}
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

    Args:
        dataset: Dataset format name (e.g. ``"aircopbench"``, ``"generic"``).
        seed: Random seed for reproducibility.

    Example:
        >>> pipeline = create_pipeline("aircopbench")
        >>> pipeline.run_full("data/raw/AirCopBench", "data/processed")
    """
    fmt_cls = DatasetFormat.get(dataset)
    return DataSynthesisPipeline(fmt_cls(), seed=seed)
