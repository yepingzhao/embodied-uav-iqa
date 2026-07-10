"""Data synthesis pipeline for UAV-IQA with multi-UAV distortion injection.

Provides:
- DatasetFormat: minimal abstract base for extract step
- AirCopBenchFormat / GenericImageDirFormat: concrete format classes
- DataSynthesisPipeline: orchestrates inject/annotate/aggregate steps
- create_pipeline: convenience factory function

The pipeline produces a flat Processed JSON format where each entry is one
(question, distortion) pair with multi-image input.  Every line is a
self-contained training sample.
"""

import hashlib
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from uav_iqa.annotations import (
    build_sample_id,
    extract_uav_id_from_question_id,
    get_dataset_name,
    group_by_scene_frame,
    normalize_subtask_type,
    seed_for_distortion,
)
from uav_iqa.distortion import UAVDistortionPipeline
from uav_iqa.utils import find_images

_log = logging.getLogger(__name__)


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
            raise ValueError(f"Unknown dataset format: '{name}'. Available: {cls.list_formats()}")
        return cls._registry[name]

    @classmethod
    def list_formats(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abstractmethod
    def get_exclude_dirs(self) -> set[str]: ...

    @abstractmethod
    def find_reference_images(self, input_root: Path) -> list[Path]: ...


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
    1. inject   — group by scene+frame, apply 36 distortions, produce flat entries
    2. annotate — VLM multi-image inference for cognitive scores
    3. aggregate — merge model scores, compute final cognitive_score

    Output is a flat list where each entry is a (question, distortion) pair.
    ``run_full()`` composes the steps end-to-end.
    """

    def __init__(self, dataset_format: DatasetFormat, seed: int = 42):
        self.format = dataset_format
        self.seed = seed

    # ---- Step 1: Inject distortions ----

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

        Reads VQA JSON files from ``input_root/{train,test}/``, groups entries
        by ``sequence_frame``, and applies the SAME distortion type+intensity
        to all UAV images in a group.

        Produces:
        - ``output_dir/distorted/{sample_id}_{uav_idx}.{fmt}`` image files
        - Enriched JSON files in ``output_dir/{train,test}/`` with
          ``distorted_uav_paths``, ``distortion_info``, and per-entry
          ``vlm_scores`` / ``cognitive_score`` fields.

        Args:
            input_root: Raw dataset root (contains VQA JSONs + reference images).
            output_dir: Processed output directory (receives distorted images +
                enriched JSONs).
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
            split_dir = input_root / split_name
            if not split_dir.is_dir():
                _log.warning("Split directory not found: %s", split_dir)
                continue
            for fpath in sorted(split_dir.glob("*_VQA_*.json")):
                vqa_files.append((split_name, fpath))

        if not vqa_files:
            raise RuntimeError(
                f"No VQA JSON files found in {input_root}/{{train,test}}/. "
                f"Ensure the raw dataset is correctly structured."
            )

        _log.info("Found %d VQA files to process", len(vqa_files))

        all_distortions = UAVDistortionPipeline.get_all_distortion_names()
        n_intensity = len(UAVDistortionPipeline.INTENSITY_LEVELS)
        _log.info(
            "%d distortion types × %d intensity levels = %d variants per group",
            len(all_distortions),
            n_intensity,
            len(all_distortions) * n_intensity,
        )

        stats = {"total_flat_entries": 0}

        for split_name, fpath in vqa_files:
            _log.info("Processing %s/%s", split_name, fpath.name)
            dataset_name = get_dataset_name(fpath.name)

            out_split_dir = output_dir / split_name
            out_split_dir.mkdir(parents=True, exist_ok=True)

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
                _log.info(
                    "  [DRY RUN] %d groups, %d distortion types",
                    len(groups),
                    len(distortions_subset),
                )
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
                base_seed=self.seed,
            )

            out_path = out_split_dir / fpath.name
            with open(out_path, "w") as f:
                json.dump(output_entries, f, indent=2, ensure_ascii=False)
            _log.info("  Wrote %d entries to %s", len(output_entries), out_path)

            stats["total_flat_entries"] += len(output_entries)

        _log.info("Inject complete: %d flat entries", stats["total_flat_entries"])
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
        base_seed: int,
    ) -> list[dict]:
        """Apply distortions and return flat (question × distortion) entries."""
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
            distortion_results: list[tuple[dict, dict[str, str]]] = []

            for dist_name in distortions_subset:
                intensity = self._pick_intensity(dist_name, dataset_name, seq_frame)
                level = int(intensity * 10)
                seed = seed_for_distortion(dataset_name, seq_frame, dist_name, base_seed)
                safe_frame = seq_frame.replace("/", "_").replace("\\", "_")
                image_prefix = f"{dataset_name}__{safe_frame}__{dist_name}_L{level:02d}"

                dist_pipeline = UAVDistortionPipeline(seed=seed)
                distorted_uav_paths: dict[str, str] = {}

                skip_dist = False
                for idx, uav_key in enumerate(uav_keys):
                    uav_rel = uav_paths.get(uav_key, "")
                    if not uav_rel:
                        skip_dist = True
                        break
                    uav_abs = self._validate_uav_path(uav_rel, input_root)
                    if uav_abs is None:
                        skip_dist = True
                        break

                    out_name = f"{image_prefix}_{idx}{ext}"
                    out_path = distorted_dir / out_name

                    if not out_path.exists():
                        ok = self._save_distorted_uav_image(
                            uav_abs, dist_pipeline, dist_name, intensity,
                            out_path, fmt, ext, compress,
                        )
                        if not ok:
                            skip_dist = True
                            break

                    distorted_uav_paths[uav_key] = str(out_path.relative_to(distorted_dir.parent))

                if skip_dist:
                    continue

                dist_info = {
                    "type": dist_name,
                    "category": UAVDistortionPipeline.get_distortion_categories().get(
                        dist_name, "unknown"
                    ),
                    "level": level,
                    "intensity": intensity,
                    "seed": seed,
                }
                distortion_results.append((dist_info, distorted_uav_paths))

            if not distortion_results:
                continue

            for e in entries:
                question_id = e.get("question_id", "")
                question_type = e.get("question_type", "")
                for dist_info, distorted_uav_paths in distortion_results:
                    flat_entry = {
                        "sample_id": build_sample_id(
                            dataset_name,
                            seq_frame,
                            dist_info["type"],
                            dist_info["level"],
                            split=split_name,
                            question_id=question_id,
                        ),
                        "dataset": dataset_name,
                        "split": split_name,
                        "sequence_frame": seq_frame,
                        "question_id": question_id,
                        "question_type": question_type,
                        "subtask_type": normalize_subtask_type(question_type),
                        "uav_id": extract_uav_id_from_question_id(question_id, question_type),
                        "question": e.get("question", ""),
                        "options": e.get("options", []),
                        "correct_answer": e.get("correct_answer", ""),
                        "uav_paths": uav_paths,
                        "distorted_uav_paths": distorted_uav_paths,
                        "distortion_info": dist_info,
                        "vlm_scores": {},
                        "cognitive_score": None,
                    }
                    output.append(flat_entry)

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

    @staticmethod
    def _validate_uav_path(uav_rel: str, input_root: Path) -> Path | None:
        """Validate and resolve a relative UAV image path.

        Returns the resolved absolute Path on success, or None if the path
        fails validation (missing file, path traversal, or OS error).
        """
        uav_abs = input_root / uav_rel.lstrip("/").replace("\\", "/")
        try:
            uav_abs = uav_abs.resolve()
            if not str(uav_abs).startswith(str(input_root.resolve())):
                _log.warning("Path traversal attempt blocked: %s", uav_rel)
                return None
        except OSError:
            _log.warning("Cannot resolve UAV image path: %s", uav_rel)
            return None
        if not uav_abs.exists():
            _log.warning("UAV image not found: %s", uav_abs)
            return None
        return uav_abs

    @staticmethod
    def _save_distorted_uav_image(
        uav_abs: Path,
        dist_pipeline: UAVDistortionPipeline,
        dist_name: str,
        intensity: float,
        out_path: Path,
        fmt: str,
        ext: str,
        compress: bool,
    ) -> bool:
        """Apply distortion to one UAV image and save to disk.

        Returns True on success, False if any step fails.
        """
        img = cv2.imread(str(uav_abs))
        if img is None:
            _log.warning("Failed to read UAV image: %s", uav_abs)
            return False
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
            return False
        out_path.write_bytes(buf.tobytes())
        return True

    @staticmethod
    def _atomic_json_write(data: object, fpath: Path) -> None:
        """Atomically write JSON data using a tempfile + os.replace."""
        tf_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", dir=fpath.parent, delete=False
            ) as tf:
                tf_path = tf.name
                json.dump(data, tf, indent=2, ensure_ascii=False)
            os.replace(tf_path, fpath)
        except Exception:
            if tf_path and os.path.exists(tf_path):
                os.unlink(tf_path)
            raise

    # ---- Step 3: Annotate scores ----

    def annotate_scores(
        self,
        output_dir: str | Path,
        scorer,
        input_root: str | Path | None = None,
        max_entries: int | None = None,
        files: list[str] | None = None,
    ) -> None:
        """Annotate flat entries with VLM cognitive scores.

        For each entry, generates reference and distorted descriptions via
        multi-image VLM inference, then computes text similarity to derive a
        ``cognitive_score``.

        Updates the main JSON ``vlm_scores`` / ``cognitive_score`` fields
        in-place.

        Args:
            output_dir: Processed output directory with train/test JSONs.
            scorer: VLM scorer instance (``VLMScorer``).
            input_root: Raw dataset root for resolving reference image paths.
                Required when reference images are not in output_dir.
            max_entries: Limit number of entries to annotate per split
                (for debugging).
            files: Specific file paths relative to output_dir to annotate
                (e.g. ``["train/Sim3_VQA_train.json"]``). If None, annotates
                all ``*_VQA_*.json`` files.
        """
        output_dir = Path(output_dir)
        input_root = Path(input_root) if input_root else output_dir.parent
        model_name = getattr(scorer, "model_name", "unknown")

        if files:
            fpaths = []
            resolved_output = output_dir.resolve()
            for f in files:
                resolved = (output_dir / f).resolve()
                if not str(resolved).startswith(str(resolved_output)):
                    raise ValueError(f"Path traversal blocked: {f}")
                fpaths.append(resolved)
        else:
            fpaths = sorted(output_dir.glob("train/*_VQA_*.json")) + sorted(
                output_dir.glob("test/*_VQA_*.json")
            )

        for fpath in fpaths:
            split_name = fpath.parent.name
            self._annotate_one_file(
                fpath=fpath,
                output_dir=output_dir,
                input_root=input_root,
                split_name=split_name,
                model_name=model_name,
                scorer=scorer,
                max_entries=max_entries,
            )

    def _annotate_one_file(
        self,
        fpath: Path,
        output_dir: Path,
        input_root: Path,
        split_name: str,
        model_name: str,
        scorer,
        max_entries: int | None,
    ) -> None:
        """Annotate one flat-entry JSON file with a single VLM model."""
        with open(fpath) as f:
            entries = json.load(f)

        if not isinstance(entries, list) or not entries:
            return

        _log.info("Annotating %s [%s/%s]", fpath.name, split_name, model_name)

        processed = 0
        for entry in tqdm(entries, desc=f"  {model_name}/{split_name}/{fpath.stem}", unit="entry"):
            uav_paths = entry.get("uav_paths", {})
            distorted_uav_paths = entry.get("distorted_uav_paths", {})
            if not uav_paths or not distorted_uav_paths:
                continue

            question = entry.get("question", "")
            if not question:
                continue

            uav_keys = sorted(uav_paths.keys())
            ref_image_paths = []
            for k in uav_keys:
                uav_abs = DataSynthesisPipeline._validate_uav_path(uav_paths[k], input_root)
                if uav_abs is not None:
                    ref_image_paths.append(str(uav_abs))
                else:
                    ref_image_paths.append("")
            dist_image_paths = [
                str(output_dir / distorted_uav_paths.get(k, "").lstrip("/")) for k in uav_keys
            ]

            cognitive = scorer.score_multi_image(
                ref_image_paths=ref_image_paths,
                dist_image_paths=dist_image_paths,
                question=question,
                subtask_type=entry.get("subtask_type", ""),
            )

            entry.setdefault("vlm_scores", {})[model_name] = cognitive["cognitive_score"]
            entry.setdefault("vlm_details", {})[model_name] = {
                "cognitive_score": cognitive["cognitive_score"],
                "bleu": cognitive["bleu"],
                "rouge_l": cognitive["rouge_l"],
                "cider": cognitive["cider"],
                "ref_description": cognitive.get("ref_description", ""),
                "dist_description": cognitive.get("dist_description", ""),
                "prompt": cognitive.get("prompt", ""),
            }
            entry["cognitive_score"] = round(
                float(sum(entry["vlm_scores"].values()) / len(entry["vlm_scores"])), 6
            )

            processed += 1
            if max_entries and processed >= max_entries:
                break

        self._atomic_json_write(entries, fpath)
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
                    entries = json.load(f)

                if not isinstance(entries, list) or not entries:
                    continue

                updated = 0
                for entry in entries:
                    scores = entry.get("vlm_scores", {})
                    if scores:
                        if strategy == "mean":
                            entry["cognitive_score"] = round(
                                float(sum(scores.values()) / len(scores)), 6
                            )
                        updated += 1

                if updated:
                    self._atomic_json_write(entries, fpath)
                _log.info("Aggregated %d entries in %s", updated, fpath.name)

    # ---- Step 4b: Merge sidecar scores ----

    def merge_sidecar_scores(
        self,
        output_dir: str | Path,
        *,
        weights: tuple[float, float, float] = (1.0, 1.0, 0.1),
        strategy: str = "mean",
        model_names: list[str] | None = None,
    ) -> None:
        """Merge VLM sidecar detail files back into main annotated JSONs.

        Reads per-sample ``bleu`` / ``rouge_l`` / ``cider`` from
        ``vlm/{model_name}/{split}/*.json`` sidecar files, computes a
        ``cognitive_score`` for each model, and updates ``vlm_scores``
        and ``cognitive_score`` in the corresponding main annotated files.

        This is the post-inference merge step: the inference framework writes
        placeholder ``vlm_scores`` (0.0) in the main files and detailed scores
        in the sidecar; this method reconciles them.

        Args:
            output_dir: Processed output directory (contains train/, test/,
                and vlm/ subdirectories).
            weights: (bleu, rouge_l, cider) weights for computing per-model
                cognitive_score. Default ``(1.0, 1.0, 0.1)``, matching
                :func:`compute_cognitive_score`.
            strategy: Aggregation strategy for final ``cognitive_score``
                across models: ``"mean"`` (default).
            model_names: Specific model names to merge. If None, merges all
                models found under ``vlm/``.
        """
        output_dir = Path(output_dir)
        vlm_root = output_dir / "vlm"

        if not vlm_root.is_dir():
            _log.warning("No vlm/ sidecar directory at %s, nothing to merge", vlm_root)
            return

        available_models = {
            p.name for p in vlm_root.iterdir() if p.is_dir()
        }
        if model_names:
            models_to_merge = [m for m in model_names if m in available_models]
            missing = set(model_names) - set(available_models)
            if missing:
                _log.warning("Models not found in vlm/: %s", ", ".join(sorted(missing)))
        else:
            models_to_merge = sorted(available_models)

        if not models_to_merge:
            _log.warning("No models to merge")
            return

        _log.info("Merging sidecar scores for models: %s", ", ".join(models_to_merge))

        w_bleu, w_rouge, w_cider = weights
        w_sum = w_bleu + w_rouge + w_cider

        for model_name in models_to_merge:
            model_vlm_dir = vlm_root / model_name
            _log.info("  Model: %s", model_name)

            for split_name in ("train", "test"):
                split_vlm_dir = model_vlm_dir / split_name
                if not split_vlm_dir.is_dir():
                    continue

                for sidecar_path in sorted(split_vlm_dir.glob("*_VQA_*.json")):
                    self._merge_one_sidecar(
                        sidecar_path=sidecar_path,
                        output_dir=output_dir,
                        split_name=split_name,
                        model_name=model_name,
                        w_bleu=w_bleu,
                        w_rouge=w_rouge,
                        w_cider=w_cider,
                        w_sum=w_sum,
                        strategy=strategy,
                    )

    def _merge_one_sidecar(
        self,
        sidecar_path: Path,
        output_dir: Path,
        split_name: str,
        model_name: str,
        w_bleu: float,
        w_rouge: float,
        w_cider: float,
        w_sum: float,
        strategy: str,
    ) -> None:
        """Merge one sidecar file into its corresponding main JSON."""
        with open(sidecar_path) as f:
            sidecar_entries = json.load(f)

        if not isinstance(sidecar_entries, list) or not sidecar_entries:
            return

        sidecar_scores: dict[str, float] = {}
        for se in sidecar_entries:
            sid = se.get("sample_id", "")
            if not sid:
                continue
            bleu = float(se.get("bleu", 0.0))
            rouge = float(se.get("rouge_l", 0.0))
            cider = float(se.get("cider", 0.0))
            cognitive = (w_bleu * bleu + w_rouge * rouge + w_cider * cider) / w_sum
            sidecar_scores[sid] = round(cognitive, 6)

        main_path = output_dir / split_name / sidecar_path.name
        if not main_path.is_file():
            _log.warning(
                "Main file not found for sidecar %s (expected %s), skipping",
                sidecar_path.name, main_path,
            )
            return

        with open(main_path) as f:
            main_entries = json.load(f)

        if not isinstance(main_entries, list):
            return

        updated = 0
        for entry in main_entries:
            sid = entry.get("sample_id", "")
            score = sidecar_scores.get(sid)
            if score is None:
                continue

            entry.setdefault("vlm_scores", {})[model_name] = score
            updated += 1

        for entry in main_entries:
            scores = entry.get("vlm_scores", {})
            if scores:
                if strategy == "mean":
                    entry["cognitive_score"] = round(
                        float(sum(scores.values()) / len(scores)), 6
                    )

        if updated:
            self._atomic_json_write(main_entries, main_path)
        _log.info(
            "  Merged %d/%d entries: %s/%s [%s]",
            updated, len(main_entries), split_name, sidecar_path.name, model_name,
        )

    # ---- Full pipeline ----

    def run_full(
        self,
        input_root: str | Path,
        output_dir: str | Path,
        steps: str = "all",
        workers: int = 0,
        compress: bool = True,
        dry_run: bool = False,
        fmt: str = "png",
        scorer=None,
        max_annotate_entries: int | None = None,
    ) -> None:
        """Run the full data synthesis pipeline end-to-end.

        Args:
            input_root: Raw dataset root directory.
            output_dir: Root output directory (creates subdirs inside).
            steps: Comma-separated step names: ``"inject"``,
                   ``"annotate"``, ``"aggregate"``, or ``"all"``.
            workers: Parallel workers for distortion injection (0 = auto).
            compress: Save distorted images with compression.
            dry_run: Process only 2 groups with 3 distortions for testing.
            fmt: Output format, ``"png"`` or ``"jpeg"``.
            scorer: VLM scorer for annotation step.
            max_annotate_entries: Limit entries during annotation (debugging).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        step_set = self._parse_steps(steps)

        if "inject" in step_set:
            _log.info("=== Step 1: Inject distortions ===")
            self.inject_distortions(
                input_root=input_root,
                output_dir=output_dir,
                workers=workers,
                compress=compress,
                dry_run=dry_run,
                fmt=fmt,
            )

        if "annotate" in step_set:
            _log.info("=== Step 2: Annotate scores ===")
            if scorer is None:
                _log.warning("No scorer provided — skipping annotation step")
            else:
                self.annotate_scores(
                    output_dir=output_dir,
                    scorer=scorer,
                    input_root=input_root,
                    max_entries=max_annotate_entries,
                )

        if "aggregate" in step_set:
            _log.info("=== Step 3: Aggregate scores ===")
            self.aggregate_scores(output_dir=output_dir)

        if "merge" in step_set:
            _log.info("=== Step 4: Merge sidecar scores ===")
            self.merge_sidecar_scores(output_dir=output_dir)

        _log.info("Pipeline complete. Output: %s", output_dir)

    @staticmethod
    def _parse_steps(steps: str) -> set[str]:
        if steps == "all":
            return {"inject", "annotate", "aggregate", "merge"}
        valid = {"inject", "annotate", "aggregate", "merge"}
        selected = {s.strip() for s in steps.split(",")}
        invalid = selected - valid
        if invalid:
            raise ValueError(f"Unknown steps: {invalid}. Valid: {sorted(valid)}")
        return selected


# ===========================================================================
# Convenience factory
# ===========================================================================


def create_pipeline(dataset: str = "aircopbench", seed: int = 42) -> DataSynthesisPipeline:
    """Create a ``DataSynthesisPipeline`` for a named dataset format.

    Example:
        >>> pipeline = create_pipeline("aircopbench")
        >>> pipeline.run_full("data/raw/AirCopBench", "data/processed")
    """
    fmt_cls = DatasetFormat.get(dataset)
    return DataSynthesisPipeline(fmt_cls(), seed=seed)
