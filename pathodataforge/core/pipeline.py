"""End-to-end processing pipeline."""

from __future__ import annotations

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from pathodataforge.core.dataset_splitter import assign_splits
from pathodataforge.core.feature_extractor import feature_consumer
from pathodataforge.core.metadata_cleaner import clean_metadata
from pathodataforge.core.patch_extractor import extract_patches_for_slide
from pathodataforge.core.runtime import describe_gpus, get_gpu_info, has_mps
from pathodataforge.core.sampling import sampling_worker
from pathodataforge.core.scanner import scan_wsi_folder
from pathodataforge.core.wsi_reader import WSIReader
from pathodataforge.utils.config import resolve_demo_defaults, save_config
from pathodataforge.utils.io import ensure_directory, safe_load_npy, write_json
from pathodataforge.utils.logger import setup_logger


ProgressCallback = Callable[[int, str], None]
LogCallback = Callable[[str], None]


def prepare_output_directories(output_dir: str | Path) -> dict[str, Path]:
    root = ensure_directory(output_dir)
    directories = {
        "root": root,
        "patches": ensure_directory(root / "patches"),
        "train": ensure_directory(root / "patches" / "train"),
        "val": ensure_directory(root / "patches" / "val"),
        "test": ensure_directory(root / "patches" / "test"),
        "metadata": ensure_directory(root / "metadata"),
        "coordinates": ensure_directory(root / "metadata" / "coordinates"),
        "logs": ensure_directory(root / "logs"),
        "configs": ensure_directory(root / "configs"),
        "reports": ensure_directory(root / "reports"),
        "overlays": ensure_directory(root / "reports" / "overlays"),
        "features": ensure_directory(root / "features"),
    }
    return directories


def _infer_column(fields: list[str], candidates: list[str]) -> str:
    lowered = {field.lower(): field for field in fields}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for field in fields:
        normalized = field.lower().replace(" ", "_")
        if normalized in candidates:
            return field
    return ""


def _validate_input_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    input_cfg = config.get("input", {})
    wsi_dir = Path(str(input_cfg.get("wsi_dir") or "")).expanduser()
    metadata_file = Path(str(input_cfg.get("metadata_file") or "")).expanduser()
    output_dir = Path(str(input_cfg.get("output_dir") or "./PathoDataForge_output")).expanduser()

    if not str(input_cfg.get("wsi_dir") or "").strip():
        raise ValueError("input.wsi_dir is required. Generate demo data or set it in the config.")
    if not str(input_cfg.get("metadata_file") or "").strip():
        raise ValueError("input.metadata_file is required. Generate demo data or set it in the config.")
    if not wsi_dir.exists():
        raise FileNotFoundError(f"WSI folder not found: {wsi_dir}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    return wsi_dir, metadata_file, output_dir


def _apply_target_file_filter(files: list[Path], config: dict[str, Any]) -> list[Path]:
    target_files = config.get("runtime", {}).get("target_files", []) or []
    if not target_files:
        return files
    target_set = {str(item) for item in target_files}
    return [path for path in files if path.name in target_set]


def _feature_devices(config: dict[str, Any]) -> list[str]:
    features_cfg = config.get("features", {})
    runtime_cfg = config.get("runtime", {})
    requested = str(features_cfg.get("device", "auto") or "auto")
    if requested != "auto":
        return [requested]
    gpus = get_gpu_info()
    if not gpus:
        if has_mps():
            return ["mps"]
        return ["cpu"]
    max_workers = max(1, int(runtime_cfg.get("gpu_workers", 1)))
    return [f"cuda:{gpu.index}" for gpu in gpus[:max_workers]]


def _run_coordinate_and_feature_stage(
    metadata_with_split: pd.DataFrame,
    output_root: Path,
    config: dict[str, Any],
    logger: Any,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    coordinates_cfg = config.get("coordinates", {})
    features_cfg = config.get("features", {})
    visualization_cfg = config.get("visualization", {})
    runtime_cfg = config.get("runtime", {})
    should_sample = (
        bool(coordinates_cfg.get("enable", True))
        or bool(features_cfg.get("enable", False))
        or bool(visualization_cfg.get("save_overlay", False))
    )
    if not should_sample:
        return {}, [], []

    logger.info("Runtime GPU status: %s", describe_gpus())
    payloads = []
    for _, row in metadata_with_split.iterrows():
        payloads.append(
            {
                "row": row.to_dict(),
                "patch_config": config.get("patch", {}),
                "visualization_config": visualization_cfg,
                "output_root": str(output_root),
                "force": bool(runtime_cfg.get("force_resample", False)),
            }
        )

    feature_enabled = bool(features_cfg.get("enable", False))
    feature_results: list[dict[str, Any]] = []
    stop_event = threading.Event()
    task_queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue(
        maxsize=max(1, int(runtime_cfg.get("queue_maxsize", 8)))
    )
    consumers: list[threading.Thread] = []
    if feature_enabled:
        for index, device in enumerate(_feature_devices(config)):
            thread = threading.Thread(
                target=feature_consumer,
                args=(
                    f"FeatureWorker-{index}",
                    task_queue,
                    output_root,
                    features_cfg,
                    device,
                    logger,
                    feature_results,
                    stop_event,
                ),
                daemon=True,
            )
            thread.start()
            consumers.append(thread)

    coordinate_lookup: dict[str, dict[str, Any]] = {}
    sampling_results: list[dict[str, Any]] = []

    def handle_sampling_result(result: dict[str, Any]) -> None:
        sampling_results.append(result)
        source_path = str(result.get("source_path", ""))
        if result.get("ok"):
            coordinate_lookup[source_path] = result
            if feature_enabled and not stop_event.is_set():
                task_queue.put(
                    {
                        "row": next(
                            payload["row"]
                            for payload in payloads
                            if payload["row"]["source_path"] == source_path
                        ),
                        "coordinate_npy": result["coordinate_npy"],
                        "level": result["level"],
                        "patch_size": result["patch_size"],
                    }
                )
        else:
            logger.error("Coordinate sampling failed for %s: %s", source_path, result.get("error"))

    cpu_workers = max(1, int(runtime_cfg.get("cpu_workers", 1)))
    logger.info("Coordinate sampling: slides=%s cpu_workers=%s", len(payloads), cpu_workers)
    if cpu_workers <= 1 or len(payloads) <= 1:
        for index, payload in enumerate(payloads, start=1):
            handle_sampling_result(sampling_worker(payload))
            if progress:
                progress(25 + int(index / max(1, len(payloads)) * 20), "Sampling coordinates")
    else:
        with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
            futures = {executor.submit(sampling_worker, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), start=1):
                handle_sampling_result(future.result())
                if progress:
                    progress(25 + int(index / max(1, len(payloads)) * 20), "Sampling coordinates")

    if feature_enabled:
        alive_consumers = [thread for thread in consumers if thread.is_alive()]
        if alive_consumers:
            for _ in alive_consumers:
                task_queue.put(None)
            task_queue.join()
        else:
            while True:
                try:
                    task_queue.get_nowait()
                    task_queue.task_done()
                except queue.Empty:
                    break
        stop_event.set()
        for thread in consumers:
            thread.join()

    return coordinate_lookup, sampling_results, feature_results


def run_pipeline(
    config: dict[str, Any],
    logger_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the full iMoonLab-PathoDataForge preprocessing workflow."""
    config = resolve_demo_defaults(config)
    wsi_dir, metadata_file, output_dir = _validate_input_paths(config)
    directories = prepare_output_directories(output_dir)
    logger = setup_logger(
        log_file=directories["logs"] / "processing.log",
        gui_callback=logger_callback,
    )

    def progress(value: int, message: str) -> None:
        if progress_callback:
            progress_callback(max(0, min(100, int(value))), message)

    started_at = datetime.now()
    progress(0, "Starting")
    logger.info("iMoonLab-PathoDataForge pipeline started")
    logger.info("WSI folder: %s", wsi_dir)
    logger.info("Metadata file: %s", metadata_file)
    logger.info("Output directory: %s", directories["root"])

    try:
        save_config(config, directories["configs"] / "used_config.yaml")

        scan_result = scan_wsi_folder(wsi_dir)
        scan_result.files = _apply_target_file_filter(scan_result.files, config)
        if not scan_result.files:
            raise ValueError(f"No supported image files found in {wsi_dir}")
        logger.info(
            "Scanned %s image files across %s supported formats",
            scan_result.image_file_count,
            scan_result.supported_format_count,
        )
        progress(10, "Scanned input folder")

        metadata_cfg = config.get("metadata", {})
        metadata_preview = pd.read_csv(metadata_file, nrows=0) if metadata_file.suffix.lower() == ".csv" else pd.read_excel(metadata_file, nrows=0)
        fields = list(metadata_preview.columns)
        case_id_column = str(metadata_cfg.get("case_id_column") or "").strip()
        label_column = str(metadata_cfg.get("label_column") or "").strip()
        filename_column = str(metadata_cfg.get("filename_column") or "").strip()
        if not case_id_column:
            case_id_column = _infer_column(fields, ["case_id", "patient_id", "patient", "病例ID"])
        if not label_column:
            label_column = _infer_column(fields, ["label", "diagnosis", "class", "target", "标签"])

        clean_result = clean_metadata(
            scan_result.files,
            metadata_file,
            case_id_column=case_id_column,
            label_column=label_column,
            filename_column=filename_column,
            case_id_regex=str(metadata_cfg.get("case_id_regex") or "").strip(),
        )
        logger.info(
            "Metadata matching complete: matched=%s unmatched=%s",
            clean_result.matched_count,
            clean_result.unmatched_count,
        )
        progress(25, "Matched metadata")

        split_cfg = config.get("split", {})
        metadata_with_split = assign_splits(
            clean_result.metadata,
            train=float(split_cfg.get("train", 0.7)),
            val=float(split_cfg.get("val", 0.15)),
            test=float(split_cfg.get("test", 0.15)),
            seed=int(split_cfg.get("seed", 42)),
        )
        metadata_cleaned_path = directories["metadata"] / "metadata_cleaned.csv"
        mapping_path = directories["metadata"] / "mapping.csv"
        metadata_with_split.to_csv(metadata_cleaned_path, index=False)
        clean_result.mapping.to_csv(mapping_path, index=False)
        logger.info("Wrote metadata outputs")

        coordinate_lookup, sampling_results, feature_results = _run_coordinate_and_feature_stage(
            metadata_with_split,
            directories["root"],
            config,
            logger,
            progress,
        )

        manifest_records: list[dict[str, Any]] = []
        slide_summaries: list[dict[str, Any]] = []
        patch_cfg = config.get("patch", {})
        quality_cfg = config.get("quality", {})
        total_slides = len(metadata_with_split)

        for slide_number, (_, row) in enumerate(metadata_with_split.iterrows(), start=1):
            progress(
                25 + int((slide_number - 1) / max(total_slides, 1) * 65),
                f"Processing slide {slide_number}/{total_slides}",
            )
            try:
                coordinate_ranges = None
                lookup = coordinate_lookup.get(str(row["source_path"]))
                if lookup and lookup.get("coordinate_npy"):
                    coordinate_ranges = safe_load_npy(lookup["coordinate_npy"], logger)
                with WSIReader(row["source_path"]) as reader:
                    records, slide_summary = extract_patches_for_slide(
                        reader=reader,
                        metadata_row=row,
                        output_root=directories["root"],
                        patch_config=patch_cfg,
                        quality_config=quality_cfg,
                        logger=logger,
                        coordinate_ranges_level0=coordinate_ranges,
                    )
                manifest_records.extend(records)
                slide_summaries.append(slide_summary)
            except Exception:
                logger.exception("Failed to process slide: %s", row.get("source_path", ""))
                slide_summaries.append(
                    {
                        "source_path": row.get("source_path", ""),
                        "case_id": row.get("case_id", ""),
                        "slide_id": row.get("slide_id", ""),
                        "error": True,
                    }
                )

        manifest_columns = [
            "patch_path",
            "case_id",
            "slide_id",
            "label",
            "x",
            "y",
            "x0_level0",
            "y0_level0",
            "x1_level0",
            "y1_level0",
            "x_center_level0",
            "y_center_level0",
            "coordinate_format",
            "level",
            "patch_size",
            "tissue_ratio",
            "blur_score",
            "mean_brightness",
            "keep",
            "discard_reason",
            "split",
            "patch_provenance_id",
            "wsi_content_sha256",
            "source_wsi_path",
            "extraction_profile_id",
            "extraction_profile",
            "declared_patch_pixel_sha256",
            "replay_pixel_sha256",
            "annotation_path",
            "annotation_geometry_sha256",
            "declared_spatial_relation",
            "computed_spatial_relation",
            "verification_predicates",
            "verification_reasons",
            "verification_status",
        ]
        manifest = pd.DataFrame(manifest_records, columns=manifest_columns)
        patch_manifest_path = directories["metadata"] / "patch_manifest.csv"
        manifest.to_csv(patch_manifest_path, index=False)
        progress(95, "Writing summary")

        kept_total = int(manifest["keep"].sum()) if not manifest.empty else 0
        discarded_total = int((~manifest["keep"]).sum()) if not manifest.empty else 0
        split_counts = (
            manifest[manifest["keep"]].groupby("split").size().to_dict()
            if not manifest.empty
            else {}
        )
        finished_at = datetime.now()
        summary = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "input": {
                "wsi_dir": str(wsi_dir),
                "metadata_file": str(metadata_file),
                "scanned_files": scan_result.image_file_count,
                "supported_format_count": scan_result.supported_format_count,
                "unsupported_count": scan_result.unsupported_count,
            },
            "metadata": {
                "matched_count": clean_result.matched_count,
                "unmatched_count": clean_result.unmatched_count,
                "metadata_cleaned_csv": str(metadata_cleaned_path),
                "mapping_csv": str(mapping_path),
            },
            "coordinates": {
                "enabled": bool(config.get("coordinates", {}).get("enable", True)),
                "format": "range_level0",
                "slides": sampling_results,
            },
            "visualization": {
                "save_overlay": bool(config.get("visualization", {}).get("save_overlay", False)),
                "first_overlay": next(
                    (item.get("overlay", "") for item in sampling_results if item.get("overlay")),
                    "",
                ),
            },
            "features": {
                "enabled": bool(config.get("features", {}).get("enable", False)),
                "model_name": config.get("features", {}).get("model_name", ""),
                "results": feature_results,
            },
            "runtime": {
                "gpu_status": describe_gpus(),
                "cpu_workers": int(config.get("runtime", {}).get("cpu_workers", 1)),
                "gpu_workers": int(config.get("runtime", {}).get("gpu_workers", 1)),
            },
            "patches": {
                "kept": kept_total,
                "discarded": discarded_total,
                "manifest_csv": str(patch_manifest_path),
                "split_counts": split_counts,
            },
            "slides": slide_summaries,
            "reports": {
                "summary_json": str(directories["reports"] / "summary.json"),
                "log_file": str(directories["logs"] / "processing.log"),
                "used_config": str(directories["configs"] / "used_config.yaml"),
            },
        }
        write_json(summary, directories["reports"] / "summary.json")
        logger.info("Pipeline finished: kept=%s discarded=%s", kept_total, discarded_total)
        progress(100, "Done")
        return summary
    except Exception:
        logger.exception("Pipeline failed")
        progress(100, "Failed")
        raise
