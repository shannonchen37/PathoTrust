"""Streaming patch extraction and patch manifest construction."""

from __future__ import annotations

import logging
import json
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pathodataforge.core.anonymizer import sanitize_filename_token
from pathodataforge.core.quality_control import evaluate_patch
from pathodataforge.core.replay_verifier import (
    VerificationStatus,
    annotation_geometry_sha256,
    build_extraction_profile,
    canonical_json,
    canonical_pixel_sha256,
    sha256_file,
    verify_patch_derivation,
)
from pathodataforge.core.tissue_detector import create_tissue_mask, mask_region_ratio
from pathodataforge.core.wsi_reader import WSIInfo, WSIReader


def select_level(info: WSIInfo, level_value: Any) -> int:
    """Select a level by index or by magnification string such as 20x."""
    if isinstance(level_value, str):
        text = level_value.strip().lower()
        if text.endswith("x"):
            target = float(text[:-1])
            base_power = info.objective_power or 40.0
            magnifications = [
                base_power / max(downsample, 1.0) for downsample in info.level_downsamples
            ]
            return min(
                range(info.level_count),
                key=lambda index: abs(magnifications[index] - target),
            )
        level_index = int(text)
    else:
        level_index = int(level_value)
    return max(0, min(level_index, info.level_count - 1))


def _grid_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    step = max(patch_size, stride)
    return list(range(0, length - patch_size + 1, step))


def _manifest_record(
    row: pd.Series | dict[str, Any],
    split_name: str,
    patch_path: str,
    x: int,
    y: int,
    level: int,
    patch_size: int,
    tissue_ratio: float,
    blur_score: float,
    mean_brightness: float,
    keep: bool,
    discard_reason: str,
    x0_level0: int | None = None,
    y0_level0: int | None = None,
    x1_level0: int | None = None,
    y1_level0: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if x0_level0 is None:
        x0_level0 = x
    if y0_level0 is None:
        y0_level0 = y
    if x1_level0 is None:
        x1_level0 = x0_level0 + patch_size
    if y1_level0 is None:
        y1_level0 = y0_level0 + patch_size
    record = {
        "patch_path": patch_path,
        "case_id": str(row.get("case_id", "")),
        "slide_id": str(row.get("slide_id", "")),
        "label": row.get("label", ""),
        "x": x,
        "y": y,
        "x0_level0": int(x0_level0),
        "y0_level0": int(y0_level0),
        "x1_level0": int(x1_level0),
        "y1_level0": int(y1_level0),
        "x_center_level0": int(round((x0_level0 + x1_level0) / 2)),
        "y_center_level0": int(round((y0_level0 + y1_level0) / 2)),
        "coordinate_format": "range_level0",
        "level": level,
        "patch_size": patch_size,
        "tissue_ratio": round(float(tissue_ratio), 6),
        "blur_score": round(float(blur_score), 6),
        "mean_brightness": round(float(mean_brightness), 6),
        "keep": bool(keep),
        "discard_reason": discard_reason,
        "split": split_name,
    }
    record.update(
        provenance
        or {
            "patch_provenance_id": "",
            "wsi_content_sha256": "",
            "source_wsi_path": "",
            "extraction_profile_id": "",
            "extraction_profile": "",
            "declared_patch_pixel_sha256": "",
            "replay_pixel_sha256": "",
            "annotation_path": "",
            "annotation_geometry_sha256": "",
            "declared_spatial_relation": "",
            "computed_spatial_relation": "",
            "verification_predicates": "",
            "verification_reasons": "",
            "verification_status": "NOT_APPLICABLE",
        }
    )
    return record


def extract_patches_for_slide(
    reader: WSIReader,
    metadata_row: pd.Series | dict[str, Any],
    output_root: str | Path,
    patch_config: dict[str, Any],
    quality_config: dict[str, Any],
    logger: logging.Logger | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    coordinate_ranges_level0: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract kept patches for one slide and return all QC manifest rows."""
    output_root = Path(output_root)
    split_name = str(metadata_row.get("split", "train") or "train")
    split_dir = output_root / "patches" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    patch_size = int(patch_config.get("patch_size", 512))
    stride = max(patch_size, int(patch_config.get("stride", patch_size)))
    sampling_mode = str(patch_config.get("sampling_mode", "dense")).lower().replace("-", "_")
    dense_all = sampling_mode == "dense"
    max_patches = int(patch_config.get("max_patches_per_slide", 2000))
    max_candidates = int(
        patch_config.get("max_candidates_per_slide", max(max_patches * 20, max_patches + 1000))
    )
    if patch_size <= 0 or stride <= 0 or (max_patches <= 0 and not dense_all):
        raise ValueError("patch_size and stride must be > 0; max_patches_per_slide must be > 0 outside dense mode")

    info = reader.info()
    level = select_level(info, patch_config.get("level", 0))
    source_path = Path(str(metadata_row.get("source_path", reader.path)))
    source_content_sha256 = sha256_file(source_path)
    extraction_profile = build_extraction_profile(
        reader,
        source_content_sha256=source_content_sha256,
        level=level,
        read_size=(patch_size, patch_size),
    )
    annotation_path_text = str(metadata_row.get("annotation_file", "") or "").strip()
    declared_spatial_relation = str(metadata_row.get("annotation_patch_relation", "") or "").strip().upper()
    annotation_payload: dict[str, Any] | None = None
    annotation_digest = ""
    if annotation_path_text:
        annotation_path = Path(annotation_path_text)
        if annotation_path.exists() and annotation_path.suffix.lower() == ".json":
            annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotation_digest = annotation_geometry_sha256(annotation_payload)
    level_width, level_height = info.level_dimensions[level]
    thumbnail = reader.thumbnail((1024, 1024))
    tissue_mask = create_tissue_mask(thumbnail)

    tissue_threshold = float(quality_config.get("tissue_threshold", 0.5))
    blur_threshold = float(quality_config.get("blur_threshold", 80.0))
    brightness_min = float(quality_config.get("brightness_min", 30))
    brightness_max = float(quality_config.get("brightness_max", 230))
    prefilter_threshold = max(0.01, tissue_threshold * 0.25)

    case_token = sanitize_filename_token(metadata_row.get("case_id", "case"))
    slide_token = sanitize_filename_token(metadata_row.get("slide_id", "slide"))
    label = metadata_row.get("label", "")
    if logger:
        logger.info(
            "Extracting %s (%sx%s, level=%s, backend=%s, label=%s)",
            metadata_row.get("original_filename", metadata_row.get("source_path", "")),
            level_width,
            level_height,
            level,
            info.reader_backend,
            label,
        )

    records: list[dict[str, Any]] = []
    kept = 0
    discarded = 0
    evaluated = 0
    coordinates_from_cache = coordinate_ranges_level0 is not None

    if coordinate_ranges_level0 is not None:
        coordinate_items = []
        for x0, y0, x1, y1 in coordinate_ranges_level0.astype(np.int64):
            x_level = int(round(int(x0) / info.level_downsamples[level]))
            y_level = int(round(int(y0) / info.level_downsamples[level]))
            coordinate_items.append((x_level, y_level, int(x0), int(y0), int(x1), int(y1)))
    else:
        coordinate_items = [
            (
                x,
                y,
                int(round(x * info.level_downsamples[level])),
                int(round(y * info.level_downsamples[level])),
                int(round(x * info.level_downsamples[level]))
                + int(round(patch_size * info.level_downsamples[level])),
                int(round(y * info.level_downsamples[level]))
                + int(round(patch_size * info.level_downsamples[level])),
            )
            for y in _grid_positions(level_height, patch_size, stride)
            for x in _grid_positions(level_width, patch_size, stride)
        ]

    for x, y, x0_level0, y0_level0, x1_level0, y1_level0 in coordinate_items:
        if not dense_all and (kept >= max_patches or evaluated >= max_candidates):
            break
        evaluated += 1

        coarse_ratio = mask_region_ratio(
            tissue_mask,
            x=x,
            y=y,
            patch_size=patch_size,
            level_dimensions=(level_width, level_height),
        )
        if coarse_ratio < prefilter_threshold:
            discarded += 1
            records.append(
                _manifest_record(
                    metadata_row,
                    split_name,
                    "",
                    x,
                    y,
                    level,
                    patch_size,
                    coarse_ratio,
                    0.0,
                    255.0,
                    False,
                    "background_prefilter",
                    x0_level0,
                    y0_level0,
                    x1_level0,
                    y1_level0,
                )
            )
            continue

        patch = (
            reader.read_region_level0(x0_level0, y0_level0, level, (patch_size, patch_size))
            if coordinates_from_cache
            else reader.read_region_level(x, y, level, (patch_size, patch_size))
        )
        quality = evaluate_patch(
            patch,
            tissue_threshold=tissue_threshold,
            blur_threshold=blur_threshold,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
        )

        patch_path = ""
        provenance: dict[str, Any] | None = None
        if quality.keep:
            patch_id = kept + 1
            filename = f"{case_token}_{slide_token}_{x}_{y}_{level}_{patch_id:06d}.png"
            output_path = split_dir / filename
            patch.save(output_path)
            patch_path = str(output_path.relative_to(output_root))
            declared_pixel_digest = canonical_pixel_sha256(patch)
            provenance_seed = canonical_json(
                {
                    "source_content_sha256": source_content_sha256,
                    "range_level0": [x0_level0, y0_level0, x1_level0, y1_level0],
                    "extraction_profile_id": extraction_profile["profile_id"],
                }
            )
            patch_provenance_id = "PATCHPROV_" + hashlib.sha256(provenance_seed.encode("utf-8")).hexdigest()[:24]
            verification = verify_patch_derivation(
                reader=reader,
                source_path=source_path,
                patch_path=output_path,
                patch_range_level0=(x0_level0, y0_level0, x1_level0, y1_level0),
                profile=extraction_profile,
                declared_patch_pixel_sha256=declared_pixel_digest,
                annotation_payload=annotation_payload if declared_spatial_relation else None,
                declared_annotation_geometry_sha256=annotation_digest or None,
                declared_spatial_relation=declared_spatial_relation or None,
            )
            provenance = {
                "patch_provenance_id": patch_provenance_id,
                "wsi_content_sha256": source_content_sha256,
                "source_wsi_path": str(source_path.resolve()),
                "extraction_profile_id": extraction_profile["profile_id"],
                "extraction_profile": canonical_json(extraction_profile),
                "declared_patch_pixel_sha256": declared_pixel_digest,
                "annotation_path": annotation_path_text if declared_spatial_relation else "",
                "annotation_geometry_sha256": annotation_digest if declared_spatial_relation else "",
                "declared_spatial_relation": declared_spatial_relation,
                **verification.to_manifest_fields(),
            }
            if verification.status is not VerificationStatus.VERIFIED:
                output_path.unlink(missing_ok=True)
                patch_path = ""
            kept += 1
        else:
            discarded += 1

        records.append(
            _manifest_record(
                metadata_row,
                split_name,
                patch_path,
                x,
                y,
                level,
                patch_size,
                quality.tissue_ratio,
                quality.blur_score,
                quality.mean_brightness,
                quality.keep,
                quality.discard_reason,
                x0_level0,
                y0_level0,
                x1_level0,
                y1_level0,
                provenance,
            )
        )

        if progress_callback and evaluated % 50 == 0:
            progress_callback(evaluated, kept)

    if logger:
        logger.info(
            "Slide done: kept=%s discarded=%s evaluated=%s",
            kept,
            discarded,
            evaluated,
        )

    summary = {
        "source_path": str(metadata_row.get("source_path", "")),
        "case_id": str(metadata_row.get("case_id", "")),
        "slide_id": str(metadata_row.get("slide_id", "")),
        "level": level,
        "width": level_width,
        "height": level_height,
        "kept_patches": kept,
        "discarded_patches": discarded,
        "evaluated_patches": evaluated,
        "backend": info.reader_backend,
        "stopped_by_max_candidates": False if dense_all else evaluated >= max_candidates and kept < max_patches,
    }
    return records, summary
