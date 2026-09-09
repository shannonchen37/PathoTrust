"""Strict deterministic provenance verification for WSI-derived patch assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, __version__ as pillow_version

from pathodataforge.core.wsi_reader import WSIReader


PROFILE_SCHEMA_VERSION = "pathodataforge.strict-replay.v1"
GEOMETRY_SCHEMA_VERSION = "pathodataforge.level0-geometry.v1"
CANONICAL_PIXEL_REPRESENTATION = "rgb-uint8-hwc-tight-v1"
LOSSLESS_PATCH_SUFFIXES = {".png", ".tif", ".tiff", ".bmp"}


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    PATCH_REPLAY_MISMATCH = "PATCH_REPLAY_MISMATCH"
    GEOMETRY_MISMATCH = "GEOMETRY_MISMATCH"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    AMBIGUOUS_SOURCE = "AMBIGUOUS_SOURCE"
    UNSUPPORTED_PROFILE = "UNSUPPORTED_PROFILE"


@dataclass(slots=True)
class VerificationResult:
    status: VerificationStatus
    predicates: dict[str, bool | None]
    replay_pixel_sha256: str = ""
    declared_pixel_sha256: str = ""
    computed_spatial_relation: str = ""
    reasons: tuple[str, ...] = ()

    def to_manifest_fields(self) -> dict[str, Any]:
        return {
            "verification_status": self.status.value,
            "verification_predicates": canonical_json(self.predicates),
            "replay_pixel_sha256": self.replay_pixel_sha256,
            "declared_pixel_sha256": self.declared_pixel_sha256,
            "computed_spatial_relation": self.computed_spatial_relation,
            "verification_reasons": "|".join(self.reasons),
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoder_identity(reader: WSIReader) -> dict[str, str]:
    if reader.backend == "openslide":
        try:
            import openslide  # type: ignore

            version = str(getattr(openslide, "__version__", "unknown"))
        except Exception:
            version = "unknown"
        return {"name": "openslide-python", "version": version}
    return {"name": "Pillow", "version": str(pillow_version)}


def build_extraction_profile(
    reader: WSIReader,
    *,
    source_content_sha256: str,
    level: int,
    read_size: tuple[int, int],
) -> dict[str, Any]:
    """Describe the existing extraction behavior without inventing new defaults."""
    info = reader.info()
    if level < 0 or level >= info.level_count:
        raise ValueError(f"level {level} is outside the WSI pyramid")
    width, height = (int(read_size[0]), int(read_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("read_size must contain positive integers")
    downsample = float(info.level_downsamples[level])
    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "source": {
            "content_sha256": source_content_sha256,
            "level0_size": [int(info.width), int(info.height)],
            "reader_backend": info.reader_backend,
            "decoder": decoder_identity(reader),
        },
        "coordinates": {
            "space": "level0",
            "origin": "upper-left",
            "x_direction": "right",
            "y_direction": "down",
            "integer_type": "int64",
            "range_rule": "half-open-[x0,y0,x1,y1)",
            "level0_extent_rule": "round(read-size-times-level-downsample)",
        },
        "extraction": {
            "level": int(level),
            "level_downsample": downsample,
            "objective_power": info.objective_power,
            "selected_magnification": (
                info.objective_power / downsample if info.objective_power is not None else None
            ),
            "read_size": [width, height],
            "output_size": [width, height],
            "resampling": "native-level" if info.reader_backend == "openslide" or level == 0 else "Pillow-BILINEAR",
            "out_of_bounds": "reject",
            "padding": "none",
            "orientation": "OpenSlide-native" if info.reader_backend == "openslide" else "Pillow-ImageOps.exif_transpose",
        },
        "normalization": {
            "pixel_mode": "RGB",
            "alpha": "discard-by-convert-RGB",
            "icc_color_space": "no-explicit-conversion-preserve-decoder-values",
            "bit_depth": 8,
            "dtype": "uint8",
            "channel_order": "RGB",
            "canonical_pixel_representation": CANONICAL_PIXEL_REPRESENTATION,
        },
        "output": {"strict_lossless": True, "allowed_suffixes": sorted(LOSSLESS_PATCH_SUFFIXES)},
    }
    profile["profile_id"] = profile_id(profile)
    return profile


def profile_id(profile: dict[str, Any]) -> str:
    payload = {key: value for key, value in profile.items() if key != "profile_id"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    header = canonical_json(
        {
            "representation": CANONICAL_PIXEL_REPRESENTATION,
            "shape": list(array.shape),
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_annotation_geometry(payload: dict[str, Any]) -> list[list[list[int]]]:
    result: list[list[list[int]]] = []
    for stroke in payload.get("annotations", []):
        points: list[list[int]] = []
        for point in stroke.get("points_level0", []):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("annotation point must be a level-0 [x, y] pair")
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError("annotation point must use integer coordinates")
            x_int, y_int = int(x), int(y)
            if float(x) != x_int or float(y) != y_int:
                raise ValueError("annotation point must use integer coordinates")
            points.append([x_int, y_int])
        if points:
            result.append(points)
    return result


def annotation_geometry_sha256(payload: dict[str, Any]) -> str:
    geometry = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "coordinate_space": "level0",
        "polylines": canonical_annotation_geometry(payload),
    }
    return hashlib.sha256(canonical_json(geometry).encode("utf-8")).hexdigest()


def build_annotation_provenance_payload(
    source_path: str | Path,
    level0_size: tuple[int, int],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_wsi": {
            "content_sha256": sha256_file(source_path),
            "level0_size": [int(level0_size[0]), int(level0_size[1])],
        },
        "coordinate_space": "level0",
        "geometry_profile": {
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "origin": "upper-left",
            "x_direction": "right",
            "y_direction": "down",
            "integer_type": "int64",
            "primitive": "polyline-centerline",
        },
        "annotation_type": "doctor_freehand_high_risk",
        "annotations": annotations,
    }
    payload["geometry_sha256"] = annotation_geometry_sha256(payload)
    return payload


def resolve_annotation_by_wsi_content(
    annotation_dir: str | Path,
    wsi_content_sha256: str,
) -> Path | None:
    """Resolve one annotation by exact WSI content identity; never select a first fuzzy match."""
    matches: list[Path] = []
    for path in sorted(Path(annotation_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("source_wsi", {}).get("content_sha256", "")) == wsi_content_sha256:
            matches.append(path)
    if len(matches) > 1:
        raise ValueError(
            f"AMBIGUOUS_SOURCE: {len(matches)} annotation files reference WSI content {wsi_content_sha256[:12]}"
        )
    return matches[0] if matches else None


def bind_annotation_relations(
    patch_manifest_path: str | Path,
    annotation_path: str | Path,
) -> int:
    """Declare spatial provenance edges for exact-content patch rows, without assigning labels."""
    import pandas as pd

    manifest_path = Path(patch_manifest_path)
    annotation_path = Path(annotation_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    source_digest = str(payload.get("source_wsi", {}).get("content_sha256", ""))
    if len(source_digest) != 64:
        raise ValueError("UNSUPPORTED_PROFILE: annotation lacks an exact WSI content instance")
    geometry = canonical_annotation_geometry(payload)
    geometry_digest = annotation_geometry_sha256(payload)
    manifest = pd.read_csv(manifest_path).fillna("")
    if "wsi_content_sha256" not in manifest:
        raise ValueError("UNSUPPORTED_PROFILE: patch manifest lacks WSI content instances")
    kept = manifest["keep"].astype(str).str.lower().isin(["true", "1"])
    matched = kept & manifest["wsi_content_sha256"].astype(str).eq(source_digest)
    if not matched.any():
        raise ValueError("SOURCE_MISMATCH: no kept patch belongs to the annotation WSI content instance")
    for column in [
        "annotation_path",
        "annotation_geometry_sha256",
        "declared_spatial_relation",
        "computed_spatial_relation",
        "verification_predicates",
        "verification_reasons",
        "verification_status",
    ]:
        if column not in manifest:
            manifest[column] = ""
    for index in manifest.index[matched]:
        patch_range = (
            int(manifest.at[index, "x0_level0"]),
            int(manifest.at[index, "y0_level0"]),
            int(manifest.at[index, "x1_level0"]),
            int(manifest.at[index, "y1_level0"]),
        )
        manifest.at[index, "annotation_path"] = str(annotation_path.resolve())
        manifest.at[index, "annotation_geometry_sha256"] = geometry_digest
        manifest.at[index, "declared_spatial_relation"] = spatial_relation(geometry, patch_range)
        manifest.at[index, "computed_spatial_relation"] = ""
        manifest.at[index, "verification_predicates"] = canonical_json(
            {
                "source_content_match": None,
                "profile_supported": None,
                "range_valid": None,
                "declared_patch_matches_record": None,
                "patch_replay_match": None,
                "annotation_source_match": None,
                "annotation_geometry_match": None,
                "spatial_relation_match": None,
            }
        )
        manifest.at[index, "verification_reasons"] = "annotation spatial declaration changed; delivery replay required"
        manifest.at[index, "verification_status"] = "CANDIDATE"
    manifest.to_csv(manifest_path, index=False)
    return int(matched.sum())


def spatial_relation(
    polylines: Iterable[Iterable[Iterable[int]]],
    patch_range_level0: tuple[int, int, int, int],
) -> str:
    """Return INTERSECTS or DISJOINT for centerline geometry; no label is inferred."""
    x0, y0, x1, y1 = map(int, patch_range_level0)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("patch range must be a non-empty half-open rectangle")

    def inside(point: tuple[int, int]) -> bool:
        return x0 <= point[0] < x1 and y0 <= point[1] < y1

    def orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
        value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        return 0 if value == 0 else (1 if value > 0 else 2)

    def on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

    def segments_intersect(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int], d: tuple[int, int]) -> bool:
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        if o1 != o2 and o3 != o4:
            return True
        return (
            (o1 == 0 and on_segment(a, c, b))
            or (o2 == 0 and on_segment(a, d, b))
            or (o3 == 0 and on_segment(c, a, d))
            or (o4 == 0 and on_segment(c, b, d))
        )

    edges = [((x0, y0), (x1 - 1, y0)), ((x1 - 1, y0), (x1 - 1, y1 - 1)), ((x1 - 1, y1 - 1), (x0, y1 - 1)), ((x0, y1 - 1), (x0, y0))]
    for raw_line in polylines:
        line = [(int(point[0]), int(point[1])) for point in raw_line]
        if any(inside(point) for point in line):
            return "INTERSECTS"
        for first, second in zip(line, line[1:]):
            if any(segments_intersect(first, second, edge[0], edge[1]) for edge in edges):
                return "INTERSECTS"
    return "DISJOINT"


def _profile_is_supported(reader: WSIReader, profile: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        reasons.append("profile schema version is unsupported")
    if profile.get("profile_id") != profile_id(profile):
        reasons.append("profile_id does not match the profile content")
    source = profile.get("source", {})
    if source.get("reader_backend") != reader.backend:
        reasons.append("reader backend differs from the recorded profile")
    if source.get("decoder") != decoder_identity(reader):
        reasons.append("decoder identity or version differs from the recorded profile")
    normalization = profile.get("normalization", {})
    expected = {
        "pixel_mode": "RGB",
        "alpha": "discard-by-convert-RGB",
        "icc_color_space": "no-explicit-conversion-preserve-decoder-values",
        "bit_depth": 8,
        "dtype": "uint8",
        "channel_order": "RGB",
        "canonical_pixel_representation": CANONICAL_PIXEL_REPRESENTATION,
    }
    if normalization != expected:
        reasons.append("normalization rules are unsupported")
    output = profile.get("output", {})
    if output.get("strict_lossless") is not True:
        reasons.append("strict replay requires a lossless declared patch")
    try:
        extraction = profile["extraction"]
        expected_profile = build_extraction_profile(
            reader,
            source_content_sha256=str(source.get("content_sha256", "")),
            level=int(extraction["level"]),
            read_size=tuple(int(value) for value in extraction["read_size"]),
        )
        if profile != expected_profile:
            reasons.append("profile does not match the current extraction implementation")
    except (KeyError, TypeError, ValueError):
        reasons.append("profile lacks required extraction fields")
    return not reasons, reasons


def verify_patch_derivation(
    *,
    reader: WSIReader,
    source_path: str | Path,
    patch_path: str | Path,
    patch_range_level0: tuple[int, int, int, int],
    profile: dict[str, Any] | None,
    declared_patch_pixel_sha256: str,
    annotation_payload: dict[str, Any] | None = None,
    declared_annotation_geometry_sha256: str | None = None,
    declared_spatial_relation: str | None = None,
) -> VerificationResult:
    predicates: dict[str, bool | None] = {
        "source_content_match": False,
        "profile_supported": False,
        "range_valid": False,
        "declared_patch_matches_record": False,
        "patch_replay_match": False,
        "annotation_source_match": None,
        "annotation_geometry_match": None,
        "spatial_relation_match": None,
    }
    if not profile:
        return VerificationResult(VerificationStatus.UNSUPPORTED_PROFILE, predicates, reasons=("extraction profile is missing",))
    supported, reasons = _profile_is_supported(reader, profile)
    predicates["profile_supported"] = supported
    if not supported:
        return VerificationResult(VerificationStatus.UNSUPPORTED_PROFILE, predicates, reasons=tuple(reasons))

    source_digest = sha256_file(source_path)
    predicates["source_content_match"] = source_digest == profile.get("source", {}).get("content_sha256")
    if not predicates["source_content_match"]:
        return VerificationResult(VerificationStatus.SOURCE_MISMATCH, predicates, reasons=("WSI content does not match the declared source instance",))

    patch = Path(patch_path)
    if patch.suffix.lower() not in LOSSLESS_PATCH_SUFFIXES:
        return VerificationResult(VerificationStatus.UNSUPPORTED_PROFILE, predicates, reasons=("declared patch encoding is not supported by strict lossless replay",))
    extraction = profile.get("extraction", {})
    try:
        level = int(extraction["level"])
        read_size = tuple(int(value) for value in extraction["read_size"])
    except (KeyError, TypeError, ValueError):
        return VerificationResult(VerificationStatus.UNSUPPORTED_PROFILE, predicates, reasons=("profile lacks a valid level or read_size",))
    info = reader.info()
    x0, y0, x1, y1 = map(int, patch_range_level0)
    if x0 < 0 or y0 < 0 or x1 > info.width or y1 > info.height or x0 >= x1 or y0 >= y1:
        return VerificationResult(VerificationStatus.OUT_OF_BOUNDS, predicates, reasons=("declared level-0 range is outside the WSI",))
    expected_x1 = x0 + int(round(read_size[0] * float(extraction.get("level_downsample", 0))))
    expected_y1 = y0 + int(round(read_size[1] * float(extraction.get("level_downsample", 0))))
    predicates["range_valid"] = (x1, y1) == (expected_x1, expected_y1)
    if not predicates["range_valid"]:
        return VerificationResult(VerificationStatus.GEOMETRY_MISMATCH, predicates, reasons=("declared range is inconsistent with read_size and downsample",))

    try:
        with Image.open(patch) as declared_image:
            declared_digest = canonical_pixel_sha256(declared_image)
    except Exception as exc:
        return VerificationResult(VerificationStatus.UNSUPPORTED_PROFILE, predicates, reasons=(f"declared patch cannot be decoded: {exc}",))
    predicates["declared_patch_matches_record"] = declared_digest == declared_patch_pixel_sha256

    replay_image = reader.read_region_level0(x0, y0, level, read_size)
    replay_digest = canonical_pixel_sha256(replay_image)
    predicates["patch_replay_match"] = replay_digest == declared_digest == declared_patch_pixel_sha256

    relation = ""
    if annotation_payload is not None or declared_spatial_relation is not None:
        if annotation_payload is None or not declared_spatial_relation:
            predicates["annotation_source_match"] = False
            predicates["annotation_geometry_match"] = False
            predicates["spatial_relation_match"] = False
        else:
            try:
                annotation_source = str(
                    annotation_payload.get("source_wsi", {}).get("content_sha256", "")
                )
                predicates["annotation_source_match"] = annotation_source == source_digest
                current_geometry_digest = annotation_geometry_sha256(annotation_payload)
                predicates["annotation_geometry_match"] = (
                    not declared_annotation_geometry_sha256
                    or current_geometry_digest == declared_annotation_geometry_sha256
                )
                geometry = canonical_annotation_geometry(annotation_payload)
                relation = spatial_relation(geometry, patch_range_level0)
                predicates["spatial_relation_match"] = relation == str(declared_spatial_relation).upper()
            except (TypeError, ValueError):
                predicates["annotation_source_match"] = False
                predicates["annotation_geometry_match"] = False
                predicates["spatial_relation_match"] = False

    if predicates["annotation_source_match"] is False:
        status = VerificationStatus.SOURCE_MISMATCH
        reasons.append("annotation references a different or unknown WSI content instance")
    elif predicates["annotation_geometry_match"] is False or predicates["spatial_relation_match"] is False:
        status = VerificationStatus.GEOMETRY_MISMATCH
        reasons.append("annotation geometry or recomputed patch relation differs from the declaration")
    elif not predicates["patch_replay_match"]:
        status = VerificationStatus.PATCH_REPLAY_MISMATCH
        reasons.append("replayed canonical pixels differ from the declared patch")
    else:
        status = VerificationStatus.VERIFIED
    return VerificationResult(
        status,
        predicates,
        replay_pixel_sha256=replay_digest,
        declared_pixel_sha256=declared_digest,
        computed_spatial_relation=relation,
        reasons=tuple(reasons),
    )


def require_verified_delivery(manifest_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return kept VERIFIED rows or fail before any delivery directory is created."""
    kept = [row for row in manifest_rows if str(row.get("keep", "")).lower() in {"true", "1"}]
    failures = [row for row in kept if row.get("verification_status") != VerificationStatus.VERIFIED.value]
    if failures:
        statuses = sorted({str(row.get("verification_status") or "MISSING") for row in failures})
        raise ValueError(f"Delivery blocked: {len(failures)} kept patch assets are not VERIFIED ({', '.join(statuses)})")
    return kept
