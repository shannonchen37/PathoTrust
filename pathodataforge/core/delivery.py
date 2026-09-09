"""Build public delivery sets only from strictly verified derived assets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

from pathodataforge.core.replay_verifier import (
    VerificationStatus,
    canonical_json,
    require_verified_delivery,
    sha256_file,
    verify_patch_derivation,
)
from pathodataforge.core.scanner import SUPPORTED_WSI_EXTENSIONS
from pathodataforge.core.wsi_reader import WSIReader


def _source_candidates(
    row: dict,
    search_roots: Iterable[str | Path],
    digest_cache: dict[Path, str],
) -> list[Path]:
    paths: set[Path] = set()
    declared = Path(str(row.get("source_wsi_path", "")))
    if declared.is_file():
        paths.add(declared.resolve())
    for raw_root in search_roots:
        root = Path(raw_root)
        if root.is_file():
            paths.add(root.resolve())
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_EXTENSIONS:
                    paths.add(path.resolve())
    expected = str(row.get("wsi_content_sha256", ""))
    matches: list[Path] = []
    for path in sorted(paths):
        if path not in digest_cache:
            digest_cache[path] = sha256_file(path)
        digest = digest_cache[path]
        if digest == expected:
            matches.append(path)
    return matches


def _reverify_rows(
    source_root: Path,
    rows: list[dict],
    source_search_roots: Iterable[str | Path],
) -> list[dict]:
    digest_cache: dict[Path, str] = {}
    source_cache: dict[str, list[Path]] = {}
    updated: list[dict] = []
    for original in rows:
        row = dict(original)
        if str(row.get("keep", "")).lower() not in {"true", "1"}:
            updated.append(row)
            continue
        expected_source = str(row.get("wsi_content_sha256", ""))
        if expected_source not in source_cache:
            source_cache[expected_source] = _source_candidates(
                row,
                source_search_roots,
                digest_cache,
            )
        matches = source_cache[expected_source]
        if len(matches) != 1:
            status = (
                VerificationStatus.AMBIGUOUS_SOURCE
                if len(matches) > 1
                else VerificationStatus.SOURCE_MISMATCH
            )
            row.update(
                {
                    "verification_status": status.value,
                    "verification_predicates": canonical_json(
                        {
                            "source_content_match": False,
                            "source_is_unambiguous": len(matches) == 1,
                            "profile_supported": None,
                            "range_valid": None,
                            "declared_patch_matches_record": None,
                            "patch_replay_match": None,
                            "annotation_source_match": None,
                            "annotation_geometry_match": None,
                            "spatial_relation_match": None,
                        }
                    ),
                    "verification_reasons": (
                        f"{len(matches)} WSI content candidates resolve the declared source instance"
                    ),
                }
            )
            updated.append(row)
            continue
        try:
            profile = json.loads(str(row.get("extraction_profile", "")))
        except Exception:
            profile = None
        annotation_payload = None
        annotation_path = Path(str(row.get("annotation_path", "")))
        if str(row.get("declared_spatial_relation", "")) and annotation_path.is_file():
            try:
                annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            except Exception:
                annotation_payload = None
        patch_path = source_root / str(row.get("patch_path", ""))
        with WSIReader(matches[0]) as reader:
            result = verify_patch_derivation(
                reader=reader,
                source_path=matches[0],
                patch_path=patch_path,
                patch_range_level0=(
                    int(row["x0_level0"]),
                    int(row["y0_level0"]),
                    int(row["x1_level0"]),
                    int(row["y1_level0"]),
                ),
                profile=profile,
                declared_patch_pixel_sha256=str(row.get("declared_patch_pixel_sha256", "")),
                annotation_payload=annotation_payload,
                declared_annotation_geometry_sha256=str(row.get("annotation_geometry_sha256", "")) or None,
                declared_spatial_relation=str(row.get("declared_spatial_relation", "")) or None,
            )
        row.update(result.to_manifest_fields())
        row["source_wsi_path"] = str(matches[0])
        updated.append(row)
    return updated


def build_verified_delivery(
    source_root: str | Path,
    destination: str | Path,
    *,
    source_search_roots: Iterable[str | Path] = (),
) -> int:
    source_root = Path(source_root)
    destination = Path(destination)
    manifest_path = source_root / "metadata" / "patch_manifest.csv"
    if not manifest_path.exists():
        raise ValueError("Missing patch manifest; deterministic provenance admission cannot run")
    manifest = pd.read_csv(manifest_path).fillna("")
    rows = _reverify_rows(
        source_root,
        manifest.to_dict(orient="records"),
        source_search_roots,
    )
    # Persist the fresh validation state even when the admission gate rejects delivery.
    refreshed_manifest = pd.DataFrame(rows, columns=manifest.columns)
    refreshed_manifest.to_csv(manifest_path, index=False)
    verified_rows = require_verified_delivery(rows)

    # No output directory exists until the complete admission predicate succeeds.
    destination.mkdir(parents=True, exist_ok=False)
    coordinates = source_root / "metadata" / "coordinates"
    if coordinates.is_dir():
        shutil.copytree(coordinates, destination / "metadata" / "coordinates")
    for row in verified_rows:
        relative_patch = Path(str(row["patch_path"]))
        source_patch = source_root / relative_patch
        if not source_patch.is_file():
            shutil.rmtree(destination)
            raise ValueError(f"VERIFIED patch is missing at delivery time: {relative_patch}")
        target = destination / relative_patch
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_patch, target)

    public_columns = [column for column in manifest.columns if column != "source_wsi_path"]
    verified_manifest = pd.DataFrame(verified_rows, columns=manifest.columns)[public_columns]
    if "annotation_path" in verified_manifest:
        verified_manifest["annotation_path"] = verified_manifest["annotation_path"].map(
            lambda value: (
                str(Path("reports") / "annotations" / Path(str(value)).name)
                if str(value)
                else ""
            )
        )
    public_manifest = destination / "metadata" / "patch_manifest.csv"
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    verified_manifest.to_csv(public_manifest, index=False)

    annotation_paths = {
        str(row.get("annotation_path", ""))
        for row in verified_rows
        if str(row.get("annotation_path", ""))
        and str(row.get("declared_spatial_relation", ""))
    }
    for annotation_path in sorted(annotation_paths):
        source = Path(annotation_path)
        if not source.is_file():
            shutil.rmtree(destination)
            raise ValueError(f"VERIFIED annotation is missing at delivery time: {source.name}")
        target = destination / "reports" / "annotations" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (destination / "DATASET_README.txt").write_text(
        "\n".join(
            [
                "iMoonLab-PathoDataForge verified data package",
                "",
                "- patches/: deterministic-replay VERIFIED patch assets only",
                "- metadata/patch_manifest.csv: VERIFIED provenance and predicates",
                "- metadata/coordinates/: level-0 coordinate records when available",
                "- reports/annotations/: only annotations referenced by VERIFIED spatial edges",
                "",
                "Private mappings, unverified features, overlays, and internal summaries are excluded.",
            ]
        ),
        encoding="utf-8",
    )
    return len(verified_rows)
