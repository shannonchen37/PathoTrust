"""Build de-identified multi-level WSI manifests."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pathodataforge.privacy_index.anonymizer import make_uid, pseudo_patient_uid
from pathodataforge.privacy_index.checksum import sha256_file
from pathodataforge.privacy_index.schema import MANIFEST_COLUMNS, PRIVATE_MAPPING_COLUMNS, age_group, month_bucket
from pathodataforge.core.replay_verifier import annotation_geometry_sha256


def build_privacy_index(
    clinical_df: pd.DataFrame,
    processed_root: str | Path,
    *,
    wsi_root: str | Path | None = None,
    annotation_root: str | Path | None = None,
    secret_key: str | bytes | None = None,
    test_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = Path(processed_root)
    anonymized_wsi_root = processed / "anonymized_wsi"
    annotation_out_root = processed / "manifests" / "annotations"
    anonymized_wsi_root.mkdir(parents=True, exist_ok=True)
    annotation_out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for row_index, row in clinical_df.fillna("").iterrows():
        source_wsi = _source_wsi_path(row, wsi_root)
        checksum = sha256_file(source_wsi)
        real_patient_id = _field(row, "real_patient_id") or _field(row, "case_id") or f"ROW_{row_index:06d}"
        patient_uid = pseudo_patient_uid(real_patient_id, secret_key, test_mode=test_mode)
        case_uid = make_uid("CASE", patient_uid, _field(row, "case_id"), secret_key=secret_key, test_mode=test_mode)
        specimen_uid = make_uid("SPEC", case_uid, _field(row, "specimen_id"), secret_key=secret_key, test_mode=test_mode)
        block_uid = make_uid("BLOCK", specimen_uid, _field(row, "block_id"), secret_key=secret_key, test_mode=test_mode)
        slide_uid = make_uid("SLIDE", block_uid, _field(row, "slide_id"), source_wsi.name, secret_key=secret_key, test_mode=test_mode)
        wsi_uid = f"WSI_{checksum[:12]}"
        patch_uid = make_uid("PATCH", slide_uid, "patch_set", secret_key=secret_key, test_mode=test_mode)
        annotation_uid = make_uid(
            "ANNO",
            slide_uid,
            _field(row, "annotation_file"),
            secret_key=secret_key,
            test_mode=test_mode,
        )

        anonymized_wsi = anonymized_wsi_root / f"{wsi_uid}{source_wsi.suffix.lower()}"
        if not anonymized_wsi.exists():
            shutil.copy2(source_wsi, anonymized_wsi)

        annotation_result = _copy_annotation(
            row,
            annotation_root,
            annotation_out_root,
            annotation_uid,
            checksum,
        )

        manifest_rows.append(
            {
                "patient_uid": patient_uid,
                "case_uid": case_uid,
                "specimen_uid": specimen_uid,
                "block_uid": block_uid,
                "slide_uid": slide_uid,
                "wsi_uid": wsi_uid,
                "patch_uid": patch_uid,
                "wsi_path": str(anonymized_wsi),
                "wsi_format": source_wsi.suffix.lower().lstrip("."),
                "checksum_sha256": checksum,
                "wsi_content_instance": f"sha256:{checksum}",
                "stain_type": _field(row, "stain_type"),
                "tumor_site": _field(row, "tumor_site"),
                "diagnosis": _field(row, "diagnosis"),
                "label": _field(row, "label"),
                "age_group": age_group(_field(row, "age")),
                "scan_month": month_bucket(_field(row, "scan_date")),
                "doctor_uid": make_uid("DOCTOR", _field(row, "doctor_id"), secret_key=secret_key, test_mode=test_mode)
                if _field(row, "doctor_id")
                else "",
                "annotation_uid": annotation_uid,
                "annotation_path": str(annotation_result["path"]) if annotation_result["path"] else "",
                "annotation_source_sha256": annotation_result["source_sha256"],
                "annotation_geometry_sha256": annotation_result["geometry_sha256"],
                "annotation_verification_status": annotation_result["status"],
                "replay_profile_ids": _field(row, "replay_profile_ids"),
                "replay_profiles": _field(row, "replay_profiles"),
                "patch_verification_predicates": _field(row, "patch_verification_predicates"),
                "verified_patch_count": _field(row, "verified_patch_count") or "0",
                "patch_verification_status": _field(row, "patch_verification_status") or "NOT_APPLICABLE",
                "created_at": created_at,
            }
        )
        private_rows.append(
            {
                "real_patient_id": real_patient_id,
                "patient_name": _field(row, "patient_name"),
                "hospital_id": _field(row, "hospital_id"),
                "case_id": _field(row, "case_id"),
                "pathology_id": _field(row, "pathology_id"),
                "pseudo_patient_uid": patient_uid,
            }
        )

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    private_mapping = pd.DataFrame(private_rows, columns=PRIVATE_MAPPING_COLUMNS).drop_duplicates()
    return manifest, private_mapping


def _source_wsi_path(row: pd.Series, wsi_root: str | Path | None) -> Path:
    source_path = _field(row, "source_path")
    if source_path:
        path = Path(source_path)
        if path.exists():
            return path
    filename = _field(row, "wsi_filename") or _field(row, "original_filename")
    if not filename:
        raise ValueError("Missing WSI filename/source_path in clinical table row.")
    path = Path(filename)
    if not path.is_absolute() and wsi_root is not None:
        path = Path(wsi_root) / path
    if not path.exists():
        raise FileNotFoundError(f"WSI file not found for privacy index: {path}")
    return path


def _copy_annotation(
    row: pd.Series,
    annotation_root: str | Path | None,
    annotation_out_root: Path,
    annotation_uid: str,
    expected_wsi_sha256: str,
) -> dict[str, Any]:
    annotation_file = _field(row, "annotation_file")
    if not annotation_file:
        return {"path": None, "source_sha256": "", "geometry_sha256": "", "status": "NOT_APPLICABLE"}
    source = Path(annotation_file)
    if not source.is_absolute() and annotation_root is not None:
        source = Path(annotation_root) / source
    if not source.exists():
        return {"path": None, "source_sha256": "", "geometry_sha256": "", "status": "SOURCE_MISMATCH"}
    if source.suffix.lower() != ".json":
        return {"path": None, "source_sha256": "", "geometry_sha256": "", "status": "UNSUPPORTED_PROFILE"}
    try:
        import json

        payload = json.loads(source.read_text(encoding="utf-8"))
        source_sha256 = str(payload.get("source_wsi", {}).get("content_sha256", ""))
        geometry_sha256 = annotation_geometry_sha256(payload)
    except Exception:
        return {"path": None, "source_sha256": "", "geometry_sha256": "", "status": "UNSUPPORTED_PROFILE"}
    if source_sha256 != expected_wsi_sha256:
        return {
            "path": None,
            "source_sha256": source_sha256,
            "geometry_sha256": geometry_sha256,
            "status": "SOURCE_MISMATCH",
        }
    target = annotation_out_root / f"{annotation_uid}{source.suffix.lower() or '.json'}"
    if not target.exists():
        shutil.copy2(source, target)
    return {
        "path": target,
        "source_sha256": source_sha256,
        "geometry_sha256": geometry_sha256,
        "status": "VERIFIED",
    }


def _field(row: pd.Series, key: str) -> str:
    value = row.get(key, "")
    return "" if pd.isna(value) else str(value).strip()
