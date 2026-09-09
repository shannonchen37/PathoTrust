"""Schema constants for privacy-preserving WSI indexing."""

from __future__ import annotations

from datetime import datetime
from typing import Any


SENSITIVE_FIELDS = {
    "patient_name",
    "id_card",
    "phone",
    "birth_date",
    "real_patient_id",
    "hospital_id",
    "raw_pathology_id",
    "姓名",
    "身份证",
    "手机号",
    "出生日期",
    "真实患者ID",
    "医院原始ID",
}

PRIVATE_MAPPING_COLUMNS = [
    "real_patient_id",
    "patient_name",
    "hospital_id",
    "case_id",
    "pathology_id",
    "pseudo_patient_uid",
]

MANIFEST_COLUMNS = [
    "patient_uid",
    "case_uid",
    "specimen_uid",
    "block_uid",
    "slide_uid",
    "wsi_uid",
    "patch_uid",
    "wsi_path",
    "wsi_format",
    "checksum_sha256",
    "wsi_content_instance",
    "stain_type",
    "tumor_site",
    "diagnosis",
    "label",
    "age_group",
    "scan_month",
    "doctor_uid",
    "annotation_uid",
    "annotation_path",
    "annotation_source_sha256",
    "annotation_geometry_sha256",
    "annotation_verification_status",
    "replay_profile_ids",
    "replay_profiles",
    "patch_verification_predicates",
    "verified_patch_count",
    "patch_verification_status",
    "created_at",
]


def age_group(value: Any) -> str:
    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return ""
    if age < 20:
        return "0-20"
    if age < 40:
        return "20-40"
    if age < 60:
        return "40-60"
    if age < 80:
        return "60-80"
    return "80+"


def month_bucket(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return text[:7] if len(text) >= 7 else ""
