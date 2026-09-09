from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin

from pathodataforge.core.patch_extractor import extract_patches_for_slide
from pathodataforge.core.delivery import build_verified_delivery
from pathodataforge.core.pipeline import run_pipeline
from pathodataforge.core.replay_verifier import (
    VerificationStatus,
    annotation_geometry_sha256,
    bind_annotation_relations,
    build_annotation_provenance_payload,
    build_extraction_profile,
    canonical_pixel_sha256,
    profile_id,
    require_verified_delivery,
    resolve_annotation_by_wsi_content,
    verify_patch_derivation,
)
from pathodataforge.core.wsi_reader import WSIReader


def _write_source(path: Path, offset: int = 0) -> None:
    yy, xx = np.mgrid[0:32, 0:32]
    array = np.stack(
        [
            (xx * 7 + offset) % 256,
            (yy * 9 + offset) % 256,
            ((xx + yy) * 5 + offset) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(array, "RGB").save(path)


def _declaration(tmp_path: Path) -> tuple[Path, Path, dict, str]:
    source = tmp_path / "slide.png"
    patch = tmp_path / "patch.png"
    _write_source(source)
    with WSIReader(source) as reader:
        profile = build_extraction_profile(
            reader,
            source_content_sha256=__import__("hashlib").sha256(source.read_bytes()).hexdigest(),
            level=0,
            read_size=(8, 8),
        )
        image = reader.read_region_level0(4, 5, 0, (8, 8))
        image.save(patch)
        digest = canonical_pixel_sha256(image)
    return source, patch, profile, digest


def _verify(source: Path, patch: Path, profile: dict | None, digest: str, **kwargs):
    with WSIReader(source) as reader:
        return verify_patch_derivation(
            reader=reader,
            source_path=source,
            patch_path=patch,
            patch_range_level0=kwargs.pop("patch_range_level0", (4, 5, 12, 13)),
            profile=profile,
            declared_patch_pixel_sha256=digest,
            **kwargs,
        )


def test_strict_replay_succeeds_for_exact_source_range_and_profile(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    result = _verify(source, patch, profile, digest)
    assert result.status is VerificationStatus.VERIFIED
    assert all(value is True for value in result.predicates.values() if value is not None)


def test_same_filename_with_different_content_and_wsi_replacement_fail(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    _write_source(source, offset=31)
    result = _verify(source, patch, profile, digest)
    assert result.status is VerificationStatus.SOURCE_MISMATCH
    assert result.predicates["source_content_match"] is False


def test_path_change_with_same_content_instance_still_replays(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    moved = tmp_path / "renamed-source.png"
    moved.write_bytes(source.read_bytes())
    result = _verify(moved, patch, profile, digest)
    assert result.status is VerificationStatus.VERIFIED


def test_range_change_or_out_of_bounds_cannot_verify(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    shifted = _verify(source, patch, profile, digest, patch_range_level0=(5, 5, 13, 13))
    outside = _verify(source, patch, profile, digest, patch_range_level0=(-1, 5, 7, 13))
    malformed = _verify(source, patch, profile, digest, patch_range_level0=(4, 5, 11, 13))
    assert shifted.status is VerificationStatus.PATCH_REPLAY_MISMATCH
    assert outside.status is VerificationStatus.OUT_OF_BOUNDS
    assert malformed.status is VerificationStatus.GEOMETRY_MISMATCH


def test_level_downsample_read_output_and_resampling_changes_fail(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    for path, value in [
        (("extraction", "level"), 1),
        (("extraction", "level_downsample"), 2.0),
        (("extraction", "read_size"), [7, 8]),
        (("extraction", "output_size"), [7, 8]),
        (("extraction", "resampling"), "Pillow-LANCZOS"),
    ]:
        changed = copy.deepcopy(profile)
        changed[path[0]][path[1]] = value
        changed["profile_id"] = profile_id(changed)
        assert _verify(source, patch, changed, digest).status is VerificationStatus.UNSUPPORTED_PROFILE


def test_pillow_multilevel_profile_replays_with_recorded_downsample(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    yy, xx = np.mgrid[0:512, 0:512]
    array = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    Image.fromarray(array).save(source)
    patch = tmp_path / "level1.png"
    with WSIReader(source) as reader:
        assert reader.info().level_count > 1
        profile = build_extraction_profile(
            reader,
            source_content_sha256=__import__("hashlib").sha256(source.read_bytes()).hexdigest(),
            level=1,
            read_size=(16, 16),
        )
        image = reader.read_region_level0(20, 30, 1, (16, 16))
        image.save(patch)
        digest = canonical_pixel_sha256(image)
        result = verify_patch_derivation(
            reader=reader,
            source_path=source,
            patch_path=patch,
            patch_range_level0=(20, 30, 52, 62),
            profile=profile,
            declared_patch_pixel_sha256=digest,
        )
    assert result.status is VerificationStatus.VERIFIED
    assert profile["extraction"]["level_downsample"] == 2.0
    assert profile["extraction"]["resampling"] == "Pillow-BILINEAR"


def test_missing_or_decoder_version_unknown_profile_never_passes(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    assert _verify(source, patch, None, digest).status is VerificationStatus.UNSUPPORTED_PROFILE
    changed = copy.deepcopy(profile)
    changed["source"]["decoder"]["version"] = "different"
    changed["profile_id"] = profile_id(changed)
    assert _verify(source, patch, changed, digest).status is VerificationStatus.UNSUPPORTED_PROFILE


def test_png_reencoding_with_identical_pixels_does_not_false_fail(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    with Image.open(patch) as image:
        pixels = image.convert("RGB").copy()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("different-container-metadata", "yes")
    pixels.save(patch, pnginfo=metadata, compress_level=9)
    assert _verify(source, patch, profile, digest).status is VerificationStatus.VERIFIED


def test_lossy_jpeg_is_unsupported_in_strict_mode(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    jpeg = patch.with_suffix(".jpg")
    with Image.open(patch) as image:
        image.save(jpeg, quality=95)
    assert _verify(source, jpeg, profile, digest).status is VerificationStatus.UNSUPPORTED_PROFILE


def test_geometry_and_replay_predicates_both_control_verified(tmp_path: Path) -> None:
    source, patch, profile, digest = _declaration(tmp_path)
    annotation = {
        "source_wsi": {"content_sha256": profile["source"]["content_sha256"]},
        "annotations": [{"label": "must-not-be-used-as-training-label", "points_level0": [[0, 8], [20, 8]]}]
    }
    passed = _verify(
        source,
        patch,
        profile,
        digest,
        annotation_payload=annotation,
        declared_annotation_geometry_sha256=annotation_geometry_sha256(annotation),
        declared_spatial_relation="INTERSECTS",
    )
    assert passed.status is VerificationStatus.VERIFIED
    changed_geometry = {
        "source_wsi": {"content_sha256": profile["source"]["content_sha256"]},
        "annotations": [{"label": "anything", "points_level0": [[20, 20], [22, 22]]}],
    }
    geometry_failed = _verify(
        source,
        patch,
        profile,
        digest,
        annotation_payload=changed_geometry,
        declared_annotation_geometry_sha256=annotation_geometry_sha256(annotation),
        declared_spatial_relation="INTERSECTS",
    )
    assert geometry_failed.status is VerificationStatus.GEOMETRY_MISMATCH
    assert geometry_failed.predicates["patch_replay_match"] is True
    with Image.open(patch) as image:
        changed_patch = np.asarray(image.convert("RGB")).copy()
    changed_patch[0, 0] = 255 - changed_patch[0, 0]
    Image.fromarray(changed_patch).save(patch)
    replay_failed = _verify(
        source,
        patch,
        profile,
        digest,
        annotation_payload=annotation,
        declared_annotation_geometry_sha256=annotation_geometry_sha256(annotation),
        declared_spatial_relation="INTERSECTS",
    )
    assert replay_failed.status is VerificationStatus.PATCH_REPLAY_MISMATCH
    assert replay_failed.predicates["spatial_relation_match"] is True
    assert "label" not in replay_failed.to_manifest_fields()


def test_annotation_geometry_digest_changes_without_creating_label_semantics() -> None:
    first = {"annotations": [{"label": "positive", "points_level0": [[1, 1], [2, 2]]}]}
    second = {"annotations": [{"label": "negative", "points_level0": [[1, 1], [2, 2]]}]}
    moved = {"annotations": [{"label": "positive", "points_level0": [[1, 1], [3, 2]]}]}
    assert annotation_geometry_sha256(first) == annotation_geometry_sha256(second)
    assert annotation_geometry_sha256(first) != annotation_geometry_sha256(moved)


def test_annotation_save_payload_uses_content_identity_and_level0_geometry(tmp_path: Path) -> None:
    source = tmp_path / "slide.png"
    _write_source(source)
    annotations = [{"label": "high_risk", "points_level0": [[1, 2], [3, 4]]}]
    payload = build_annotation_provenance_payload(source, (32, 32), annotations)
    assert payload["source_wsi"]["content_sha256"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    assert payload["coordinate_space"] == "level0"
    assert payload["geometry_profile"]["primitive"] == "polyline-centerline"
    assert payload["geometry_sha256"] == annotation_geometry_sha256(payload)
    assert "source_path" not in payload


def test_annotation_source_resolution_uses_exact_content_and_rejects_ambiguity(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    payload = {"source_wsi": {"content_sha256": "a" * 64}, "annotations": []}
    first = annotation_dir / "unrelated-name.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    assert resolve_annotation_by_wsi_content(annotation_dir, "a" * 64) == first
    (annotation_dir / "second.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        resolve_annotation_by_wsi_content(annotation_dir, "a" * 64)
    except ValueError as exc:
        assert "AMBIGUOUS_SOURCE" in str(exc)
    else:
        raise AssertionError("ambiguous annotation sources were silently first-matched")


def test_delivery_gate_rejects_every_non_verified_kept_asset() -> None:
    verified = {"keep": True, "patch_path": "patches/a.png", "verification_status": "VERIFIED"}
    assert require_verified_delivery([verified]) == [verified]
    for status in [
        "SOURCE_MISMATCH",
        "PATCH_REPLAY_MISMATCH",
        "GEOMETRY_MISMATCH",
        "OUT_OF_BOUNDS",
        "AMBIGUOUS_SOURCE",
        "UNSUPPORTED_PROFILE",
        "",
    ]:
        row = {"keep": True, "patch_path": "patches/b.png", "verification_status": status}
        try:
            require_verified_delivery([verified, row])
        except ValueError as exc:
            assert "Delivery blocked" in str(exc)
        else:
            raise AssertionError(f"delivery unexpectedly accepted {status!r}")


def test_real_patch_extractor_records_and_validates_replay_provenance(tmp_path: Path) -> None:
    source = tmp_path / "slide.png"
    _write_source(source)
    output = tmp_path / "output"
    with WSIReader(source) as reader:
        rows, summary = extract_patches_for_slide(
            reader,
            metadata_row={"case_id": "C1", "slide_id": "S1", "source_path": str(source), "split": "train"},
            output_root=output,
            patch_config={"patch_size": 8, "stride": 8, "level": 0, "sampling_mode": "dense"},
            quality_config={"tissue_threshold": 0.0, "blur_threshold": 0.0, "brightness_min": 0, "brightness_max": 255},
        )
    kept = [row for row in rows if row["keep"]]
    assert summary["evaluated_patches"] > 0
    assert kept
    assert all(row["verification_status"] == "VERIFIED" for row in kept)
    assert all(row["extraction_profile_id"] and row["patch_provenance_id"] for row in kept)
    assert all(Path(output / row["patch_path"]).exists() for row in kept)


def test_real_pipeline_regression_writes_verified_patch_manifest(tmp_path: Path) -> None:
    wsi_dir = tmp_path / "wsi"
    wsi_dir.mkdir()
    source = wsi_dir / "CASE1_slide.png"
    _write_source(source)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame([{"case": "CASE1", "label": "A", "filename": source.name}]).to_csv(metadata, index=False)
    output = tmp_path / "pipeline-output"
    summary = run_pipeline(
        {
            "input": {"wsi_dir": str(wsi_dir), "metadata_file": str(metadata), "output_dir": str(output)},
            "metadata": {"case_id_column": "case", "label_column": "label", "filename_column": "filename"},
            "patch": {"patch_size": 8, "stride": 8, "level": 0, "sampling_mode": "dense"},
            "quality": {"tissue_threshold": 0.0, "blur_threshold": 0.0, "brightness_min": 0, "brightness_max": 255},
            "coordinates": {"enable": False},
            "features": {"enable": False},
            "visualization": {"save_overlay": False},
            "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 1},
            "runtime": {"cpu_workers": 1, "gpu_workers": 1},
        }
    )
    manifest = pd.read_csv(summary["patches"]["manifest_csv"])
    kept = manifest[manifest["keep"]]
    assert not kept.empty
    assert kept["verification_status"].eq("VERIFIED").all()
    assert kept["extraction_profile"].str.contains("strict-replay.v1", regex=False).all()


def test_verified_delivery_copies_only_verified_assets_and_blocks_failures(tmp_path: Path) -> None:
    source_root = tmp_path / "processed"
    source = tmp_path / "source.png"
    _write_source(source)
    with WSIReader(source) as reader:
        rows, _ = extract_patches_for_slide(
            reader,
            {"case_id": "C1", "slide_id": "S1", "source_path": str(source), "split": "train"},
            source_root,
            {"patch_size": 8, "stride": 32, "level": 0, "sampling_mode": "dense"},
            {"tissue_threshold": 0.0, "blur_threshold": 0.0, "brightness_min": 0, "brightness_max": 255},
        )
    manifest_dir = source_root / "metadata"
    manifest_dir.mkdir()
    kept_rows = [row for row in rows if row["keep"]]
    assert len(kept_rows) == 1
    pd.DataFrame(kept_rows).to_csv(manifest_dir / "patch_manifest.csv", index=False)
    destination = tmp_path / "delivery"
    assert build_verified_delivery(source_root, destination) == 1
    assert (destination / kept_rows[0]["patch_path"]).exists()
    assert pd.read_csv(destination / "metadata" / "patch_manifest.csv")["verification_status"].eq("VERIFIED").all()
    assert "source_wsi_path" not in pd.read_csv(destination / "metadata" / "patch_manifest.csv").columns

    failed_root = source_root
    internal = pd.read_csv(failed_root / "metadata" / "patch_manifest.csv").fillna("")
    failed_patch = failed_root / str(internal.iloc[0]["patch_path"])
    Image.new("RGB", (8, 8), "red").save(failed_patch)
    blocked = tmp_path / "blocked-delivery"
    try:
        build_verified_delivery(failed_root, blocked)
    except ValueError as exc:
        assert "Delivery blocked" in str(exc)
    else:
        raise AssertionError("non-VERIFIED delivery was created")
    assert not blocked.exists()
    refreshed = pd.read_csv(failed_root / "metadata" / "patch_manifest.csv")
    assert refreshed.iloc[0]["verification_status"] == "PATCH_REPLAY_MISMATCH"


def test_delivery_re_resolves_moved_source_by_content_and_rejects_ambiguity(tmp_path: Path) -> None:
    source_root = tmp_path / "processed"
    original = tmp_path / "original.png"
    _write_source(original)
    with WSIReader(original) as reader:
        rows, _ = extract_patches_for_slide(
            reader,
            {"case_id": "C1", "slide_id": "S1", "source_path": str(original), "split": "train"},
            source_root,
            {"patch_size": 8, "stride": 32, "level": 0, "sampling_mode": "dense"},
            {"tissue_threshold": 0.0, "blur_threshold": 0.0, "brightness_min": 0, "brightness_max": 255},
        )
    kept = [row for row in rows if row["keep"]]
    (source_root / "metadata").mkdir()
    pd.DataFrame(kept).to_csv(source_root / "metadata" / "patch_manifest.csv", index=False)

    moved_root = tmp_path / "moved"
    moved_root.mkdir()
    moved = moved_root / "renamed.png"
    moved.write_bytes(original.read_bytes())
    original.unlink()
    assert build_verified_delivery(
        source_root,
        tmp_path / "moved-delivery",
        source_search_roots=[moved_root],
    ) == 1

    duplicate = moved_root / "duplicate.png"
    duplicate.write_bytes(moved.read_bytes())
    try:
        build_verified_delivery(
            source_root,
            tmp_path / "ambiguous-delivery",
            source_search_roots=[moved_root],
        )
    except ValueError as exc:
        assert "AMBIGUOUS_SOURCE" in str(exc)
    else:
        raise AssertionError("multiple content-identical WSI candidates were silently selected")
    assert not (tmp_path / "ambiguous-delivery").exists()


def test_bound_annotation_relation_is_recomputed_at_delivery_without_labels(tmp_path: Path) -> None:
    source_root = tmp_path / "processed"
    source = tmp_path / "source.png"
    _write_source(source)
    with WSIReader(source) as reader:
        rows, _ = extract_patches_for_slide(
            reader,
            {"case_id": "C1", "slide_id": "S1", "source_path": str(source), "split": "train"},
            source_root,
            {"patch_size": 8, "stride": 32, "level": 0, "sampling_mode": "dense"},
            {"tissue_threshold": 0.0, "blur_threshold": 0.0, "brightness_min": 0, "brightness_max": 255},
        )
    manifest_path = source_root / "metadata" / "patch_manifest.csv"
    manifest_path.parent.mkdir()
    kept = [row for row in rows if row["keep"]]
    pd.DataFrame(kept).to_csv(manifest_path, index=False)
    annotation = tmp_path / "annotation.json"
    payload = {
        "source_wsi": {"content_sha256": kept[0]["wsi_content_sha256"]},
        "annotations": [{"label": "positive-must-not-propagate", "points_level0": [[1, 1], [4, 4]]}],
    }
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    assert bind_annotation_relations(manifest_path, annotation) == 1
    bound = pd.read_csv(manifest_path)
    assert bound.iloc[0]["verification_status"] == "CANDIDATE"
    assert bound.iloc[0]["declared_spatial_relation"] == "INTERSECTS"
    assert "training_label" not in bound.columns

    assert build_verified_delivery(source_root, tmp_path / "annotation-delivery") == 1
    public_row = pd.read_csv(tmp_path / "annotation-delivery" / "metadata" / "patch_manifest.csv").iloc[0]
    assert public_row["annotation_path"] == "reports/annotations/annotation.json"
    assert (tmp_path / "annotation-delivery" / public_row["annotation_path"]).exists()
    payload["annotations"][0]["points_level0"] = [[2, 2], [5, 5]]
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    try:
        build_verified_delivery(source_root, tmp_path / "changed-geometry-delivery")
    except ValueError as exc:
        assert "GEOMETRY_MISMATCH" in str(exc)
    else:
        raise AssertionError("changed annotation geometry was not detected")
    assert not (tmp_path / "changed-geometry-delivery").exists()
