"""Optional photographic inputs remain bound to immutable source and mask bytes."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageProxy, ProxyTransform
from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v4.companions import (
    calibrated_depth,
    freeze_companions,
    load_materials,
    validate_record,
)

pytestmark = pytest.mark.unit


def _calibration():
    return {
        "width": 60,
        "height": 40,
        "fx": 1200.0,
        "fy": 1000.0,
        "cx": 29.5,
        "cy": 19.5,
        "source": "camera calibration chart 2026-09-19",
        "coordinate_space": "canonical_master",
    }


def _source_record():
    return {"path": "photo.tif", "source_sha256": "a" * 64, "calibration": _calibration()}


def _inputs():
    return [{"id": "input-0000", "path": "photo.tif", "sha256": "a" * 64, "size_bytes": 100}]


def _write_manifest(tmp_path, record):
    manifest = tmp_path / "companions.json"
    manifest.write_text(dumps_json({"schema": "tp.lux.companions.v1", "inputs": [record]}, allow_nan=False))
    return manifest


def _material_record(tmp_path, array=None):
    if array is None:
        array = np.full((40, 60), 0.75, np.float32)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    data = stream.getvalue()
    (tmp_path / "wood.npy").write_bytes(data)
    record = _source_record()
    record["materials"] = {
        "masks": {"wood": {"path": "wood.npy", "sha256": hashlib.sha256(data).hexdigest()}},
        "confidences": {"wood": 0.9},
        "coordinate_space": "canonical_master",
    }
    return record


def _freeze(tmp_path, record, **kwargs):
    return freeze_companions(
        _write_manifest(tmp_path, record),
        _inputs(),
        max_input_bytes=kwargs.get("max_input_bytes", 100_000),
        max_pixels=kwargs.get("max_pixels", 10_000),
    )


def _proxy():
    transform = ProxyTransform((40, 60), (20, 30), (28, 42))
    return ImageProxy(np.zeros((28, 42, 3), np.uint8), transform, "b" * 64)


def test_freeze_and_load_bind_complete_masks_and_source(tmp_path):
    records, root, receipt = _freeze(tmp_path, _material_record(tmp_path))
    record = records["photo.tif"]
    assert receipt == {
        "path": "companions.json",
        "sha256": hashlib.sha256((tmp_path / "companions.json").read_bytes()).hexdigest(),
        "size_bytes": (tmp_path / "companions.json").stat().st_size,
    }
    validate_record(record)
    assert record["materials"]["masks"]["wood"] == {
        "path": "wood.npy",
        "sha256": hashlib.sha256((tmp_path / "wood.npy").read_bytes()).hexdigest(),
        "size_bytes": (tmp_path / "wood.npy").stat().st_size,
        "shape": [40, 60],
        "dtype": "float32",
    }
    masks, confidences = load_materials(root, record, max_input_bytes=100_000, max_pixels=10_000)
    np.testing.assert_array_equal(masks["wood"], np.full((40, 60), 0.75, np.float32))
    assert confidences == {"wood": 0.9}
    with pytest.raises(ValueError):
        masks["wood"].setflags(write=True)


def test_missing_materials_abstains_without_filesystem_access(tmp_path):
    for record in (None, _source_record()):
        assert load_materials(tmp_path / "missing", record, max_input_bytes=1000, max_pixels=1000) == ({}, {})


def test_missing_per_material_confidence_is_retained_for_abstention(tmp_path):
    record = _material_record(tmp_path)
    record["materials"]["confidences"] = {}
    records, root, receipt = _freeze(tmp_path, record)
    masks, confidence = load_materials(root, records["photo.tif"], max_input_bytes=100_000, max_pixels=10_000)
    assert set(masks) == {"wood"}
    assert confidence == {}


@pytest.mark.parametrize("field,value", [("path", "unknown.tif"), ("source_sha256", "c" * 64)])
def test_companions_reject_unknown_or_changed_source(tmp_path, field, value):
    record = _source_record()
    record[field] = value
    with pytest.raises(ValueError, match="source binding"):
        _freeze(tmp_path, record)


@pytest.mark.parametrize("path", ["../wood.npy", "/wood.npy", "a/../wood.npy", "a//wood.npy", "a\\wood.npy", "a:wood.npy"])
def test_mask_paths_cannot_escape_or_alias_namespace(tmp_path, path):
    record = _material_record(tmp_path)
    record["materials"]["masks"]["wood"]["path"] = path
    with pytest.raises(ValueError, match="portable"):
        _freeze(tmp_path, record)


def test_symlinked_mask_rejected(tmp_path):
    record = _material_record(tmp_path)
    (tmp_path / "real.npy").write_bytes((tmp_path / "wood.npy").read_bytes())
    (tmp_path / "wood.npy").unlink()
    (tmp_path / "wood.npy").symlink_to(tmp_path / "real.npy")
    with pytest.raises((ArtifactEvidenceError, ValueError, OSError)):
        _freeze(tmp_path, record)


def test_tampering_after_prepare_never_reaches_material_operations(tmp_path):
    records, root, receipt = _freeze(tmp_path, _material_record(tmp_path))
    path = tmp_path / "wood.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="changed after preparation"):
        load_materials(root, records["photo.tif"], max_input_bytes=100_000, max_pixels=10_000)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((40, 60), np.float64),
        np.zeros((40, 60, 1), np.float32),
        np.full((40, 60), np.nan, np.float32),
        np.full((40, 60), 1.1, np.float32),
        np.asfortranarray(np.zeros((40, 60), np.float32)),
    ],
)
def test_masks_reject_unsupported_dtype_shape_values_and_layout(tmp_path, array):
    with pytest.raises(ValueError, match="Material masks"):
        _freeze(tmp_path, _material_record(tmp_path, array))


def test_mask_header_bomb_rejected_without_numpy_load(tmp_path, monkeypatch):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": False, "shape": (10**9, 10**9)})
    data = stream.getvalue()
    (tmp_path / "bomb.npy").write_bytes(data)
    record = {
        "path": "photo.tif",
        "source_sha256": "a" * 64,
        "materials": {
            "masks": {"wood": {"path": "bomb.npy", "sha256": hashlib.sha256(data).hexdigest()}},
            "confidences": {},
            "coordinate_space": "canonical_master",
        },
    }
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Unsafe np.load must not be called"))
    with pytest.raises(ValueError, match="bounded"):
        _freeze(tmp_path, record)


def test_combined_mask_bytes_bound_applies_at_prepare_and_reload(tmp_path):
    record = _material_record(tmp_path)
    data = (tmp_path / "wood.npy").read_bytes()
    (tmp_path / "stone.npy").write_bytes(data)
    record["materials"]["masks"]["stone"] = {"path": "stone.npy", "sha256": hashlib.sha256(data).hexdigest()}
    with pytest.raises(ValueError, match="Combined material masks"):
        _freeze(tmp_path, record, max_input_bytes=len(data) + 1)
    records, root, receipt = _freeze(tmp_path, record)
    with pytest.raises(ValueError, match="Combined material masks"):
        load_materials(root, records["photo.tif"], max_input_bytes=len(data) + 1, max_pixels=10_000)


def test_conflicting_mask_calibration_geometry_fails(tmp_path):
    with pytest.raises(ValueError, match="same canonical master geometry"):
        _freeze(tmp_path, _material_record(tmp_path, np.zeros((20, 30), np.float32)))


def test_invalid_frozen_geometry_or_unknown_authorizing_field_fails(tmp_path):
    records, _, _ = _freeze(tmp_path, _material_record(tmp_path))
    record = records["photo.tif"]
    record["materials"]["masks"]["wood"]["untrusted"] = True
    with pytest.raises(ValueError, match="unsupported members"):
        validate_record(record)


@pytest.mark.parametrize(
    "field,value", [("coordinate_space", "source"), ("fx", 0), ("fy", True), ("source", "estimated"), ("cx", -1), ("cy", 40)]
)
def test_invalid_calibration_rejected_during_preparation(tmp_path, field, value):
    record = _source_record()
    record["calibration"][field] = value
    with pytest.raises(ValueError):
        _freeze(tmp_path, record)


def test_duplicate_json_members_rejected(tmp_path):
    path = tmp_path / "companions.json"
    path.write_text('{"schema":"tp.lux.companions.v1","schema":"tp.lux.companions.v1","inputs":[]}')
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_companions(path, _inputs(), 100_000, 10_000)


def test_calibrated_depth_uses_effective_intrinsics_and_retains_native_values():
    native = np.linspace(1, 3, 28 * 42, dtype=np.float32).reshape(28, 42)
    result = calibrated_depth(native, _proxy(), _source_record(), "a" * 64)
    np.testing.assert_array_equal(result.native_depth, native)
    np.testing.assert_allclose(result.metric_map_m, native * (550 / 300))
    assert result.intrinsics.to_payload() == {
        "fx": 600.0,
        "fy": 500.0,
        "cx": 14.5,
        "cy": 9.5,
        "width": 42,
        "height": 28,
        "source": _calibration()["source"],
    }
    assert result.native_semantics == "da3_metric_uncalibrated"
    assert result.confidence is None
    assert result.calibration["method"] == "da3_metric_focal_pixels_div_300"
    uncalibrated = calibrated_depth(native, _proxy(), None, "a" * 64)
    assert uncalibrated.metric_map_m is None
    assert uncalibrated.intrinsics is None
    assert uncalibrated.calibration is None
    assert result.content_hash() != uncalibrated.content_hash()


def test_calibration_invalidates_nonpositive_native_samples_without_overwriting_them():
    native = np.ones((28, 42), np.float32)
    native[0, :2] = [0, -1]
    result = calibrated_depth(native, _proxy(), _source_record(), "a" * 64)
    np.testing.assert_array_equal(result.native_depth, native)
    assert not result.valid_mask[0, :2].any()
    assert result.valid_mask[1:].all()


def test_calibration_cannot_silently_use_different_image_geometry_or_source():
    record = _source_record()
    record["calibration"]["width"] = 61
    native = np.ones((28, 42), np.float32)
    with pytest.raises(ValueError, match="canonical photographic master"):
        calibrated_depth(native, _proxy(), record, "a" * 64)
    with pytest.raises(ValueError, match="source digest"):
        calibrated_depth(native, _proxy(), _source_record(), "c" * 64)
    with pytest.raises(ValueError, match="proxy grid"):
        calibrated_depth(np.ones((20, 30), np.float32), _proxy(), _source_record(), "a" * 64)


@pytest.mark.parametrize("coordinate_space", [None, "source", "inference_proxy"])
def test_material_masks_require_explicit_canonical_coordinates(tmp_path, coordinate_space):
    record = _material_record(tmp_path)
    if coordinate_space is None:
        record["materials"].pop("coordinate_space")
    else:
        record["materials"]["coordinate_space"] = coordinate_space
    with pytest.raises(ValueError):
        _freeze(tmp_path, record)
