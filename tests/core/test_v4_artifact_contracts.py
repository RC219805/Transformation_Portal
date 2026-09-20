"""Semantic, calibration and immutability tests for V4 core artifacts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.depth_artifact import CameraIntrinsics, DepthArtifact
from transformation_portal.core.image_artifact import ImageMaster

pytestmark = pytest.mark.unit


def _depth(**kwargs):
    base = {
        "native_depth": np.arange(1, 7, dtype=np.float32).reshape(2, 3),
        "native_semantics": "da3_metric_uncalibrated",
        "valid_mask": np.ones((2, 3), dtype=bool),
        "source_sha256": "a" * 64,
    }
    base.update(kwargs)
    return DepthArtifact(**base)


def _intrinsics():
    return CameraIntrinsics(300.0, 300.0, 1.5, 1.0, 3, 2, "calibrated_camera")


def test_image_arrays_and_metadata_are_deeply_immutable():
    pixels = np.full((2, 3, 3), 0.5, np.float32)
    alpha = np.ones((2, 3), np.float32)
    metadata = {"nested": {"choices": [1, 2]}}
    master = ImageMaster(pixels, "a" * 64, 16, alpha, metadata)
    initial_hash = master.content_hash()
    pixels[:] = 0
    alpha[:] = 0
    metadata["nested"]["choices"].append(3)
    assert master.content_hash() == initial_hash
    with pytest.raises(ValueError):
        master.pixels.setflags(write=True)
    with pytest.raises(TypeError):
        master.metadata["nested"]["new"] = "value"
    assert master.metadata["nested"]["choices"] == (1, 2)
    assert replace(master, alpha=np.zeros((2, 3), np.float32)).content_hash() != initial_hash
    assert replace(master, source_icc=b"source-profile").content_hash() != initial_hash


def test_image_master_rejects_nonfinite_pixels_and_bad_alpha():
    with pytest.raises(ValueError, match="finite"):
        ImageMaster(np.full((2, 2, 3), np.nan, np.float32), "a" * 64, 16)
    with pytest.raises(ValueError, match="alpha"):
        ImageMaster(np.zeros((2, 2, 3), np.float32), "a" * 64, 16, np.full((2, 2), 1.1, np.float32))


def test_uncalibrated_da3_remains_uncalibrated_and_native_values_survive():
    artifact = _depth()
    assert artifact.metric_map_m is None
    np.testing.assert_array_equal(artifact.native_depth, np.arange(1, 7, dtype=np.float32).reshape(2, 3))
    assert artifact.relative_depth().min() == 0
    assert artifact.relative_depth().max() == 1
    assert artifact.native_depth.max() == 6
    inverse = replace(artifact, native_semantics="relative_inverse_depth")
    np.testing.assert_allclose(inverse.relative_depth(), 1 - artifact.relative_depth(), atol=1e-7)


def test_metric_depth_requires_grid_intrinsics_and_calibration():
    metric = np.ones((2, 3), np.float32)
    with pytest.raises(ValueError, match="effective intrinsics"):
        _depth(metric_map_m=metric)
    with pytest.raises(ValueError, match="calibration"):
        _depth(metric_map_m=metric, intrinsics=_intrinsics())
    calibrated = _depth(metric_map_m=metric, intrinsics=_intrinsics(), calibration={"method": "da3_focal_px_over_300"})
    assert calibrated.to_payload()["has_metric_depth"]
    with pytest.raises(ValueError, match="depth grid"):
        _depth(intrinsics=replace(_intrinsics(), width=4))
    with pytest.raises(ValueError, match="provenance"):
        replace(_intrinsics(), source="estimated")


def test_depth_hash_binds_every_semantic_input_and_detaches_metadata():
    source = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    metadata = {"runtime": {"revision": "one"}}
    artifact = _depth(native_depth=source, metadata=metadata)
    baseline = artifact.content_hash()
    source[:] = 99
    metadata["runtime"]["revision"] = "two"
    assert artifact.content_hash() == baseline
    with pytest.raises(ValueError):
        artifact.native_depth.setflags(write=True)
    with pytest.raises(TypeError):
        artifact.metadata["runtime"]["revision"] = "two"
    assert replace(artifact, native_semantics="relative_distance").content_hash() != baseline
    assert replace(artifact, source_sha256="b" * 64).content_hash() != baseline
    assert replace(artifact, confidence=np.full((2, 3), 0.5, np.float32)).content_hash() != baseline
    mask = np.ones((2, 3), bool)
    mask[0, 0] = False
    assert replace(artifact, valid_mask=mask).content_hash() != baseline
    calibrated = replace(
        artifact, metric_map_m=np.ones((2, 3), np.float32), intrinsics=_intrinsics(), calibration={"method": "test"}
    )
    assert calibrated.content_hash() != baseline
    assert replace(calibrated, calibration={"method": "another"}).content_hash() != calibrated.content_hash()


def test_depth_validity_and_confidence_fail_closed():
    with pytest.raises(ValueError, match="valid samples"):
        _depth(valid_mask=np.zeros((2, 3), bool))
    with pytest.raises(ValueError, match="Confidence"):
        _depth(confidence=np.full((2, 3), 1.01, np.float32))
    with pytest.raises(ValueError, match="finite JSON"):
        _depth(metadata={"threshold": float("nan")})
    with pytest.raises(ValueError, match="Unsupported native"):
        _depth(native_semantics="meters_probably")
    with pytest.raises(ValueError, match="must agree"):
        _depth(
            native_semantics="metric_distance_m",
            metric_map_m=np.ones((2, 3), np.float32),
            intrinsics=_intrinsics(),
            calibration={"method": "native_metric"},
        )
