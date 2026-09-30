"""Preparation receipts cannot silently change the requested color interpretation."""

from __future__ import annotations

import copy
import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from transformation_portal.core.image_artifact import metadata_payload
from transformation_portal.lux_depth_v4.color_preparation import prepare_input_color
from transformation_portal.lux_depth_v4.color_preparation_evidence import (
    retained_source_icc,
    validate_color_preparation_evidence,
)
from transformation_portal.lux_depth_v4.photography import decode_master, output_srgb_icc

pytestmark = pytest.mark.unit


def _master(*, mode="auto", profile=None):
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), (128, 64, 32)).save(stream, format="PNG", icc_profile=profile)
    return decode_master(stream.getvalue(), source_name="photo.png", input_color=mode)


@pytest.mark.parametrize(
    ("mode", "profile"),
    [
        (mode, profile)
        for mode in ("auto", "auto_assume_srgb", "srgb", "linear_srgb")
        for profile in (None, output_srgb_icc())
        if mode != "auto" or profile is not None
    ],
)
def test_receipt_replays_exact_preparation(mode, profile):
    master = _master(mode=mode, profile=profile)
    validate_color_preparation_evidence(master.metadata, input_color=mode, source_icc=profile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "tp.color.preparation.v2"),
        ("requested_input_color", "srgb"),
        ("assumed_srgb", True),
        ("assumed_srgb", 0),
        ("action", "assume"),
        ("source_icc_sha256", "a" * 64),
        ("target_icc_sha256", "b" * 64),
        ("output_precision", "uint8"),
        ("engine_version", "different"),
        ("unrecognized", "injected"),
    ],
)
def test_rehashed_receipt_mutation_fails(field, value):
    profile = output_srgb_icc()
    metadata = metadata_payload(_master(profile=profile).metadata)
    metadata["color_preparation"][field] = value
    with pytest.raises(ValueError, match="Color preparation"):
        validate_color_preparation_evidence(metadata, input_color="auto", source_icc=profile)


def test_metadata_conflict_and_retained_profile_replacement_fail():
    profile = output_srgb_icc()
    metadata = metadata_payload(_master(profile=profile).metadata)
    conflicting = copy.deepcopy(metadata)
    conflicting["color_preparation"]["source_metadata"]["declared_color"] = "linear_srgb"
    with pytest.raises(ValueError, match="disagree"):
        validate_color_preparation_evidence(conflicting, input_color="auto", source_icc=profile)
    with pytest.raises(ValueError, match="retained source ICC"):
        validate_color_preparation_evidence(metadata, input_color="auto", source_icc=None)
    metadata["source_format"] = "JPEG"
    with pytest.raises(ValueError, match="master"):
        validate_color_preparation_evidence(metadata, input_color="auto", source_icc=profile)


def test_assumption_receipt_is_required_but_historical_modes_remain_valid():
    for mode in ("auto", "srgb", "linear_srgb"):
        validate_color_preparation_evidence({}, input_color=mode, source_icc=None)
    with pytest.raises(ValueError, match="requires color preparation evidence"):
        validate_color_preparation_evidence({}, input_color="auto_assume_srgb", source_icc=None)


@pytest.mark.parametrize("mode", ["auto", "auto_assume_srgb", "srgb", "linear_srgb"])
def test_present_null_receipt_is_not_historical_metadata(mode):
    with pytest.raises(ValueError, match="bounded object contract"):
        validate_color_preparation_evidence({"color_preparation": None}, input_color=mode, source_icc=None)


@pytest.mark.parametrize("mode", ["auto", "srgb"])
@pytest.mark.parametrize("field", ["declared_color", "exif_color"])
@pytest.mark.parametrize("value", [[], {}, True, 1.0, "unexpected"])
def test_receipt_source_field_types_fail_closed(mode, field, value):
    profile = output_srgb_icc()
    metadata = metadata_payload(_master(mode=mode, profile=profile).metadata)
    metadata["color_preparation"]["source_metadata"][field] = value
    with pytest.raises(ValueError, match="invalid source metadata"):
        validate_color_preparation_evidence(metadata, input_color=mode, source_icc=profile)


@pytest.mark.parametrize("mode", ["srgb", "linear_srgb"])
def test_explicit_override_receipt_preserves_normalized_unsupported_metadata(mode):
    prepared = prepare_input_color(
        input_color=mode, image_format="TIFF", profile=None, declared_color="unknown", exif_color="unknown"
    )
    metadata = {
        "source_format": "TIFF",
        "input_color": prepared.source_color,
        "color_resolution": prepared.resolution,
        "color_preparation": prepared.evidence,
    }
    assert prepared.evidence["source_metadata"]["declared_color"] == "unsupported"
    assert prepared.evidence["source_metadata"]["exif_color"] == "unsupported"
    validate_color_preparation_evidence(metadata, input_color=mode, source_icc=None)


def test_receipt_numeric_metadata_errors_are_validation_errors():
    profile = output_srgb_icc()
    metadata = metadata_payload(_master(profile=profile).metadata)
    metadata["color_preparation"]["source_metadata"]["png"] = {"gamma": 10**100}
    with pytest.raises(ValueError, match="invalid source metadata"):
        validate_color_preparation_evidence(metadata, input_color="auto", source_icc=profile)


def _inventory(tmp_path, array):
    path = tmp_path / "source-icc.npy"
    np.save(path, array, allow_pickle=False)
    raw = path.read_bytes()
    record = {"kind": "array", "path": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return path, {path.name: record}


def test_retained_source_profile_is_bounded_and_inventory_bound(tmp_path):
    profile = output_srgb_icc()
    path, declared = _inventory(tmp_path, np.frombuffer(profile, np.uint8))
    assert retained_source_icc(tmp_path, path.name, declared) == profile
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="changed"):
        retained_source_icc(tmp_path, path.name, declared)


@pytest.mark.parametrize("array", [np.zeros((2, 2), np.uint8), np.zeros(4, np.uint16), np.zeros(0, np.uint8)])
def test_retained_source_profile_rejects_invalid_array_shape_or_type(tmp_path, array):
    path, declared = _inventory(tmp_path, array)
    with pytest.raises(ValueError, match="bounded one-dimensional byte array"):
        retained_source_icc(tmp_path, path.name, declared)


@pytest.mark.parametrize("version", ["1_0", "2_0"])
def test_retained_source_profile_rejects_boolean_dimension(tmp_path, version):
    stream = io.BytesIO()
    writer = getattr(np.lib.format, f"write_array_header_{version}")
    writer(stream, {"descr": "|u1", "fortran_order": False, "shape": (True,)})
    raw = stream.getvalue() + b"x"
    path = tmp_path / "source-icc.npy"
    path.write_bytes(raw)
    record = {"kind": "array", "path": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="bounded one-dimensional byte array"):
        retained_source_icc(tmp_path, path.name, {path.name: record})
