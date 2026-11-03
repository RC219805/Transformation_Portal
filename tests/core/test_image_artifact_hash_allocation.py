"""Exact historical artifact hashes without copying immutable native rasters."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageMaster, artifact_content_hash, immutable_array

pytestmark = pytest.mark.unit


def _legacy_hash(payload: dict[str, Any], arrays: dict[str, np.ndarray | None]) -> str:
    """Independent original C-order snapshot and descriptor serialization."""
    descriptors = {
        name: (
            None
            if value is None
            else {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
        )
        for name, value in arrays.items()
    }
    encoded = json.dumps(
        {"metadata": payload, "arrays": descriptors}, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("dtype", ["u1", "<u2", ">u2", "i2", "f2", "f4", "f8", "c8", "?", "M8[ns]", "V4"])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "negative", "transpose"])
@pytest.mark.parametrize("readonly", [False, True])
def test_hash_matches_legacy_bytes_for_dtypes_layouts_alpha_and_mutability(dtype, layout, readonly):
    kind = np.dtype(dtype)
    values = np.frombuffer(bytes(range(24 * kind.itemsize)), kind).reshape(4, 6).copy()
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[:, ::2]
    elif layout == "negative":
        values = values[::-1, ::-1]
    elif layout == "transpose":
        values = values.T
    if readonly and layout == "c":
        values = immutable_array(values, values.dtype)
    values.setflags(write=not readonly)
    alpha = immutable_array(np.arange(6).reshape(2, 3) / 5, np.float32)
    arrays = {"pixels": values, "alpha": alpha, "missing": None}
    payload = {"schema": "test.artifact.v1", "domain": "linear_srgb", "label": "café", "tags": ("native", "retained")}
    assert artifact_content_hash(payload, arrays) == _legacy_hash(payload, arrays)


@pytest.mark.parametrize("shape", [(), (0,), (0, 3), (3, 0), (1, 1)])
def test_empty_and_scalar_arrays_keep_original_hash_contract(shape):
    values = immutable_array(np.zeros(shape, np.float32), np.float32)
    assert artifact_content_hash({}, {"samples": values}) == _legacy_hash({}, {"samples": values})


def test_immutable_native_rgb_and_alpha_hash_without_tobytes():
    master = ImageMaster(
        np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 36,
        "a" * 64,
        16,
        np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4),
    )
    arrays = {"pixels": master.pixels, "alpha": master.alpha}
    assert all(value.flags.c_contiguous and not value.flags.writeable for value in arrays.values())
    expected = _legacy_hash(master.to_payload(), arrays)

    def forbid_native_snapshot(_frame, event, function):
        if event == "c_call" and getattr(function, "__name__", "") == "tobytes":
            if any(getattr(function, "__self__", None) is value for value in arrays.values()):
                pytest.fail("Immutable native RGB/alpha must not allocate a byte snapshot")

    previous = sys.getprofile()
    try:
        sys.setprofile(forbid_native_snapshot)
        actual = artifact_content_hash(master.to_payload(), arrays)
    finally:
        sys.setprofile(previous)
    assert actual == expected


@pytest.mark.parametrize(
    "layout", ["writable", "readonly_fortran", "readonly_strided", "readonly_owned", "readonly_alias", "readonly_bytearray"]
)
def test_writable_or_noncontiguous_arrays_keep_snapshot_fallback(layout):
    values = np.arange(24, dtype=np.uint16).reshape(4, 6)
    if layout == "readonly_fortran":
        values = np.asfortranarray(values)
    elif layout == "readonly_strided":
        values = values[:, ::2]
    elif layout == "readonly_owned":
        values = values.copy()
    elif layout == "readonly_bytearray":
        values = np.frombuffer(bytearray(values.tobytes()), values.dtype).reshape(values.shape)
    values.setflags(write=layout == "writable")
    if layout == "readonly_owned":
        assert values.flags.owndata and values.base is None
    elif layout == "readonly_alias":
        assert not values.flags.owndata and values.base.flags.writeable
    expected = _legacy_hash({}, {"pixels": values})
    calls = []

    def observe_snapshot(_frame, event, function):
        if (
            event == "c_call"
            and getattr(function, "__name__", "") == "tobytes"
            and getattr(function, "__self__", None) is values
        ):
            calls.append(True)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe_snapshot)
        actual = artifact_content_hash({}, {"pixels": values})
    finally:
        sys.setprofile(previous)
    assert actual == expected
    assert calls == [True]


def test_array_subclass_keeps_historical_tobytes_semantics():
    values = np.ma.MaskedArray(
        immutable_array(np.arange(6), np.int16), mask=[False, True, False, False, False, False], fill_value=17, copy=False
    )
    assert values.flags.c_contiguous and not values.flags.writeable
    assert artifact_content_hash({}, {"pixels": values}) == _legacy_hash({}, {"pixels": values})
    assert artifact_content_hash({}, {"pixels": values}) != _legacy_hash({}, {"pixels": np.asarray(values)})


def test_readonly_array_cannot_trust_subclass_owner_base_override():
    class MutableOwner(np.ndarray):
        @property
        def base(self):
            return b"misleading-immutable-owner"

    owner = np.ndarray.__new__(MutableOwner, (6,), np.uint16)
    owner[:] = np.arange(6)
    values = np.asarray(owner)
    values.setflags(write=False)
    # Exact type is necessary to reproduce the owner-chain bypass.
    assert type(values) is np.ndarray  # pylint: disable=unidiomatic-typecheck
    assert values.base is owner
    assert values.flags.c_contiguous and owner.flags.writeable
    expected = _legacy_hash({}, {"pixels": values})
    calls = []

    def observe_snapshot(_frame, event, function):
        if (
            event == "c_call"
            and getattr(function, "__name__", "") == "tobytes"
            and getattr(function, "__self__", None) is values
        ):
            calls.append(True)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe_snapshot)
        actual = artifact_content_hash({}, {"pixels": values})
    finally:
        sys.setprofile(previous)
    assert actual == expected
    assert calls == [True]


def test_shape_dtype_names_and_domain_remain_distinct_hash_authority():
    values = immutable_array(np.arange(6, dtype=np.uint16), np.uint16)
    baseline = artifact_content_hash({"domain": "linear_srgb"}, {"pixels": values})
    assert artifact_content_hash({"domain": "display_srgb"}, {"pixels": values}) != baseline
    assert artifact_content_hash({"domain": "linear_srgb"}, {"other": values}) != baseline
    assert artifact_content_hash({"domain": "linear_srgb"}, {"pixels": values.reshape(2, 3)}) != baseline
    assert artifact_content_hash({"domain": "linear_srgb"}, {"pixels": values.view(np.int16)}) != baseline
