"""Native mask identity preserves historical bytes without payload-time copies."""

import hashlib
import struct

import numpy as np
import pytest

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.materials_v4 import contracts
from transformation_portal.materials_v4.contracts import MaterialEvidence, MaterialsError, RegionEvidence
from transformation_portal.materials_v4.taxonomy import TAXONOMY_VERSION

pytestmark = pytest.mark.unit


def legacy_mask_digest(source):
    """Independent oracle for the historical little-endian C-order float bytes."""
    samples = [float(value) for value in source.flat]
    return hashlib.sha256(struct.pack("<" + "f" * len(samples), *samples)).hexdigest()


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed", "boolean", "little_endian"])
def test_region_digest_matches_legacy_bytes_after_constructor_freezes_layout(layout):
    source = np.arange(48, dtype=np.float32).reshape(6, 8) / np.float32(47)
    if layout == "fortran":
        source = np.asfortranarray(source)
    elif layout == "strided":
        source = source[::2, 1::2]
    elif layout == "reversed":
        source = source[::-1, ::-1]
    elif layout == "boolean":
        source = source > np.float32(0.5)
    elif layout == "little_endian":
        source = source.astype("<f4")
    expected = legacy_mask_digest(source)
    item = RegionEvidence("native-region", "water", source)

    assert item.mask.dtype == np.dtype("<f4")
    assert item.mask.flags.c_contiguous
    assert not item.mask.flags.writeable
    assert item.to_payload()["mask"] == {"shape": list(source.shape), "dtype": "float32", "sha256": expected}
    source[...] = 0
    assert item.to_payload()["mask"]["sha256"] == expected
    with pytest.raises(ValueError):
        item.mask.setflags(write=True)


def test_non_native_endian_mask_remains_rejected_before_freezing():
    source = np.ones((3, 4), dtype=np.dtype(np.float32).newbyteorder("S"))
    with pytest.raises(MaterialsError, match="bool or float32"):
        RegionEvidence("native-region", "water", source)


def test_region_payload_hash_uses_the_frozen_mask_buffer_without_tobytes(monkeypatch):
    class NoPayloadCopy(np.ndarray):
        def tobytes(self, *_args, **_kwargs):
            raise AssertionError("Payload hashing must not allocate a native-mask bytes copy")

    item = RegionEvidence("native-region", "water", np.full((4, 5), 0.75, np.float32))
    expected = legacy_mask_digest(item.mask)
    # Preserve every frozen-array invariant while exposing accidental tobytes.
    guarded = item.mask.view(NoPayloadCopy)
    object.__setattr__(item, "mask", guarded)
    assert guarded.flags.c_contiguous and not guarded.flags.writeable
    original_sha256 = hashlib.sha256
    observed = []

    def direct_buffer_hash(value):
        assert isinstance(value, memoryview)
        assert value.obj is guarded and value.format == "B" and value.nbytes == guarded.nbytes
        observed.append(value.nbytes)
        return original_sha256(value)

    monkeypatch.setattr(contracts.hashlib, "sha256", direct_buffer_hash)
    assert item.to_payload()["mask"]["sha256"] == expected
    assert observed == [guarded.nbytes]


def test_native_float32_mask_precision_preserves_complete_evidence_identity():
    mask = np.array([[0, -0.0], [np.nextafter(np.float32(0.5), np.float32(1)), 1]], dtype=np.float32)
    item = MaterialEvidence("b" * 64, mask.shape, (RegionEvidence("native-region", "water", mask),))
    expected_region = {
        "region_id": "native-region",
        "label": "water",
        "mask": {"shape": [2, 2], "dtype": "float32", "sha256": legacy_mask_digest(mask)},
        "semantic_confidence": None,
        "geometric_quality": None,
        "provenance": "supplied",
        "score_type": None,
        "calibration_sha256": None,
    }
    expected = {
        "schema": "tp.materials.evidence.v1",
        "taxonomy": TAXONOMY_VERSION,
        "coordinate_space": "canonical_master",
        "source_sha256": "b" * 64,
        "shape": [2, 2],
        "regions": [expected_region],
        "producer": {},
        "calibration": None,
        "status": "available",
        "reason": None,
    }
    assert item.to_payload() == expected
    assert item.content_hash() == hashlib.sha256(canonicalize_json(expected)).hexdigest()
