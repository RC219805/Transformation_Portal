"""Material bundles reject stale sources, substituted files and hostile headers."""

import hashlib
import io
import json
import os
import struct

import numpy as np
import pytest

from transformation_portal.materials_v4.artifacts import load_evidence, write_evidence
from transformation_portal.materials_v4.contracts import MaterialEvidence, MaterialLimits, MaterialsError, RegionEvidence

pytestmark = pytest.mark.unit


def _bundle(tmp_path):
    evidence = MaterialEvidence(
        source_sha256="a" * 64,
        shape=(4, 5),
        regions=(RegionEvidence("water-1", "water", np.full((4, 5), 0.75, np.float32), 0.9),),
        producer={"backend": "supplied", "revision": "frozen"},
    )
    path = tmp_path / "evidence.json"
    write_evidence(evidence, path)
    return evidence, path


def _load(path, **kwargs):
    return load_evidence(path, "a" * 64, (4, 5), **kwargs)


def _mask(path):
    record = json.loads(path.read_text())
    return record, path.parent / record["regions"][0]["mask"]["path"]


def _replace_mask(path, data):
    record, mask = _mask(path)
    mask.write_bytes(data)
    record["regions"][0]["mask"].update(file_sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    path.write_text(json.dumps(record))


def test_roundtrip_portable_identity_and_readonly_pixels(tmp_path):
    evidence, path = _bundle(tmp_path)
    loaded = _load(path, expected_content_hash=evidence.content_hash())
    assert loaded.content_hash() == evidence.content_hash()
    np.testing.assert_array_equal(loaded.regions[0].mask, evidence.regions[0].mask)
    with pytest.raises(ValueError):
        loaded.regions[0].mask.setflags(write=True)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    write_evidence(loaded, relocated / "manifest.json")
    assert _load(relocated / "manifest.json").content_hash() == evidence.content_hash()


def test_stale_source_geometry_and_frozen_identity_fail(tmp_path):
    _, path = _bundle(tmp_path)
    with pytest.raises(MaterialsError, match="source or geometry"):
        load_evidence(path, "b" * 64, (4, 5))
    with pytest.raises(MaterialsError, match="source or geometry"):
        load_evidence(path, "a" * 64, (5, 4))
    with pytest.raises(MaterialsError, match="frozen execution"):
        _load(path, expected_content_hash="b" * 64)


def test_swapped_numeric_file_and_semantic_metadata_are_detected(tmp_path):
    _, path = _bundle(tmp_path)
    record, mask = _mask(path)
    original = mask.read_bytes()
    mask.write_bytes(original[:-4] + b"\x00\x00\x00\x00")
    with pytest.raises(MaterialsError, match="digest"):
        _load(path)
    mask.write_bytes(original)
    record["regions"][0]["semantic_confidence"] = 1.0
    path.write_text(json.dumps(record))
    with pytest.raises(MaterialsError, match="semantic content"):
        _load(path)


def test_all_mask_descriptors_are_preflighted_before_arrays_are_loaded(tmp_path, monkeypatch):
    _, path = _bundle(tmp_path)
    from transformation_portal.materials_v4 import artifacts

    monkeypatch.setattr(artifacts, "_mask_array", lambda *args: pytest.fail("Numeric allocation attempted before preflight"))
    with pytest.raises(MaterialsError, match="aggregate"):
        _load(path, limits=MaterialLimits(max_mask_bytes=79))
    with pytest.raises(MaterialsError, match="encoded bundle"):
        _load(path, limits=MaterialLimits(max_bundle_bytes=80))


def test_calibration_receipt_survives_bundle_without_becoming_trusted(tmp_path):
    from transformation_portal.materials_v4.contracts import CalibrationReceipt

    identities = {
        name: "b" * 64
        for name in (
            "classifier_sha256",
            "proposal_sha256",
            "prompt_sha256",
            "preprocessing_sha256",
            "region_construction_sha256",
            "split_sha256",
            "artifact_sha256",
        )
    }
    receipt = CalibrationReceipt(**identities, method="held_out", classes=("water",))
    evidence = MaterialEvidence(
        "a" * 64,
        (4, 5),
        (
            RegionEvidence(
                "inferred",
                "water",
                np.ones((4, 5), np.float32),
                0.95,
                provenance="inferred",
                calibration_sha256=receipt.content_hash(),
            ),
        ),
        calibration=receipt,
    )
    path = tmp_path / "calibrated.json"
    write_evidence(evidence, path)
    loaded = _load(path)
    assert loaded.calibration.content_hash() == receipt.content_hash()
    assert loaded.regions[0].provenance == "inferred"
    assert "trusted" not in loaded.to_payload()["calibration"]


@pytest.mark.parametrize("replacement", ["../outside.npy", "/tmp/outside.npy", "a/../mask.npy", "a\\mask.npy", "./mask.npy"])
def test_noncanonical_and_escaping_paths_fail(tmp_path, replacement):
    _, path = _bundle(tmp_path)
    record, _ = _mask(path)
    record["regions"][0]["mask"]["path"] = replacement
    path.write_text(json.dumps(record))
    with pytest.raises(MaterialsError):
        _load(path)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_linked_mask_aliases_are_rejected(tmp_path, kind):
    _, path = _bundle(tmp_path)
    _, mask = _mask(path)
    real = tmp_path / "other.npy"
    mask.rename(real)
    if kind == "symlink":
        mask.symlink_to(real)
    else:
        os.link(real, mask)
    with pytest.raises(MaterialsError):
        _load(path)


def test_symlinked_manifest_and_parent_are_rejected(tmp_path):
    _, path = _bundle(tmp_path)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(MaterialsError):
        _load(link)
    parent = tmp_path / "linked-dir"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(MaterialsError):
        _load(parent / "evidence.json")


@pytest.mark.parametrize(
    "array",
    [
        np.ones((4, 5), object),
        np.ones((5, 4), np.float32),
        np.ones((4, 5), np.float64),
        np.full((4, 5), np.nan, np.float32),
        np.asfortranarray(np.ones((4, 5), np.float32)),
    ],
)
def test_unsafe_or_wrong_npy_headers_and_samples_fail_without_pickle_load(tmp_path, array):
    _, path = _bundle(tmp_path)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=True)
    _replace_mask(path, stream.getvalue())
    with pytest.raises(MaterialsError):
        _load(path)


def test_huge_npy_shape_and_header_fail_before_array_construction(tmp_path, monkeypatch):
    _, path = _bundle(tmp_path)
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (999999999, 999999999), }\n"
    data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\x00" * 80
    _replace_mask(path, data)
    from transformation_portal.materials_v4 import artifacts

    monkeypatch.setattr(
        artifacts.np, "frombuffer", lambda *args, **kwargs: pytest.fail("Hostile header reached array constructor")
    )
    with pytest.raises(MaterialsError, match="geometry"):
        _load(path)
    _, mask = _mask(path)
    data = bytearray(mask.read_bytes())
    data[8:10] = struct.pack("<H", 65535)
    _replace_mask(path, bytes(data))
    with pytest.raises(MaterialsError, match="header"):
        _load(path)


def test_duplicate_json_keys_and_unknown_schema_are_rejected(tmp_path):
    _, path = _bundle(tmp_path)
    original = path.read_text()
    path.write_text(original[:-1] + ',"schema":"tp.materials.evidence.v1"}')
    with pytest.raises(MaterialsError):
        _load(path)
    record = json.loads(original)
    record["schema"] = "tp.materials.evidence.v999"
    path.write_text(json.dumps(record))
    with pytest.raises(MaterialsError, match="Unsupported"):
        _load(path)


def test_publication_does_not_follow_preexisting_numeric_symlink(tmp_path):
    evidence, path = _bundle(tmp_path)
    _, mask = _mask(path)
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    mask.unlink()
    mask.symlink_to(victim)
    with pytest.raises(MaterialsError):
        write_evidence(evidence, path)
    assert victim.read_bytes() == b"keep"
