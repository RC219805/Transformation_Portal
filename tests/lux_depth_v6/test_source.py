"""Bounded source admission and replayable retained V5 image inputs."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tests.lux_depth_v5 import test_evidence as v5_evidence
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v6 import source as source_module
from transformation_portal.lux_depth_v6.source import (
    SourceLimits,
    load_depth_inputs,
    load_master,
    prepare_source,
    validate_source,
)

pytestmark = pytest.mark.unit
completed = v5_evidence.completed


def test_admits_verified_v5_and_loads_exact_retained_images(completed):
    source = prepare_source(completed.result.output_root)
    assert source.plan_sha256 == hashlib.sha256(source.canonical_plan_bytes).hexdigest()
    assert source.evidence_sha256 == hashlib.sha256(source.canonical_evidence_bytes).hexdigest()
    assert len(source.images) == 1
    assert source.images[0].shape == (29, 43)
    master = load_master(source, "input-0000")
    original, depth, proxy = load_depth_inputs(source, "input-0000")
    np.testing.assert_array_equal(master.pixels, np.load(source.root / "input-0000/master.npy"))
    np.testing.assert_array_equal(original.pixels, np.load(source.root / "input-0000/source-master.npy"))
    assert depth.content_hash() == source.images[0].descriptor["depth_content_sha256"]
    assert depth.shape == proxy.transform.padded_shape
    assert not original.pixels.flags.writeable
    validate_source(source)


@pytest.mark.parametrize(
    "completed", [{"alpha": True, "icc": True}, {"calibration": True}, {"unknown_sky": True}], indirect=True
)
def test_loads_alpha_icc_calibration_and_unavailable_sky(completed):
    source = prepare_source(completed.result.output_root)
    original, depth, _ = load_depth_inputs(source, "input-0000")
    master = load_master(source, "input-0000")
    assert original.to_payload() == source.images[0].descriptor["source"]
    assert master.to_payload() == source.images[0].descriptor["master"]
    assert depth.to_payload() == source.images[0].descriptor["depth"]


@pytest.mark.parametrize(
    "field,value", [("max_input_bytes", True), ("max_pixels", 0), ("memory_mib", 1.5), ("max_pixels", 200_000_001)]
)
def test_rejects_invalid_source_limits(field, value):
    with pytest.raises(ValueError):
        SourceLimits(**{field: value})


@pytest.mark.parametrize(
    "limit", [SourceLimits(max_input_bytes=20000), SourceLimits(max_pixels=1), SourceLimits(memory_mib=1)]
)
def test_preflights_limits_before_semantic_array_verification(completed, monkeypatch, limit):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Source budget must reject before semantic verifier allocation")

    monkeypatch.setattr(source_module, "verify_execution_evidence_v3", unexpected)
    with pytest.raises(ValueError, match="budget"):
        prepare_source(completed.result.output_root, limits=limit)


@pytest.mark.parametrize(
    "name", ["execution-plan.json", "execution-evidence.json", "input-0000/master.npy", "input-0000/native-sky.npy"]
)
def test_validation_rejects_changed_retained_artifacts(completed, name):
    source = prepare_source(completed.result.output_root)
    path = source.root / name
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="changed"):
        validate_source(source)


def test_master_loader_rechecks_array_hash(completed):
    source = prepare_source(completed.result.output_root)
    path = source.root / "input-0000/master.npy"
    pixels = np.load(path)
    pixels[0, 0, 0] += 0.01
    np.save(path, pixels, allow_pickle=False)
    with pytest.raises(ValueError, match="changed"):
        load_master(source, "input-0000")


def test_descriptor_views_cannot_mutate_frozen_source(completed):
    source = prepare_source(completed.result.output_root)
    descriptor = source.images[0].descriptor
    descriptor["master"]["shape"][0] = 100
    assert source.images[0].descriptor["master"]["shape"] == [29, 43]
    with pytest.raises(ValueError, match="requested image"):
        load_master(source, "not-admitted")
    with pytest.raises(ValueError, match="digest"):
        validate_source(replace(source, source_digest="0" * 64))


def test_rejects_applied_materials_before_replay(completed, monkeypatch):
    root = completed.result.output_root
    relative = "input-0000/photograph.json"
    descriptor = completed.descriptor
    descriptor["materials"] = {"status": "applied", "changed_pixels": 1}
    raw = canonicalize_json(descriptor)
    (root / relative).write_bytes(raw)
    for artifact in completed.evidence["artifacts"]:
        if artifact["path"] == relative:
            artifact.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    (root / "execution-evidence.json").write_bytes(canonicalize_json(completed.evidence))
    monkeypatch.setattr(source_module, "verify_execution_evidence_v3", lambda **_kwargs: pytest.fail("must reject early"))
    with pytest.raises(ValueError, match="does not retain their masks"):
        prepare_source(root)


def test_cancellation_prevents_admission_and_validation(completed):
    with pytest.raises(RuntimeError, match="cancelled"):
        prepare_source(completed.result.output_root, cancellation=lambda: True)
    source = prepare_source(completed.result.output_root)
    with pytest.raises(RuntimeError, match="cancelled"):
        validate_source(source, cancellation=lambda: True)


def test_rejects_linked_source_roots_and_artifacts(completed, tmp_path):
    alias = tmp_path / "linked-source"
    alias.symlink_to(completed.result.output_root, target_is_directory=True)
    with pytest.raises((ValueError, OSError, ArtifactEvidenceError)):
        prepare_source(alias)
    source = prepare_source(completed.result.output_root)
    path = source.root / "input-0000/master.npy"
    retained = tmp_path / "master-retained.npy"
    path.rename(retained)
    path.symlink_to(retained)
    with pytest.raises((ValueError, OSError, ArtifactEvidenceError)):
        validate_source(source)


def test_rejects_hardlinked_retained_artifact(completed, tmp_path):
    source = prepare_source(completed.result.output_root)
    (tmp_path / "another-name.npy").hardlink_to(source.root / "input-0000/master.npy")
    with pytest.raises(ValueError, match="without link aliases"):
        validate_source(source)


def test_revalidates_frozen_geometry_and_admission_limits(completed):
    source = prepare_source(completed.result.output_root)
    changed = replace(source.images[0], shape=(1, 1))
    with pytest.raises(ValueError, match="frozen descriptors"):
        validate_source(replace(source, images=(changed,)))
    with pytest.raises(ValueError, match="pixel budget"):
        validate_source(replace(source, limits=SourceLimits(max_pixels=1)))
    with pytest.raises(ValueError, match="pixel budget"):
        load_master(replace(source, limits=SourceLimits(max_pixels=1)), "input-0000")
    with pytest.raises(ValueError, match="frozen completion"):
        validate_source(replace(source, inventory=source.inventory[:-1]))


def test_rejects_added_undeclared_source_artifact(completed):
    source = prepare_source(completed.result.output_root)
    (source.root / "unadmitted.txt").write_text("new artifact")
    with pytest.raises(ValueError, match="inventory changed"):
        validate_source(source)


def test_source_geometry_is_admitted_before_independent_replay(completed, monkeypatch):
    root = completed.result.output_root
    relative = "input-0000/photograph.json"
    completed.descriptor["source"]["shape"] = [10000, 10000]
    raw = canonicalize_json(completed.descriptor)
    (root / relative).write_bytes(raw)
    for artifact in completed.evidence["artifacts"]:
        if artifact["path"] == relative:
            artifact.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    (root / "execution-evidence.json").write_bytes(canonicalize_json(completed.evidence))

    def unexpected(*_args, **_kwargs):
        pytest.fail("Geometry admission must precede replay allocations")

    monkeypatch.setattr(source_module, "verify_execution_evidence_v3", unexpected)
    with pytest.raises(ValueError, match="geometry must agree"):
        prepare_source(root)
