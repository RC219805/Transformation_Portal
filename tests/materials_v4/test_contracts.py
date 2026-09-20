"""Evidence cannot acquire semantic authority through mutation or score coercion."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.materials_v4.contracts import (
    CalibrationReceipt,
    MaterialEvidence,
    MaterialLimits,
    MaterialsError,
    RegionEvidence,
)
from transformation_portal.materials_v4.taxonomy import canonical_label

pytestmark = pytest.mark.unit


def receipt(**changes):
    values = {
        name: "a" * 64
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
    values.update(method="held_out_isotonic", classes=("water",), metrics={"risk": 0.01})
    values.update(changes)
    return CalibrationReceipt(**values)


def region(**changes):
    values = dict(region_id="region-a", label="water", mask=np.ones((4, 5), np.float32), semantic_confidence=0.9)
    values.update(changes)
    return RegionEvidence(**values)


def evidence(**changes):
    values = dict(source_sha256="b" * 64, shape=(4, 5), regions=(region(),))
    values.update(changes)
    return MaterialEvidence(**values)


def test_source_arrays_and_nested_metadata_cannot_mutate_evidence():
    mask = np.full((4, 5), 0.75, np.float32)
    producer = {"model": {"revision": "immutable"}, "stages": ["proposal", "semantic"]}
    item = evidence(regions=(region(mask=mask),), producer=producer)
    digest = item.content_hash()
    mask[:] = 0
    producer["model"]["revision"] = "changed"
    producer["stages"].append("fallback")
    assert item.content_hash() == digest
    assert np.all(item.regions[0].mask == 0.75)
    with pytest.raises(ValueError):
        item.regions[0].mask.setflags(write=True)
    with pytest.raises(TypeError):
        item.producer["model"]["revision"] = "changed"
    payload = item.to_payload()
    payload["producer"]["model"]["revision"] = "changed"
    assert item.content_hash() == digest


def test_canonical_order_and_telemetry_do_not_change_semantic_identity():
    first, second = region(region_id="a"), region(region_id="b", label="glass")
    one = evidence(regions=(second, first), producer={"model": "pinned", "timing_ms": {"load": 1}})
    two = evidence(regions=(first, second), producer={"timing_ms": {"load": 99}, "model": "pinned"})
    assert tuple(item.region_id for item in one.regions) == ("a", "b")
    assert one.content_hash() == two.content_hash()
    assert "timing_ms" not in one.to_payload()["producer"]


@pytest.mark.parametrize(
    "changes",
    [
        {"semantic_confidence": 0.91},
        {"geometric_quality": 0.4},
        {"provenance": "heuristic"},
        {"label": "glass"},
        {"score_type": "ranking_only"},
        {"mask": np.zeros((4, 5), np.float32)},
    ],
)
def test_every_authorizing_region_component_changes_identity(changes):
    assert evidence().content_hash() != evidence(regions=(region(**changes),)).content_hash()


@pytest.mark.parametrize("value", [True, np.bool_(True), float("nan"), float("inf"), -0.1, 1.1, "0.9"])
def test_geometric_or_semantic_scores_cannot_be_forged_by_coercion(value):
    for key in ("semantic_confidence", "geometric_quality"):
        with pytest.raises(MaterialsError):
            region(**{key: value})


def test_missing_semantic_confidence_is_not_synthesized_from_geometry():
    item = region(semantic_confidence=None, geometric_quality=0.99)
    assert item.semantic_confidence is None
    assert item.score_type is None
    assert item.calibration_sha256 is None


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((4, 5), np.int16),
        np.ones((4, 5), object),
        np.ones((4, 5, 1), np.float32),
        np.full((4, 5), np.nan, np.float32),
        np.full((4, 5), 1.1, np.float32),
    ],
)
def test_invalid_numeric_masks_fail_before_freezing(mask):
    with pytest.raises(MaterialsError):
        region(mask=mask)


def test_duplicate_region_ids_and_mixed_coordinate_grids_fail():
    with pytest.raises(MaterialsError, match="unique"):
        evidence(regions=(region(), region()))
    with pytest.raises(MaterialsError, match="geometry"):
        evidence(regions=(region(mask=np.ones((5, 4), np.float32)),))


def test_limits_apply_to_aggregate_numeric_storage():
    item = evidence(regions=(region(region_id="a"), region(region_id="b")))
    with pytest.raises(MaterialsError, match="aggregate"):
        item.validate_limits(MaterialLimits(max_mask_bytes=159))
    for changes in ({"max_pixels": True}, {"max_regions": 0}, {"max_mask_bytes": 10**20}):
        with pytest.raises(MaterialsError):
            MaterialLimits(**changes)


def test_unavailable_is_distinct_from_successful_empty_evidence():
    empty = evidence(regions=())
    unavailable = evidence(regions=(), status="unavailable", reason="model_missing")
    assert empty.content_hash() != unavailable.content_hash()
    with pytest.raises(MaterialsError):
        evidence(status="unavailable", reason="model_missing")
    with pytest.raises(MaterialsError):
        evidence(regions=(), status="unavailable")


def test_calibration_identity_binds_region_construction_and_does_not_imply_trust():
    calibration = receipt()
    changed = replace(calibration, region_construction_sha256="c" * 64)
    assert changed.content_hash() != calibration.content_hash()
    bound = region(provenance="inferred", calibration_sha256=calibration.content_hash())
    item = evidence(regions=(bound,), calibration=calibration)
    assert "trusted" not in item.to_payload()["calibration"]
    with pytest.raises(MaterialsError, match="calibration binding"):
        evidence(regions=(bound,), calibration=changed)
    with pytest.raises(MaterialsError, match="calibration binding"):
        evidence(regions=(bound,))


@pytest.mark.parametrize(
    "label,expected",
    [
        ("brushed steel", "metal"),
        ("silk curtain", "fabric"),
        ("painted wall", "paint"),
        ("porcelain surface", "ceramic"),
        ("seaweed", "unknown"),
        ("windowless painted wall", "unknown"),
        ("glass and water", "unknown"),
        ("WoOd Floor", "wood"),
    ],
)
def test_taxonomy_uses_exact_aliases_and_safe_unknown(label, expected):
    assert canonical_label(label) == expected
    assert region(label=label).label == expected
