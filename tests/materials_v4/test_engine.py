"""Evidence authority, deterministic composition, and honest response receipts."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.materials_v4.contracts import CalibrationReceipt, MaterialEvidence, RegionEvidence
from transformation_portal.materials_v4.engine import (
    MaterialOperation,
    ResponsePolicy,
    apply_response,
    plan_response,
)

pytestmark = pytest.mark.unit
SOURCE = "a" * 64


def _master(pixels=None, alpha=None):
    if pixels is None:
        pixels = np.full((16, 16, 3), 0.4, dtype=np.float32)
    return ImageMaster(pixels, SOURCE, 16, alpha=alpha, metadata={"author": "fixture"}, source_icc=b"icc-fixture")


def _region(name="region-1", label="water", mask=None, **kwargs):
    if mask is None:
        mask = np.ones((16, 16), dtype=np.float32)
    return RegionEvidence(name, label, mask, semantic_confidence=kwargs.pop("semantic_confidence", 0.95), **kwargs)


def _evidence(*regions, **kwargs):
    return MaterialEvidence(SOURCE, (16, 16), tuple(regions), **kwargs)


def _policy(**kwargs):
    return ResponsePolicy(min_coverage_px=4, tile_size=5, **kwargs)


def test_receipt_measures_actual_changed_pixels_and_preserves_master_metadata():
    master = _master()
    mask = np.zeros(master.shape, dtype=np.float32)
    mask[4:12, 4:12] = 0.6
    evidence = _evidence(_region(mask=mask))
    plan = plan_response(master, evidence, _policy())
    output, receipt = apply_response(master, evidence, plan)
    actual = np.abs(output.pixels.astype(np.float64) - master.pixels.astype(np.float64))
    assert receipt["changed_pixels"] == 64
    assert receipt["max_abs_delta"] == actual.max()
    assert receipt["mean_abs_delta"] == pytest.approx(actual.mean(), abs=1e-15)
    assert receipt["max_abs_delta"] > 0
    # Exactly one alpha blend: 0.4 * 0.01 gain * 0.6 mask.
    assert actual[8, 8, 0] == pytest.approx(0.0024, abs=1e-7)
    np.testing.assert_array_equal(output.pixels[mask == 0], master.pixels[mask == 0])
    assert output.metadata == master.metadata
    assert output.source_icc == master.source_icc
    assert output.source_bit_depth == master.source_bit_depth
    assert receipt["schema"] == "tp.materials.execution.v1"


@pytest.mark.parametrize("label", ["wood", "metal", "fabric", "stucco", "unknown"])
def test_unsupported_taxonomy_values_abstain_without_crashing(label):
    master = _master()
    evidence = _evidence(_region(label=label))
    plan = plan_response(master, evidence, _policy())
    output, receipt = apply_response(master, evidence, plan)
    assert output is master
    assert plan.regions[0].reason == "unsupported_material"
    assert receipt["changed_pixels"] == 0


def test_small_high_confidence_region_is_not_confused_with_frame_coverage():
    mask = np.zeros((16, 16), np.float32)
    mask[0:2, 0:2] = 1
    plan = plan_response(_master(), _evidence(_region(mask=mask)), _policy())
    assert plan.regions[0].status == "eligible"
    assert plan.regions[0].coverage_px == 4


def test_uncertain_unknown_and_overlapping_support_protects_pixels():
    master = _master()
    known = np.zeros(master.shape, np.float32)
    known[2:14, 2:14] = 1
    uncertain = np.zeros(master.shape, np.float32)
    uncertain[5:10, 5:10] = 0.1
    evidence = _evidence(_region(mask=known), _region("unknown-region", "unknown", uncertain))
    output, receipt = apply_response(master, evidence, plan_response(master, evidence, _policy()))
    np.testing.assert_array_equal(output.pixels[uncertain > 0], master.pixels[uncertain > 0])
    assert receipt["changed_pixels"] == 144 - 25


def test_confident_conflicts_protected_and_order_independent():
    master = _master()
    first = np.zeros(master.shape, np.float32)
    first[2:12, 2:12] = 1
    second = np.zeros(master.shape, np.float32)
    second[6:14, 6:14] = 1
    regions = [_region("water", "water", first), _region("glass", "glass", second)]
    evidence = _evidence(*regions)
    reversed_evidence = _evidence(*reversed(regions))
    plan = plan_response(master, evidence, _policy())
    reversed_plan = plan_response(master, reversed_evidence, _policy())
    assert plan.content_hash() == reversed_plan.content_hash()
    output, receipt = apply_response(master, evidence, plan)
    reversed_output, reversed_receipt = apply_response(master, reversed_evidence, reversed_plan)
    np.testing.assert_array_equal(output.pixels, reversed_output.pixels)
    np.testing.assert_array_equal(output.pixels[(first > 0) & (second > 0)], master.pixels[(first > 0) & (second > 0)])
    assert receipt == reversed_receipt


def test_post_conflict_support_is_rechecked():
    mask = np.ones((16, 16), np.float32)
    unknown = mask.copy()
    unknown[0, 0] = 0
    evidence = _evidence(_region(mask=mask), _region("protected", "unknown", unknown))
    plan = plan_response(_master(), evidence, _policy())
    water = next(region for region in plan.regions if region.label == "water")
    assert water.coverage_px == 256
    assert water.resolved_coverage_px == 1
    assert water.reason == "below_resolved_coverage_threshold"
    output, receipt = apply_response(_master(), evidence, plan)
    assert receipt["changed_pixels"] == 0


def test_hdr_negative_and_fully_transparent_samples_preserved_exactly():
    pixels = np.full((16, 16, 3), 0.5, np.float32)
    pixels[3, 4] = [4.0, 2.0, 1.1]
    pixels[5, 6] = [-0.1, 0.2, 0.4]
    alpha = np.full((16, 16), 0.5, np.float32)
    alpha[7, 8] = 0
    master = _master(pixels, alpha)
    evidence = _evidence(_region())
    output, _ = apply_response(master, evidence, plan_response(master, evidence, _policy()))
    for coord in [(3, 4), (5, 6), (7, 8)]:
        np.testing.assert_array_equal(output.pixels[coord], master.pixels[coord])
    np.testing.assert_array_equal(output.alpha, master.alpha)
    assert output.pixels[0, 0, 0] > master.pixels[0, 0, 0]


@pytest.mark.parametrize("label", ["glass", "water", "stone", "foliage", "sky"])
def test_tiled_and_full_frame_response_are_bit_identical(label):
    master = _master(np.random.default_rng(7).random((16, 16, 3), dtype=np.float32))
    evidence = _evidence(_region(label=label, mask=np.full((16, 16), 0.7, np.float32)))
    tiled = plan_response(master, evidence, _policy())
    full = plan_response(master, evidence, replace(_policy(), tile_size=32))
    output, _ = apply_response(master, evidence, tiled)
    full_output, _ = apply_response(master, evidence, full)
    np.testing.assert_array_equal(output.pixels, full_output.pixels)


def test_final_stored_delta_honors_budget_for_soft_masks_and_float_rounding():
    master = _master(np.full((16, 16, 3), 0.9, np.float32))
    evidence = _evidence(_region(mask=np.full((16, 16), 0.51, np.float32)))
    policy = _policy(operations=(MaterialOperation("water", "linear_gain_v1", 0.1),), max_abs_delta=0.001)
    output, receipt = apply_response(master, evidence, plan_response(master, evidence, policy))
    actual = np.abs(output.pixels.astype(np.float64) - master.pixels.astype(np.float64))
    assert 0 < actual.max() <= policy.max_abs_delta
    assert receipt["max_abs_delta"] == actual.max()


def test_no_operations_and_zero_budget_are_exact_noops():
    master = _master()
    evidence = _evidence(_region())
    for policy in [_policy(operations=()), _policy(max_abs_delta=0.0)]:
        output, receipt = apply_response(master, evidence, plan_response(master, evidence, policy))
        assert output is master
        assert receipt["changed_pixels"] == 0


def test_stale_master_mask_confidence_operation_and_forged_decision_are_rejected():
    master = _master()
    evidence = _evidence(_region())
    plan = plan_response(master, evidence, _policy())
    changed_master = _master(master.pixels + 0.01)
    changed_mask = _evidence(_region(mask=np.full((16, 16), 0.9, np.float32)))
    changed_score = _evidence(_region(semantic_confidence=0.5))
    wrong_operations = replace(plan, operations_sha256="b" * 64)
    forged_decision = replace(plan, regions=(replace(plan.regions[0], coverage_px=300),))
    for subject, proof, candidate in [
        (changed_master, evidence, plan),
        (master, changed_mask, plan),
        (master, changed_score, plan),
        (master, evidence, wrong_operations),
        (master, evidence, forged_decision),
    ]:
        with pytest.raises(ValueError, match="Stale or invalid"):
            apply_response(subject, proof, candidate)


def test_uncalibrated_inference_and_disabled_supplied_confidence_abstain():
    master = _master()
    evidence = _evidence(_region(provenance="inferred", score_type="clip_softmax_margin_v1"))
    assert plan_response(master, evidence, _policy()).regions[0].reason == "untrusted_inference"
    evidence = _evidence(_region())
    assert (
        plan_response(master, evidence, _policy(allow_supplied_confidence=False)).regions[0].reason
        == "supplied_confidence_disabled"
    )


def test_calibration_requires_explicit_admission_and_matching_region_identity():
    names = ["classifier", "proposal", "prompt", "preprocessing", "region_construction", "split", "artifact"]
    calibration = CalibrationReceipt(
        **{f"{name}_sha256": "c" * 64 for name in names}, method="temperature_scaling", classes=("water",)
    )
    region = _region(
        provenance="inferred", score_type="calibrated_material_probability_v1", calibration_sha256=calibration.content_hash()
    )
    recipe = {f"{name}_sha256": "c" * 64 for name in names[:5]}
    evidence = _evidence(region, calibration=calibration, producer=recipe)
    assert plan_response(_master(), evidence, _policy()).regions[0].reason == "untrusted_inference"
    admitted = _policy(trusted_calibration_sha256=(calibration.content_hash(),))
    assert plan_response(_master(), evidence, admitted).regions[0].status == "eligible"
    with pytest.raises(ValueError, match="calibration binding"):
        _evidence(replace(region, calibration_sha256="d" * 64), calibration=calibration)
    wrong_producer = _evidence(region, calibration=calibration)
    assert plan_response(_master(), wrong_producer, admitted).regions[0].reason == "calibration_producer_mismatch"


def test_resource_budget_rejects_before_rendering():
    with pytest.raises(ValueError, match="scratch"):
        plan_response(_master(), _evidence(_region()), _policy(max_working_bytes=1))


def test_source_and_geometry_mismatches_reject():
    master = _master()
    evidence = _evidence(_region())
    with pytest.raises(ValueError, match="source or geometry"):
        plan_response(replace(master, source_sha256="f" * 64), evidence, _policy())


def test_frozen_policy_roundtrips_and_rejects_unknown_or_missing_fields():
    policy = _policy()
    payload = policy.to_payload()
    assert ResponsePolicy.from_payload(payload) == policy
    for key in payload:
        incomplete = dict(payload)
        del incomplete[key]
        with pytest.raises(ValueError):
            ResponsePolicy.from_payload(incomplete)
    with pytest.raises(ValueError):
        ResponsePolicy.from_payload({**payload, "unexpected": 0})
    with pytest.raises(ValueError):
        ResponsePolicy.from_payload({**payload, "protected_master_samples": "none"})


@pytest.mark.parametrize(
    "name",
    ["min_confidence", "min_coverage_px", "support_threshold", "max_abs_delta", "tile_size", "halo", "max_working_bytes"],
)
def test_frozen_policy_rejects_bool_numeric(name):
    with pytest.raises(ValueError):
        ResponsePolicy.from_payload({**_policy().to_payload(), name: True})


def test_missing_evidence_receipt_preserves_reason_and_reviewable_plan():
    master = _master()
    evidence = _evidence(status="unavailable", reason="no_evidence_for_source")
    plan = plan_response(master, evidence, _policy())
    output, receipt = apply_response(master, evidence, plan)
    assert output is master
    assert receipt["evidence_status"] == "unavailable"
    assert receipt["evidence_reason"] == "no_evidence_for_source"
    assert receipt["response_plan"] == plan.to_payload()
    assert receipt["protected_changed_pixels"] == 0
    assert receipt["protected_max_abs_delta"] == 0
