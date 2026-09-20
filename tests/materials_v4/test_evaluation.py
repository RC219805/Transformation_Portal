"""Small exact fixtures validate measurements, never production acceptance."""

from copy import deepcopy

import numpy as np
import pytest

from transformation_portal.materials_v4.evaluation import EVALUATION_INPUT_SCHEMA, evaluate_dataset

pytestmark = pytest.mark.unit


def _record(index, **overrides):
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    return {
        "record_id": f"record-{index}",
        "source_sha256": f"{index + 1:064x}",
        "property_group": f"property-{index}",
        "split": "test",
        "true_label": "water",
        "predicted_label": "water",
        "confidence": 0.8,
        "predicted_mask": mask.tolist(),
        "target_mask": mask.tolist(),
        **overrides,
    }


def _payload(*records, **overrides):
    return {
        "schema": EVALUATION_INPUT_SCHEMA,
        "records": list(records),
        "selection_sources": [],
        "min_class_support": 2,
        **overrides,
    }


def test_exact_masks_and_known_calibration_errors_are_measured_honestly():
    report = evaluate_dataset(_payload(_record(0), _record(1)))
    assert report["status"] == "measured"
    assert report["acceptance"] == "not_assessed"
    assert report["per_class"]["water"]["mask_iou"] == 1.0
    assert report["per_class"]["water"]["boundary_f1"]["f1"] == 1.0
    assert report["calibration"]["ece"] == pytest.approx(0.2)
    assert report["calibration"]["top_label_binary_brier"] == pytest.approx(0.04)
    assert report["unknown_coverage"]["record_fraction"] == 0
    assert report["split_receipt"]["scope"] == "declared_records_only"


def test_wrong_class_masks_do_not_receive_spurious_geometric_credit():
    report = evaluate_dataset(_payload(_record(0, predicted_label="glass"), _record(1, predicted_label="glass")))
    assert report["per_class"]["water"]["mask_iou"] == 0
    assert report["per_class"]["water"]["boundary_f1"]["f1"] == 0
    assert report["confusion_records"] == {"water": {"glass": 2}}
    assert report["calibration"]["top_label_binary_brier"] == pytest.approx(0.64)


def test_boundary_tolerance_is_spatial_and_explicit():
    shifted = np.zeros((7, 7), dtype=np.uint8)
    shifted[2:5, 3:6] = 1
    records = [_record(index, predicted_mask=shifted.tolist()) for index in range(2)]
    exact = evaluate_dataset(_payload(*records, boundary_tolerance=0))
    tolerant = evaluate_dataset(_payload(*records, boundary_tolerance=1))
    assert exact["per_class"]["water"]["mask_iou"] == pytest.approx(0.5)
    assert exact["per_class"]["water"]["boundary_f1"]["f1"] < 1
    assert tolerant["per_class"]["water"]["boundary_f1"]["f1"] == 1
    assert tolerant["per_class"]["water"]["boundary_f1"]["distance"] == "chebyshev"


def test_unknowns_reduce_selective_coverage_and_remain_visible():
    records = [
        _record(0, confidence=0.9),
        _record(1, confidence=0.9),
        _record(2, predicted_label="unknown", confidence=None),
        _record(3, predicted_label="unknown", confidence=None),
    ]
    report = evaluate_dataset(_payload(*records))
    assert report["unknown_coverage"]["record_fraction"] == 0.5
    point = next(point for point in report["selective_risk"]["points"] if point["threshold"] == 0.8)
    assert point["coverage"] == 0.5
    assert point["risk"] == 0
    absent = next(point for point in report["selective_risk"]["points"] if point["threshold"] == 1.0)
    assert absent["status"] == "unavailable"
    assert absent["risk"] is None


@pytest.mark.parametrize("failure", ["source_leak", "property_leak", "sealed_selection", "unknown_selection"])
def test_split_receipt_rejects_leakage_and_test_selection(failure):
    first, second = _record(0), _record(1)
    payload = _payload(first, second)
    if failure == "source_leak":
        second.update(source_sha256=first["source_sha256"], split="train")
    elif failure == "property_leak":
        second.update(property_group=first["property_group"], split="calibration")
    elif failure == "sealed_selection":
        payload["selection_sources"] = [first["source_sha256"]]
    else:
        payload["selection_sources"] = ["a" * 64]
    report = evaluate_dataset(payload)
    assert report["status"] == "unavailable"
    assert report["reason"] == "invalid_evaluation_input"
    assert "split_receipt" not in report


def test_selection_on_declared_validation_sources_is_bound_in_receipt():
    selection = _record(2, split="validation")
    payload = _payload(_record(0), _record(1), selection, selection_sources=[selection["source_sha256"]])
    report = evaluate_dataset(payload)
    assert report["status"] == "measured"
    assert report["evaluation_records"] == 2
    assert report["split_receipt"]["selection_sources"] == [selection["source_sha256"]]
    assert report["split_receipt"]["sealed_test_used_for_selection"] is False


def test_record_order_does_not_change_any_metric_or_identity():
    records = [_record(0), _record(1, predicted_label="glass"), _record(2, confidence=0.6)]
    first = evaluate_dataset(_payload(*records))
    second = evaluate_dataset(_payload(*reversed(records)))
    assert first == second


def test_missing_masks_and_scores_never_become_perfect_metrics():
    records = [_record(index, predicted_mask=None, target_mask=None, confidence=None) for index in range(2)]
    report = evaluate_dataset(_payload(*records))
    assert report["status"] == "measured"  # Confusion counts remain valid.
    assert report["per_class"]["water"]["status"] == "unavailable"
    assert report["calibration"]["status"] == "unavailable"
    assert report["selective_risk"]["status"] == "unavailable"
    assert report["unknown_coverage"]["mask_canvas_fraction"] is None


def test_small_support_is_unavailable_for_dataset_and_per_class():
    report = evaluate_dataset(_payload(_record(0)))
    assert report["status"] == "unavailable"
    assert report["reason"] == "insufficient_evaluation_support"
    report = evaluate_dataset(_payload(_record(0), _record(1, true_label="glass", predicted_label="glass")))
    assert report["per_class"]["water"]["status"] == "unavailable"
    assert report["per_class"]["glass"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_selection",
        "missing_record_field",
        "nan_confidence",
        "bool_confidence",
        "nonbinary_mask",
        "geometry",
        "unknown_field",
    ],
)
def test_malformed_evidence_returns_unavailable(mutation):
    payload = _payload(_record(0), _record(1))
    if mutation == "missing_selection":
        del payload["selection_sources"]
    elif mutation == "missing_record_field":
        del payload["records"][0]["target_mask"]
    elif mutation == "nan_confidence":
        payload["records"][0]["confidence"] = float("nan")
    elif mutation == "bool_confidence":
        payload["records"][0]["confidence"] = True
    elif mutation == "nonbinary_mask":
        payload["records"][0]["target_mask"][0][0] = 0.5
    elif mutation == "geometry":
        payload["records"][0]["target_mask"] = [[0]]
    else:
        payload["unexpected"] = True
    report = evaluate_dataset(payload)
    assert report["status"] == "unavailable"
    assert report["acceptance"] == "not_assessed"


def test_dataset_identity_binds_masks_scores_and_declarations():
    payload = _payload(_record(0), _record(1))
    original = evaluate_dataset(payload)
    altered = deepcopy(payload)
    altered["records"][0]["confidence"] = 0.7
    assert evaluate_dataset(altered)["dataset_sha256"] != original["dataset_sha256"]
    altered = deepcopy(payload)
    altered["records"][0]["predicted_mask"][0][0] = 1
    assert evaluate_dataset(altered)["dataset_sha256"] != original["dataset_sha256"]
