"""Bounded measurements for declared Materials V4 evaluation records.

This module measures supplied annotations. It does not certify their quality,
calibration adequacy, or that a producer actually honored its declared splits.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

from transformation_portal.ingest.canonical_json import canonicalize_json

from .contracts import MaterialsError, validate_digest
from .taxonomy import MATERIAL_LABELS

EVALUATION_INPUT_SCHEMA = "tp.materials.evaluation_input.v1"
EVALUATION_REPORT_SCHEMA = "tp.materials.evaluation.v1"
_MAX_RECORDS = 10_000
_MAX_MASK_PIXELS = 16_777_216
_MAX_TOTAL_MASK_PIXELS = 67_108_864
_DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, **extra}


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise MaterialsError(f"{name} must be an integer in [{low},{high}]")
    return value


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise MaterialsError("Confidence and thresholds must be finite numbers in [0,1]")
    try:
        number = float(value)
    except OverflowError as exc:
        raise MaterialsError("Confidence is outside [0,1]") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise MaterialsError("Confidence and thresholds must be finite numbers in [0,1]")
    return number


def _mask(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value or not isinstance(value[0], list) or not value[0]:
            raise MaterialsError("Evaluation masks must be nonempty binary HW arrays")
        if len(value) * len(value[0]) > _MAX_MASK_PIXELS:
            raise MaterialsError("Evaluation mask exceeds pixel budget")
    elif not isinstance(value, np.ndarray) or value.size > _MAX_MASK_PIXELS:
        raise MaterialsError("Evaluation mask exceeds pixel budget or has unsupported type")
    try:
        array = np.asarray(value)
    except ValueError as exc:
        raise MaterialsError("Evaluation mask must have rectangular geometry") from exc
    if array.ndim != 2 or min(array.shape) == 0 or array.dtype.kind not in "buif":
        raise MaterialsError("Evaluation masks must be nonempty binary HW arrays")
    if not np.isfinite(array).all() or np.any((array != 0) & (array != 1)):
        raise MaterialsError("Evaluation masks must be finite and exactly binary")
    return array.astype(bool)


def _parse(payload: Mapping[str, Any]) -> tuple[list[dict], dict, list[dict]]:
    required = {"schema", "records", "selection_sources"}
    optional = {"evaluation_split", "min_class_support", "calibration_bins", "boundary_tolerance", "selective_thresholds"}
    if not isinstance(payload, Mapping) or not required <= set(payload) or set(payload) - required - optional:
        raise MaterialsError("Evaluation input has missing or unknown fields")
    if payload["schema"] != EVALUATION_INPUT_SCHEMA:
        raise MaterialsError("Unsupported evaluation input schema")
    settings = {
        "evaluation_split": payload.get("evaluation_split", "test"),
        "min_class_support": _integer(payload.get("min_class_support", 5), "min_class_support", 2, _MAX_RECORDS),
        "calibration_bins": _integer(payload.get("calibration_bins", 10), "calibration_bins", 2, 100),
        "boundary_tolerance": _integer(payload.get("boundary_tolerance", 1), "boundary_tolerance", 0, 8),
    }
    if settings["evaluation_split"] not in {"validation", "test"}:
        raise MaterialsError("Only validation or sealed test records may be evaluated")
    thresholds = payload.get("selective_thresholds", list(_DEFAULT_THRESHOLDS))
    if not isinstance(thresholds, (list, tuple)) or not 1 <= len(thresholds) <= 100:
        raise MaterialsError("Selective thresholds must contain 1..100 values")
    settings["selective_thresholds"] = sorted(set(_score(value) for value in thresholds))
    raw_records = payload["records"]
    if not isinstance(raw_records, (list, tuple)) or len(raw_records) > _MAX_RECORDS:
        raise MaterialsError("Evaluation records exceed record budget")
    fields = {
        "record_id",
        "source_sha256",
        "property_group",
        "split",
        "true_label",
        "predicted_label",
        "confidence",
        "predicted_mask",
        "target_mask",
    }
    records, descriptors = [], []
    record_ids: set[str] = set()
    source_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    source_groups: dict[str, str] = {}
    total_pixels = 0
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise MaterialsError("Each evaluation record requires the exact record schema")
        for name in ("record_id", "property_group"):
            value = raw[name]
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or not value.isascii()
                or any(ord(c) < 32 for c in value)
            ):
                raise MaterialsError(f"{name} must be bounded nonempty ASCII text")
        if raw["record_id"] in record_ids:
            raise MaterialsError("Evaluation record IDs must be unique")
        record_ids.add(raw["record_id"])
        source = validate_digest(raw["source_sha256"], "source_sha256")
        split, group = raw["split"], raw["property_group"]
        if split not in {"train", "calibration", "validation", "test"}:
            raise MaterialsError("Unknown dataset split")
        if source in source_splits and source_splits[source] != split:
            raise MaterialsError("Source leak across dataset splits")
        if group in group_splits and group_splits[group] != split:
            raise MaterialsError("Property-group leak across dataset splits")
        if source in source_groups and source_groups[source] != group:
            raise MaterialsError("A source has inconsistent property-group identity")
        source_splits[source], group_splits[group], source_groups[source] = split, split, group
        if raw["true_label"] not in MATERIAL_LABELS or raw["predicted_label"] not in MATERIAL_LABELS:
            raise MaterialsError("Evaluation labels must be canonical taxonomy labels")
        confidence = None if raw["confidence"] is None else _score(raw["confidence"])
        predicted, target = _mask(raw["predicted_mask"]), _mask(raw["target_mask"])
        if (predicted is None) != (target is None):
            raise MaterialsError("Predicted and target masks must both be present or both absent")
        if predicted is not None and target is not None and predicted.shape != target.shape:
            raise MaterialsError("Predicted and target masks must have matching geometry")
        total_pixels += predicted.size if predicted is not None else 0
        if total_pixels > _MAX_TOTAL_MASK_PIXELS:
            raise MaterialsError("Evaluation masks exceed aggregate pixel budget")
        record = {**dict(raw), "confidence": confidence, "predicted_mask": predicted, "target_mask": target}
        descriptor = {key: value for key, value in record.items() if key not in {"predicted_mask", "target_mask"}}
        for key in ("predicted_mask", "target_mask"):
            mask = record[key]
            descriptor[key] = (
                None if mask is None else {"shape": list(mask.shape), "sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
            )
        records.append(record)
        descriptors.append(descriptor)
    selected = payload["selection_sources"]
    if not isinstance(selected, (tuple, list)) or len(selected) > _MAX_RECORDS:
        raise MaterialsError("Selection sources must be a bounded explicit sequence")
    selection = sorted(set(validate_digest(value, "selection source") for value in selected))
    if any(source not in source_splits for source in selection):
        raise MaterialsError("Selection source has no declared dataset membership")
    if any(source_splits[source] == "test" for source in selection):
        raise MaterialsError("Sealed test source was used for model or policy selection")
    settings["selection_sources"] = selection
    return (
        sorted(records, key=lambda item: item["record_id"]),
        settings,
        sorted(descriptors, key=lambda item: item["record_id"]),
    )


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return mask & ~eroded


def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance == 0:
        return mask
    padded = np.pad(mask, tolerance, constant_values=False)
    dilated = np.zeros_like(mask)
    height, width = mask.shape
    for dy in range(2 * tolerance + 1):
        for dx in range(2 * tolerance + 1):
            dilated |= padded[dy : dy + height, dx : dx + width]
    return dilated


def _mask_metrics(records: list[dict], minimum: int, tolerance: int) -> dict:
    accumulators: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        predicted, target = record["predicted_mask"], record["target_mask"]
        if predicted is None:
            continue
        for label in {record["true_label"], record["predicted_label"]}:
            actual = target if record["true_label"] == label else np.zeros_like(target)
            proposed = predicted if record["predicted_label"] == label else np.zeros_like(predicted)
            accumulator = accumulators[label]
            accumulator["support_records"] += 1
            accumulator["true_records"] += int(record["true_label"] == label)
            accumulator["intersection"] += int(np.count_nonzero(actual & proposed))
            accumulator["union"] += int(np.count_nonzero(actual | proposed))
            target_boundary, predicted_boundary = _boundary(actual), _boundary(proposed)
            accumulator["target_boundary"] += int(target_boundary.sum())
            accumulator["predicted_boundary"] += int(predicted_boundary.sum())
            accumulator["matched_prediction"] += int(
                np.count_nonzero(predicted_boundary & _dilate(target_boundary, tolerance))
            )
            accumulator["matched_target"] += int(np.count_nonzero(target_boundary & _dilate(predicted_boundary, tolerance)))
    result = {}
    labels = sorted({record["true_label"] for record in records} | {record["predicted_label"] for record in records})
    for label in labels:
        counts = accumulators[label]
        if counts["true_records"] < minimum:
            result[label] = _unavailable("insufficient_annotated_class_support", support_records=counts["true_records"])
            continue
        if counts["union"] == 0:
            result[label] = _unavailable("empty_mask_union", support_records=counts["true_records"])
            continue
        precision = counts["matched_prediction"] / counts["predicted_boundary"] if counts["predicted_boundary"] else 0.0
        recall = counts["matched_target"] / counts["target_boundary"] if counts["target_boundary"] else 0.0
        boundary = _unavailable("empty_target_boundary")
        if counts["target_boundary"]:
            boundary = {
                "status": "measured",
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "tolerance_px": tolerance,
                "distance": "chebyshev",
                "boundary": "inner_8_connected",
            }
        result[label] = {
            "status": "measured",
            "support_records": counts["true_records"],
            "mask_iou": counts["intersection"] / counts["union"],
            "boundary_f1": boundary,
            "aggregation": "pooled_record_instance_pixels",
        }
    return result


def _confidence_metrics(records: list[dict], settings: dict) -> tuple[dict, dict]:
    candidates = [
        record
        for record in records
        if record["predicted_label"] not in {"unknown", "mixed"} and record["confidence"] is not None
    ]
    minimum = settings["min_class_support"]
    if len(candidates) < minimum:
        unavailable = _unavailable("insufficient_scored_support", support_records=len(candidates))
        return unavailable, unavailable.copy()
    confidence = np.array([record["confidence"] for record in candidates], dtype=np.float64)
    correct = np.array([record["predicted_label"] == record["true_label"] for record in candidates], dtype=np.float64)
    bin_ids = np.minimum((confidence * settings["calibration_bins"]).astype(np.int64), settings["calibration_bins"] - 1)
    bins, ece = [], 0.0
    for index in range(settings["calibration_bins"]):
        included = bin_ids == index
        count = int(included.sum())
        mean_confidence: float | None = None
        accuracy: float | None = None
        if count:
            mean_confidence = float(confidence[included].mean())
            accuracy = float(correct[included].mean())
            ece += count / len(candidates) * abs(mean_confidence - accuracy)
        bins.append(
            {
                "lower": index / settings["calibration_bins"],
                "upper": (index + 1) / settings["calibration_bins"],
                "upper_inclusive": index == settings["calibration_bins"] - 1,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    calibration = {
        "status": "measured",
        "support_records": len(candidates),
        "ece": ece,
        "top_label_binary_brier": float(np.mean((confidence - correct) ** 2)),
        "bins": bins,
        "scope": "top_label_correctness; not multiclass_brier_or_calibration_certification",
    }
    points = []
    for threshold in settings["selective_thresholds"]:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        points.append(
            {
                "threshold": threshold,
                "accepted_records": count,
                "coverage": count / len(records),
                "risk": float(1 - correct[accepted].mean()) if count >= minimum else None,
                "status": "measured" if count >= minimum else "unavailable",
                "reason": None if count >= minimum else "insufficient_accepted_support",
            }
        )
    selective = {
        "status": "measured",
        "denominator_records": len(records),
        "points": points,
        "abstention": "unknown_mixed_or_missing_confidence",
    }
    return calibration, selective


def evaluate_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Measure strict declared records, returning unavailable for invalid evidence.

    Required input keys: schema, records, selection_sources. Every record has
    record_id, source_sha256, property_group, split, true_label, predicted_label,
    confidence, predicted_mask, target_mask. Masks are paired binary HW arrays or
    paired nulls. Optional settings: evaluation_split (test), min_class_support
    (5), calibration_bins (10), boundary_tolerance (1), selective_thresholds.
    """
    base = {
        "schema": EVALUATION_REPORT_SCHEMA,
        "acceptance": "not_assessed",
        "evidence_scope": "caller_declared_annotations_and_split_membership",
    }
    try:
        records, settings, descriptors = _parse(payload)
    except (MaterialsError, TypeError, ValueError) as exc:
        return {**base, **_unavailable("invalid_evaluation_input", diagnostic=str(exc))}
    evaluation = [record for record in records if record["split"] == settings["evaluation_split"]]
    split_membership = sorted({(record["source_sha256"], record["property_group"], record["split"]) for record in records})
    split_receipt = {
        "schema": "tp.materials.dataset_split.v1",
        "membership_sha256": _digest(split_membership),
        "selection_sources": settings["selection_sources"],
        "sealed_test_used_for_selection": False,
        "source_and_property_group_disjoint": True,
        "scope": "declared_records_only",
    }
    split_receipt["content_sha256"] = _digest(split_receipt)
    report = {
        **base,
        "dataset_sha256": _digest(descriptors),
        "settings": settings,
        "split_receipt": split_receipt,
        "evaluation_records": len(evaluation),
    }
    if len(evaluation) < settings["min_class_support"]:
        return {**report, **_unavailable("insufficient_evaluation_support")}
    confusion: dict[str, dict[str, int]] = {}
    for record in evaluation:
        row = confusion.setdefault(record["true_label"], {})
        row[record["predicted_label"]] = row.get(record["predicted_label"], 0) + 1
    calibration, selective = _confidence_metrics(evaluation, settings)
    known_masks = [record for record in evaluation if record["predicted_mask"] is not None]
    unknown_mask_pixels = sum(
        int(record["predicted_mask"].sum()) for record in known_masks if record["predicted_label"] in {"unknown", "mixed"}
    )
    canvas_pixels = sum(record["predicted_mask"].size for record in known_masks)
    return {
        **report,
        "status": "measured",
        "confusion_records": confusion,
        "unknown_coverage": {
            "record_fraction": sum(record["predicted_label"] in {"unknown", "mixed"} for record in evaluation)
            / len(evaluation),
            "mask_canvas_fraction": unknown_mask_pixels / canvas_pixels if canvas_pixels else None,
            "mask_denominator": "sum_of_annotated_record_canvas_pixels; not_unique_scene_area",
        },
        "per_class": _mask_metrics(evaluation, settings["min_class_support"], settings["boundary_tolerance"]),
        "calibration": calibration,
        "selective_risk": selective,
    }
