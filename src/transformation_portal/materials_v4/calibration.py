"""Source-bound, draft top-label calibration; fitting never grants edit authority.

Classwise isotonic regression estimates whether an accepted predicted label is
correct. It does not estimate segmentation quality or certify calibration on
unseen properties. Human review and a separate policy allowlist remain required.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from PIL import Image

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.image_artifact import metadata_payload
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.io import directory_path, snapshot

from .artifacts import _calibration, _mask_array, _object, _relative_path, load_evidence
from .contracts import CalibrationReceipt, MaterialLimits, MaterialsError, validate_digest
from .evaluation import _parse, _score
from .semantics import SEMANTIC_SCORE_TYPE
from .taxonomy import MATERIAL_LABELS, TAXONOMY_VERSION

CALIBRATION_INPUT_SCHEMA = "tp.materials.calibration_input.v1"
CALIBRATION_ARTIFACT_SCHEMA = "tp.materials.calibration_artifact.v1"
CALIBRATED_SCORE_TYPE = "calibrated_material_probability_v1"
METHOD = "classwise_isotonic_top_label_v1"
RECIPE_KEYS = {
    "classifier_sha256",
    "proposal_sha256",
    "prompt_sha256",
    "preprocessing_sha256",
    "region_construction_sha256",
    "taxonomy_sha256",
}
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_SOURCES = 256
_MAX_RECORDS = 10_000


def _fit_metadata(value: Mapping[str, Any], *, maximum_bytes: int = _MAX_JSON_BYTES) -> Mapping[str, Any]:
    """Preserve and freeze finite fitting JSON within its own artifact budget.

    Evidence metadata has a separate, smaller contract. Count canonical tokens
    while freezing rather than widening that contract or allocating an oversized
    encoded artifact. Every key is retained, including timing-like names.
    """
    encoded_bytes = 0

    def account(size: int) -> None:
        nonlocal encoded_bytes
        encoded_bytes += size
        if encoded_bytes > maximum_bytes:
            raise MaterialsError("Calibration metadata exceeds byte budget")

    def freeze(item: Any, depth: int = 0) -> Any:
        if depth > 12:
            raise MaterialsError("Calibration metadata nesting exceeds budget")
        if isinstance(item, Mapping):
            if len(item) > 128 or not all(type(key) is str and len(key) <= 128 for key in item):
                raise MaterialsError("Calibration metadata keys exceed budget")
            account(2 + max(0, len(item) - 1))
            result = {}
            for key, child in item.items():
                account(len(canonicalize_json(key)) + 1)
                result[key] = freeze(child, depth + 1)
            return MappingProxyType(result)
        if isinstance(item, (tuple, list)):
            if len(item) > _MAX_RECORDS:
                raise MaterialsError("Calibration metadata sequence exceeds budget")
            account(2 + max(0, len(item) - 1))
            return tuple(freeze(child, depth + 1) for child in item)
        if type(item) is str:
            if len(item) > 4096:
                raise MaterialsError("Calibration metadata string exceeds budget")
        elif item is None or type(item) is bool:
            pass
        elif type(item) is int and -(2**63) <= item < 2**63:
            pass
        elif type(item) is float and math.isfinite(item):
            pass
        else:
            raise MaterialsError("Calibration metadata must contain bounded finite JSON values")
        account(len(canonicalize_json(item)))
        return item

    if not isinstance(value, Mapping):
        raise MaterialsError("Calibration metadata must be an object")
    try:
        return freeze(value)
    except (UnicodeError, RecursionError) as exc:
        raise MaterialsError("Calibration metadata must be bounded valid UTF-8 JSON") from exc


def decode_calibration_json_object(data: bytes) -> dict[str, Any]:
    """Decode strict calibration JSON without changing execution-plan limits."""
    if not isinstance(data, bytes) or not 0 < len(data) <= _MAX_JSON_BYTES:
        raise MaterialsError("Calibration JSON exceeds byte budget or is empty")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise MaterialsError("Calibration JSON contains duplicate object keys")
            result[key] = value
        return result

    def constant(_: str) -> Any:
        raise MaterialsError("Calibration JSON must contain finite numbers")

    try:
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs, parse_constant=constant)
        _fit_metadata(payload)
        return payload
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise MaterialsError("Calibration JSON must be a bounded finite UTF-8 object") from exc


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonicalize_json(metadata_payload(payload))).hexdigest()


def taxonomy_digest() -> str:
    return _digest({"taxonomy": TAXONOMY_VERSION, "labels": sorted(MATERIAL_LABELS)})


def _identity(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or not value.isascii()
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise MaterialsError(f"{name} requires a bounded explicit identity")
    return value


def _read(root: Path, relative: Any, maximum: int) -> tuple[bytes, dict]:
    path = _relative_path(relative)
    return snapshot(root, root / path, maximum_bytes=maximum)


def _recipe(value: Any) -> dict[str, str]:
    record = _object(value, RECIPE_KEYS)
    result = {name: validate_digest(record[name], name) for name in sorted(RECIPE_KEYS)}
    if result["taxonomy_sha256"] != taxonomy_digest():
        raise MaterialsError("Calibration taxonomy differs from the current taxonomy")
    return result


def _isotonic(pairs: list[tuple[float, bool]]) -> list[list[float | int]]:
    """Weighted pool-adjacent-violators, with equal scores pooled first.

    Knots store the observed score interval, empirical correctness, and support
    of each monotone block. Application is constant within each observed block,
    interpolates gaps between blocks, and abstains outside observed support.
    """
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for score, correct in pairs:
        grouped[score][0] += int(correct)
        grouped[score][1] += 1
    blocks: list[list[float | int]] = []
    for score, (successes, count) in sorted(grouped.items()):
        blocks.append([score, score, successes, count])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            left, right = blocks[-2:]
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    return [[low, high, successes / count, count] for low, high, successes, count in blocks]


@dataclass(frozen=True)
class CalibrationModel:
    """Validated fitting artifact and receipt; trust is deliberately absent."""

    fit: Mapping[str, Any]
    receipt: CalibrationReceipt

    def __post_init__(self) -> None:
        fit = _fit_metadata(
            _object(
                self.fit,
                {
                    "schema",
                    "method",
                    "recipe",
                    "split_receipt",
                    "dataset_sha256",
                    "annotation_provenance",
                    "classes",
                    "excluded_classes",
                    "scope",
                    "acceptance",
                },
            )
        )
        if fit["schema"] != "tp.materials.calibration_fit.v1" or fit["method"] != METHOD:
            raise MaterialsError("Unsupported calibration fitting method")
        if fit["acceptance"] != "not_assessed" or fit["scope"] != "semantic_top_label_correctness_only":
            raise MaterialsError("A fitting artifact cannot assert acceptance")
        recipe = _recipe(fit["recipe"])
        if not isinstance(self.receipt, CalibrationReceipt) or self.receipt.method != METHOD:
            raise MaterialsError("Calibration requires the matching procedure receipt")
        if self.receipt.domain != "photography" or self.receipt.artifact_sha256 != _digest(fit):
            raise MaterialsError("Calibration receipt does not bind the photographic fitting artifact")
        if any(getattr(self.receipt, key) != value for key, value in recipe.items()):
            raise MaterialsError("Calibration receipt and fitting recipe differ")
        split = _object(
            fit["split_receipt"],
            {"schema", "membership_sha256", "selection_sources", "sealed_test_used_for_selection", "scope", "content_sha256"},
        )
        if (
            split["schema"] != "tp.materials.dataset_split.v1"
            or split["sealed_test_used_for_selection"] is not False
            or split["scope"] != "observed_sources_with_declared_property_membership"
            or split["content_sha256"] != _digest({key: value for key, value in split.items() if key != "content_sha256"})
            or self.receipt.split_sha256 != split["content_sha256"]
        ):
            raise MaterialsError("Calibration split receipt is invalid")
        validate_digest(split["membership_sha256"], "membership_sha256")
        validate_digest(fit["dataset_sha256"], "dataset_sha256")
        if (
            not isinstance(split["selection_sources"], (list, tuple))
            or not 1 <= len(split["selection_sources"]) <= _MAX_SOURCES
        ):
            raise MaterialsError("Calibration selection sources exceed budget")
        for source in split["selection_sources"]:
            validate_digest(source, "selection source")
        classes = fit["classes"]
        if not isinstance(classes, Mapping) or not classes or set(classes) != set(self.receipt.classes):
            raise MaterialsError("Calibration class coverage differs from its receipt")
        for label, model in classes.items():
            if label not in MATERIAL_LABELS - {"unknown", "mixed"}:
                raise MaterialsError("Only known labels can be calibrated")
            _object(model, {"knots", "support_records", "property_groups"})
            if type(model["support_records"]) is not int or not 5 <= model["support_records"] <= 10_000:
                raise MaterialsError("Calibration class requires at least five fitting records")
            if (
                not isinstance(model["property_groups"], (tuple, list))
                or not 1 <= len(model["property_groups"]) <= _MAX_SOURCES
            ):
                raise MaterialsError("Calibration property-group support is invalid")
            for group in model["property_groups"]:
                _identity(group, "property_group")
            knots = model["knots"]
            if not isinstance(knots, (list, tuple)) or not 1 <= len(knots) <= 10_000:
                raise MaterialsError("Calibration knots exceed budget")
            previous_high, previous_probability, count = -1.0, -1.0, 0
            for knot in knots:
                if not isinstance(knot, (tuple, list)) or len(knot) != 4:
                    raise MaterialsError("Invalid isotonic knot")
                low, high, probability = (_score(value) for value in knot[:3])
                if low > high or low <= previous_high or probability < previous_probability:
                    raise MaterialsError("Calibration knots must be ordered, disjoint and monotone")
                if type(knot[3]) is not int or not 1 <= knot[3] <= 10_000:
                    raise MaterialsError("Invalid isotonic support")
                previous_high, previous_probability = high, probability
                count += knot[3]
            if count != model["support_records"]:
                raise MaterialsError("Calibration knot support differs from class support")
        if not isinstance(fit["annotation_provenance"], (tuple, list)) or not 5 <= len(fit["annotation_provenance"]) <= 10_000:
            raise MaterialsError("Calibration requires bounded annotation provenance")
        if not isinstance(fit["excluded_classes"], Mapping):
            raise MaterialsError("Calibration exclusions must be explicit")
        object.__setattr__(self, "fit", fit)
        if len(canonicalize_json(self.to_payload())) > _MAX_JSON_BYTES:
            raise MaterialsError("Calibration artifact exceeds byte budget")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_ARTIFACT_SCHEMA,
            "fit": metadata_payload(self.fit),
            "receipt": self.receipt.to_payload(),
            "review_status": "unreviewed",
        }

    def probability(self, label: str, score: float) -> float | None:
        value = _score(score)
        model = self.fit["classes"].get(label)
        if model is None:
            return None
        knots = model["knots"]
        if value < knots[0][0] or value > knots[-1][1]:
            return None
        scores, probabilities = [], []
        for low, high, probability, _ in knots:
            scores.append(low)
            probabilities.append(probability)
            if high != low:
                scores.append(high)
                probabilities.append(probability)
        return float(np.interp(value, scores, probabilities))


def load_calibration(path: Path) -> tuple[CalibrationModel, dict]:
    """Snapshot a bounded local artifact; no pickle or executable content."""
    path = path.expanduser().absolute()
    root = directory_path(path.parent)
    data, receipt = _read(root, path.name, _MAX_JSON_BYTES)
    payload = _object(decode_calibration_json_object(data), {"schema", "fit", "receipt", "review_status"})
    if payload["schema"] != CALIBRATION_ARTIFACT_SCHEMA or payload["review_status"] != "unreviewed":
        raise MaterialsError("Calibration fitting artifacts are always unreviewed")
    calibration = _calibration(payload["receipt"])
    if calibration is None:
        raise MaterialsError("Calibration artifact requires a receipt")
    return CalibrationModel(payload["fit"], calibration), receipt


def _membership(raw: Any, source_root: Path) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_SOURCES:
        raise MaterialsError("Calibration requires 1..256 explicit photographic sources")
    sources, groups, membership = {}, {}, []
    for value in raw:
        item = _object(value, {"source_path", "source_sha256", "property_group", "split"})
        source = validate_digest(item["source_sha256"], "source_sha256")
        group = _identity(item["property_group"], "property_group")
        split = item["split"]
        if not isinstance(split, str) or split not in {"train", "calibration", "validation", "test"} or source in sources:
            raise MaterialsError("Source membership must be unique with a supported split")
        if group in groups and groups[group] != split:
            raise MaterialsError("Property-group leak across dataset splits")
        relative = _relative_path(item["source_path"])
        _, observed = snapshot(source_root, source_root / relative, maximum_bytes=1024**3, retain_bytes=False)
        if observed["sha256"] != source:
            raise MaterialsError("Calibration source bytes differ from declared membership")
        sources[source], groups[group] = dict(item), split
        membership.append((source, group, split))
    return sources, sorted(membership)


def fit_calibration(payload: Mapping[str, Any], *, source_root: Path, data_root: Path) -> CalibrationModel:
    """Fit calibration-only observations bound to actual files and annotations.

    Reviewer identities and property memberships are declarations, not verified
    credentials. The result remains unreviewed and cannot alter the policy
    allowlist. Sealed test labels/scores are neither accepted nor used here.
    """
    _object(payload, {"schema", "evaluation", "split_membership", "bindings"})
    if payload["schema"] != CALIBRATION_INPUT_SCHEMA:
        raise MaterialsError("Unsupported calibration input schema")
    try:
        records, settings, descriptors = _parse(payload["evaluation"])
    except (TypeError, ValueError) as exc:
        raise MaterialsError(f"Invalid calibration evaluation input: {exc}") from exc
    if len(records) < 5 or any(record["split"] != "calibration" for record in records):
        raise MaterialsError("Fitting accepts at least five calibration records only; sealed test labels are forbidden")
    source_root, data_root = directory_path(source_root), directory_path(data_root)
    sources, membership = _membership(payload["split_membership"], source_root)
    selected = sorted({record["source_sha256"] for record in records})
    if settings["selection_sources"] != selected:
        raise MaterialsError("Selection sources must explicitly cover exactly the fitting sources")
    raw_bindings = payload["bindings"]
    if not isinstance(raw_bindings, list) or len(raw_bindings) != len(records):
        raise MaterialsError("Each fitting record requires an exact source/evidence/annotation binding")
    bindings = {}
    for item in raw_bindings:
        _object(item, {"record_id", "evidence_path", "evidence_sha256", "region_id", "annotation_path", "annotation_sha256"})
        _identity(item["record_id"], "record_id")
        _identity(item["region_id"], "region_id")
        if item["record_id"] in bindings:
            raise MaterialsError("Fitting bindings must have unique record IDs")
        bindings[item["record_id"]] = item
    if set(bindings) != {record["record_id"] for record in records}:
        raise MaterialsError("Fitting binding IDs differ from calibration record IDs")
    recipe, provenance, pairs, properties, observed_regions = None, [], defaultdict(list), defaultdict(set), set()
    for record in records:
        membership_item = sources.get(record["source_sha256"])
        if (
            membership_item is None
            or membership_item["split"] != "calibration"
            or membership_item["property_group"] != record["property_group"]
        ):
            raise MaterialsError("Fitting record differs from its source/property membership")
        if record["predicted_mask"] is not None:
            raise MaterialsError("Fitting uses file-bound masks, not declared evaluation mask arrays")
        binding = bindings[record["record_id"]]
        annotation_data, annotation_file = _read(data_root, binding["annotation_path"], 1024 * 1024)
        if annotation_file["sha256"] != validate_digest(binding["annotation_sha256"], "annotation_sha256"):
            raise MaterialsError("Annotation bytes differ from the fitting binding")
        annotation = _object(
            decode_bounded_json_object(annotation_data),
            {
                "schema",
                "source_sha256",
                "shape",
                "property_group",
                "label",
                "mask_path",
                "mask_sha256",
                "annotator",
                "reviewer",
                "review_status",
            },
        )
        if (
            annotation["schema"] != "tp.materials.annotation.v1"
            or annotation["source_sha256"] != record["source_sha256"]
            or annotation["property_group"] != record["property_group"]
            or annotation["label"] != record["true_label"]
            or annotation["review_status"] != "reviewed"
            or _identity(annotation["annotator"], "annotator") == _identity(annotation["reviewer"], "reviewer")
        ):
            raise MaterialsError(
                "Annotation requires matching source/label/group and distinct declared human review identities"
            )
        source_data, _ = _read(source_root, membership_item["source_path"], 1024**3)
        if hashlib.sha256(source_data).hexdigest() != record["source_sha256"]:
            raise MaterialsError("Calibration source changed during fitting")
        with Image.open(io.BytesIO(source_data)) as image:
            width, height = image.size
            if image.getexif().get(274) in {5, 6, 7, 8}:
                height, width = width, height
        shape = (height, width)
        if annotation["shape"] != list(shape):
            raise MaterialsError("Annotation frame differs from orientation-normalized photographic geometry")
        evidence_path = data_root / _relative_path(binding["evidence_path"])
        evidence_data, evidence_file = snapshot(data_root, evidence_path, maximum_bytes=1024 * 1024)
        if evidence_file["sha256"] != validate_digest(binding["evidence_sha256"], "evidence_sha256"):
            raise MaterialsError("Inference evidence bytes differ from fitting binding")
        evidence_content = decode_bounded_json_object(evidence_data).get("content_sha256")
        validate_digest(evidence_content, "evidence content_sha256")
        evidence = load_evidence(evidence_path, record["source_sha256"], shape, expected_content_hash=evidence_content)
        region = next((item for item in evidence.regions if item.region_id == binding["region_id"]), None)
        if (
            region is None
            or region.provenance != "inferred"
            or region.score_type != SEMANTIC_SCORE_TYPE
            or evidence.calibration is not None
            or region.calibration_sha256 is not None
            or region.label != record["predicted_label"]
            or region.semantic_confidence != record["confidence"]
        ):
            raise MaterialsError("Fitting scores must match an original uncalibrated inference region")
        current_recipe = _recipe({key: evidence.producer.get(key) for key in RECIPE_KEYS})
        if recipe is not None and recipe != current_recipe:
            raise MaterialsError("Mixed inference recipes cannot share a calibration artifact")
        recipe = current_recipe
        # Repackaging or renaming a region cannot create an independent observation.
        evidence_key = (record["source_sha256"], _digest(current_recipe), hashlib.sha256(region.mask.tobytes()).hexdigest())
        if evidence_key in observed_regions:
            raise MaterialsError("Duplicate inference region cannot increase calibration support")
        observed_regions.add(evidence_key)
        annotation_root = directory_path((data_root / _relative_path(binding["annotation_path"])).parent)
        mask_data, mask_file = _read(annotation_root, annotation["mask_path"], MaterialLimits().max_mask_bytes)
        if mask_file["sha256"] != validate_digest(annotation["mask_sha256"], "mask_sha256"):
            raise MaterialsError("Annotation mask bytes differ from their declaration")
        mask = _mask_array(mask_data, shape, MaterialLimits())
        if not np.any(mask) or np.any((mask != 0) & (mask != 1)):
            raise MaterialsError("Reviewed annotation requires a nonempty binary source-frame mask")
        predicted = region.mask > 0
        intersection, union = int(np.count_nonzero(predicted & (mask > 0))), int(np.count_nonzero(predicted | (mask > 0)))
        predicted_pixels = int(np.count_nonzero(predicted))
        if predicted_pixels == 0 or intersection / predicted_pixels < 0.5:
            raise MaterialsError("Reviewed annotation must cover at least half of its predicted region")
        provenance.append(
            {
                "record_id": record["record_id"],
                "source_sha256": record["source_sha256"],
                "evidence_sha256": evidence_file["sha256"],
                "annotation_sha256": annotation_file["sha256"],
                "mask_sha256": mask_file["sha256"],
                "annotator": annotation["annotator"],
                "reviewer": annotation["reviewer"],
                "mask_iou": intersection / union,
                "review_scope": "declared_identities_and_file_bindings_only",
            }
        )
        if region.label not in {"unknown", "mixed"} and region.semantic_confidence is not None:
            pairs[region.label].append((region.semantic_confidence, region.label == record["true_label"]))
            properties[region.label].add(record["property_group"])
    classes, excluded = {}, {}
    minimum = max(5, settings["min_class_support"])
    for label, observations in sorted(pairs.items()):
        if (
            len(observations) < minimum
            or len({score for score, _ in observations}) < 2
            or len({correct for _, correct in observations}) < 2
        ):
            excluded[label] = "requires_minimum_support_distinct_scores_and_correct_and_incorrect_examples"
            continue
        classes[label] = {
            "knots": _isotonic(observations),
            "support_records": len(observations),
            "property_groups": sorted(properties[label]),
        }
    if not classes or recipe is None:
        raise MaterialsError("No class has adequate calibration fitting support; no artifact was produced")
    split_receipt = {
        "schema": "tp.materials.dataset_split.v1",
        "membership_sha256": _digest(membership),
        "selection_sources": selected,
        "sealed_test_used_for_selection": False,
        "scope": "observed_sources_with_declared_property_membership",
    }
    split_receipt["content_sha256"] = _digest(split_receipt)
    fit = {
        "schema": "tp.materials.calibration_fit.v1",
        "method": METHOD,
        "recipe": recipe,
        "split_receipt": split_receipt,
        "dataset_sha256": _digest(descriptors),
        "annotation_provenance": provenance,
        "classes": classes,
        "excluded_classes": excluded,
        "scope": "semantic_top_label_correctness_only",
        "acceptance": "not_assessed",
    }
    receipt = CalibrationReceipt(
        **recipe,
        split_sha256=split_receipt["content_sha256"],
        artifact_sha256=_digest(fit),
        method=METHOD,
        classes=tuple(classes),
        metrics={"fit_records": len(records), "held_out_acceptance": "not_assessed", "review_status": "unreviewed"},
    )
    return CalibrationModel(fit, receipt)
