"""Evidence-bound planning and a single conservative Materials V4 compositor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np

from transformation_portal.core.image_artifact import ImageMaster, validate_source_sha256
from transformation_portal.ingest.canonical_json import canonicalize_json

from .contracts import MaterialEvidence, MaterialsError
from .operations import operation_contract_hash, operation_delta, validate_operation


@dataclass(frozen=True)
class MaterialOperation:
    """An explicit response family, versioned kernel, and bounded strength."""

    label: str
    operation_id: str
    strength: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or self.label not in {"glass", "stone", "water", "foliage", "sky"}:
            raise MaterialsError("Response operations support glass, stone, water, foliage, and sky")
        validate_operation(self.operation_id, self.strength)
        object.__setattr__(self, "strength", float(self.strength))

    def to_payload(self) -> dict[str, Any]:
        return {"label": self.label, "operation_id": self.operation_id, "strength": self.strength}


def _default_operations() -> tuple[MaterialOperation, ...]:
    return (
        MaterialOperation("glass", "linear_gain_v1", 0.015),
        MaterialOperation("water", "linear_gain_v1", 0.01),
        MaterialOperation("stone", "luminance_detail_gain_v1", 0.03),
        MaterialOperation("foliage", "luminance_detail_gain_v1", 0.02),
        MaterialOperation("sky", "luminance_detail_attenuation_v1", 0.02),
    )


@dataclass(frozen=True)
class ResponsePolicy:
    """Complete editing, confidence, conflict, color, and resource policy.

    Any positive uncertain or overlapping support protects that pixel. Positive
    alpha is an editing weight, whereas ``support_threshold`` only measures the
    core area needed for eligibility. HDR and fully transparent pixels are
    protected, and output values are never clipped to the display range.
    """

    operations: tuple[MaterialOperation, ...] = field(default_factory=_default_operations)
    min_confidence: float = 0.8
    min_coverage_px: int = 500
    support_threshold: float = 0.5
    allow_supplied_confidence: bool = True
    trusted_calibration_sha256: tuple[str, ...] = ()
    color_space: str = "linear_srgb"
    conflict_rule: str = "protect_all_positive_overlaps"
    max_abs_delta: float = 0.025
    tile_size: int = 512
    halo: int = 1
    max_working_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not all(isinstance(item, MaterialOperation) for item in operations):
            raise MaterialsError("Policy operations must be MaterialOperation values")
        operations = tuple(sorted(operations, key=lambda item: item.label))
        if len({item.label for item in operations}) != len(operations):
            raise MaterialsError("Policy may contain only one operation per material")
        object.__setattr__(self, "operations", operations)
        for name in ("min_confidence", "support_threshold", "max_abs_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise MaterialsError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0 <= self.min_confidence <= 1 or not 0 < self.support_threshold <= 1:
            raise MaterialsError("Confidence and support thresholds are outside [0,1]")
        if not 0 <= self.max_abs_delta <= 0.25:
            raise MaterialsError("max_abs_delta must be in [0,0.25]")
        for name in ("min_coverage_px", "tile_size", "halo", "max_working_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MaterialsError(f"{name} must be a positive integer")
        if self.halo != 1 or self.tile_size > 4096:
            raise MaterialsError("V1 requires a one-pixel halo and tile_size <= 4096")
        if not isinstance(self.allow_supplied_confidence, bool):
            raise MaterialsError("allow_supplied_confidence must be boolean")
        if self.color_space != "linear_srgb" or self.conflict_rule != "protect_all_positive_overlaps":
            raise MaterialsError("Unsupported color or conflict policy")
        trusted = tuple(sorted(set(self.trusted_calibration_sha256)))
        for digest in trusted:
            validate_source_sha256(digest)
        object.__setattr__(self, "trusted_calibration_sha256", trusted)

    def to_payload(self) -> dict[str, Any]:
        return {
            "operations": [item.to_payload() for item in self.operations],
            "min_confidence": self.min_confidence,
            "min_coverage_px": self.min_coverage_px,
            "support_threshold": self.support_threshold,
            "allow_supplied_confidence": self.allow_supplied_confidence,
            "trusted_calibration_sha256": list(self.trusted_calibration_sha256),
            "color_space": self.color_space,
            "conflict_rule": self.conflict_rule,
            "max_abs_delta": self.max_abs_delta,
            "tile_size": self.tile_size,
            "halo": self.halo,
            "max_working_bytes": self.max_working_bytes,
            "protected_master_samples": "outside_unit_rgb_or_zero_alpha",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ResponsePolicy:
        """Parse the complete frozen policy; no missing fields or silent defaults."""
        expected = set(cls().to_payload())
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise MaterialsError("Response policy requires its exact complete schema")
        if payload["protected_master_samples"] != "outside_unit_rgb_or_zero_alpha":
            raise MaterialsError("Unsupported master sample protection policy")
        raw_operations = payload["operations"]
        if not isinstance(raw_operations, list):
            raise MaterialsError("Policy operations must be a JSON array")
        operations = []
        for item in raw_operations:
            if not isinstance(item, Mapping) or set(item) != {"label", "operation_id", "strength"}:
                raise MaterialsError("Operation policy requires label, operation_id, and strength")
            operations.append(MaterialOperation(**dict(item)))
        if not isinstance(payload["trusted_calibration_sha256"], list):
            raise MaterialsError("Trusted calibration identities must be a JSON array")
        kwargs = dict(payload)
        del kwargs["protected_master_samples"]
        kwargs["operations"] = tuple(operations)
        kwargs["trusted_calibration_sha256"] = tuple(payload["trusted_calibration_sha256"])
        return cls(**kwargs)


@dataclass(frozen=True)
class RegionResponse:
    region_id: str
    label: str
    status: str
    reason: str
    coverage_px: int
    resolved_coverage_px: int
    operation_id: str | None = None
    strength: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class PreparedResponse:
    """Immutable proposed response, bound to exact evidence and baseline bytes."""

    source_sha256: str
    master_sha256: str
    evidence_sha256: str
    operations_sha256: str
    policy: ResponsePolicy
    regions: tuple[RegionResponse, ...]

    def __post_init__(self) -> None:
        for digest in (self.source_sha256, self.master_sha256, self.evidence_sha256, self.operations_sha256):
            validate_source_sha256(digest)
        if not isinstance(self.policy, ResponsePolicy):
            raise MaterialsError("Prepared response requires a ResponsePolicy")
        regions = tuple(self.regions)
        if not all(isinstance(item, RegionResponse) for item in regions):
            raise MaterialsError("Prepared response contains invalid region decisions")
        object.__setattr__(self, "regions", regions)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.materials.response_plan.v1",
            "source_sha256": self.source_sha256,
            "master_sha256": self.master_sha256,
            "evidence_sha256": self.evidence_sha256,
            "operations_sha256": self.operations_sha256,
            "policy": self.policy.to_payload(),
            "regions": [region.to_payload() for region in self.regions],
        }

    def content_hash(self) -> str:
        return hashlib.sha256(canonicalize_json(self.to_payload())).hexdigest()


def _tiles(shape: tuple[int, int], size: int) -> Iterator[tuple[slice, slice]]:
    for y in range(0, shape[0], size):
        for x in range(0, shape[1], size):
            yield slice(y, min(shape[0], y + size)), slice(x, min(shape[1], x + size))


def _validate_inputs(master: ImageMaster, evidence: MaterialEvidence, policy: ResponsePolicy) -> None:
    if not isinstance(master, ImageMaster) or not isinstance(evidence, MaterialEvidence):
        raise MaterialsError("Materials response requires ImageMaster and MaterialEvidence")
    if not isinstance(policy, ResponsePolicy):
        raise MaterialsError("Materials response requires a ResponsePolicy")
    if master.source_sha256 != evidence.source_sha256 or master.shape != tuple(evidence.shape):
        raise MaterialsError("Material evidence source or geometry does not match the master")
    # Conservative bound on concurrent numerical scratch; evidence and immutable
    # master/output storage are separately owned artifacts, not tile scratch.
    height = min(master.shape[0], policy.tile_size + 2 * policy.halo)
    width = min(master.shape[1], policy.tile_size + 2 * policy.halo)
    if height * width * 256 > policy.max_working_bytes:
        raise MaterialsError("Tile scratch exceeds max_working_bytes; choose a smaller tile")


def _reason(region: Any, evidence: MaterialEvidence, policy: ResponsePolicy, operations: dict) -> str | None:
    if evidence.status != "available":
        return "evidence_unavailable"
    if region.label not in operations:
        return "unsupported_material"
    if region.semantic_confidence is None:
        return "missing_semantic_confidence"
    if region.semantic_confidence < policy.min_confidence:
        return "below_semantic_confidence"
    if region.provenance == "supplied":
        return None if policy.allow_supplied_confidence else "supplied_confidence_disabled"
    if region.provenance != "inferred":
        return "unsupported_provenance"
    receipt = evidence.calibration
    if receipt is None:
        return "untrusted_inference"
    calibration_hash = receipt.content_hash()
    if (
        calibration_hash not in policy.trusted_calibration_sha256
        or region.calibration_sha256 != calibration_hash
        or region.score_type != "calibrated_material_probability_v1"
        or receipt.domain != "photography"
        or region.label not in receipt.classes
    ):
        return "untrusted_inference"
    recipe_fields = (
        "classifier_sha256",
        "proposal_sha256",
        "prompt_sha256",
        "preprocessing_sha256",
        "region_construction_sha256",
    )
    if any(evidence.producer.get(name) != getattr(receipt, name) for name in recipe_fields):
        return "calibration_producer_mismatch"
    if receipt.taxonomy_sha256 is not None and evidence.producer.get("taxonomy_sha256") != receipt.taxonomy_sha256:
        return "calibration_producer_mismatch"
    return None


def _owners(master: ImageMaster, evidence: MaterialEvidence, eligible: tuple[bool, ...], ys: slice, xs: slice) -> np.ndarray:
    baseline = master.pixels[ys, xs]
    shape = baseline.shape[:2]
    owner = np.full(shape, -1, dtype=np.int32)
    occupied = np.zeros(shape, dtype=bool)
    protected = np.any((baseline < 0) | (baseline > 1), axis=2)
    if master.alpha is not None:
        protected |= master.alpha[ys, xs] == 0
    for index, region in enumerate(evidence.regions):
        positive = region.mask[ys, xs] > 0
        if eligible[index]:
            protected |= positive & occupied
            owner[positive] = index
            occupied |= positive
        else:
            protected |= positive
    owner[protected] = -1
    return owner


def plan_response(master: ImageMaster, evidence: MaterialEvidence, policy: ResponsePolicy | None = None) -> PreparedResponse:
    """Validate evidence, abstain before conflicts, then measure resolved support."""
    policy = ResponsePolicy() if policy is None else policy
    _validate_inputs(master, evidence, policy)
    operations = {operation.label: operation for operation in policy.operations}
    coverage = [0] * len(evidence.regions)
    for ys, xs in _tiles(master.shape, policy.tile_size):
        for index, region in enumerate(evidence.regions):
            coverage[index] += int(np.count_nonzero(region.mask[ys, xs] >= policy.support_threshold))
    reasons = [_reason(region, evidence, policy, operations) for region in evidence.regions]
    for index, reason in enumerate(reasons):
        if reason is None and coverage[index] < policy.min_coverage_px:
            reasons[index] = "below_coverage_threshold"
    eligible = tuple(reason is None for reason in reasons)
    resolved = [0] * len(evidence.regions)
    for ys, xs in _tiles(master.shape, policy.tile_size):
        owner = _owners(master, evidence, eligible, ys, xs)
        for index, region in enumerate(evidence.regions):
            if eligible[index]:
                resolved[index] += int(np.count_nonzero((owner == index) & (region.mask[ys, xs] >= policy.support_threshold)))
    decisions = []
    for index, region in enumerate(evidence.regions):
        reason = reasons[index]
        if reason is None and resolved[index] < policy.min_coverage_px:
            reason = "below_resolved_coverage_threshold"
        operation = operations.get(region.label)
        if reason is None and operation is None:
            raise MaterialsError("Eligible material has no response operation")
        decisions.append(
            RegionResponse(
                region.region_id,
                region.label,
                "eligible" if reason is None else "abstained",
                reason or "evidence_accepted",
                coverage[index],
                resolved[index],
                operation.operation_id if reason is None and operation is not None else None,
                operation.strength if reason is None and operation is not None else None,
            )
        )
    return PreparedResponse(
        master.source_sha256,
        master.content_hash(),
        evidence.content_hash(),
        operation_contract_hash(),
        policy,
        tuple(decisions),
    )


def _bounded_candidate(baseline: np.ndarray, delta: np.ndarray, maximum: float) -> np.ndarray:
    candidate = (baseline.astype(np.float64) + np.clip(delta, -maximum, maximum)).astype(np.float32)
    # Float32 rounding must not move a final stored sample past the hard bound.
    outside = np.abs(candidate.astype(np.float64) - baseline.astype(np.float64)) > maximum
    candidate[outside] = np.nextafter(candidate[outside], baseline[outside])
    return candidate


def apply_response(
    master: ImageMaster, evidence: MaterialEvidence, plan: PreparedResponse
) -> tuple[ImageMaster, dict[str, Any]]:
    """Execute only an exact current plan; composite once and measure real pixels."""
    if not isinstance(plan, PreparedResponse):
        raise MaterialsError("Materials execution requires a PreparedResponse")
    current = plan_response(master, evidence, plan.policy)
    if current.content_hash() != plan.content_hash():
        raise MaterialsError("Stale or invalid materials response plan; replan from current evidence")
    eligible = tuple(region.status == "eligible" for region in plan.regions)
    output = master.pixels.copy() if any(eligible) else None
    changed_pixels = 0
    maximum_delta = 0.0
    protected_changed_pixels = 0
    protected_max_abs_delta = 0.0
    sum_abs_delta = 0.0
    regions_changed = [0] * len(plan.regions)
    for ys, xs in _tiles(master.shape, plan.policy.tile_size):
        if output is None:
            break
        owner = _owners(master, evidence, eligible, ys, xs)
        y0, x0 = max(0, ys.start - plan.policy.halo), max(0, xs.start - plan.policy.halo)
        y1, x1 = min(master.shape[0], ys.stop + plan.policy.halo), min(master.shape[1], xs.stop + plan.policy.halo)
        extended = master.pixels[y0:y1, x0:x1]
        crop = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
        baseline = master.pixels[ys, xs]
        for index, decision in enumerate(plan.regions):
            assigned = owner == index
            if not eligible[index] or not assigned.any():
                continue
            if decision.operation_id is None or decision.strength is None:
                raise MaterialsError("Eligible material has an incomplete response operation")
            delta = operation_delta(extended, decision.operation_id, decision.strength)[crop]
            delta *= evidence.regions[index].mask[ys, xs, None]
            candidate = _bounded_candidate(baseline, delta, plan.policy.max_abs_delta)
            output[ys, xs][assigned] = candidate[assigned]
            regions_changed[index] += int(np.count_nonzero(assigned & np.any(candidate != baseline, axis=2)))
        actual = np.abs(output[ys, xs].astype(np.float64) - baseline.astype(np.float64))
        changed_pixels += int(np.count_nonzero(np.any(actual != 0, axis=2)))
        maximum_delta = max(maximum_delta, float(actual.max(initial=0)))
        protected_actual = actual[owner < 0]
        protected_changed_pixels += int(np.count_nonzero(np.any(protected_actual != 0, axis=1)))
        protected_max_abs_delta = max(protected_max_abs_delta, float(protected_actual.max(initial=0)))
        sum_abs_delta += float(actual.sum(dtype=np.float64))
    result = master
    if changed_pixels and output is not None:
        result = ImageMaster(
            pixels=output,
            source_sha256=master.source_sha256,
            source_bit_depth=master.source_bit_depth,
            alpha=master.alpha,
            metadata=master.metadata,
            source_icc=master.source_icc,
        )
    receipt = {
        "schema": "tp.materials.execution.v1",
        "plan_sha256": plan.content_hash(),
        "input_master_sha256": plan.master_sha256,
        "output_master_sha256": result.content_hash(),
        "evidence_sha256": plan.evidence_sha256,
        "evidence_status": evidence.status,
        "evidence_reason": evidence.reason,
        "response_plan": plan.to_payload(),
        "operations_sha256": plan.operations_sha256,
        "status": "applied" if changed_pixels else "abstained",
        "reason": "pixels_changed" if changed_pixels else "no_pixel_change",
        "changed_pixels": changed_pixels,
        "max_abs_delta": maximum_delta,
        "protected_changed_pixels": protected_changed_pixels,
        "protected_max_abs_delta": protected_max_abs_delta,
        "mean_abs_delta": sum_abs_delta / master.pixels.size,
        "regions": [
            {**region.to_payload(), "changed_pixels": regions_changed[index]} for index, region in enumerate(plan.regions)
        ],
    }
    return result, receipt
