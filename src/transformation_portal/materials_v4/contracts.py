"""Immutable, content-bound evidence for Materials V4.

Calibration receipts describe evidence; constructing or loading one never grants
automatic-edit authority. The response policy must explicitly admit its digest.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from transformation_portal.core.image_artifact import metadata_payload
from transformation_portal.ingest.canonical_json import canonicalize_json

from .taxonomy import TAXONOMY_VERSION, canonical_label

EVIDENCE_SCHEMA = "tp.materials.evidence.v1"
CALIBRATION_SCHEMA = "tp.materials.calibration.v1"
_TIMING_KEYS = frozenset({"timing", "timing_ms", "elapsed_seconds", "runtime_seconds", "duration_ms", "duration_seconds"})


class MaterialsError(ValueError):
    """Malformed, unbound, or over-budget Materials V4 evidence."""


def validate_digest(value: Any, name: str = "digest") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MaterialsError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, name: str, *, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or not value.isascii()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise MaterialsError(f"{name} must be bounded, nonempty ASCII text without controls")
    return value


def unit_score(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise MaterialsError(f"{name} must be a finite number in [0,1]")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise MaterialsError(f"{name} must be a finite number in [0,1]") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise MaterialsError(f"{name} must be a finite number in [0,1]")
    return number


def _shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise MaterialsError("Material geometry must be a positive integer HW pair")
    return value[0], value[1]


def _freeze(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        raise MaterialsError("Evidence metadata nesting exceeds budget")
    if isinstance(value, Mapping):
        if len(value) > 128 or not all(isinstance(key, str) and len(key) <= 128 for key in value):
            raise MaterialsError("Evidence metadata keys exceed budget")
        return MappingProxyType(
            {key: _freeze(item, depth=depth + 1) for key, item in value.items() if key not in _TIMING_KEYS}
        )
    if isinstance(value, (tuple, list)):
        if len(value) > 1024:
            raise MaterialsError("Evidence metadata sequence exceeds budget")
        return tuple(_freeze(item, depth=depth + 1) for item in value)
    if isinstance(value, str):
        if len(value) > 4096:
            raise MaterialsError("Evidence metadata string exceeds budget")
        return value
    if value is None or type(value) is bool:
        return value
    if type(value) is int and -(2**63) <= value < 2**63:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise MaterialsError("Evidence metadata must contain bounded finite JSON values")


def _metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterialsError("Evidence metadata must be a mapping")
    frozen = _freeze(value)
    try:
        encoded = canonicalize_json(metadata_payload(frozen))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise MaterialsError("Evidence metadata must be valid UTF-8 JSON") from exc
    if len(encoded) > 65536:
        raise MaterialsError("Evidence metadata exceeds byte budget")
    return frozen


@dataclass(frozen=True)
class MaterialLimits:
    """Admission budgets; larger values require an explicit deployment policy."""

    max_pixels: int = 100_000_000
    max_regions: int = 256
    max_mask_bytes: int = 512_000_000
    max_manifest_bytes: int = 1_048_576
    max_header_bytes: int = 4096
    max_bundle_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        ceilings = {
            "max_pixels": 1_000_000_000,
            "max_regions": 4096,
            "max_mask_bytes": 8_000_000_000,
            "max_manifest_bytes": 4_194_304,
            "max_header_bytes": 65536,
            "max_bundle_bytes": 8_589_934_592,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise MaterialsError(f"{name} must be a positive integer no greater than {ceiling}")


@dataclass(frozen=True)
class CalibrationReceipt:
    """Identity of a calibration procedure, never a declaration of trusted authority."""

    classifier_sha256: str
    proposal_sha256: str
    prompt_sha256: str
    preprocessing_sha256: str
    region_construction_sha256: str
    split_sha256: str
    artifact_sha256: str
    method: str
    taxonomy_sha256: str | None = None
    domain: str = "photography"
    classes: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "classifier_sha256",
            "proposal_sha256",
            "prompt_sha256",
            "preprocessing_sha256",
            "region_construction_sha256",
            "split_sha256",
            "artifact_sha256",
        ):
            validate_digest(getattr(self, name), name)
        if self.taxonomy_sha256 is not None:
            validate_digest(self.taxonomy_sha256, "taxonomy_sha256")
        _text(self.method, "calibration method")
        _text(self.domain, "calibration domain")
        if not isinstance(self.classes, (tuple, list)) or len(self.classes) > 64:
            raise MaterialsError("Calibration classes must be a bounded sequence")
        try:
            labels = tuple(canonical_label(label) for label in self.classes)
        except ValueError as exc:
            raise MaterialsError(str(exc)) from exc
        if len(set(labels)) != len(labels) or "unknown" in labels or "mixed" in labels:
            raise MaterialsError("Calibration classes must be unique known labels")
        object.__setattr__(self, "classes", tuple(sorted(labels)))
        object.__setattr__(self, "metrics", _metadata(self.metrics))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_SCHEMA,
            **{name: metadata_payload(getattr(self, name)) for name in self.__dataclass_fields__},
        }

    def content_hash(self) -> str:
        return hashlib.sha256(canonicalize_json(self.to_payload())).hexdigest()


@dataclass(frozen=True, eq=False)
class RegionEvidence:
    """One source-space region; geometric quality never substitutes for semantics."""

    region_id: str
    label: str
    mask: np.ndarray
    semantic_confidence: float | None = None
    geometric_quality: float | None = None
    provenance: str = "supplied"
    score_type: str | None = None
    calibration_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.region_id) is None:
            raise MaterialsError("region_id must be a bounded portable identifier")
        try:
            object.__setattr__(self, "label", canonical_label(self.label))
        except ValueError as exc:
            raise MaterialsError(str(exc)) from exc
        if not isinstance(self.provenance, str) or self.provenance not in {"supplied", "reviewed", "inferred", "heuristic"}:
            raise MaterialsError("Region provenance is unsupported")
        for name in ("semantic_confidence", "geometric_quality"):
            object.__setattr__(self, name, unit_score(getattr(self, name), name))
        if self.score_type is not None:
            _text(self.score_type, "score_type")
        if self.calibration_sha256 is not None:
            validate_digest(self.calibration_sha256, "calibration_sha256")
        if not isinstance(self.mask, np.ndarray) or self.mask.dtype not in {np.dtype("bool"), np.dtype("float32")}:
            raise MaterialsError("Region mask must be a bool or float32 numpy array")
        shape = _shape(self.mask.shape)
        if math.prod(shape) > MaterialLimits().max_pixels:
            raise MaterialsError("Region mask exceeds pixel budget")
        if not np.isfinite(self.mask).all() or np.any((self.mask < 0) | (self.mask > 1)):
            raise MaterialsError("Region mask must be finite in [0,1]")
        array = np.asarray(self.mask, dtype=np.dtype("<f4"), order="C")
        immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f4")).reshape(shape)
        object.__setattr__(self, "mask", immutable)

    def to_payload(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "label": self.label,
            "mask": {
                "shape": list(self.mask.shape),
                "dtype": "float32",
                "sha256": hashlib.sha256(self.mask.tobytes()).hexdigest(),
            },
            "semantic_confidence": self.semantic_confidence,
            "geometric_quality": self.geometric_quality,
            "provenance": self.provenance,
            "score_type": self.score_type,
            "calibration_sha256": self.calibration_sha256,
        }


@dataclass(frozen=True, eq=False)
class MaterialEvidence:
    """Canonical material observations bound to photographic source bytes."""

    source_sha256: str
    shape: tuple[int, int]
    regions: tuple[RegionEvidence, ...]
    producer: Mapping[str, Any] = field(default_factory=dict)
    calibration: CalibrationReceipt | None = None
    status: str = "available"
    reason: str | None = None

    def __post_init__(self) -> None:
        validate_digest(self.source_sha256, "source_sha256")
        object.__setattr__(self, "shape", _shape(self.shape))
        if not isinstance(self.regions, (tuple, list)) or not all(
            isinstance(region, RegionEvidence) for region in self.regions
        ):
            raise MaterialsError("Evidence regions must contain RegionEvidence values")
        regions = tuple(sorted(self.regions, key=lambda region: region.region_id))
        if len({region.region_id for region in regions}) != len(regions):
            raise MaterialsError("Evidence region IDs must be unique")
        if any(region.mask.shape != self.shape for region in regions):
            raise MaterialsError("Region masks must match canonical source geometry")
        if self.calibration is not None and not isinstance(self.calibration, CalibrationReceipt):
            raise MaterialsError("Evidence calibration must be a CalibrationReceipt")
        receipt_digest = self.calibration.content_hash() if self.calibration is not None else None
        if any(region.calibration_sha256 is not None and region.calibration_sha256 != receipt_digest for region in regions):
            raise MaterialsError("Region calibration binding does not match supplied receipt")
        if not isinstance(self.status, str) or self.status not in {"available", "unavailable", "abstained", "degraded"}:
            raise MaterialsError("Evidence status is unsupported")
        if self.status in {"unavailable", "abstained"} and regions:
            raise MaterialsError("Unavailable or abstained evidence cannot carry actionable regions")
        if self.reason is not None:
            _text(self.reason, "evidence reason", maximum=512)
        if self.status != "available" and self.reason is None:
            raise MaterialsError("Non-available evidence requires an explicit reason")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "producer", _metadata(self.producer))
        self.validate_limits(MaterialLimits())

    def validate_limits(self, limits: MaterialLimits) -> None:
        if not isinstance(limits, MaterialLimits):
            raise MaterialsError("Evidence limits must be MaterialLimits")
        if math.prod(self.shape) > limits.max_pixels or len(self.regions) > limits.max_regions:
            raise MaterialsError("Evidence exceeds pixel or region budget")
        if sum(region.mask.nbytes for region in self.regions) > limits.max_mask_bytes:
            raise MaterialsError("Evidence exceeds aggregate mask byte budget")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "taxonomy": TAXONOMY_VERSION,
            "coordinate_space": "canonical_master",
            "source_sha256": self.source_sha256,
            "shape": list(self.shape),
            "regions": [region.to_payload() for region in self.regions],
            "producer": metadata_payload(self.producer),
            "calibration": self.calibration.to_payload() if self.calibration is not None else None,
            "status": self.status,
            "reason": self.reason,
        }

    def content_hash(self) -> str:
        return hashlib.sha256(canonicalize_json(self.to_payload())).hexdigest()
