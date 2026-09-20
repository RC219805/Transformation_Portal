"""Versioned, immutable depth semantics for photographic pipeline stages.

This successor does not change the public Lux V3 DepthArtifact contract or the
backend DepthResult protocol. Native model output is distinct from calibrated
meters and from a relative display/finishing derivative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .image_artifact import artifact_content_hash, freeze_metadata, immutable_array, metadata_payload, validate_source_sha256

DEPTH_ARTIFACT_SCHEMA = "tp.depth.artifact.v2"
NATIVE_DEPTH_SEMANTICS = frozenset(
    {"relative_distance", "relative_inverse_depth", "da3_metric_uncalibrated", "metric_distance_m"}
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Effective pinhole intrinsics on the artifact grid, with provenance."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Intrinsics require finite values and positive focal lengths")
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (self.width, self.height)):
            raise ValueError("Intrinsics require positive integer dimensions")
        if not isinstance(self.source, str) or not self.source.strip() or self.source == "estimated":
            raise ValueError("Metric calibration requires explicit intrinsics provenance")

    def to_payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ("fx", "fy", "cx", "cy", "width", "height", "source")}


@dataclass(frozen=True)
class DepthArtifact:
    """Native depth and optional measured/calibrated evidence on one grid."""

    native_depth: np.ndarray
    native_semantics: str
    valid_mask: np.ndarray
    source_sha256: str
    metric_map_m: np.ndarray | None = None
    confidence: np.ndarray | None = None
    intrinsics: CameraIntrinsics | None = None
    calibration: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_source_sha256(self.source_sha256)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Depth metadata must be a mapping")
        if self.native_semantics not in NATIVE_DEPTH_SEMANTICS:
            raise ValueError("Unsupported native depth semantics")
        native = immutable_array(self.native_depth, np.float32)
        mask = np.asarray(self.valid_mask)
        if native.ndim != 2 or min(native.shape) <= 0 or mask.dtype != np.bool_ or mask.shape != native.shape:
            raise ValueError("Depth requires a non-empty HW raster and matching boolean validity mask")
        mask = immutable_array(mask, np.bool_)
        if not mask.any() or not np.isfinite(native[mask]).all():
            raise ValueError("Depth must contain finite valid samples")
        object.__setattr__(self, "native_depth", native)
        object.__setattr__(self, "valid_mask", mask)
        for name in ("metric_map_m", "confidence"):
            value = getattr(self, name)
            if value is None:
                continue
            array = immutable_array(value, np.float32)
            if array.shape != native.shape or not np.isfinite(array[mask]).all():
                raise ValueError(f"{name} must match depth geometry and be finite on valid samples")
            if name == "metric_map_m" and np.any(array[mask] <= 0):
                raise ValueError("Calibrated metric depth must be positive")
            if name == "confidence" and np.any((array[mask] < 0) | (array[mask] > 1)):
                raise ValueError("Confidence must be in [0,1]")
            object.__setattr__(self, name, array)
        if self.intrinsics is not None:
            if not isinstance(self.intrinsics, CameraIntrinsics):
                raise ValueError("intrinsics must be CameraIntrinsics")
            if (self.intrinsics.height, self.intrinsics.width) != native.shape:
                raise ValueError("Effective intrinsics must match the depth grid")
        if self.metric_map_m is not None:
            if self.intrinsics is None or not isinstance(self.calibration, Mapping) or not self.calibration.get("method"):
                raise ValueError("Metric maps require effective intrinsics and an explicit calibration method")
        elif self.calibration is not None:
            raise ValueError("Calibration must accompany a metric map")
        if self.native_semantics == "metric_distance_m" and self.metric_map_m is None:
            raise ValueError("Native metric semantics require calibrated metric evidence")
        if (
            self.native_semantics == "metric_distance_m"
            and self.metric_map_m is not None
            and not np.array_equal(native[mask], self.metric_map_m[mask])
        ):
            raise ValueError("Native metric values must agree with the calibrated metric map")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        if self.calibration is not None:
            object.__setattr__(self, "calibration", freeze_metadata(self.calibration))

    @property
    def shape(self) -> tuple[int, int]:
        return self.native_depth.shape

    def relative_depth(self) -> np.ndarray:
        """Derive near=0/far=1 depth without replacing native values."""
        values = self.native_depth[self.valid_mask]
        low, high = np.percentile(values, [1, 99])
        result = np.zeros(self.shape, dtype=np.float32)
        if high > low:
            result[self.valid_mask] = np.clip((values - low) / (high - low), 0, 1)
            if self.native_semantics == "relative_inverse_depth":
                result[self.valid_mask] = 1 - result[self.valid_mask]
        return immutable_array(result, np.float32)

    def to_payload(self) -> dict[str, Any]:
        values = self.native_depth[self.valid_mask]
        low, high = np.percentile(values, [1, 99])
        return {
            "schema": DEPTH_ARTIFACT_SCHEMA,
            "native_semantics": self.native_semantics,
            "source_sha256": self.source_sha256,
            "shape": list(self.shape),
            "has_metric_depth": self.metric_map_m is not None,
            "has_confidence": self.confidence is not None,
            "intrinsics": self.intrinsics.to_payload() if self.intrinsics is not None else None,
            "calibration": metadata_payload(self.calibration),
            "relative_derivative": {"method": "valid_percentile_1_99", "low": float(low), "high": float(high)},
            "metadata": metadata_payload(self.metadata),
        }

    def content_hash(self) -> str:
        return artifact_content_hash(
            self.to_payload(),
            {
                "native_depth": self.native_depth,
                "valid_mask": self.valid_mask,
                "metric_map_m": self.metric_map_m,
                "confidence": self.confidence,
            },
        )
