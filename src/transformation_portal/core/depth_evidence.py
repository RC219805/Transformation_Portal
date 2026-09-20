"""Immutable DA3 depth evidence with separate numeric, image, and surface support.

Native API samples are never replaced by calibrated or photographic derivatives.
Unknown sky evidence disables surface use; finite numbers alone are insufficient
to authorize either photographic editing or metric surface interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .depth_artifact import CameraIntrinsics
from .image_artifact import (
    ImageProxy,
    ProxyTransform,
    artifact_content_hash,
    freeze_metadata,
    immutable_array,
    metadata_payload,
    validate_source_sha256,
)

DEPTH_EVIDENCE_SCHEMA = "tp.depth.artifact.v3"
_PRECISIONS = frozenset({"fp32", "fp16"})


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must be a boolean raster on the native grid")
    return immutable_array(array, np.bool_)


@dataclass(frozen=True)
class DepthEvidence:
    """Source-bound native output and independently described usable geometry."""

    native_depth: np.ndarray
    numeric_valid: np.ndarray
    support_mask: np.ndarray
    sky_mask: np.ndarray | None
    valid_mask: np.ndarray
    transform: ProxyTransform
    source_sha256: str
    precision: str = "fp32"
    metric_map_m: np.ndarray | None = None
    intrinsics: CameraIntrinsics | None = None
    calibration: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_source_sha256(self.source_sha256)
        if not isinstance(self.transform, ProxyTransform) or self.precision not in _PRECISIONS:
            raise ValueError("Depth evidence requires explicit proxy geometry and supported compute precision")
        native = np.asarray(self.native_depth)
        if native.dtype != np.float32 or native.shape != self.transform.padded_shape:
            raise ValueError("Native API depth must retain its float32 raster on the exact proxy grid")
        native = immutable_array(native, np.float32)
        numeric = _mask(self.numeric_valid, native.shape, "numeric_valid")
        support = _mask(self.support_mask, native.shape, "support_mask")
        usable = _mask(self.valid_mask, native.shape, "valid_mask")
        expected_support = np.zeros(native.shape, dtype=bool)
        height, width = self.transform.resized_shape
        expected_support[:height, :width] = True
        if not np.array_equal(numeric, np.isfinite(native) & (native > 0)):
            raise ValueError("DA3 numeric validity must mean finite positive native output")
        if not np.array_equal(support, expected_support):
            raise ValueError("Image support must match the declared unpadded proxy")
        sky = None if self.sky_mask is None else _mask(self.sky_mask, native.shape, "sky_mask")
        expected_usable = np.zeros(native.shape, dtype=bool) if sky is None else numeric & support & ~sky
        if not np.array_equal(usable, expected_usable):
            raise ValueError("Usable surface validity must exclude unknown sky, invalid numbers, and padding")
        for name, value in (
            ("native_depth", native),
            ("numeric_valid", numeric),
            ("support_mask", support),
            ("sky_mask", sky),
            ("valid_mask", usable),
        ):
            object.__setattr__(self, name, value)
        if self.metric_map_m is None:
            if self.intrinsics is not None or self.calibration is not None:
                raise ValueError("Camera calibration must accompany its metric derivative")
        else:
            if not isinstance(self.intrinsics, CameraIntrinsics) or not isinstance(self.calibration, Mapping):
                raise ValueError("Metric derivatives require effective intrinsics and explicit calibration")
            if (self.intrinsics.height, self.intrinsics.width) != native.shape:
                raise ValueError("Effective intrinsics must describe the native grid")
            if self.calibration.get("method") != "da3_metric_focal_pixels_div_300":
                raise ValueError("Unsupported DA3 calibration recipe")
            expected_intrinsics, expected_calibration = _camera_calibration(
                self.calibration.get("input_intrinsics"), self.transform, self.source_sha256
            )
            if self.intrinsics != expected_intrinsics or metadata_payload(self.calibration) != expected_calibration:
                raise ValueError("Calibration receipt must match its independently transformed camera input")
            metric = np.asarray(self.metric_map_m)
            if metric.dtype != np.float32 or metric.shape != native.shape:
                raise ValueError("Metric derivatives must be float32 on the native grid")
            expected = np.zeros(native.shape, np.float32)
            factor = (self.intrinsics.fx + self.intrinsics.fy) / 600.0
            with np.errstate(over="ignore", invalid="ignore"):
                expected[numeric] = native[numeric] * factor
            if not np.isfinite(expected).all() or np.any(expected[numeric] <= 0) or not np.array_equal(metric, expected):
                raise ValueError("Metric samples must match finite native focal calibration exactly")
            object.__setattr__(self, "metric_map_m", immutable_array(metric, np.float32))
            object.__setattr__(self, "calibration", freeze_metadata(self.calibration))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Depth evidence metadata must be a mapping")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def shape(self) -> tuple[int, int]:
        return self.native_depth.shape

    @property
    def confidence(self) -> None:
        """The governed metric model supplies no contracted accuracy confidence."""
        return None

    def _relative_limits(self) -> tuple[float, float] | None:
        if not self.valid_mask.any():
            return None
        low, high = np.percentile(self.native_depth[self.valid_mask], [1, 99])
        return float(low), float(high)

    def relative_depth(self) -> np.ndarray:
        """Near=0/far=1, normalized only over usable, in-frame surface samples."""
        result = np.zeros(self.shape, dtype=np.float32)
        limits = self._relative_limits()
        if limits is not None and limits[1] > limits[0]:
            result[self.valid_mask] = np.clip((self.native_depth[self.valid_mask] - limits[0]) / (limits[1] - limits[0]), 0, 1)
        return immutable_array(result, np.float32)

    def to_payload(self) -> dict[str, Any]:
        limits = self._relative_limits()
        return {
            "schema": DEPTH_EVIDENCE_SCHEMA,
            "source_sha256": self.source_sha256,
            "shape": list(self.shape),
            "native_semantics": "da3_metric_uncalibrated",
            "native_representation": "api_output_after_model_postprocessing",
            "numeric_domain": "finite_positive",
            "direction": "near_small_far_large",
            "geometry": self.transform.to_payload(),
            "sky_status": "unavailable" if self.sky_mask is None else "model_mask",
            "surface_policy": "require_known_non_sky_and_numeric_and_in_frame",
            "confidence_status": "unavailable",
            "precision": {
                "compute": self.precision,
                "weights": "float32",
                "depth_head": "float32",
                "storage": "float32",
                "autocast": {"enabled": self.precision == "fp16", "dtype": "float16" if self.precision == "fp16" else None},
            },
            "has_metric_depth": self.metric_map_m is not None,
            "metric_status": "unavailable" if self.metric_map_m is None else "inferred_with_supplied_camera_calibration",
            "intrinsics": None if self.intrinsics is None else self.intrinsics.to_payload(),
            "calibration": metadata_payload(self.calibration),
            "relative_derivative": {
                "method": "usable_in_frame_percentile_1_99",
                "status": "unavailable" if limits is None else "constant" if limits[1] <= limits[0] else "available",
                "low": None if limits is None else limits[0],
                "high": None if limits is None else limits[1],
                "invalid_value": 0,
            },
            "sample_counts": {
                "numeric_valid": int(self.numeric_valid.sum()),
                "in_frame": int(self.support_mask.sum()),
                "usable_surface": int(self.valid_mask.sum()),
                "sky_in_frame": None if self.sky_mask is None else int((self.sky_mask & self.support_mask).sum()),
            },
            "metadata": metadata_payload(self.metadata),
        }

    def content_hash(self) -> str:
        return artifact_content_hash(
            self.to_payload(),
            {
                "native_depth": self.native_depth,
                "numeric_valid": self.numeric_valid,
                "support_mask": self.support_mask,
                "sky_mask": self.sky_mask,
                "valid_mask": self.valid_mask,
                "metric_map_m": self.metric_map_m,
            },
        )


def _camera_calibration(
    supplied: Any, transform: ProxyTransform, source_sha256: str
) -> tuple[CameraIntrinsics, dict[str, Any]]:
    from transformation_portal.lux_depth_v4.companions import validate_record

    validate_record({"path": "camera", "source_sha256": source_sha256, "calibration": supplied})
    original_height, original_width = transform.original_shape
    if (supplied["height"], supplied["width"]) != (original_height, original_width):
        raise ValueError("Supplied camera geometry differs from the canonical master")
    height, width = transform.resized_shape
    scale_x, scale_y = width / original_width, height / original_height
    intrinsics = CameraIntrinsics(
        fx=float(supplied["fx"]) * scale_x,
        fy=float(supplied["fy"]) * scale_y,
        cx=(float(supplied["cx"]) + 0.5) * scale_x - 0.5,
        cy=(float(supplied["cy"]) + 0.5) * scale_y - 0.5,
        width=transform.padded_shape[1],
        height=transform.padded_shape[0],
        source=supplied["source"],
    )
    return intrinsics, {
        "method": "da3_metric_focal_pixels_div_300",
        "input_intrinsics": metadata_payload(supplied),
        "meters_per_native_unit": (intrinsics.fx + intrinsics.fy) / 600.0,
        "measurement_status": "model_inference_not_measured_scene_distance",
    }


def build_depth_evidence(
    native: np.ndarray,
    sky: np.ndarray | None,
    proxy: ImageProxy,
    source_sha256: str,
    *,
    companion: Mapping[str, Any] | None = None,
    precision: str = "fp32",
) -> DepthEvidence:
    """Reconstruct DA3 evidence from raw arrays and independently bound inputs."""
    validate_source_sha256(source_sha256)
    if not isinstance(proxy, ImageProxy):
        raise ValueError("Depth evidence requires an immutable photographic proxy")
    native = np.asarray(native)
    if native.dtype != np.float32 or native.shape != proxy.transform.padded_shape:
        raise ValueError("Native API depth must be float32 on the prepared proxy grid")
    numeric = np.isfinite(native) & (native > 0)
    support = np.zeros(native.shape, dtype=bool)
    height, width = proxy.transform.resized_shape
    support[:height, :width] = True
    sky = None if sky is None else _mask(sky, native.shape, "sky_mask")
    usable = np.zeros_like(support) if sky is None else numeric & support & ~sky
    intrinsics = metric = calibration = None
    if companion is not None:
        # Reuse the existing strict, source-bound companion input contract. Its
        # V4 finite-only output mask is deliberately not reused here.
        from transformation_portal.lux_depth_v4.companions import validate_record

        validate_record(companion)
        if companion["source_sha256"] != source_sha256:
            raise ValueError("Calibration source differs from the photographic source")
        supplied = companion.get("calibration")
        if supplied is not None:
            intrinsics, calibration = _camera_calibration(supplied, proxy.transform, source_sha256)
            factor = (intrinsics.fx + intrinsics.fy) / 600.0
            metric = np.zeros(native.shape, np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                metric[numeric] = native[numeric] * factor
    return DepthEvidence(
        native,
        numeric,
        support,
        sky,
        usable,
        proxy.transform,
        source_sha256,
        precision,
        metric,
        intrinsics,
        calibration,
        {
            "proxy_master_content_hash": proxy.master_content_hash,
            "proxy_content_hash": artifact_content_hash(
                {"transform": proxy.transform.to_payload(), "master_content_hash": proxy.master_content_hash},
                {"pixels": proxy.pixels},
            ),
        },
    )
