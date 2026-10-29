"""Versioned, source-bound geometry for the opt-in legacy Lux correction.

This describes the legacy model crop, not a lossless photographic inverse.
Removed source borders remain unsupported instead of acquiring stretched depth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ..core.image_artifact import immutable_array, validate_source_sha256
from ..ingest.canonical_json import dumps_json

LEGACY_GEOMETRY_RECIPE = "legacy_encoded_crop_stretch_v1"
NORMALIZED_GEOMETRY_RECIPE = "normalized_crop_support_v1"
CORRECTED_GEOMETRY_RECIPE = NORMALIZED_GEOMETRY_RECIPE
# Match DA3's historical default budget while making its model-grid resize
# explicit and source-bound in the corrected recipe.
CORRECTED_MODEL_MAX_LONG_EDGE = 504


def validate_geometry_recipe(value: str) -> str:
    if not isinstance(value, str) or value not in {LEGACY_GEOMETRY_RECIPE, NORMALIZED_GEOMETRY_RECIPE}:
        raise ValueError("Unsupported Lux geometry recipe")
    return value


def _shape(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a positive integer pair")
    if any(type(side) is not int or side < 1 for side in value):
        raise ValueError(f"{name} must be a positive integer pair")
    return value[0], value[1]


@dataclass(frozen=True)
class GeometryTransform:
    """Exact normalized source → resized crop/pad model-grid authority."""

    source_sha256: str
    encoded_shape: tuple[int, int]
    orientation: int
    target_size: int | None = None

    def __post_init__(self) -> None:
        validate_source_sha256(self.source_sha256)
        object.__setattr__(self, "encoded_shape", _shape(self.encoded_shape, "encoded_shape"))
        if (
            isinstance(self.orientation, bool)
            or not isinstance(self.orientation, Integral)
            or self.orientation not in range(1, 9)
        ):
            raise ValueError("Unsupported EXIF orientation")
        object.__setattr__(self, "orientation", int(self.orientation))
        if self.target_size is not None and (type(self.target_size) is not int or self.target_size < 1):
            raise ValueError("target_size must be a positive integer or null")
        if self.encoded_shape[0] * self.encoded_shape[1] > 67_108_864 or (
            self.target_size is not None and self.target_size > 67_108_864
        ):
            raise ValueError("Geometry exceeds its decoded pixel budget")
        if min(self.resized_shape) < 1:
            raise ValueError("Target size eliminates a photographic dimension")
        if any(h * w > 67_108_864 for h, w in (self.encoded_shape, self.resized_shape, self.model_shape)):
            raise ValueError("Geometry exceeds its decoded pixel budget")

    @property
    def normalized_shape(self) -> tuple[int, int]:
        return self.encoded_shape[::-1] if self.orientation >= 5 else self.encoded_shape

    @property
    def resized_shape(self) -> tuple[int, int]:
        h, w = self.normalized_shape
        if self.target_size is None:
            return h, w
        size = self.target_size
        return (size, int(w * (size / h))) if h > w else (int(h * (size / w)), size)

    @property
    def model_shape(self) -> tuple[int, int]:
        h, w = self.resized_shape
        return max(14, h // 14 * 14), max(14, w // 14 * 14)

    @property
    def crop_shape(self) -> tuple[int, int]:
        h, w = self.resized_shape
        mh, mw = self.model_shape
        return min(h, mh), min(w, mw)

    @property
    def crop_origin(self) -> tuple[int, int]:
        h, w = self.resized_shape
        ch, cw = self.crop_shape
        return (h - ch) // 2, (w - cw) // 2

    @property
    def padding(self) -> tuple[int, int, int, int]:
        h, w = self.crop_shape
        mh, mw = self.model_shape
        top, left = (mh - h) // 2, (mw - w) // 2
        return top, mh - h - top, left, mw - w - left

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.lux.geometry.v1",
            "recipe": NORMALIZED_GEOMETRY_RECIPE,
            "source_sha256": self.source_sha256,
            "encoded_shape": list(self.encoded_shape),
            "orientation": self.orientation,
            "target_size": self.target_size,
            "normalized_shape": list(self.normalized_shape),
            "resized_shape": list(self.resized_shape),
            "model_shape": list(self.model_shape),
            "crop_origin": list(self.crop_origin),
            "crop_shape": list(self.crop_shape),
            "padding": list(self.padding),
            "resize": "lanczos_pixel_centers",
            "depth_restoration": "bilinear_valid_support_nearest_domain",
            "mask_restoration": "nearest_supported",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GeometryTransform:
        if not isinstance(payload, Mapping):
            raise ValueError("Geometry authority must be an object")
        try:
            transform = cls(payload["source_sha256"], payload["encoded_shape"], payload["orientation"], payload["target_size"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete geometry authority") from exc
        if dumps_json(dict(payload), sort_keys=True, allow_nan=False) != dumps_json(transform.to_payload(), sort_keys=True):
            raise ValueError("Geometry authority disagrees with the versioned transform")
        return transform

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            dumps_json(self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def _canvas(self, model: np.ndarray) -> np.ndarray:
        values = np.asarray(model)
        if values.ndim != 2 or values.shape != self.model_shape:
            raise ValueError("Model array must match the exact recorded geometry grid")
        if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_) or np.iscomplexobj(values):
            raise ValueError("Model array must contain finite real samples")
        if not np.isfinite(values).all():
            raise ValueError("Model array must contain finite real samples")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Model array cannot be represented as finite float32 samples")
        top, _, left, _ = self.padding
        h, w = self.crop_shape
        y, x = self.crop_origin
        canvas = np.zeros(self.resized_shape, dtype=np.float32)
        canvas[y : y + h, x : x + w] = values[top : top + h, left : left + w]
        return canvas

    def _resize(self, values: np.ndarray, resample: Image.Resampling) -> np.ndarray:
        if values.shape == self.normalized_shape:
            return np.array(values, dtype=np.float32, copy=True)
        with Image.fromarray(values.astype(np.float32)) as image:
            return np.asarray(image.resize(self.normalized_shape[::-1], resample), dtype=np.float32)

    def support(self) -> np.ndarray:
        canvas = self._canvas(np.ones(self.model_shape, dtype=np.float32))
        return self._resize(canvas, Image.Resampling.NEAREST) > 0

    def restore_numeric(self, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        canvas = self._canvas(model)
        weights = self._resize(self._canvas(np.ones(self.model_shape, dtype=np.float32)), Image.Resampling.BILINEAR)
        support = self.support() & (weights > 0)
        restored = np.zeros(self.normalized_shape, dtype=np.float32)
        np.divide(self._resize(canvas, Image.Resampling.BILINEAR), weights, out=restored, where=support)
        return restored, support

    def restore_mask(self, model: np.ndarray) -> np.ndarray:
        return np.where(self.support(), self._resize(self._canvas(model), Image.Resampling.NEAREST), 0).astype(np.float32)


@dataclass(frozen=True)
class GeometryPreprocessResult:
    pixels: np.ndarray
    canonical_rgb: np.ndarray
    alpha: np.ndarray | None
    transform: GeometryTransform
    source_bit_depth: int = 8

    def __post_init__(self) -> None:
        if self.source_bit_depth not in {8, 16} or type(self.source_bit_depth) is not int:
            raise ValueError("Unsupported photographic source bit depth")
        if self.pixels.shape != (*self.transform.model_shape, 3) or self.canonical_rgb.shape != (
            *self.transform.normalized_shape,
            3,
        ):
            raise ValueError("Decoded pixels disagree with geometry authority")
        if self.pixels.dtype != np.float32 or not np.isfinite(self.pixels).all() or self.canonical_rgb.dtype != np.uint8:
            raise ValueError("Decoded geometry requires finite float32 model pixels and uint8 photographic proxy pixels")
        if np.any((self.pixels < 0) | (self.pixels > 1)):
            raise ValueError("Decoded model pixels must be bounded in [0,1]")
        if self.alpha is not None and self.alpha.shape != self.transform.normalized_shape:
            raise ValueError("Decoded alpha disagrees with geometry authority")
        if self.alpha is not None and self.alpha.dtype != np.uint8:
            raise ValueError("Decoded alpha proxy must be uint8")
        object.__setattr__(self, "pixels", immutable_array(self.pixels, np.float32))
        object.__setattr__(self, "canonical_rgb", immutable_array(self.canonical_rgb, np.uint8))
        if self.alpha is not None:
            object.__setattr__(self, "alpha", immutable_array(self.alpha, np.uint8))
