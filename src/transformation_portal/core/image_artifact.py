"""Immutable photographic pixels and geometry for Lux Depth V4.

The master is finite, full-resolution linear sRGB. Model proxies never own
the photographic pixels. Array storage is backed by immutable bytes so callers
cannot undo the read-only flag and mutate authorizing content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from transformation_portal.ingest.canonical_json import dumps_json

IMAGE_ARTIFACT_SCHEMA = "tp.image.master.v1"


def freeze_metadata(value: Any) -> Any:
    """Validate JSON metadata and detach it from mutable caller state."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Artifact metadata keys must be strings")
        return MappingProxyType({key: freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("Artifact metadata must contain finite JSON values")


def metadata_payload(value: Any) -> Any:
    """Return a detached JSON-serializable view of immutable metadata."""
    if isinstance(value, Mapping):
        return {key: metadata_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [metadata_payload(item) for item in value]
    return value


def immutable_array(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def validate_source_sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def artifact_content_hash(payload: Mapping[str, Any], arrays: Mapping[str, np.ndarray | None]) -> str:
    """Bind named arrays, their dtype/shape, and semantic metadata."""
    descriptors = {}
    for name, array in arrays.items():
        descriptors[name] = (
            None
            if array is None
            else {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    encoded = dumps_json(
        {"metadata": metadata_payload(payload), "arrays": descriptors},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ImageMaster:
    """Canonical, unbounded linear-sRGB master with separate straight alpha."""

    pixels: np.ndarray
    source_sha256: str
    source_bit_depth: int
    alpha: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_icc: bytes | None = None

    def __post_init__(self) -> None:
        validate_source_sha256(self.source_sha256)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Master metadata must be a mapping")
        if isinstance(self.source_bit_depth, bool) or self.source_bit_depth not in {8, 16, 32}:
            raise ValueError("source_bit_depth must be 8, 16, or 32")
        pixels = immutable_array(self.pixels, np.float32)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
            raise ValueError("Master pixels must be non-empty HxWx3")
        if not np.isfinite(pixels).all():
            raise ValueError("Master pixels must be finite")
        object.__setattr__(self, "pixels", pixels)
        if self.alpha is not None:
            alpha = immutable_array(self.alpha, np.float32)
            if alpha.shape != pixels.shape[:2] or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
                raise ValueError("Master alpha must be finite HxW in [0,1]")
            object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
        if self.source_icc is not None:
            object.__setattr__(self, "source_icc", bytes(self.source_icc))

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape[:2]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": IMAGE_ARTIFACT_SCHEMA,
            "color_space": "linear_srgb",
            "alpha_mode": "straight" if self.alpha is not None else None,
            "shape": list(self.shape),
            "source_sha256": self.source_sha256,
            "source_bit_depth": self.source_bit_depth,
            "source_icc_sha256": hashlib.sha256(self.source_icc).hexdigest() if self.source_icc is not None else None,
            "metadata": metadata_payload(self.metadata),
        }

    def content_hash(self) -> str:
        return artifact_content_hash(self.to_payload(), {"pixels": self.pixels, "alpha": self.alpha})


@dataclass(frozen=True)
class ProxyTransform:
    """Aspect-preserving resize followed by right/bottom edge padding."""

    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    padded_shape: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("original_shape", "resized_shape", "padded_shape"):
            shape = tuple(getattr(self, name))
            if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
                raise ValueError("Proxy dimensions must be positive integer pairs")
            object.__setattr__(self, name, shape)
        if any(r > p for r, p in zip(self.resized_shape, self.padded_shape)):
            raise ValueError("Proxy padding cannot remove pixels")
        if any(p % 14 for p in self.padded_shape):
            raise ValueError("Proxy padded dimensions must be multiples of 14")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.image.proxy_transform.v1",
            "original_shape": list(self.original_shape),
            "resized_shape": list(self.resized_shape),
            "padded_shape": list(self.padded_shape),
            "padding": "bottom_right_edge",
            "resize": "bilinear_pixel_centers",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ProxyTransform:
        expected = {"schema", "original_shape", "resized_shape", "padded_shape", "padding", "resize"}
        if (
            set(payload) != expected
            or payload["schema"] != "tp.image.proxy_transform.v1"
            or payload["padding"] != "bottom_right_edge"
            or payload["resize"] != "bilinear_pixel_centers"
        ):
            raise ValueError("Unsupported proxy transform")
        return cls(tuple(payload["original_shape"]), tuple(payload["resized_shape"]), tuple(payload["padded_shape"]))


@dataclass(frozen=True)
class ImageProxy:
    """Explicitly bounded, encoded model input; never a photographic master."""

    pixels: np.ndarray
    transform: ProxyTransform
    master_content_hash: str

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels)
        if pixels.dtype != np.uint8 or pixels.shape != (*self.transform.padded_shape, 3):
            raise ValueError("Proxy pixels must be uint8 RGB matching the padded geometry")
        validate_source_sha256(self.master_content_hash)
        object.__setattr__(self, "pixels", immutable_array(pixels, np.uint8))
