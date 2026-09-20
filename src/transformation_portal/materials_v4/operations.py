"""Small, versioned photographic delta kernels in scene-linear sRGB.

Kernels read immutable baseline pixels and return float64 deltas. They do not
own masks, compositing, output encoding, or execution authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from transformation_portal.ingest.canonical_json import canonicalize_json


@dataclass(frozen=True)
class OperationSpec:
    """Numerical meaning and spatial support of a versioned operation."""

    operation_id: str
    radius: int
    max_strength: float
    meaning: str
    color_space: str = "linear_srgb"


OPERATION_SPECS: Mapping[str, OperationSpec] = MappingProxyType(
    {
        "linear_gain_v1": OperationSpec("linear_gain_v1", 0, 0.1, "delta = baseline * strength"),
        "luminance_detail_gain_v1": OperationSpec(
            "luminance_detail_gain_v1", 1, 0.2, "RGB-scaled Rec.709 luminance detail above an edge-padded 3x3 box mean"
        ),
        "luminance_detail_attenuation_v1": OperationSpec(
            "luminance_detail_attenuation_v1",
            1,
            0.2,
            "negative RGB-scaled Rec.709 luminance detail above an edge-padded 3x3 box mean",
        ),
    }
)


def operation_contract_hash() -> str:
    """Bind the immutable operation versions, domains, bounds, and semantics."""
    payload = {
        key: {
            "operation_id": spec.operation_id,
            "radius": spec.radius,
            "max_strength": spec.max_strength,
            "meaning": spec.meaning,
            "color_space": spec.color_space,
        }
        for key, spec in sorted(OPERATION_SPECS.items())
    }
    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


def validate_operation(operation_id: str, strength: float) -> None:
    """Fail closed on unversioned operations or unbounded strengths."""
    if not isinstance(operation_id, str) or operation_id not in OPERATION_SPECS:
        raise ValueError(f"Unsupported Materials V4 operation: {operation_id!r}")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not np.isfinite(strength):
        raise ValueError("Operation strength must be a finite number")
    if not 0 <= strength <= OPERATION_SPECS[operation_id].max_strength:
        raise ValueError("Operation strength is outside the versioned bound")


def operation_delta(pixels: np.ndarray, operation_id: str, strength: float) -> np.ndarray:
    """Return a neutral, unclipped delta from the supplied linear RGB baseline.

    The caller must supply a one-pixel halo for spatial operations and crop it
    after this call. At actual image boundaries, edge replication applies.
    Luminance detail scales all RGB channels together, retaining neutral pixels
    and RGB chromaticity; no operation adds a material-specific color cast.
    """
    validate_operation(operation_id, strength)
    pixels = np.asarray(pixels)
    if pixels.ndim != 3 or pixels.shape[-1] != 3 or min(pixels.shape[:2]) < 1:
        raise ValueError("Operation pixels must be non-empty HxWx3")
    if pixels.dtype.kind != "f" or not np.isfinite(pixels).all():
        raise ValueError("Operation pixels must be finite floating-point linear RGB")
    baseline = pixels.astype(np.float64, copy=False)
    if operation_id == "linear_gain_v1":
        return baseline * float(strength)
    luminance = baseline[..., 0] * 0.2126 + baseline[..., 1] * 0.7152 + baseline[..., 2] * 0.0722
    padded = np.pad(luminance, 1, mode="edge")
    local_mean = np.zeros_like(luminance)
    height, width = luminance.shape
    for dy in range(3):
        for dx in range(3):
            local_mean += padded[dy : dy + height, dx : dx + width]
    local_mean /= 9.0
    ratio = (luminance - local_mean) / np.maximum(np.abs(luminance), 1e-6)
    sign = -1.0 if operation_id == "luminance_detail_attenuation_v1" else 1.0
    return baseline * (ratio * float(strength) * sign)[..., None]
