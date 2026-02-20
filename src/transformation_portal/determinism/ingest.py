from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Canonical Phase II implementation lives in spatial_ai/ingest; re-export for backward compat.
from transformation_portal.spatial_ai.ingest.phase2_camera_native_linear import (  # noqa: F401
    BRADFORD_D65_TO_D50_F32,
    _apply_3x3_f32_hwc,
    _rawpy_demosaic,
    ingest_phase2_xyz_d50_linear_fp32,
)

from .cas import sha256_file as _sha256_file
from .tensor import canonicalize_tensor_f32_le_c


def sha256_file(path: Path) -> str:
    """Compatibility wrapper; delegates to CAS helper implementation."""
    return _sha256_file(path)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def probe_subnormals_preserved() -> bool:
    """Best-effort FTZ/DAZ probe.

    Returns True if float32 subnormals appear preserved under basic operations.
    This is not a definitive hardware register read, but it detects common FTZ/DAZ
    configurations in NumPy kernels.
    """
    x = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    if x == np.float32(0.0):
        return False
    y = x * np.float32(1.0)
    z = x + np.float32(0.0)
    # NumPy comparisons produce np.bool_; normalize to Python bool for JSON/JCS callers.
    return bool((y != np.float32(0.0)) and (z != np.float32(0.0)))


def ingest_from_npy(path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    arr = np.load(path, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError("Expected .npy to contain a numpy ndarray")
    # Do not permit implicit dtype promotion.
    if arr.dtype != np.float32:
        raise ValueError(f"Expected float32 tensor, got {arr.dtype}")
    arr = canonicalize_tensor_f32_le_c(arr).astype(np.float32, copy=False)  # normalize endian/order
    fingerprint = {
        "contract": "npy_tensor",
        "input_kind": "npy",
        "input_path": str(path),
        "shape": list(arr.shape),
        "dtype": "float32",
        "order": "C",
    }
    return arr, fingerprint
