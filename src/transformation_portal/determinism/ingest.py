from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .tensor import canonicalize_tensor_f32_le_c

BRADFORD_D65_TO_D50_F32 = np.array(
    [
        [1.0478112, 0.0228866, -0.0501270],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092319, 0.0150436, 0.7521316],
    ],
    dtype=np.float32,
)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _apply_3x3_f32_hwc(vec3: np.ndarray, mat3x3: np.ndarray) -> np.ndarray:
    """Apply 3x3 matrix to HWC vec3 using explicit float32 multiply/add sequence."""
    if vec3.dtype != np.float32:
        raise ValueError("vec3 must be float32")
    if mat3x3.dtype != np.float32 or mat3x3.shape != (3, 3):
        raise ValueError("mat3x3 must be float32 shape (3,3)")

    r = vec3[..., 0]
    g = vec3[..., 1]
    b = vec3[..., 2]

    # X
    x = r * mat3x3[0, 0]
    x = x + (g * mat3x3[0, 1])
    x = x + (b * mat3x3[0, 2])

    # Y
    y = r * mat3x3[1, 0]
    y = y + (g * mat3x3[1, 1])
    y = y + (b * mat3x3[1, 2])

    # Z
    z = r * mat3x3[2, 0]
    z = z + (g * mat3x3[2, 1])
    z = z + (b * mat3x3[2, 2])

    out = np.stack((x, y, z), axis=-1)
    return out.astype(np.float32, copy=False)


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


def _rawpy_demosaic(name: str):
    import rawpy  # type: ignore

    n = name.strip().upper()
    try:
        return getattr(rawpy.DemosaicAlgorithm, n)
    except AttributeError as e:
        raise ValueError(f"Unknown demosaic algorithm: {name}") from e


def ingest_phase2_xyz_d50_linear_fp32(
    path: Path,
    *,
    wb_mode: str = "camera",
    demosaic: str = "AHD",
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Certified Phase II decode: RAW -> camera RGB (linear) -> XYZ(D65) -> Bradford(D50) -> float32 HWC.

    This path intentionally avoids BLAS/GEMM for the 3x3 transforms by using explicit
    multiply/add sequences.
    """
    try:
        import rawpy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "rawpy is required for contract camera_native_linear. " "Install with the project's raw extra."
        ) from e

    wb_mode_n = wb_mode.strip().lower()
    if wb_mode_n not in {"none", "camera", "auto"}:
        raise ValueError("wb_mode must be one of: none, camera, auto")

    with rawpy.imread(str(path)) as raw:
        demosaic_alg = _rawpy_demosaic(demosaic)

        use_camera_wb = wb_mode_n == "camera"
        use_auto_wb = wb_mode_n == "auto"

        rgb16 = raw.postprocess(
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            no_auto_bright=True,
            no_auto_scale=True,
            use_camera_wb=use_camera_wb,
            use_auto_wb=use_auto_wb,
            demosaic_algorithm=demosaic_alg,
            output_bps=16,
            user_flip=0,
        )

        if rgb16.ndim != 3 or rgb16.shape[2] != 3:
            raise ValueError(f"Unexpected RGB output shape: {rgb16.shape}")

        # Normalize to [0, 1] in float32 with explicit float32 scalar.
        scale = np.float32(1.0 / 65535.0)
        rgb = rgb16.astype(np.float32) * scale

        # LibRaw provides camera RGB -> XYZ (D65) matrix.
        # rawpy exposes it as rgb_xyz_matrix; cast to float32.
        rgb_to_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float32)
        if rgb_to_xyz.shape != (3, 3):
            raise ValueError(f"Unexpected rgb_xyz_matrix shape: {rgb_to_xyz.shape}")

        xyz_d65 = _apply_3x3_f32_hwc(rgb, rgb_to_xyz)
        xyz_d50 = _apply_3x3_f32_hwc(xyz_d65, BRADFORD_D65_TO_D50_F32)

        xyz_d50 = canonicalize_tensor_f32_le_c(xyz_d50).astype(np.float32, copy=False)

        camera_wb = None
        try:
            camera_wb = [float(x) for x in raw.camera_whitebalance]
        except Exception:
            camera_wb = None

        fingerprint = {
            "contract": "camera_native_linear",
            "input_kind": "raw",
            "input_path": str(path),
            "wb_mode": wb_mode_n,
            "camera_whitebalance": camera_wb,
            "demosaic": demosaic_alg.name if hasattr(demosaic_alg, "name") else str(demosaic_alg),
            "rgb_xyz_matrix_f32": [[float(x) for x in row] for row in rgb_to_xyz.tolist()],
            "bradford_d65_to_d50_f32": [[float(x) for x in row] for row in BRADFORD_D65_TO_D50_F32.tolist()],
            "shape": list(xyz_d50.shape),
            "dtype": "float32",
            "order": "C",
        }
        return xyz_d50, fingerprint
