"""Phase II certified ingest: camera_native_linear -> xyz_d50_linear_fp32.

This is the canonical home for the camera_native_linear ingest contract.
The determinism harness (transformation_portal.determinism.ingest) delegates here.

Constraints (ADR-030, Phase II):
  - FTZ/DAZ must be disabled (fail closed via C-extension register probe)
  - rawpy decode: gamma=(1,1), no_auto_bright, no_auto_scale, output_bps=16
  - float32 discipline throughout; no implicit promotion
  - 3x3 matrix application uses explicit multiply/add (no BLAS/GEMM)
  - Fingerprint is deterministic and schema-versioned
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from transformation_portal.core.raw_runtime import run_raw_worker
from transformation_portal.determinism.fpstate import enforce_ftz_daz_disabled
from transformation_portal.determinism.tensor import canonicalize_tensor_f32_le_c

# Bradford chromatic adaptation: D65 -> D50 (frozen Phase II constant)
BRADFORD_D65_TO_D50_F32: np.ndarray = np.array(
    [
        [1.0478112, 0.0228866, -0.0501270],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092319, 0.0150436, 0.7521316],
    ],
    dtype=np.float32,
)
PHASE2_FINGERPRINT_SCHEMA_VERSION = "1.0.0"


def _apply_3x3_f32_hwc(vec3: np.ndarray, mat3x3: np.ndarray) -> np.ndarray:
    """Apply 3×3 matrix to HWC float32 image using explicit multiply/add.

    Vectorised over H×W via NumPy channel indexing; no BLAS/GEMM path.
    """
    if vec3.dtype != np.float32:
        raise ValueError("vec3 must be float32")
    if vec3.ndim != 3 or vec3.shape[2] != 3:
        raise ValueError(f"vec3 must have HWC shape with 3 channels, got {vec3.shape}")
    if mat3x3.dtype != np.float32 or mat3x3.shape != (3, 3):
        raise ValueError("mat3x3 must be float32 shape (3,3)")

    r, g, b = vec3[..., 0], vec3[..., 1], vec3[..., 2]

    x = r * mat3x3[0, 0]
    x = x + (g * mat3x3[0, 1])
    x = x + (b * mat3x3[0, 2])

    y = r * mat3x3[1, 0]
    y = y + (g * mat3x3[1, 1])
    y = y + (b * mat3x3[1, 2])

    z = r * mat3x3[2, 0]
    z = z + (g * mat3x3[2, 1])
    z = z + (b * mat3x3[2, 2])

    return np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)


def _rawpy_demosaic(name: str):
    """Backwards-compatible re-export of the canonical demosaic resolver."""
    from transformation_portal.core.raw_runtime import resolve_demosaic_algorithm

    return resolve_demosaic_algorithm(name)


def ingest_phase2_xyz_d50_linear_fp32(
    path: Path | str,
    *,
    wb_mode: Literal["none", "camera", "auto"] = "camera",
    demosaic: str = "AHD",
    raw_python_executable: str | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Certified Phase II decode: RAW -> camera RGB (linear) -> XYZ(D65) -> Bradford(D50) -> float32 HWC.

    Raises FPStateError before any processing if FTZ/DAZ are enabled.
    Returns (tensor, fingerprint) where fingerprint is schema-versioned provenance.
    """
    path = Path(path)
    enforce_ftz_daz_disabled()

    if raw_python_executable is not None:
        tensor, metadata = run_raw_worker(
            python_executable=raw_python_executable,
            command_name="phase2_decode",
            input_path=path,
            payload={
                "wb_mode": wb_mode,
                "demosaic": demosaic,
            },
            start=Path(__file__),
        )
        fingerprint = metadata.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise RuntimeError(f"RAW worker returned invalid Phase II fingerprint payload: {fingerprint!r}")
        tensor_f32 = np.asarray(tensor, dtype=np.float32)
        if tensor_f32.ndim != 3 or tensor_f32.shape[2] != 3:
            raise RuntimeError(
                "RAW worker returned invalid Phase II tensor shape; expected float32 HWC with 3 channels, "
                f"got shape {tensor_f32.shape!r}"
            )
        return canonicalize_tensor_f32_le_c(tensor_f32), fingerprint

    try:
        import rawpy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "rawpy is required for contract camera_native_linear. "
            "Use `./scripts/setup/install_raw_runtime.sh` for the isolated RAW runtime, "
            "or deliberately install the RAW extra into this active interpreter for development."
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

        scale = np.float32(1.0 / 65535.0)
        rgb = rgb16.astype(np.float32) * scale

        # LibRaw camera RGB -> XYZ(D65) matrix, cast to float32.
        rgb_to_xyz = np.asarray(raw.rgb_xyz_matrix, dtype=np.float32)
        if rgb_to_xyz.shape != (3, 3):
            raise ValueError(f"Unexpected rgb_xyz_matrix shape: {rgb_to_xyz.shape}")

        xyz_d65 = _apply_3x3_f32_hwc(rgb, rgb_to_xyz)
        xyz_d50 = _apply_3x3_f32_hwc(xyz_d65, BRADFORD_D65_TO_D50_F32)
        xyz_d50 = canonicalize_tensor_f32_le_c(xyz_d50).astype(np.float32, copy=False)

        camera_wb: Optional[list] = None
        try:
            camera_wb = [float(x) for x in raw.camera_whitebalance]
        except (TypeError, ValueError, AttributeError):
            camera_wb = None

        fingerprint: Dict[str, Any] = {
            "schema_version": PHASE2_FINGERPRINT_SCHEMA_VERSION,
            "contract": "camera_native_linear",
            "input_kind": "raw",
            "input_path": Path(path).name,
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
