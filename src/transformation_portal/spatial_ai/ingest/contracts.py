"""Ingest contract dispatcher for spatial_ai/ingest.

Usage:
    from transformation_portal.spatial_ai.ingest.contracts import IngestOptions, decode_contract

    opts = IngestOptions(contract="camera_native_linear")
    tensor = decode_contract("/path/to/file.CR3", opts)

For the full (tensor, fingerprint) tuple needed by the determinism harness, use
ingest_phase2_xyz_d50_linear_fp32 directly from phase2_camera_native_linear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

IngestContractName = Literal["legacy_linear_srgb", "camera_native_linear"]


@dataclass(frozen=True)
class IngestOptions:
    contract: IngestContractName
    tensor_role: str = "xyz_d50_linear_fp32"
    wb_mode: Literal["none", "camera", "auto"] = "camera"
    demosaic: str = field(default="AHD")
    raw_python_executable: str | None = None
    no_auto_bright: bool = True
    no_auto_scale: bool = True
    gamma_mode: Literal["linear"] = "linear"


def _validate_deterministic_raw_policy(opts: IngestOptions) -> None:
    """Require deterministic RAW decode semantics for all contracts."""
    if not opts.no_auto_bright:
        raise ValueError("decode_contract requires no_auto_bright=True for deterministic ingest.")
    if not opts.no_auto_scale:
        raise ValueError("decode_contract requires no_auto_scale=True for deterministic ingest.")
    if opts.gamma_mode != "linear":
        raise ValueError("decode_contract requires gamma_mode='linear' for deterministic ingest.")


def decode_contract(input_path: Path | str, opts: IngestOptions) -> np.ndarray:
    """Contract dispatcher: returns a float32 HWC tensor for the given contract.

    - camera_native_linear: Phase II certified path (xyz_d50_linear_fp32).
      Fails closed if FTZ/DAZ are enabled. Requires rawpy.
    - legacy_linear_srgb: routes to LinearDecoder (Phase I, not certified).
    """
    _validate_deterministic_raw_policy(opts)

    if opts.contract == "camera_native_linear":
        if opts.tensor_role != "xyz_d50_linear_fp32":
            raise ValueError("camera_native_linear requires tensor_role='xyz_d50_linear_fp32' for Phase II certification.")
        from .phase2_camera_native_linear import ingest_phase2_xyz_d50_linear_fp32

        tensor, _ = ingest_phase2_xyz_d50_linear_fp32(
            path=input_path,
            wb_mode=opts.wb_mode,
            demosaic=opts.demosaic,
            raw_python_executable=opts.raw_python_executable,
        )
        return tensor

    if opts.contract == "legacy_linear_srgb":
        if opts.wb_mode != "camera":
            raise ValueError("legacy_linear_srgb currently supports wb_mode='camera' only.")
        from .linear_decoder import LinearDecoder

        result = LinearDecoder(
            gamma=1.0,
            strict_ingest=True,
            raw_python_executable=opts.raw_python_executable,
            demosaic=opts.demosaic,
        ).decode(input_path)
        return result.linear_rgb

    raise ValueError(
        f"Unknown ingest contract: {opts.contract!r}. " "Valid values: 'camera_native_linear', 'legacy_linear_srgb'."
    )
