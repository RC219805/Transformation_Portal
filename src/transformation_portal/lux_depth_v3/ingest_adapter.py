"""Adapter from lux_depth_v3 into canonical spatial_ai ingest contracts.

Phase C1 design rule:
- RAW decode policy is owned by ``spatial_ai.ingest.decode_contract``.
- lux_depth_v3 only adapts decoded tensors
  for existing inference/preprocess paths.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from PIL import Image

import transformation_portal.spatial_ai.ingest.contracts as ingest_contracts
from transformation_portal.core.raw_runtime import run_raw_worker
from transformation_portal.ingest.canonical_json import canonicalize_json

from .raw_loader import is_raw_file

if TYPE_CHECKING:
    from .config import EnhanceConfig

logger = logging.getLogger(__name__)

RAW_PREVIEW_ESCAPE_ENV = "TP_ALLOW_RAW_PREVIEW"
RAW_INGEST_PROFILE = "tp.raw_ingest.deterministic_v1"
RAW_CAMERA_WB_NON_POSITIVE_TOKEN = "camera_whitebalance has zero or negative gain"


class RawIngestError(RuntimeError):
    """Raised when RAW ingest cannot satisfy deterministic policy."""


def _validated_raw_dimensions(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise RawIngestError(f"RAW dimension probe returned invalid input_size: {value!r}")
    height, width = value
    return width, height


def probe_raw_dimensions(path: Path, config: "EnhanceConfig") -> tuple[int, int]:
    """Return visible RAW dimensions without allocating a demosaiced frame."""

    path = Path(path)
    mode = _normalized_ingest_mode(config)
    preview_allowed = _preview_escape_enabled(config)
    if mode == "force_preview":
        if not preview_allowed:
            raise RawIngestError(
                f"raw_ingest_mode=force_preview requires {RAW_PREVIEW_ESCAPE_ENV}=1 (debug-only escape hatch)."
            )
        with Image.open(path) as preview:
            return _validated_raw_dimensions((preview.height, preview.width))

    try:
        python_executable = getattr(config, "raw_python_executable", None)
        if python_executable:
            _array, metadata = run_raw_worker(
                python_executable=python_executable,
                command_name="probe",
                input_path=path,
                payload={},
                start=path,
            )
            return _validated_raw_dimensions(metadata.get("input_size"))

        import rawpy

        with rawpy.imread(str(path)) as raw:
            return _validated_raw_dimensions((int(raw.sizes.height), int(raw.sizes.width)))
    except Exception as exc:
        if preview_allowed and mode == "auto":
            try:
                with Image.open(path) as preview:
                    return _validated_raw_dimensions((preview.height, preview.width))
            except Exception:
                pass
        if isinstance(exc, RawIngestError):
            raise
        raise RawIngestError(f"RAW dimension probe failed for {path.name}: {exc}") from exc


def _normalized_ingest_mode(config: "EnhanceConfig") -> str:
    allowed_modes = ("auto", "force_rawpy", "force_preview")
    mode = str(getattr(config, "raw_ingest_mode", "auto")).strip().lower()
    if mode not in allowed_modes:
        message = "raw_ingest_mode must be one of: {}".format(
            ", ".join(allowed_modes),
        )
        raise ValueError(message)
    return mode


def _preview_escape_enabled(config: "EnhanceConfig" | None = None) -> bool:
    if config is not None and getattr(config, "execution_plan_authority", None) is not None:
        return bool(getattr(config, "raw_preview_escape_enabled", False))
    val = (
        os.getenv(
            RAW_PREVIEW_ESCAPE_ENV,
            "0",
        )
        .strip()
        .lower()
    )
    return val in {"1", "true", "yes", "on"}


def build_raw_ingest_options(
    config: "EnhanceConfig",
) -> ingest_contracts.IngestOptions:
    """Build canonical deterministic ingest options from config.

    ``legacy_linear_srgb`` is used for
    lux_depth_v3 because the stage expects an
    RGB-like float tensor. Deterministic RAW
    policy is still enforced by the contract
    fields.
    """
    return ingest_contracts.IngestOptions(
        contract="legacy_linear_srgb",
        tensor_role="xyz_d50_linear_fp32",
        wb_mode=str(  # type: ignore[arg-type]
            getattr(config, "raw_wb_mode", "camera"),
        )
        .strip()
        .lower(),
        demosaic=str(
            getattr(config, "raw_demosaic", "AHD"),
        )
        .strip()
        .upper(),
        raw_python_executable=getattr(config, "raw_python_executable", None),
        no_auto_bright=True,
        no_auto_scale=True,
        gamma_mode="linear",
    )


def raw_ingest_summary(
    config: "EnhanceConfig",
    *,
    raw_python_executable: str | None = None,
) -> Dict[str, Any]:
    """Return deterministic digest summary.

    For provenance and run-card metadata.
    """
    mode = _normalized_ingest_mode(config)
    options = build_raw_ingest_options(config)
    payload: Dict[str, Any] = {
        "profile": RAW_INGEST_PROFILE,
        "mode": mode,
        "contract": options.contract,
        "wb_mode": options.wb_mode,
        "demosaic": options.demosaic,
        "raw_python_executable": raw_python_executable or getattr(config, "raw_python_executable", None),
        "no_auto_bright": options.no_auto_bright,
        "no_auto_scale": options.no_auto_scale,
        "gamma_mode": options.gamma_mode,
        "preview_escape_env": RAW_PREVIEW_ESCAPE_ENV,
        "preview_escape_enabled": _preview_escape_enabled(config),
    }
    payload["settings_hash"] = hashlib.sha256(
        canonicalize_json(payload),
    ).hexdigest()
    return payload


def _decode_preview_rgb(path: Path) -> np.ndarray:
    try:
        # Best-effort preview path for containers PIL can decode.
        # Camera RAW formats typically require contract decode path.
        with Image.open(path) as preview_image:
            preview = preview_image.convert("RGB")
        return np.asarray(preview, dtype=np.float32) / 255.0
    except Exception as exc:
        raise RawIngestError(
            "RAW preview decode failed for"
            f" {path.name}. The preview escape hatch"
            " requires a preview-decodable container"
            " or a dedicated RAW preview extractor."
            f" Underlying error: {exc}"
        ) from exc


def _is_camera_wb_metadata_failure(exc: Exception) -> bool:
    return RAW_CAMERA_WB_NON_POSITIVE_TOKEN in str(exc)


def _decode_auto_wb_linear_rgb(
    path: Path,
    *,
    python_executable: str | None,
) -> np.ndarray:
    from .raw_loader import load_raw_as_rgb

    linear_rgb_u16 = load_raw_as_rgb(
        path,
        use_camera_wb=False,
        half_size=False,
        output_bps=16,
        output_linear=True,
        python_executable=python_executable,
    )
    return np.clip(linear_rgb_u16.astype(np.float32) / 65535.0, 0.0, 1.0).astype(np.float32)


def decode_for_lux_depth(
    path: Path,
    config: "EnhanceConfig",
) -> np.ndarray:
    """Decode RAW input via canonical ingest.

    Uses ingest contract for lux_depth_v3
    consumers.

    Returns float32 HWC in [0,1] (clipped) for
    compatibility with existing depth code.
    """
    path = Path(path)
    if not is_raw_file(path):
        message = "{}{}".format(
            "decode_for_lux_depth only supports RAW files, got: ",
            path,
        )
        raise ValueError(message)

    mode = _normalized_ingest_mode(config)
    preview_allowed = _preview_escape_enabled(config)

    if mode == "force_preview":
        if not preview_allowed:
            message = "{}{}{}".format(
                "raw_ingest_mode=force_preview requires ",
                "{}=1".format(RAW_PREVIEW_ESCAPE_ENV),
                " (debug-only escape hatch).",
            )
            raise RawIngestError(message)
        logger.warning("RAW preview escape hatch enabled for %s", path.name)
        return np.clip(_decode_preview_rgb(path), 0.0, 1.0).astype(np.float32)

    options = build_raw_ingest_options(config)

    try:
        tensor = ingest_contracts.decode_contract(path, options)
    except Exception as exc:
        if mode != "force_preview" and _is_camera_wb_metadata_failure(exc):
            logger.warning(
                "Canonical RAW decode rejected camera white balance for %s; retrying with rawpy auto white balance.",
                path.name,
            )
            try:
                return _decode_auto_wb_linear_rgb(
                    path,
                    python_executable=getattr(config, "raw_python_executable", None),
                )
            except Exception as fallback_exc:
                raise RawIngestError(
                    "Canonical RAW decode failed for {}: {}. "
                    "Auto-WB retry also failed: {}".format(path.name, exc, fallback_exc)
                ) from fallback_exc
        if preview_allowed and mode == "auto":
            preview_message = "{}{}".format(
                "Canonical RAW decode failed for %s;",
                " falling back to preview decode because %s is enabled.",
            )
            logger.warning(
                preview_message,
                path.name,
                RAW_PREVIEW_ESCAPE_ENV,
            )
            return np.clip(
                _decode_preview_rgb(path),
                0.0,
                1.0,
            ).astype(np.float32)
        message = "Canonical RAW decode failed for {}: {}".format(
            path.name,
            exc,
        )
        raise RawIngestError(message) from exc

    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        message = "{}{} for {}".format(
            "Canonical RAW decode returned invalid shape ",
            array.shape,
            path.name,
        )
        raise RawIngestError(message)

    return np.clip(array, 0.0, 1.0).astype(np.float32)
