"""Adapter from lux_depth_v3 into canonical spatial_ai ingest contracts.

Phase C1 design rule:
- RAW decode policy is owned by ``spatial_ai.ingest.decode_contract``.
- lux_depth_v3 only adapts decoded tensors for existing inference/preprocess paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from PIL import Image

from transformation_portal.spatial_ai.ingest import contracts as ingest_contracts

from .raw_loader import is_raw_file

if TYPE_CHECKING:
    from .config import EnhanceConfig

logger = logging.getLogger(__name__)

RAW_PREVIEW_ESCAPE_ENV = "TP_ALLOW_RAW_PREVIEW"
RAW_INGEST_PROFILE = "tp.raw_ingest.deterministic_v1"


class RawIngestError(RuntimeError):
    """Raised when RAW ingest cannot satisfy deterministic policy."""


def _normalized_ingest_mode(config: "EnhanceConfig") -> str:
    mode = str(getattr(config, "raw_ingest_mode", "auto")).strip().lower()
    if mode not in {"auto", "force_rawpy", "force_preview"}:
        raise ValueError("raw_ingest_mode must be one of: auto, force_rawpy, force_preview")
    return mode


def _preview_escape_enabled() -> bool:
    return os.getenv(RAW_PREVIEW_ESCAPE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def build_raw_ingest_options(config: "EnhanceConfig") -> ingest_contracts.IngestOptions:
    """Build canonical deterministic ingest options from config.

    ``legacy_linear_srgb`` is used for lux_depth_v3 because the stage expects an RGB-like
    float tensor. Deterministic RAW policy is still enforced by the contract fields.
    """
    return ingest_contracts.IngestOptions(
        contract="legacy_linear_srgb",
        tensor_role="xyz_d50_linear_fp32",
        wb_mode=str(getattr(config, "raw_wb_mode", "camera")).strip().lower(),
        demosaic=str(getattr(config, "raw_demosaic", "AHD")).strip().upper(),
        no_auto_bright=True,
        no_auto_scale=True,
        gamma_mode="linear",
    )


def raw_ingest_summary(config: "EnhanceConfig") -> Dict[str, Any]:
    """Return deterministic digest summary for provenance and run-card metadata."""
    mode = _normalized_ingest_mode(config)
    options = build_raw_ingest_options(config)
    payload: Dict[str, Any] = {
        "profile": RAW_INGEST_PROFILE,
        "mode": mode,
        "contract": options.contract,
        "wb_mode": options.wb_mode,
        "demosaic": options.demosaic,
        "no_auto_bright": options.no_auto_bright,
        "no_auto_scale": options.no_auto_scale,
        "gamma_mode": options.gamma_mode,
        "preview_escape_env": RAW_PREVIEW_ESCAPE_ENV,
        "preview_escape_enabled": _preview_escape_enabled(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload["settings_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _decode_preview_rgb(path: Path) -> np.ndarray:
    preview = Image.open(path).convert("RGB")
    return np.asarray(preview, dtype=np.float32) / 255.0


def decode_for_lux_depth(path: Path, config: "EnhanceConfig") -> np.ndarray:
    """Decode RAW input via canonical ingest contract for lux_depth_v3 consumers.

    Returns float32 HWC in [0,1] (clipped) for compatibility with existing depth code.
    """
    path = Path(path)
    if not is_raw_file(path):
        raise ValueError(f"decode_for_lux_depth only supports RAW files, got: {path}")

    mode = _normalized_ingest_mode(config)
    preview_allowed = _preview_escape_enabled()

    if mode == "force_preview":
        if not preview_allowed:
            raise RawIngestError(
                f"raw_ingest_mode=force_preview requires {RAW_PREVIEW_ESCAPE_ENV}=1 (debug-only escape hatch)."
            )
        logger.warning("RAW preview escape hatch enabled for %s", path.name)
        return np.clip(_decode_preview_rgb(path), 0.0, 1.0).astype(np.float32)

    options = build_raw_ingest_options(config)

    try:
        tensor = ingest_contracts.decode_contract(path, options)
    except Exception as exc:
        if preview_allowed and mode == "auto":
            logger.warning(
                "Canonical RAW decode failed for %s; falling back to preview decode because %s is enabled.",
                path.name,
                RAW_PREVIEW_ESCAPE_ENV,
            )
            return np.clip(_decode_preview_rgb(path), 0.0, 1.0).astype(np.float32)
        raise RawIngestError(f"Canonical RAW decode failed for {path.name}: {exc}") from exc

    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RawIngestError(f"Canonical RAW decode returned invalid shape {array.shape} for {path.name}")

    return np.clip(array, 0.0, 1.0).astype(np.float32)
