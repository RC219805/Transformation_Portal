"""Deterministic, model-free APEX visual metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

METRIC_STATUS_OK = "ok"
METRIC_STATUS_NOT_APPLICABLE = "not_applicable"
METRIC_STATUS_INVALID_INPUT = "invalid_input"
METRIC_STATUS_UNSUPPORTED_BIT_DEPTH = "unsupported_bit_depth"
METRIC_STATUS_DIMENSION_MISMATCH = "dimension_mismatch"
METRIC_STATUS_MASK_MISSING = "mask_missing"

METRIC_STATUSES = frozenset(
    {
        METRIC_STATUS_OK,
        METRIC_STATUS_NOT_APPLICABLE,
        METRIC_STATUS_INVALID_INPUT,
        METRIC_STATUS_UNSUPPORTED_BIT_DEPTH,
        METRIC_STATUS_DIMENSION_MISMATCH,
        METRIC_STATUS_MASK_MISSING,
    }
)


def _bit_depth(array: np.ndarray) -> int | None:
    dtype = np.dtype(array.dtype)
    if dtype.kind in {"u", "i", "f"}:
        return int(dtype.itemsize * 8)
    if dtype.kind == "b":
        return 1
    return None


def _normalize(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.kind == "f":
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    if arr.dtype.kind in {"u", "i"}:
        info = np.iinfo(arr.dtype)
        return np.clip(arr.astype(np.float32) / float(info.max), 0.0, 1.0)
    return arr.astype(np.float32)


def _comparison_metadata(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    working_color_space: str | None,
    working_transfer_function: str | None,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "reference_bit_depth": _bit_depth(reference),
        "candidate_bit_depth": _bit_depth(candidate),
        "reference_dimensions": [int(reference.shape[1]), int(reference.shape[0])] if reference.ndim >= 2 else None,
        "candidate_dimensions": [int(candidate.shape[1]), int(candidate.shape[0])] if candidate.ndim >= 2 else None,
        "working_color_space": working_color_space,
        "working_transfer_function": working_transfer_function,
        "normalized_range": "0..1",
        "mask_dimensions": [int(mask.shape[1]), int(mask.shape[0])] if mask is not None and mask.ndim >= 2 else None,
    }


def _metric(
    status: str,
    *,
    value: float | None = None,
    reason: str | None = None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "value": value,
        "comparison": dict(metadata),
    }


def _invalid_metrics(status: str, reason: str, metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _metric(status, reason=reason, metadata=metadata)
        for name in (
            "visible_delta",
            "p95_delta",
            "outside_mask_delta",
            "seam_halo_score",
            "highlight_clipping",
            "shadow_lift_risk",
            "banding_risk",
            "local_contrast_delta",
            "color_drift",
            "delivery_conversion_delta",
        )
    }


def _edge_band(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if binary.ndim == 3:
        binary = np.any(binary, axis=2)
    padded = np.pad(binary, 1, mode="edge")
    neighbors = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    return binary != neighbors


def _gradient_mean(arr: np.ndarray) -> float:
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    gy, gx = np.gradient(arr.astype(np.float32))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def compute_apex_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    delivery: np.ndarray | None = None,
    working_color_space: str | None = None,
    working_transfer_function: str | None = None,
    candidate_working_color_space: str | None = None,
    candidate_working_transfer_function: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute deterministic APEX metrics over already-loaded arrays."""
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    metadata = _comparison_metadata(
        ref,
        cand,
        working_color_space=working_color_space,
        working_transfer_function=working_transfer_function,
        mask=mask,
    )
    if ref.shape != cand.shape:
        return _invalid_metrics(METRIC_STATUS_DIMENSION_MISMATCH, "dimension_mismatch", metadata)
    if candidate_working_color_space and working_color_space and candidate_working_color_space != working_color_space:
        return _invalid_metrics(METRIC_STATUS_INVALID_INPUT, "working_space_mismatch", metadata)
    if (
        candidate_working_transfer_function
        and working_transfer_function
        and candidate_working_transfer_function != working_transfer_function
    ):
        return _invalid_metrics(METRIC_STATUS_INVALID_INPUT, "working_space_mismatch", metadata)

    ref_norm = _normalize(ref)
    cand_norm = _normalize(cand)
    delta = np.abs(cand_norm - ref_norm)
    mean_delta = float(np.mean(delta))
    p95_delta = float(np.percentile(delta, 95))
    highlight_clip = float(max(0.0, np.mean(cand_norm >= 0.995) - np.mean(ref_norm >= 0.995)))
    shadow_lift = float(np.mean(np.clip(cand_norm - ref_norm, 0.0, 1.0)[ref_norm < 0.15])) if np.any(ref_norm < 0.15) else 0.0
    grad_ref = _gradient_mean(ref_norm)
    grad_cand = _gradient_mean(cand_norm)
    local_contrast_delta = float(abs(grad_cand - grad_ref))
    color_drift = (
        float(np.mean(np.abs(cand_norm.mean(axis=(0, 1)) - ref_norm.mean(axis=(0, 1))))) if ref_norm.ndim == 3 else 0.0
    )

    if cand_norm.ndim == 3:
        gray = cand_norm.mean(axis=2)
    else:
        gray = cand_norm
    unique_levels = np.sort(np.unique(np.rint(gray * 255.0).astype(np.int16)))
    if unique_levels.size <= 1:
        banding_risk = 0.0
    else:
        banding_risk = float(np.mean(np.diff(unique_levels) > 4))
    if delivery is not None:
        delivery_arr = _normalize(np.asarray(delivery))
        delivery_conversion = (
            float(np.mean(np.abs(delivery_arr - cand_norm))) if delivery_arr.shape == cand_norm.shape else None
        )
        delivery_status = METRIC_STATUS_OK if delivery_conversion is not None else METRIC_STATUS_DIMENSION_MISMATCH
        delivery_reason = None if delivery_conversion is not None else "dimension_mismatch"
    else:
        delivery_conversion = None
        delivery_status = METRIC_STATUS_NOT_APPLICABLE
        delivery_reason = "delivery_missing"

    metrics = {
        "visible_delta": _metric(METRIC_STATUS_OK, value=mean_delta, metadata=metadata),
        "p95_delta": _metric(METRIC_STATUS_OK, value=p95_delta, metadata=metadata),
        "highlight_clipping": _metric(METRIC_STATUS_OK, value=highlight_clip, metadata=metadata),
        "shadow_lift_risk": _metric(METRIC_STATUS_OK, value=shadow_lift, metadata=metadata),
        "banding_risk": _metric(METRIC_STATUS_OK, value=banding_risk, metadata=metadata),
        "local_contrast_delta": _metric(METRIC_STATUS_OK, value=local_contrast_delta, metadata=metadata),
        "color_drift": _metric(METRIC_STATUS_OK, value=color_drift, metadata=metadata),
        "delivery_conversion_delta": _metric(
            delivery_status,
            value=delivery_conversion,
            reason=delivery_reason,
            metadata=metadata,
        ),
    }

    if mask is None:
        metrics["outside_mask_delta"] = _metric(METRIC_STATUS_MASK_MISSING, reason="mask_missing", metadata=metadata)
        metrics["seam_halo_score"] = _metric(METRIC_STATUS_NOT_APPLICABLE, reason="mask_missing", metadata=metadata)
    else:
        mask_arr = np.asarray(mask)
        if mask_arr.shape[:2] != ref.shape[:2]:
            metrics["outside_mask_delta"] = _metric(
                METRIC_STATUS_INVALID_INPUT,
                reason="mask_dimension_mismatch",
                metadata=metadata,
            )
            metrics["seam_halo_score"] = _metric(
                METRIC_STATUS_INVALID_INPUT, reason="mask_dimension_mismatch", metadata=metadata
            )
        else:
            mask_bool = mask_arr.astype(bool)
            if mask_bool.ndim == 3:
                mask_bool = np.any(mask_bool, axis=2)
            outside = ~mask_bool
            outside_delta = float(np.mean(delta[outside])) if np.any(outside) else 0.0
            band = _edge_band(mask_bool)
            seam_delta = float(np.mean(delta[band])) if np.any(band) else 0.0
            metrics["outside_mask_delta"] = _metric(METRIC_STATUS_OK, value=outside_delta, metadata=metadata)
            metrics["seam_halo_score"] = _metric(METRIC_STATUS_OK, value=seam_delta, metadata=metadata)

    return metrics
