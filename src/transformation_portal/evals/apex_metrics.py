"""Deterministic, model-free APEX visual signals.

``spatial_v2`` fixes cancellation in color/clipping comparisons and measures
newly flattened smooth gradients before quantization. Values from older
algorithm revisions are not interchangeable. These are registered-image
diagnostics, not perceptual Delta E, calibrated pass/fail thresholds, or human
photographic acceptance. Exposure/geometry edits, denoising and deliberate
clipping can raise a signal; inspect full frames and native crops together.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

METRIC_STATUS_OK = "ok"
METRIC_STATUS_NOT_APPLICABLE = "not_applicable"
METRIC_STATUS_INVALID_INPUT = "invalid_input"
METRIC_STATUS_UNSUPPORTED_BIT_DEPTH = "unsupported_bit_depth"
METRIC_STATUS_DIMENSION_MISMATCH = "dimension_mismatch"
METRIC_STATUS_MASK_MISSING = "mask_missing"
ALGORITHM_REVISION = "spatial_v2"
_METHODS = {
    "visible_delta": "normalized_sample_mae",
    "p95_delta": "normalized_sample_absolute_delta_p95",
    "outside_mask_delta": "outside_mask_sample_mae",
    "seam_halo_score": "mask_boundary_sample_mae_not_causal_halo_evidence",
    "highlight_clipping": "newly_clipped_samples",
    "shadow_lift_risk": "positive_delta_in_reference_shadows_below_0.15",
    "banding_risk": "newly_flattened_smooth_steps",
    "local_contrast_delta": "absolute_mean_adjacent_luminance_gradient_change",
    "color_drift": "spatial_centered_rgb_mae",
    "delivery_conversion_delta": "candidate_delivery_normalized_sample_mae",
    "edge_halo_score": "reference_5px_edge_envelope_overshoot",
}

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
        "algorithm_revision": ALGORITHM_REVISION,
        "acceptance": "not_assessed",
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
        name: _metric(status, reason=reason, metadata={**metadata, "method": _METHODS[name]})
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
            "edge_halo_score",
        )
    }


def _edge_band(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if binary.ndim == 3:
        binary = np.any(binary, axis=2)
    padded = np.pad(binary, 1, mode="edge")
    return (
        (binary != padded[:-2, 1:-1])
        | (binary != padded[2:, 1:-1])
        | (binary != padded[1:-1, :-2])
        | (binary != padded[1:-1, 2:])
    )


def _gradient_mean(arr: np.ndarray) -> float:
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    gradients = [np.abs(np.diff(arr, axis=axis)).reshape(-1) for axis in (0, 1) if arr.shape[axis] > 1]
    return float(np.mean(np.concatenate(gradients))) if gradients else 0.0


def _valid_image(array: np.ndarray) -> bool:
    return (
        array.ndim in (2, 3)
        and all(side > 0 for side in array.shape)
        and (array.ndim != 3 or array.shape[2] in (1, 3, 4))
        and array.dtype.kind in {"u", "i", "f"}
        and bool(np.all(np.isfinite(array)))
    )


def _gray(array: np.ndarray) -> np.ndarray:
    # Ignore alpha when present; it is not a color channel.
    return array[..., :3].mean(axis=2) if array.ndim == 3 else array


def _banding_risk(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, int]:
    """Fraction of smooth, nonzero reference steps flattened by >=90%.

    Work at source precision rather than first reducing everything to 8-bit.
    Hard edges (>5% normalized change) are excluded. A flat reference supplies
    no banding evidence. This detects new posterization; it cannot certify an
    already posterized reference or distinguish intentional smoothing by itself.
    """
    ref_gray, cand_gray = _gray(reference), _gray(candidate)
    eligible_count = flattened_count = 0
    for axis in (0, 1):
        ref_step = np.abs(np.diff(ref_gray, axis=axis))
        cand_step = np.abs(np.diff(cand_gray, axis=axis))
        eligible = (ref_step > 1e-7) & (ref_step <= 0.05)
        eligible_count += int(np.count_nonzero(eligible))
        flattened_count += int(np.count_nonzero(eligible & (cand_step <= np.maximum(1e-7, ref_step * 0.1))))
    return (float(flattened_count / eligible_count) if eligible_count else 0.0), eligible_count


def _peak_tile_fraction(array: np.ndarray, tile_side: int = 64) -> float:
    """Largest affected fraction in a native, fixed-size tile (edge tiles included)."""
    return max(
        float(np.mean(array[y : y + tile_side, x : x + tile_side]))
        for y in range(0, array.shape[0], tile_side)
        for x in range(0, array.shape[1], tile_side)
    )


def _edge_overshoot(reference: np.ndarray, candidate: np.ndarray) -> float:
    """New overshoot outside the reference's 5x5 local color envelope.

    Only reference edges contribute. A neutral step/edge has no overshoot;
    ringing brighter or darker than either side raises this advisory signal.
    Legitimate local contrast or exposure changes may also raise it.
    """
    height, width = reference.shape[:2]
    overshoot_sum, edge_count = 0.0, 0
    # Bound temporary buffers for full-resolution masters, retaining a two-pixel
    # halo on each strip so strip boundaries cannot change the measurement.
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        top, bottom = max(0, start - 2), min(height, stop + 2)
        padding = ((max(0, 2 - start), max(0, stop + 2 - height)), (2, 2))
        if reference.ndim == 3:
            padding += ((0, 0),)
        padded = np.pad(reference[top:bottom], padding, mode="edge")
        local_min, local_max = reference[start:stop].copy(), reference[start:stop].copy()
        for y in range(5):
            for x in range(5):
                neighbor = padded[y : y + stop - start, x : x + width]
                np.minimum(local_min, neighbor, out=local_min)
                np.maximum(local_max, neighbor, out=local_max)
        edge = (local_max - local_min) > 0.05
        overshoot = np.maximum(candidate[start:stop] - local_max, local_min - candidate[start:stop])
        overshoot_sum += float(np.sum(np.maximum(overshoot[edge], 0.0), dtype=np.float64))
        edge_count += int(np.count_nonzero(edge))
    return overshoot_sum / edge_count if edge_count else 0.0


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
    """Compare spatially registered, same-encoding arrays without resizing.

    Float samples retain the normalized 0..1 contract. Diagnostics are relative
    to the supplied reference and are not calibrated production-quality gates.
    ``comparison.algorithm_revision`` must accompany stored metric values.
    """
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
    if not _valid_image(ref) or not _valid_image(cand):
        return _invalid_metrics(METRIC_STATUS_INVALID_INPUT, "invalid_image_array", metadata)
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
    newly_clipped = (cand_norm >= 0.995) & (ref_norm < 0.995)
    highlight_clip = float(np.mean(newly_clipped))
    shadow_lift = float(np.mean(np.clip(cand_norm - ref_norm, 0.0, 1.0)[ref_norm < 0.15])) if np.any(ref_norm < 0.15) else 0.0
    grad_ref = _gradient_mean(ref_norm)
    grad_cand = _gradient_mean(cand_norm)
    local_contrast_delta = float(abs(grad_cand - grad_ref))
    if ref_norm.ndim == 3 and ref_norm.shape[2] >= 3:
        ref_rgb, cand_rgb = ref_norm[..., :3], cand_norm[..., :3]
        ref_chroma = ref_rgb - ref_rgb.mean(axis=2, keepdims=True)
        cand_chroma = cand_rgb - cand_rgb.mean(axis=2, keepdims=True)
        # Measure before spatial reduction: swapped color regions cannot cancel.
        color_drift = float(np.mean(np.abs(cand_chroma - ref_chroma)))
    else:
        color_drift = 0.0
    banding_risk, smooth_steps = _banding_risk(ref_norm, cand_norm)
    if delivery is not None:
        delivery_source = np.asarray(delivery)
        delivery_conversion = None
        if not _valid_image(delivery_source):
            delivery_status, delivery_reason = METRIC_STATUS_INVALID_INPUT, "invalid_delivery_array"
        elif delivery_source.shape != cand_norm.shape:
            delivery_status, delivery_reason = METRIC_STATUS_DIMENSION_MISMATCH, "dimension_mismatch"
        else:
            delivery_conversion = float(np.mean(np.abs(_normalize(delivery_source) - cand_norm)))
            delivery_status, delivery_reason = METRIC_STATUS_OK, None
    else:
        delivery_conversion = None
        delivery_status = METRIC_STATUS_NOT_APPLICABLE
        delivery_reason = "delivery_missing"

    metrics = {
        "visible_delta": _metric(METRIC_STATUS_OK, value=mean_delta, metadata=metadata),
        "p95_delta": _metric(METRIC_STATUS_OK, value=p95_delta, metadata=metadata),
        "highlight_clipping": _metric(
            METRIC_STATUS_OK,
            value=highlight_clip,
            metadata={
                **metadata,
                "method": "newly_clipped_samples",
                "normalized_threshold": 0.995,
                "peak_native_64px_tile_fraction": _peak_tile_fraction(newly_clipped),
            },
        ),
        "shadow_lift_risk": _metric(METRIC_STATUS_OK, value=shadow_lift, metadata=metadata),
        "banding_risk": _metric(
            METRIC_STATUS_OK,
            value=banding_risk,
            metadata={**metadata, "method": "newly_flattened_smooth_steps", "eligible_reference_steps": smooth_steps},
        ),
        "local_contrast_delta": _metric(METRIC_STATUS_OK, value=local_contrast_delta, metadata=metadata),
        "color_drift": _metric(
            METRIC_STATUS_OK, value=color_drift, metadata={**metadata, "method": "spatial_centered_rgb_mae"}
        ),
        "edge_halo_score": _metric(
            METRIC_STATUS_OK,
            value=_edge_overshoot(ref_norm, cand_norm),
            metadata={**metadata, "method": "reference_5px_edge_envelope_overshoot"},
        ),
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
        if (
            mask_arr.ndim not in (2, 3)
            or mask_arr.shape[:2] != ref.shape[:2]
            or mask_arr.dtype.kind not in {"u", "i", "f", "b"}
            or not np.all(np.isfinite(mask_arr))
        ):
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

    for name, metric in metrics.items():
        metric["comparison"].setdefault("method", _METHODS[name])
    return metrics
