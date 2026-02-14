"""Executor for Materials V3 pixel operations."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .pixel_ops_decider import decide_pixel_ops
from .pixel_ops_registry import OP_REGISTRY


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    """Canonicalize mask to 2D (H, W) float32 format.

    Handles edge cases:
    - (H, W, 1): squeeze last dimension
    - (1, H, W): squeeze first dimension
    - (H, W): already canonical, ensure float32

    Args:
        mask: Input mask, may be 2D or 3D

    Returns:
        2D float32 mask of shape (H, W)

    Raises:
        ValueError: If mask cannot be canonicalized to 2D
    """
    if mask.ndim == 2:
        return mask.astype(np.float32)
    elif mask.ndim == 3:
        if mask.shape[-1] == 1:
            # (H, W, 1) -> (H, W)
            return mask.squeeze(axis=-1).astype(np.float32)
        elif mask.shape[0] == 1:
            # (1, H, W) -> (H, W)
            return mask.squeeze(axis=0).astype(np.float32)
        else:
            raise ValueError(f"Cannot canonicalize 3D mask with shape {mask.shape} - expected (H,W,1) or (1,H,W)")
    else:
        raise ValueError(f"Cannot canonicalize mask with {mask.ndim} dimensions - expected 2D or 3D")


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Compute bounding box from 2D mask.

    Args:
        mask: 2D mask of shape (H, W)

    Returns:
        Bounding box (x0, y0, x1, y1) or None if mask is empty
    """
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _feather_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian feathering to mask edges.

    Args:
        mask: 2D mask of shape (H, W), values in [0, 1]
        sigma: Gaussian blur sigma (0 = no feathering)

    Returns:
        Feathered mask of same shape
    """
    if sigma <= 0:
        return mask

    # Use OpenCV's GaussianBlur for efficiency
    # Kernel size should be odd and approximately 6*sigma
    ksize = int(np.ceil(sigma * 6))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)  # Minimum size of 3

    feathered = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
    return np.clip(feathered, 0.0, 1.0).astype(np.float32)


def _expand_bbox_with_padding(
    bbox: Tuple[int, int, int, int], pad: int, img_height: int, img_width: int
) -> Tuple[int, int, int, int]:
    """Expand bounding box by padding, clipping to image boundaries.

    Args:
        bbox: Original bounding box (x0, y0, x1, y1)
        pad: Padding amount in pixels
        img_height: Image height
        img_width: Image width

    Returns:
        Expanded bounding box (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(img_width, x1 + pad),
        min(img_height, y1 + pad),
    )


def _resolve_overlaps(
    materials: Dict[str, np.ndarray], material_metadata: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Resolve overlapping masks using priority-based assignment.

    For pixels covered by multiple materials, assign to the highest-priority material.
    Creates non-overlapping masks and reports overlap statistics.

    Args:
        materials: Dict mapping material name to 2D mask (H, W)
        material_metadata: Dict mapping material name to metadata (must include 'priority')

    Returns:
        Tuple of:
        - Dict of non-overlapping masks (same keys as input)
        - Telemetry dict with overlap statistics
    """
    if not materials:
        return {}, {"overlap_percent": 0.0, "reassignments": {}}

    # Get image dimensions from first mask
    first_mask = next(iter(materials.values()))
    h, w = first_mask.shape[:2]

    # Sort materials by priority (highest first)
    sorted_materials = sorted(
        materials.items(), key=lambda x: material_metadata.get(x[0], {}).get("priority", 0), reverse=True
    )

    # Track overlaps
    total_pixels = 0
    overlapping_pixels = 0
    reassignments = {name: 0 for name, _ in materials.items()}

    # First pass: identify all material pixels
    pixel_count_map = np.zeros((h, w), dtype=np.int32)
    for material_name, mask in materials.items():
        mask_2d = _canonical_mask(mask)
        mask_bool = mask_2d > 0.5
        pixel_count_map[mask_bool] += 1
        total_pixels += mask_bool.sum()

    overlapping_pixels = (pixel_count_map > 1).sum()

    # Second pass: assign pixels to highest priority material
    assigned_pixels = np.zeros((h, w), dtype=bool)
    resolved_masks = {}

    for material_name, mask in sorted_materials:
        mask_2d = _canonical_mask(mask)
        mask_bool = mask_2d > 0.5

        # Find pixels that belong to this material but aren't assigned yet
        available = mask_bool & ~assigned_pixels

        # Find pixels that were reassigned from this material
        lost = mask_bool & assigned_pixels
        reassignments[material_name] = int(lost.sum())

        # Create resolved mask (only unassigned pixels)
        resolved_mask = np.zeros((h, w), dtype=np.float32)
        resolved_mask[available] = mask_2d[available]
        resolved_masks[material_name] = resolved_mask

        # Mark these pixels as assigned
        assigned_pixels[available] = True

    # Calculate overlap percentage (overlapping pixels / total material pixels)
    # Note: This counts each overlapping pixel once, even if covered by multiple materials
    overlap_percent = (overlapping_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    telemetry = {
        "overlap_percent": round(overlap_percent, 2),  # % of material pixels that overlap
        "reassignments": {k: v for k, v in reassignments.items() if v > 0},
        "total_pixels": int(total_pixels),
        "overlapping_pixels": int(overlapping_pixels),
    }

    return resolved_masks, telemetry


def _compute_delta_stats(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    delta = np.abs(after.astype(np.float32) - before.astype(np.float32))
    mask = np.squeeze(mask) if mask.ndim == 3 else mask
    mask_bool = mask > 0.5
    inside = float(delta[mask_bool].mean()) if mask_bool.any() else 0.0
    outside = float(delta[~mask_bool].mean()) if (~mask_bool).any() else 0.0
    return {
        "inside_mask_mean_abs": round(inside, 6),
        "outside_mask_mean_abs": round(outside, 6),
    }


def apply_pixel_ops(
    image: np.ndarray,
    segmentation_result: Dict[str, Any],
    response_plan: Dict[str, Any],
    config: Any,
    registry: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply pixel ops and emit telemetry (never null).

    Implements:
    - A1: 3D mask canonicalization
    - A2: Feathering with bbox padding
    - A3: Configurable feathering per material
    - A5: Priority-based overlap resolution

    Args:
        image: Input image (uint8 or uint16)
        segmentation_result: Contains "materials" dict
        response_plan: Contains "per_class" decision data
        config: Configuration object with feathering settings
        registry: Pixel ops registry (defaults to OP_REGISTRY)

    Returns:
        Tuple of (output_image, telemetry_dict)
    """
    registry = registry or OP_REGISTRY
    telemetry = {
        "enabled": bool(getattr(config, "apply_pixel_ops", False)),
        "applied": [],
        "blocked": [],
        "timing_ms": {},
    }

    if not telemetry["enabled"]:
        return image, telemetry

    start_total = time.perf_counter()
    output = image.copy()
    materials = segmentation_result.get("materials", {})
    plan_per_class = response_plan.get("per_class", {})

    # A5: Resolve overlapping masks using priority
    from .materials_v3_taxonomy import DEFAULT_MATERIAL_METADATA

    resolved_materials, overlap_telemetry = _resolve_overlaps(materials, DEFAULT_MATERIAL_METADATA)
    telemetry["overlap_resolution"] = overlap_telemetry

    # Get feathering configuration (A3)
    feather_default = float(getattr(config, "mask_feather_sigma_default", 3.0))
    feather_overrides = getattr(config, "mask_feather_sigma_overrides", {}) or {}
    feather_disabled = set(getattr(config, "mask_feather_disabled_materials", []) or [])

    # Sort by priority (highest first) to process in order
    sorted_materials = sorted(
        resolved_materials.items(),
        key=lambda x: DEFAULT_MATERIAL_METADATA.get(x[0], {}).get("priority", 0),
        reverse=True,
    )

    for material_key, mask in sorted_materials:
        # A1: Canonicalize mask to 2D early
        try:
            mask_2d = _canonical_mask(mask)
        except ValueError as e:
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": "invalid_mask_shape",
                    "blocked_by": [str(e)],
                    "recommended_ops": [],
                }
            )
            continue

        plan_entry = plan_per_class.get(material_key, {})
        if not plan_entry:
            continue

        plan_decision = plan_entry.get("pixel_ops")
        decision = plan_decision or decide_pixel_ops(material_key, plan_entry, config, registry=registry)
        ops_for_material = registry.get(material_key, {})
        recommended_ops = decision.get("recommended_ops") or list(ops_for_material.keys())
        if not decision.get("will_apply", False):
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": decision.get("reason", "not_recommended"),
                    "blocked_by": decision.get("blocked_by", []),
                    "recommended_ops": recommended_ops,
                }
            )
            continue

        implemented_ops = [
            op_name for op_name in recommended_ops if (op_def := ops_for_material.get(op_name)) and op_def.implemented
        ]
        if not implemented_ops:
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": "no_implementation",
                    "blocked_by": ["no_implementation"],
                    "recommended_ops": recommended_ops,
                }
            )
            continue

        bbox = _bounding_box(mask_2d)
        if bbox is None:
            telemetry["blocked"].append(
                {
                    "material": material_key,
                    "reason": "empty_mask",
                    "blocked_by": ["empty_mask"],
                    "recommended_ops": decision["recommended_ops"],
                }
            )
            continue

        # A3: Get material-specific feathering sigma
        if material_key in feather_disabled:
            feather_sigma = 0.0
        else:
            feather_sigma = float(feather_overrides.get(material_key, feather_default))

        # A2: Expand bbox by feathering padding
        img_h, img_w = image.shape[:2]
        pad = math.ceil(3 * feather_sigma) if feather_sigma > 0 else 0
        x0, y0, x1, y1 = bbox
        x0_padded, y0_padded, x1_padded, y1_padded = _expand_bbox_with_padding(bbox, pad, img_h, img_w)

        # Extract padded ROI
        mask_roi_padded = mask_2d[y0_padded:y1_padded, x0_padded:x1_padded]
        before_padded = output[y0_padded:y1_padded, x0_padded:x1_padded]

        # Apply feathering to padded mask
        mask_roi_feathered = _feather_mask(mask_roi_padded, feather_sigma)

        # Process with pixel ops
        start_material = time.perf_counter()
        applied_ops = []
        working = before_padded.copy()
        original_dtype = before_padded.dtype

        # Normalize to [0, 1] (A4: single normalization point)
        if original_dtype == np.uint8:
            working = working.astype(np.float32) / 255.0
        elif original_dtype == np.uint16:
            working = working.astype(np.float32) / 65535.0
        # else: already float, no normalization needed

        for op_name in implemented_ops:
            op_def = ops_for_material.get(op_name)
            if not op_def or not op_def.implemented:
                continue
            working = op_def.op(
                working,
                mask_roi_feathered,
                {"material": material_key, "normalized": working, "scale": 1.0},
            )
            applied_ops.append(op_name)

        # Denormalize and write back (A4)
        if original_dtype == np.uint8:
            after_padded = np.clip(working * 255.0, 0.0, 255.0).astype(np.uint8)
        elif original_dtype == np.uint16:
            after_padded = np.clip(working * 65535.0, 0.0, 65535.0).astype(np.uint16)
        else:
            after_padded = working.astype(original_dtype)

        # Write back only the original (non-padded) ROI
        output[y0:y1, x0:x1] = after_padded[(y0 - y0_padded) : (y1 - y0_padded), (x0 - x0_padded) : (x1 - x0_padded)]

        elapsed_ms = (time.perf_counter() - start_material) * 1000.0

        # Compute stats on original ROI
        delta_stats = _compute_delta_stats(
            before_padded[(y0 - y0_padded) : (y1 - y0_padded), (x0 - x0_padded) : (x1 - x0_padded)],
            after_padded[(y0 - y0_padded) : (y1 - y0_padded), (x0 - x0_padded) : (x1 - x0_padded)],
            mask_roi_feathered[(y0 - y0_padded) : (y1 - y0_padded), (x0 - x0_padded) : (x1 - x0_padded)],
        )

        telemetry["applied"].append(
            {
                "material": material_key,
                "ops": applied_ops,
                "timing_ms": round(elapsed_ms, 3),
                "delta_stats": delta_stats,
                "feather_sigma": feather_sigma,
                "bbox_padding": pad,
            }
        )

    telemetry["timing_ms"]["total"] = round((time.perf_counter() - start_total) * 1000.0, 3)
    return output, telemetry
