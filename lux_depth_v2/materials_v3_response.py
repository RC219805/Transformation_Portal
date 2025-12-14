"""Materials V3 Response Planning Module.

Generates per-class response plans (strength, edge/core treatment) without pixel ops.
This is PR-4A: planning only. PR-4B will apply pixel responses.

Response planning computes:
- Core vs edge band extraction (deterministic pixel widths)
- Per-class confidence stats
- Planned response strengths
- Should-refine decisions

All outputs are emitted as `materials_v3_response_plan` in report JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from .logging_utils import setup_logging
from .materials_v3_taxonomy import normalize_material_name, get_material_metadata


log = setup_logging(__name__)


def _build_reason_histogram(reasons: list) -> dict:
    """Build histogram of decision reasons for summary.
    
    Args:
        reasons: List of reason strings
        
    Returns:
        Dict of reason → count
    """
    histogram = {}
    for reason in reasons:
        histogram[reason] = histogram.get(reason, 0) + 1
    return histogram


@dataclass
class ResponsePlanConfig:
    """Configuration for Materials V3 response planning."""
    
    # Edge band extraction (pixel-based, deterministic)
    edge_band_width_px: int = 5  # Fixed pixel width for edge band
    
    # Response strength defaults
    default_core_strength: float = 1.00  # Full response in core region
    default_edge_strength: float = 0.80  # Conservative on edges (avoid halos)
    
    # Per-material overrides
    material_core_strengths: Dict[str, float] = field(default_factory=lambda: {
        'glass': 0.90,      # Subtle on glass
        'water': 0.95,      # Subtle on water
        'foliage': 0.85,    # Conservative (avoid neon)
        'metal': 1.05,      # Slight boost
        'wood': 1.00,       # Neutral
        'stone': 1.00,      # Neutral
    })
    
    material_edge_strengths: Dict[str, float] = field(default_factory=lambda: {
        'glass': 0.70,      # Very conservative
        'water': 0.75,      # Conservative
        'foliage': 0.65,    # Very conservative (halo risk)
        'metal': 0.85,      # Moderate
        'wood': 0.80,       # Conservative
        'stone': 0.80,      # Conservative
    })
    
    # Refinement thresholds
    min_coverage_px: int = 500  # Skip classes below this pixel count
    min_mean_conf: float = 0.20  # Skip classes below this mean confidence
    
    # Decision thresholds for should_refine
    refine_coverage_threshold_px: int = 1000
    refine_conf_ambiguity_threshold: float = 0.70  # Refine if mean conf < this (PR-4C: raised from 0.50)


def extract_edge_band(
    mask: np.ndarray,
    edge_width_px: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract core and edge bands from a binary or confidence mask.
    
    Args:
        mask: HxW float32 or bool mask
        edge_width_px: Pixel width of edge band
        
    Returns:
        core_mask: bool HxW (interior, excluding edge)
        edge_mask: bool HxW (boundary band only)
    """
    if mask.dtype != bool:
        binary_mask = mask > 0.5
    else:
        binary_mask = mask
    
    # Erode to get core (interior away from boundary)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    core_mask = ndimage.binary_erosion(
        binary_mask,
        structure=struct,
        iterations=edge_width_px,
    )
    
    # Edge = original mask − core
    edge_mask = binary_mask & (~core_mask)
    
    return core_mask, edge_mask


def compute_class_stats(
    mask: np.ndarray,
    edge_width_px: int = 5,
) -> dict:
    """Compute per-class statistics for response planning.
    
    Args:
        mask: HxW float32 mask (confidence values in [0,1])
        edge_width_px: Edge band width in pixels
        
    Returns:
        Stats dict with coverage, mean_conf, edge_conf, core_conf, etc.
    """
    if mask is None or mask.size == 0:
        return {
            "coverage": 0.0,
            "coverage_px": 0,
            "mean_conf": 0.0,
            "edge_conf": 0.0,
            "core_conf": 0.0,
            "edge_pixels": 0,
            "core_pixels": 0,
        }
    
    # Convert to float32 if needed
    if mask.dtype == bool:
        mask_f = mask.astype(np.float32)
    else:
        mask_f = mask.astype(np.float32, copy=False)
    
    H, W = mask.shape
    total_pixels = H * W
    
    # Coverage
    binary_mask = mask_f > 0.5
    coverage_px = int(binary_mask.sum())
    coverage = float(coverage_px) / float(total_pixels)
    
    # Mean confidence
    if coverage_px > 0:
        mean_conf = float(mask_f[binary_mask].mean())
    else:
        mean_conf = 0.0
    
    # Extract core vs edge
    core_mask, edge_mask = extract_edge_band(mask_f, edge_width_px)
    
    core_px = int(core_mask.sum())
    edge_px = int(edge_mask.sum())
    
    # Core confidence
    if core_px > 0:
        core_conf = float(mask_f[core_mask].mean())
    else:
        core_conf = 0.0
    
    # Edge confidence
    if edge_px > 0:
        edge_conf = float(mask_f[edge_mask].mean())
    else:
        edge_conf = 0.0
    
    return {
        "coverage": coverage,
        "coverage_px": coverage_px,
        "mean_conf": mean_conf,
        "edge_conf": edge_conf,
        "core_conf": core_conf,
        "edge_pixels": edge_px,
        "core_pixels": core_px,
    }


def compute_response_strengths(
    class_name: str,
    stats: dict,
    config: ResponsePlanConfig,
) -> Tuple[float, float]:
    """Compute planned response strengths (core and edge).
    
    Args:
        class_name: Canonical material class name
        stats: Output from compute_class_stats
        config: ResponsePlanConfig
        
    Returns:
        (core_strength, edge_strength) as floats in [0, ~1.2]
    """
    # Get material-specific defaults or fall back to global
    core_strength = config.material_core_strengths.get(
        class_name,
        config.default_core_strength,
    )
    edge_strength = config.material_edge_strengths.get(
        class_name,
        config.default_edge_strength,
    )
    
    # Attenuate if coverage is low
    if stats["coverage_px"] < config.min_coverage_px:
        # Very low coverage → reduce strength to avoid noise
        attenuation = float(stats["coverage_px"]) / float(config.min_coverage_px)
        core_strength *= attenuation
        edge_strength *= attenuation
    
    # Attenuate edge strength if edge confidence is very low
    if stats["edge_conf"] < 0.25:
        edge_strength *= 0.70  # Extra conservative on uncertain edges
    
    return core_strength, edge_strength


def decide_should_refine(
    class_name: str,
    stats: dict,
    config: ResponsePlanConfig,
    strategy: str = "canary",
) -> Tuple[bool, str]:
    """Decide whether EfficientSAM refinement should be attempted.
    
    Args:
        class_name: Canonical material class name
        stats: Output from compute_class_stats
        config: ResponsePlanConfig
        strategy: "off" | "canary" | "selective" | "aggressive"
        
    Returns:
        (should_refine: bool, reason: str)
    """
    # Strategy: OFF
    if strategy == "off":
        return False, "strategy_off"
    
    # Coverage too low
    if stats["coverage_px"] < config.min_coverage_px:
        return False, "below_coverage_threshold"
    
    # Mean confidence too low (degenerate mask)
    if stats["mean_conf"] < config.min_mean_conf:
        return False, "below_confidence_threshold"
    
    # Strategy: CANARY
    if strategy == "canary":
        # Only refine validated classes from Stage 6
        canary_classes = {"glass", "water", "foliage"}
        if class_name not in canary_classes:
            return False, "not_in_canary_set"
        # Canary classes must have ambiguous confidence
        if stats["mean_conf"] >= config.refine_conf_ambiguity_threshold:
            return False, "confidence_already_high"
        return True, "canary_eligible"
    
    # Strategy: SELECTIVE
    if strategy == "selective":
        # Refine if confidence is ambiguous
        if stats["mean_conf"] < config.refine_conf_ambiguity_threshold:
            return True, "selective_ambiguous_confidence"
        return False, "selective_confidence_high"
    
    # Strategy: AGGRESSIVE
    if strategy == "aggressive":
        return True, "aggressive_all_classes"
    
    return False, f"unknown_strategy_{strategy}"


def compute_edge_signals(
    rgb_image: Optional[np.ndarray],
    edge_band_mask: np.ndarray,
) -> dict:
    """Compute edge signals (boundary pixels, gradient alignment).
    
    Args:
        rgb_image: HxWx3 float32 RGB image (optional, for gradient alignment)
        edge_band_mask: HxW bool edge band mask
        
    Returns:
        Edge signals dict with boundary_pixels and edge_alignment
    """
    boundary_pixels = int(edge_band_mask.sum())
    
    # If no image or boundary too small, return minimal signals
    if rgb_image is None or boundary_pixels < 250:
        return {
            "boundary_pixels": boundary_pixels,
            "edge_alignment": 0.0,
            "notes": ["boundary_too_small"] if boundary_pixels < 250 else [],
        }
    
    # Compute gradient magnitude (Sobel)
    gray = 0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] + 0.114 * rgb_image[..., 2]
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    
    # Edge alignment = mean gradient magnitude at boundary pixels
    edge_alignment = float(grad_mag[edge_band_mask].mean()) if boundary_pixels > 0 else 0.0
    
    return {
        "boundary_pixels": boundary_pixels,
        "edge_alignment": edge_alignment,
        "notes": [],
    }


def decide_refinement(
    class_name: str,
    stats: dict,
    edge_signals: dict,
    config: ResponsePlanConfig,
    strategy: str = "canary",
) -> Tuple[bool, bool, str]:
    """Decide EfficientSAM edge refinement eligibility and recommendation.
    
    Args:
        class_name: Canonical material name
        stats: Class stats from compute_class_stats
        edge_signals: Edge signals from compute_edge_signals
        config: ResponsePlanConfig
        strategy: RefinementStrategy value
        
    Returns:
        (eligible, should_refine_edges, reason)
    """
    # Strategy: OFF
    if strategy == "off":
        return False, False, "strategy_off"
    
    # Coverage too low
    if stats["coverage_px"] < config.min_coverage_px:
        return False, False, "below_coverage_threshold"
    
    # Mean confidence too low (degenerate mask)
    if stats["mean_conf"] < config.min_mean_conf:
        return False, False, "below_confidence_threshold"
    
    # Canary-only eligibility
    canary_classes = {"glass", "water", "foliage"}
    if class_name not in canary_classes:
        return False, False, "not_in_canary_set"
    
    # Eligible for refinement
    eligible = True
    
    # PR-4C: Edge signal guards (learned from foliage regression)
    if edge_signals["boundary_pixels"] < 250:
        return eligible, False, "boundary_too_small"
    
    if edge_signals["edge_alignment"] < 0.10:
        return eligible, False, "weak_edge_alignment"
    
    # Strategy-specific recommendation
    if strategy == "canary":
        if stats["mean_conf"] >= config.refine_conf_ambiguity_threshold:
            return eligible, False, "confidence_already_high"
        return eligible, True, "canary_eligible"
    
    if strategy == "selective":
        if stats["mean_conf"] < config.refine_conf_ambiguity_threshold:
            return eligible, True, "selective_ambiguous_confidence"
        return eligible, False, "selective_confidence_high"
    
    if strategy == "aggressive":
        return eligible, True, "aggressive_all_classes"
    
    return eligible, False, f"unknown_strategy_{strategy}"


def decide_pixel_ops(
    class_name: str,
    stats: dict,
    config: ResponsePlanConfig,
) -> Tuple[bool, bool, str, list]:
    """Decide pixel ops application (glass only for PR-4C).
    
    Args:
        class_name: Canonical material name
        stats: Class stats
        config: ResponsePlanConfig
        
    Returns:
        (eligible, should_apply, reason, recommended_ops)
    """
    # PR-4C: Only glass has pixel ops implementation
    if class_name == "glass":
        # Eligible if present with sufficient coverage
        if stats["coverage_px"] < 1000:
            return False, False, "below_coverage_threshold", []
        
        eligible = True
        
        # Recommend apply if confidence or edge quality is low
        if stats["mean_conf"] < 0.80:
            return eligible, True, "low_mean_confidence", ["glass_response"]
        
        if stats["edge_conf"] < 0.55:
            return eligible, True, "low_edge_confidence", ["glass_response"]
        
        return eligible, False, "confidence_already_high", ["glass_response"]
    
    # Other materials: report-only (no implementation yet)
    recommended_ops_map = {
        "wood": ["microcontrast"],
        "stone": ["microcontrast"],
        "metal": ["microcontrast", "highlight_boost"],
        "fabric": ["texture_clarity"],
        "foliage": ["color_pop", "edge_clarity"],
        "water": ["reflection_boost"],
    }
    
    recommended = recommended_ops_map.get(class_name, [])
    return False, False, "no_implementation", recommended


def generate_response_plan(
    canonical_materials: Dict[str, np.ndarray],
    config: ResponsePlanConfig,
    strategy: str = "canary",
    intent: str = "client",
    quality_tier: str = "max",
    rgb_image: Optional[np.ndarray] = None,
) -> dict:
    """Generate Materials V3 response plan for all present classes.
    
    PR-4C: Schema v3.1 with separated decisions + edge signals.
    
    Args:
        canonical_materials: Dict of canonical_name → mask (HxW float32)
        config: ResponsePlanConfig
        strategy: RefinementStrategy value
        intent: Auto-preset intent (preview/client/hero)
        quality_tier: Auto-preset tier (standard/max/apex)
        rgb_image: Optional HxWx3 RGB image for edge signal computation
        
    Returns:
        Response plan dict (v3.1 schema) suitable for report JSON
    """
    per_class = {}
    
    for class_name, mask in canonical_materials.items():
        stats = compute_class_stats(mask, config.edge_band_width_px)
        
        # Compute edge band mask for edge signals
        core_mask, edge_mask = extract_edge_band(mask, config.edge_band_width_px)
        
        # Compute edge signals (PR-4C)
        edge_signals = compute_edge_signals(rgb_image, edge_mask)
        
        # Compute planned strengths
        core_strength, edge_strength = compute_response_strengths(
            class_name,
            stats,
            config,
        )
        
        # PR-4C: Separate decisions
        refine_eligible, should_refine_edges, refine_reason = decide_refinement(
            class_name,
            stats,
            edge_signals,
            config,
            strategy,
        )
        
        pixelops_eligible, should_apply_pixelops, pixelops_reason, recommended_ops = decide_pixel_ops(
            class_name,
            stats,
            config,
        )
        
        # Assemble class plan (v3.1 schema)
        class_plan = {
            # Core stats
            "present": True,
            "coverage": stats["coverage"],
            "coverage_px": stats["coverage_px"],
            "mean_conf": stats["mean_conf"],
            "edge_conf": stats["edge_conf"],
            "core_conf": stats["core_conf"],
            "edge_pixels": stats["edge_pixels"],
            "core_pixels": stats["core_pixels"],
            
            # Planned strengths (v3.1: nested, but keep deprecated flat keys)
            "strengths": {
                "core": core_strength,
                "edge": edge_strength,
            },
            # Backward compatibility (deprecated in v3.1)
            "core_strength": core_strength,
            "edge_strength": edge_strength,
            
            # PR-4C: Refinement decision (EfficientSAM)
            "refinement": {
                "eligible": refine_eligible,
                "should_refine_edges": should_refine_edges,
                "reason": refine_reason,
                "strategy": strategy,
            },
            
            # PR-4C: Pixel ops decision (glass only for now)
            "pixel_ops": {
                "eligible": pixelops_eligible,
                "should_apply": should_apply_pixelops,
                "reason": pixelops_reason,
                "recommended_ops": recommended_ops,
            },
            
            # PR-4C: Edge signals
            "edge_signals": edge_signals,
            
            # Backward compatibility (deprecated in v3.1)
            # should_refine now explicitly means refinement.should_refine_edges
            "should_refine": should_refine_edges,
            "refine_reason": refine_reason,
            "skip_reason": None if should_refine_edges else refine_reason,
        }
        
        per_class[class_name] = class_plan
    
    # Assemble full plan (v3.1)
    plan = {
        "version": "v3.1",  # PR-4C: Schema version
        "enabled": True,
        "taxonomy": "base",
        "strategy": strategy,
        "scene": {
            "intent": intent,
            "quality_tier": quality_tier,
        },
        "per_class": per_class,
        "summary": {
            "present_classes": list(per_class.keys()),
            "eligible_for_pixel_ops": [k for k, v in per_class.items() if v["pixel_ops"]["eligible"]],
            "eligible_for_refinement": [k for k, v in per_class.items() if v["refinement"]["eligible"]],
            # PR-4C: Reason histograms for actionable insights
            "pixel_ops_reasons": _build_reason_histogram([v["pixel_ops"]["reason"] for v in per_class.values()]),
            "refinement_reasons": _build_reason_histogram([v["refinement"]["reason"] for v in per_class.values()]),
        },
        "notes": ["PR-4C: separated refinement + pixel ops decisions, added edge signals"],
    }
    
    return plan
