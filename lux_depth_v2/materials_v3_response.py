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
    refine_conf_ambiguity_threshold: float = 0.50  # Refine if mean conf < this


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


def generate_response_plan(
    canonical_materials: Dict[str, np.ndarray],
    config: ResponsePlanConfig,
    strategy: str = "canary",
    intent: str = "client",
    quality_tier: str = "max",
) -> dict:
    """Generate Materials V3 response plan for all present classes.
    
    Args:
        canonical_materials: Dict of canonical_name → mask (HxW float32)
        config: ResponsePlanConfig
        strategy: RefinementStrategy value
        intent: Auto-preset intent (preview/client/hero)
        quality_tier: Auto-preset tier (standard/max/apex)
        
    Returns:
        Response plan dict suitable for report JSON
    """
    per_class = {}
    
    for class_name, mask in canonical_materials.items():
        stats = compute_class_stats(mask, config.edge_band_width_px)
        
        # Compute planned strengths
        core_strength, edge_strength = compute_response_strengths(
            class_name,
            stats,
            config,
        )
        
        # Decide refinement
        should_refine, refine_reason = decide_should_refine(
            class_name,
            stats,
            config,
            strategy,
        )
        
        # Assemble class plan
        class_plan = {
            "present": True,
            "coverage": stats["coverage"],
            "coverage_px": stats["coverage_px"],
            "mean_conf": stats["mean_conf"],
            "edge_conf": stats["edge_conf"],
            "core_conf": stats["core_conf"],
            "edge_pixels": stats["edge_pixels"],
            "core_pixels": stats["core_pixels"],
            "core_strength": core_strength,
            "edge_strength": edge_strength,
            "should_refine": should_refine,
            "refine_reason": refine_reason,
            "skip_reason": None if should_refine else refine_reason,
        }
        
        per_class[class_name] = class_plan
    
    # Assemble full plan
    plan = {
        "enabled": True,
        "taxonomy": "base",  # Will be expanded in future PRs
        "strategy": strategy,
        "scene": {
            "intent": intent,
            "quality_tier": quality_tier,
        },
        "per_class": per_class,
        "notes": ["PR-4A: no pixel ops applied; planning only"],
    }
    
    return plan
