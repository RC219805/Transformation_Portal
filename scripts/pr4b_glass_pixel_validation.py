#!/usr/bin/env python3
"""
PR-4B Glass Pixel Response Validation Script

Measures PIXEL-LEVEL impact (not mask quality):
- Local edge contrast around glass boundaries
- Halo detection
- Color shift control
- Gradient magnitude changes

Runs baseline APEX vs canary (glass-enabled) APEX.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2 import torch_ops, io_utils

log = logging.getLogger(__name__)


def compute_edge_band_stats(
    image: np.ndarray,
    mask: np.ndarray,
    band_width: int = 10,
) -> dict:
    """
    Compute gradient stats in a boundary band around a mask.
    
    Args:
        image: HxWx3 RGB in [0,1]
        mask: HxW binary or confidence mask
        band_width: pixels on each side of boundary
        
    Returns:
        Stats dict with gradient magnitude, contrast, etc.
    """
    from scipy.ndimage import sobel, binary_dilation, binary_erosion
    
    # Create boundary band
    if mask.max() > 1.0:
        binary_mask = mask > 0.5
    else:
        binary_mask = mask > 0.5
        
    dilated = binary_dilation(binary_mask, iterations=band_width)
    eroded = binary_erosion(binary_mask, iterations=band_width)
    boundary_band = dilated & ~eroded
    
    if boundary_band.sum() == 0:
        return {
            "boundary_pixels": 0,
            "mean_gradient_mag": 0.0,
            "median_gradient_mag": 0.0,
            "max_gradient_mag": 0.0,
        }
    
    # Compute gradients (luma)
    if image.ndim == 3:
        luma = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        luma = image
        
    grad_x = sobel(luma, axis=1)
    grad_y = sobel(luma, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Stats in boundary band only
    boundary_grads = grad_mag[boundary_band]
    
    return {
        "boundary_pixels": int(boundary_band.sum()),
        "mean_gradient_mag": float(boundary_grads.mean()),
        "median_gradient_mag": float(np.median(boundary_grads)),
        "max_gradient_mag": float(boundary_grads.max()),
        "p95_gradient_mag": float(np.percentile(boundary_grads, 95)),
    }


def detect_halos(
    baseline_img: np.ndarray,
    canary_img: np.ndarray,
    mask: np.ndarray,
    band_width: int = 10,
) -> dict:
    """
    Detect halos by looking for excessive delta near boundaries.
    
    Returns:
        Halo risk score and stats
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    binary_mask = mask > 0.5
    dilated = binary_dilation(binary_mask, iterations=band_width)
    eroded = binary_erosion(binary_mask, iterations=band_width)
    boundary_band = dilated & ~eroded
    
    if boundary_band.sum() == 0:
        return {"halo_risk": "unknown", "boundary_pixels": 0}
    
    # Delta magnitude in boundary
    delta = np.abs(canary_img - baseline_img)
    if delta.ndim == 3:
        delta_mag = np.linalg.norm(delta, axis=2)
    else:
        delta_mag = delta
        
    boundary_delta = delta_mag[boundary_band]
    
    # Heuristic: halo if P95 delta > 0.1 (10% RGB change)
    p95_delta = float(np.percentile(boundary_delta, 95))
    mean_delta = float(boundary_delta.mean())
    
    halo_risk = "high" if p95_delta > 0.1 else ("moderate" if p95_delta > 0.05 else "low")
    
    return {
        "halo_risk": halo_risk,
        "boundary_pixels": int(boundary_band.sum()),
        "mean_delta_boundary": mean_delta,
        "p95_delta_boundary": p95_delta,
        "max_delta_boundary": float(boundary_delta.max()),
    }


def run_single_scene(
    scene_name: str,
    input_path: Path,
    output_root: Path,
) -> dict:
    """Run baseline and canary on one scene and compare pixel impact."""
    
    log.info(f"=== Processing {scene_name} ===")
    
    # Create output dirs
    baseline_dir = output_root / f"{scene_name}_A_baseline"
    canary_dir = output_root / f"{scene_name}_B_glass_canary"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    canary_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baseline
    log.info(f"Running baseline APEX for {scene_name}...")
    baseline_cfg = PipelineConfig(
        output_dir=baseline_dir,
        preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
    )
    baseline_pipe = LuxPipelineV2(baseline_cfg)
    baseline_result = baseline_pipe.process_one(input_path)
    baseline_report = baseline_result.get("report", {})
    
    # Run canary
    log.info(f"Running glass canary APEX for {scene_name}...")
    canary_cfg = PipelineConfig(
        output_dir=canary_dir,
        preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS,
    )
    canary_pipe = LuxPipelineV2(canary_cfg)
    canary_result = canary_pipe.process_one(input_path)
    canary_report = canary_result.get("report", {})
    
    # Load output images for comparison
    baseline_outputs = sorted(baseline_dir.glob("*.png")) + sorted(baseline_dir.glob("*.tif"))
    canary_outputs = sorted(canary_dir.glob("*.png")) + sorted(canary_dir.glob("*.tif"))
    
    if not baseline_outputs or not canary_outputs:
        log.warning(f"Missing outputs for {scene_name}")
        return {
            "scene": scene_name,
            "status": "outputs_missing",
        }
    
    # Use the "master" output (typically the largest or final enhanced image)
    baseline_img_path = baseline_outputs[0]
    canary_img_path = canary_outputs[0]
    
    baseline_img, _ = io_utils.read_rgb_any(baseline_img_path)
    canary_img, _ = io_utils.read_rgb_any(canary_img_path)
    
    # Ensure same shape
    if baseline_img.shape != canary_img.shape:
        log.warning(f"Shape mismatch: baseline {baseline_img.shape} vs canary {canary_img.shape}")
        return {"scene": scene_name, "status": "shape_mismatch"}
    
    # Check if pixel ops were actually applied
    pixel_ops_report = canary_report.get("materials_v3_pixel_ops", {})
    applied = pixel_ops_report.get("enabled", False)
    
    if not applied:
        return {
            "scene": scene_name,
            "status": "pixel_ops_not_applied",
            "reason": pixel_ops_report.get("reason", "unknown"),
        }
    
    # Extract glass mask from canary report (normalized)
    # Approximation: if we don't have direct access, compute delta mask
    delta_mag = np.linalg.norm(canary_img - baseline_img, axis=2)
    glass_mask_approx = (delta_mag > 0.001).astype(np.float32)
    
    # Compute pixel-level metrics
    baseline_edge_stats = compute_edge_band_stats(baseline_img, glass_mask_approx)
    canary_edge_stats = compute_edge_band_stats(canary_img, glass_mask_approx)
    halo_stats = detect_halos(baseline_img, canary_img, glass_mask_approx)
    
    # Overall pixel delta
    overall_delta = np.abs(canary_img - baseline_img)
    mean_delta_global = float(overall_delta.mean())
    max_delta_global = float(overall_delta.max())
    
    # Glass region delta
    glass_region = glass_mask_approx > 0.5
    if glass_region.sum() > 0:
        glass_delta = overall_delta[glass_region]
        mean_delta_glass = float(glass_delta.mean())
        max_delta_glass = float(glass_delta.max())
    else:
        mean_delta_glass = 0.0
        max_delta_glass = 0.0
    
    # Gradient change (edge contrast improvement)
    gradient_delta = canary_edge_stats["mean_gradient_mag"] - baseline_edge_stats["mean_gradient_mag"]
    
    return {
        "scene": scene_name,
        "status": "success",
        "pixel_ops_applied": applied,
        "pixel_ops_details": pixel_ops_report,
        "overall_delta": {
            "mean": mean_delta_global,
            "max": max_delta_global,
        },
        "glass_region_delta": {
            "mean": mean_delta_glass,
            "max": max_delta_glass,
            "pixels": int(glass_region.sum()),
        },
        "edge_stats": {
            "baseline": baseline_edge_stats,
            "canary": canary_edge_stats,
            "gradient_delta": gradient_delta,
        },
        "halo_detection": halo_stats,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Define validation set (scenes with glass)
    benchmark_scenes = {
        "Kitchen": repo_root / "assets/phase2_bench/interior_kitchen_750.tiff",
        "Bedroom": repo_root / "assets/phase2_bench/interior_bedroom.tiff",
        "Bathroom": repo_root / "assets/phase2_bench/interior_bathroom.tiff",
    }
    
    output_root = repo_root / "outputs/pr4b_glass_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = []
    for scene_name, input_path in benchmark_scenes.items():
        if not input_path.exists():
            log.warning(f"Missing input: {input_path}")
            continue
            
        scene_result = run_single_scene(scene_name, input_path, output_root)
        results.append(scene_result)
    
    # Aggregate and decide
    applied_count = sum(1 for r in results if r.get("pixel_ops_applied"))
    halo_risks = [r["halo_detection"]["halo_risk"] for r in results if "halo_detection" in r]
    high_halo_count = sum(1 for risk in halo_risks if risk == "high")
    
    gradient_improvements = [
        r["edge_stats"]["gradient_delta"]
        for r in results
        if "edge_stats" in r and r["edge_stats"]["gradient_delta"] > 0.0
    ]
    
    # Decision criteria
    promotion_ok = (
        applied_count >= 2  # At least 2/3 scenes applied
        and high_halo_count == 0  # No high halo risk
        and len(gradient_improvements) >= 2  # At least 2 scenes improved edges
    )
    
    summary = {
        "validation_date": "2025-12-14",
        "pr": "PR-4B",
        "feature": "glass_pixel_response",
        "scenes_tested": len(results),
        "scenes_applied": applied_count,
        "high_halo_risks": high_halo_count,
        "gradient_improvements": len(gradient_improvements),
        "promotion_recommended": promotion_ok,
        "details": results,
    }
    
    summary_path = output_root / "pr4b_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"\n=== PR-4B Validation Summary ===")
    log.info(f"Scenes tested: {len(results)}")
    log.info(f"Pixel ops applied: {applied_count}/{len(results)}")
    log.info(f"High halo risks: {high_halo_count}")
    log.info(f"Gradient improvements: {len(gradient_improvements)}")
    log.info(f"Promotion recommended: {promotion_ok}")
    log.info(f"\nFull report: {summary_path}")
    
    return 0 if promotion_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
