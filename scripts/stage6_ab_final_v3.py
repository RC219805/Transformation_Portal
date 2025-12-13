#!/usr/bin/env python3
"""
Stage 6 A/B Test with Boundary Metrics (PR-3C Final)

This script runs the golden baseline A/B comparison with:
- In-memory mask extraction (no disk I/O)
- Material name canonicalization
- Boundary F1 as regression guard
- Edge alignment as improvement metric
- Robust error handling
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Lux Depth V2 imports
from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.material_segmentation import create_material_segmenter
from lux_depth_v2.metrics.boundary_metrics import compute_full_boundary_metrics
from lux_depth_v2.materials_v3_taxonomy import normalize_material_dict
from lux_depth_v2 import io_utils, torch_ops
from scipy.ndimage import sobel

# Configuration
FORCE_DEVICE = "cpu"  # Stable, reproducible
MIN_BOUNDARY_PX = 250  # Minimum boundary pixels for valid metric
EDGE_ALIGN_DELTA_THRESHOLD = 0.02  # Improvement threshold
BF1_GUARD_THRESHOLD = 0.85  # Regression guard

BENCHMARK_SET = {
    "kitchen": {
        "path": "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Kitchen_UltraQuality.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "target_classes": ["glass"],
    },
    "bedroom": {
        "path": "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_PrimaryBedroom_UltraQuality.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "target_classes": ["glass"],
    },
    "bathroom": {
        "path": "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_PrimaryBathroom_UltraQuality.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "target_classes": ["glass"],
    },
    "pool": {
        "path": "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Pool_UltraQuality.tif",
        "baseline_preset": Preset.EXTERIOR_POOL_APEX_QUALITY,
        "canary_preset": Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM,
        "target_classes": ["water", "foliage"],
    },
    "aerial": {
        "path": "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Aerial_UltraQuality.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,  # Fixed: use APEX
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "target_classes": ["foliage"],
    },
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _as_dict(x) -> dict:
    """Safe conversion of metrics result to dict."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if hasattr(x, "to_dict"):
        return x.to_dict()
    raise TypeError(f"Unexpected metrics type: {type(x)}")


def compute_image_gradients(rgb: np.ndarray) -> np.ndarray:
    """Compute image gradient magnitude using Sobel operator.
    
    Parameters
    ----------
    rgb : np.ndarray
        RGB image (H, W, 3) in [0, 1]
    
    Returns
    -------
    np.ndarray
        Gradient magnitude (H, W) in [0, 1]
    """
    # Convert to grayscale
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    
    # Compute gradients
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    
    # Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to [0, 1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    
    return magnitude.astype(np.float32)


def extract_masks_in_memory(
    input_path: Path,
    preset: Preset,
    device: torch.device,
    target_classes: list[str],
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Extract masks in-memory using the segmenter.
    
    Returns:
        (masks_dict, rgb01) where masks_dict is canonicalized
    """
    # Load image
    rgb01, _ = io_utils.read_rgb_any(input_path)
    
    # Create config for this preset
    cfg = PipelineConfig(preset=preset)
    
    # Create segmenter
    seg = create_material_segmenter(cfg.segmentation, device)
    
    # Convert to torch
    rgb_t = torch_ops.to_torch_rgb(rgb01, device)
    
    # Predict masks
    masks_torch = seg.predict(rgb_t)  # Dict[str, torch.Tensor] (1,1,H,W)
    
    # Convert to numpy dict
    raw_masks = {
        k: v[0, 0].cpu().numpy().astype(np.float32)
        for k, v in masks_torch.items()
    }
    
    # Canonicalize material names
    canonical_masks = normalize_material_dict(raw_masks)
    
    # Extract only target classes
    result = {}
    for cls in target_classes:
        if cls in canonical_masks:
            result[cls] = canonical_masks[cls]
        else:
            log.warning(f"Target class '{cls}' not found in segmentation output")
            result[cls] = None
    
    return result, rgb01


def compute_class_metrics(
    base_mask: np.ndarray,
    canary_mask: np.ndarray,
    rgb01: np.ndarray,
    class_name: str,
) -> dict:
    """
    Compute boundary metrics for a single class.
    
    Returns dict with:
        - boundary_f1 (canary vs baseline)
        - edge_align_baseline (baseline vs gradients)
        - edge_align_canary (canary vs gradients)
        - edge_align_delta
        - boundary_pixels
        - status
    """
    H, W = base_mask.shape
    
    # Resize RGB to mask resolution using PIL (no scikit-image dependency)
    rgb_u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    rgb_resized_pil = Image.fromarray(rgb_u8).resize((W, H), resample=Image.BILINEAR)
    rgb_resized = np.array(rgb_resized_pil, dtype=np.float32) / 255.0
    
    # Compute image gradients
    gradients = compute_image_gradients(rgb_resized)
    
    # Baseline metrics (baseline vs gradients for edge alignment)
    base_result = compute_full_boundary_metrics(
        pred_mask=base_mask,
        ref_mask=base_mask,  # ref unused for edge alignment
        image_gradients=gradients,
        band_width_px=5,
    )
    base_metrics = _as_dict(base_result)
    
    # Canary metrics (canary vs gradients)
    canary_result = compute_full_boundary_metrics(
        pred_mask=canary_mask,
        ref_mask=canary_mask,  # ref unused for edge alignment
        image_gradients=gradients,
        band_width_px=5,
    )
    canary_metrics = _as_dict(canary_result)
    
    # Regression check (canary vs baseline boundary agreement)
    regression_result = compute_full_boundary_metrics(
        pred_mask=canary_mask,
        ref_mask=base_mask,
        image_gradients=None,  # Not needed for BF1
        band_width_px=5,
    )
    regression_metrics = _as_dict(regression_result)
    
    edge_align_baseline = base_metrics.get("edge_alignment", 0.0)
    edge_align_canary = canary_metrics.get("edge_alignment", 0.0)
    edge_align_delta = edge_align_canary - edge_align_baseline
    
    bf1 = regression_metrics.get("boundary_f1", 0.0)
    boundary_px = regression_metrics.get("boundary_pixels", 0)
    
    # Determine status
    if boundary_px < MIN_BOUNDARY_PX:
        status = "skipped_low_coverage"
    elif edge_align_delta > EDGE_ALIGN_DELTA_THRESHOLD and bf1 >= BF1_GUARD_THRESHOLD:
        status = "improved"
    elif bf1 < BF1_GUARD_THRESHOLD:
        status = "regressed"
    else:
        status = "unchanged"
    
    return {
        "class": class_name,
        "boundary_f1": float(bf1),
        "edge_align_baseline": float(edge_align_baseline),
        "edge_align_canary": float(edge_align_canary),
        "edge_align_delta": float(edge_align_delta),
        "boundary_pixels": int(boundary_px),
        "status": status,
    }


def run_scene_ab(
    scene_name: str,
    scene_config: dict,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run A/B comparison for a single scene."""
    log.info(f"\n{'='*60}")
    log.info(f"Processing scene: {scene_name}")
    log.info(f"{'='*60}")
    
    input_path = Path(scene_config["path"])
    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        return {
            "scene": scene_name,
            "success": False,
            "error": f"Input not found: {input_path}",
        }
    
    baseline_preset = scene_config["baseline_preset"]
    canary_preset = scene_config["canary_preset"]
    target_classes = scene_config["target_classes"]
    
    try:
        # Extract baseline masks
        log.info(f"Extracting baseline masks ({baseline_preset.value})...")
        base_masks, rgb01 = extract_masks_in_memory(
            input_path, baseline_preset, device, target_classes
        )
        
        # Extract canary masks
        log.info(f"Extracting canary masks ({canary_preset.value})...")
        canary_masks, _ = extract_masks_in_memory(
            input_path, canary_preset, device, target_classes
        )
        
        # Compute metrics per class
        class_results = []
        improvements = []
        regressions = []
        
        for cls in target_classes:
            base_mask = base_masks.get(cls)
            canary_mask = canary_masks.get(cls)
            
            if base_mask is None or canary_mask is None:
                log.warning(f"  {cls}: missing mask (skipped)")
                class_results.append({
                    "class": cls,
                    "status": "missing_mask",
                })
                continue
            
            log.info(f"  Computing metrics for {cls}...")
            metrics = compute_class_metrics(base_mask, canary_mask, rgb01, cls)
            class_results.append(metrics)
            
            if metrics["status"] == "improved":
                improvements.append(cls)
                log.info(f"    ✓ IMPROVED (Δ={metrics['edge_align_delta']:.4f})")
            elif metrics["status"] == "regressed":
                regressions.append(cls)
                log.warning(f"    ✗ REGRESSED (BF1={metrics['boundary_f1']:.3f})")
            elif metrics["status"] == "skipped_low_coverage":
                log.info(f"    - SKIPPED (boundary_px={metrics['boundary_pixels']})")
            else:
                log.info(f"    = UNCHANGED (Δ={metrics['edge_align_delta']:.4f})")
        
        # Scene-level decision
        scene_improved = len(improvements) > 0 and len(regressions) == 0
        
        result = {
            "scene": scene_name,
            "success": True,
            "input_path": str(input_path),
            "baseline_preset": baseline_preset.value,
            "canary_preset": canary_preset.value,
            "target_classes": target_classes,
            "class_results": class_results,
            "improvements": improvements,
            "regressions": regressions,
            "scene_improved": scene_improved,
        }
        
        log.info(f"\n{scene_name} summary:")
        log.info(f"  Improvements: {improvements}")
        log.info(f"  Regressions: {regressions}")
        log.info(f"  Overall: {'✓ IMPROVED' if scene_improved else '✗ NOT IMPROVED'}")
        
        return result
        
    except Exception as e:
        log.exception(f"Error processing {scene_name}")
        return {
            "scene": scene_name,
            "success": False,
            "error": str(e),
        }


def main():
    """Run full Stage 6 A/B benchmark with boundary metrics."""
    output_dir = Path("outputs/stage6_ab_pr3c_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(FORCE_DEVICE)
    log.info(f"Using device: {device}")
    log.info(f"Output directory: {output_dir}")
    
    results = []
    
    for scene_name, scene_config in BENCHMARK_SET.items():
        result = run_scene_ab(scene_name, scene_config, device, output_dir)
        results.append(result)
    
    # Aggregate summary
    success_count = sum(1 for r in results if r.get("success", False))
    improved_count = sum(1 for r in results if r.get("scene_improved", False))
    
    all_improvements = []
    all_regressions = []
    for r in results:
        if r.get("success"):
            all_improvements.extend(r.get("improvements", []))
            all_regressions.extend(r.get("regressions", []))
    
    summary = {
        "test": "Stage 6 A/B with Boundary Metrics (PR-3C Final)",
        "device": str(device),
        "total_scenes": len(BENCHMARK_SET),
        "successful_scenes": success_count,
        "improved_scenes": improved_count,
        "all_improvements": all_improvements,
        "all_regressions": all_regressions,
        "promotion_recommended": (
            improved_count >= 3
            and len(all_regressions) == 0
            and any("pool" in r["scene"] and r.get("scene_improved") for r in results)
        ),
        "results": results,
    }
    
    # Write summary
    summary_path = output_dir / "stage6_ab_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"\n{'='*60}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Total scenes: {len(BENCHMARK_SET)}")
    log.info(f"Successful: {success_count}")
    log.info(f"Improved: {improved_count}")
    log.info(f"All improvements: {all_improvements}")
    log.info(f"All regressions: {all_regressions}")
    log.info(f"Promotion recommended: {summary['promotion_recommended']}")
    log.info(f"\nSummary written to: {summary_path}")
    
    return 0 if summary["promotion_recommended"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
