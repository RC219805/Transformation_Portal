#!/usr/bin/env python3
"""
PR-4D Stone Pixel Response Validation Script

Two-pass validation approach:
- Pass 1: Normal gating (should skip when already high quality)
- Pass 2: Forced apply (prove ops correctness + safety)

Metrics per scene:
- coverage_px, core_px, edge_px
- mean_delta (stone region)
- p95_edge_delta + halo risk
- clamp count
- gradient change localized to stone mask

Acceptance criteria:
- forced apply: applied==true for ≥2 scenes
- halo risk: 0 HIGH cases
- mean_delta < 0.02 (stone region)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2 import torch_ops, io_utils

log = logging.getLogger(__name__)


def _coerce_report(obj: object) -> dict:
    """Coerce LuxPipelineV2.process_one() return into a report dict."""
    if isinstance(obj, dict):
        if "report" in obj and isinstance(obj["report"], dict):
            return obj["report"]
        return obj
    return {}


def compute_stone_region_stats(
    baseline_img: np.ndarray,
    canary_img: np.ndarray,
    stone_mask: np.ndarray,
) -> dict:
    """
    Compute delta stats in stone region.
    
    Args:
        baseline_img: HxWx3 RGB in [0,1]
        canary_img: HxWx3 RGB in [0,1]
        stone_mask: HxW stone confidence mask
        
    Returns:
        Stats dict with mean/max delta, etc.
    """
    binary_mask = stone_mask > 0.5
    
    if binary_mask.sum() == 0:
        return {
            "stone_pixels": 0,
            "mean_delta": 0.0,
            "max_delta": 0.0,
            "p95_delta": 0.0,
        }
    
    # Compute delta magnitude
    delta = np.abs(canary_img - baseline_img)
    delta_mag = np.linalg.norm(delta, axis=2) if delta.ndim == 3 else delta
    
    stone_delta = delta_mag[binary_mask]
    
    return {
        "stone_pixels": int(binary_mask.sum()),
        "mean_delta": float(stone_delta.mean()),
        "max_delta": float(stone_delta.max()),
        "p95_delta": float(np.percentile(stone_delta, 95)),
        "median_delta": float(np.median(stone_delta)),
    }


def compute_edge_band_stats(
    image: np.ndarray,
    mask: np.ndarray,
    band_width: int = 5,
) -> dict:
    """
    Compute gradient stats in edge band around stone.
    
    Args:
        image: HxWx3 RGB in [0,1]
        mask: HxW stone mask
        band_width: Edge band width in pixels
        
    Returns:
        Stats dict with gradient magnitude, contrast, etc.
    """
    from scipy.ndimage import sobel, binary_erosion
    
    binary_mask = mask > 0.5
    
    if binary_mask.sum() == 0:
        return {
            "boundary_pixels": 0,
            "mean_gradient_mag": 0.0,
            "p95_gradient_mag": 0.0,
        }
    
    # Create edge band using erosion
    struct = np.ones((band_width * 2 + 1, band_width * 2 + 1), dtype=bool)
    core = binary_erosion(binary_mask, structure=struct)
    edge_band = binary_mask & ~core
    
    if edge_band.sum() == 0:
        return {
            "boundary_pixels": 0,
            "mean_gradient_mag": 0.0,
            "p95_gradient_mag": 0.0,
        }
    
    # Compute gradients (luma)
    luma = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    grad_x = sobel(luma, axis=1)
    grad_y = sobel(luma, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Stats in edge band only
    edge_grads = grad_mag[edge_band]
    
    return {
        "boundary_pixels": int(edge_band.sum()),
        "mean_gradient_mag": float(edge_grads.mean()),
        "median_gradient_mag": float(np.median(edge_grads)),
        "max_gradient_mag": float(edge_grads.max()),
        "p95_gradient_mag": float(np.percentile(edge_grads, 95)),
    }


def detect_halos(
    baseline_img: np.ndarray,
    canary_img: np.ndarray,
    mask: np.ndarray,
    band_width: int = 5,
    p95_threshold: float = 0.06,
) -> dict:
    """
    Detect halos by looking for excessive delta near boundaries.
    
    Args:
        baseline_img: HxWx3 RGB baseline
        canary_img: HxWx3 RGB canary
        mask: HxW stone mask
        band_width: Edge band width
        p95_threshold: P95 threshold for HIGH risk
        
    Returns:
        Halo risk score and stats
    """
    from scipy.ndimage import binary_erosion
    
    binary_mask = mask > 0.5
    
    if binary_mask.sum() == 0:
        return {"halo_risk": "NONE", "boundary_pixels": 0}
    
    # Create edge band
    struct = np.ones((band_width * 2 + 1, band_width * 2 + 1), dtype=bool)
    core = binary_erosion(binary_mask, structure=struct)
    edge_band = binary_mask & ~core
    
    if edge_band.sum() == 0:
        return {"halo_risk": "NONE", "boundary_pixels": 0}
    
    # Delta magnitude in boundary
    delta = np.abs(canary_img - baseline_img)
    delta_mag = np.linalg.norm(delta, axis=2) if delta.ndim == 3 else delta
    
    boundary_delta = delta_mag[edge_band]
    
    # Heuristic matching StoneResponseConfig
    p95_delta = float(np.percentile(boundary_delta, 95))
    mean_delta = float(boundary_delta.mean())
    
    if p95_delta > p95_threshold:
        halo_risk = "HIGH"
    elif p95_delta > p95_threshold * 0.7:
        halo_risk = "MEDIUM"
    else:
        halo_risk = "LOW"
    
    return {
        "halo_risk": halo_risk,
        "boundary_pixels": int(edge_band.sum()),
        "mean_delta_boundary": mean_delta,
        "p95_delta_boundary": p95_delta,
        "max_delta_boundary": float(boundary_delta.max()),
    }


def should_skip_scene_for_device(
    scene_name: str,
    input_path: Path,
    device_type: str,
    max_mp_mps: float = 30.0,
) -> Optional[str]:
    """Determine if scene should be skipped due to device limitations."""
    if device_type != "mps":
        return None
    
    # Check megapixels for MPS OOM guard
    try:
        img, _ = io_utils.read_rgb_any(input_path)
        h, w = img.shape[:2]
        megapixels = (h * w) / 1_000_000
        
        if megapixels > max_mp_mps:
            return f"mps_oom_guard_mp={megapixels:.1f}>limit={max_mp_mps}"
    except Exception as e:
        return f"image_load_failed: {e}"
    
    return None


def run_single_scene(
    scene_name: str,
    input_path: Path,
    output_root: Path,
    device: Optional[str] = None,
    force_apply: bool = False,
) -> dict:
    """Run baseline and canary on one scene and compare stone pixel impact."""
    
    log.info(f"=== Processing {scene_name} (force_apply={force_apply}) ===")
    
    # Check device and skip if needed
    if device:
        device_type = device
    else:
        import torch
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    skip_reason = should_skip_scene_for_device(scene_name, input_path, device_type)
    if skip_reason:
        log.warning(f"Skipping {scene_name}: {skip_reason}")
        return {
            "scene": scene_name,
            "status": "skipped",
            "skip_reason": skip_reason,
        }
    
    # Create output dirs
    pass_label = "forced" if force_apply else "normal"
    baseline_dir = output_root / f"{scene_name}_A_baseline_{pass_label}"
    canary_dir = output_root / f"{scene_name}_B_stone_{pass_label}"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    canary_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run baseline
        log.info(f"Running baseline APEX for {scene_name}...")
        baseline_cfg = PipelineConfig(
            output_dir=baseline_dir,
            preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
        )
        if device:
            baseline_cfg.device = device
        baseline_pipe = LuxPipelineV2(baseline_cfg)
        baseline_result = baseline_pipe.process_one(input_path)
        baseline_report = _coerce_report(baseline_result)
    except Exception as e:
        log.error(f"Baseline processing failed for {scene_name}: {e}")
        return {
            "scene": scene_name,
            "status": "baseline_failed",
            "error": str(e),
        }
    
    try:
        # Run canary
        log.info(f"Running stone canary APEX for {scene_name}...")
        
        # Select preset based on force_apply flag
        canary_preset = (
            Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_STONE_VALIDATE
            if force_apply
            else Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_STONE
        )
        
        canary_cfg = PipelineConfig(
            output_dir=canary_dir,
            preset=canary_preset,
        )
        if device:
            canary_cfg.device = device
        canary_pipe = LuxPipelineV2(canary_cfg)
        canary_result = canary_pipe.process_one(input_path)
        canary_report = _coerce_report(canary_result)
    except Exception as e:
        log.error(f"Canary processing failed for {scene_name}: {e}")
        return {
            "scene": scene_name,
            "status": "canary_failed",
            "error": str(e),
        }
    
    # Load output images for comparison
    baseline_outputs = sorted(baseline_dir.glob("*.png")) + sorted(baseline_dir.glob("*.tif"))
    canary_outputs = sorted(canary_dir.glob("*.png")) + sorted(canary_dir.glob("*.tif"))
    
    if not baseline_outputs or not canary_outputs:
        log.warning(f"Missing outputs for {scene_name}")
        return {
            "scene": scene_name,
            "status": "outputs_missing",
        }
    
    # Use first output (typically master)
    baseline_img_path = baseline_outputs[0]
    canary_img_path = canary_outputs[0]
    
    baseline_img, _ = io_utils.read_rgb_any(baseline_img_path)
    canary_img, _ = io_utils.read_rgb_any(canary_img_path)
    
    # Ensure same shape
    if baseline_img.shape != canary_img.shape:
        log.warning(f"Shape mismatch: baseline {baseline_img.shape} vs canary {canary_img.shape}")
        return {"scene": scene_name, "status": "shape_mismatch"}
    
    # Check if pixel ops were actually applied
    pixel_ops_report = canary_report.get("materials_v3_pixel_ops", {}) or {}
    stone_ops = pixel_ops_report.get("stone", {})
    stone_stats = stone_ops.get("stone_stats", {})
    
    applied = stone_stats.get("applied", False)
    
    log.info(f"Stone pixel ops applied: {applied}")
    
    # Extract stone mask from segmentation
    materials_v3_meta = canary_report.get("materials_v3_metadata", {}) or {}
    
    # Try to get stone mask
    stone_mask = None
    if "materials" in materials_v3_meta:
        materials = materials_v3_meta["materials"]
        # Normalize to find stone
        from lux_depth_v2.materials_v3_taxonomy import normalize_material_dict
        normalized = normalize_material_dict(materials)
        stone_mask_raw = normalized.get("stone")
        if stone_mask_raw is not None:
            if hasattr(stone_mask_raw, 'cpu'):
                stone_mask = stone_mask_raw.cpu().numpy()
            else:
                stone_mask = stone_mask_raw
            if stone_mask.ndim == 4:
                stone_mask = stone_mask[0, 0]
            elif stone_mask.ndim == 3:
                stone_mask = stone_mask[0]
            stone_mask = stone_mask.astype(np.float32)
    
    # Compute metrics
    result = {
        "scene": scene_name,
        "status": "success",
        "force_apply": force_apply,
        "pixel_ops_applied": applied,
    }
    
    if applied:
        result.update({
            "coverage_px": stone_stats.get("coverage_px", 0),
            "core_px": stone_stats.get("core_px", 0),
            "edge_px": stone_stats.get("edge_px", 0),
            "mean_delta": stone_stats.get("mean_delta", 0.0),
            "halo_risk": stone_stats.get("halo_risk", "UNKNOWN"),
            "clamp_count": stone_stats.get("clamp_count", 0),
            "edge_clamp_count": stone_stats.get("edge_clamp_count", 0),
        })
    else:
        result["skip_reason"] = stone_stats.get("reason", "unknown")
    
    # Additional validation metrics if stone mask available
    if stone_mask is not None and stone_mask.sum() > 0:
        stone_region_stats = compute_stone_region_stats(baseline_img, canary_img, stone_mask)
        result["validation_stone_region"] = stone_region_stats
        
        edge_stats = compute_edge_band_stats(canary_img, stone_mask, band_width=3)
        result["validation_edge_band"] = edge_stats
        
        halo_stats = detect_halos(baseline_img, canary_img, stone_mask, band_width=3, p95_threshold=0.06)
        result["validation_halo"] = halo_stats
    
    return result


def main():
    parser = argparse.ArgumentParser(description="PR-4D Stone Pixel Response Validation")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["Kitchen", "GreatRoom", "Pool", "Bedroom", "Bathroom"],
        help="Scenes to validate",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/sample_images"),
        help="Input directory with test images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pr4d_stone_validation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--force-apply",
        action="store_true",
        help="Run forced apply pass (validation preset)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for scene in args.scenes:
        # Look for input image
        input_candidates = list(args.input_dir.glob(f"{scene}.*"))
        if not input_candidates:
            log.warning(f"No input found for {scene} in {args.input_dir}")
            results.append({
                "scene": scene,
                "status": "input_not_found",
            })
            continue
        
        input_path = input_candidates[0]
        
        scene_result = run_single_scene(
            scene_name=scene,
            input_path=input_path,
            output_root=args.output_dir,
            device=args.device,
            force_apply=args.force_apply,
        )
        
        results.append(scene_result)
    
    # Save summary
    summary_path = args.output_dir / f"pr4d_validation_summary_{'forced' if args.force_apply else 'normal'}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Validation complete. Summary saved to {summary_path}")
    
    # Check acceptance criteria
    applied_count = sum(1 for r in results if r.get("pixel_ops_applied", False))
    high_halo_count = sum(1 for r in results if r.get("halo_risk") == "HIGH")
    max_mean_delta = max((r.get("mean_delta", 0.0) for r in results), default=0.0)
    
    log.info(f"--- Acceptance Criteria ---")
    log.info(f"Applied count: {applied_count} (expected ≥2 for forced)")
    log.info(f"HIGH halo risk count: {high_halo_count} (expected 0)")
    log.info(f"Max mean delta: {max_mean_delta:.4f} (expected <0.02)")
    
    if args.force_apply:
        if applied_count >= 2 and high_halo_count == 0 and max_mean_delta < 0.02:
            log.info("✅ VALIDATION PASSED")
            return 0
        else:
            log.error("❌ VALIDATION FAILED")
            return 1
    else:
        log.info("Normal pass complete (no strict criteria)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
