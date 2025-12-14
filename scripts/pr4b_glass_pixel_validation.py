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
    """Coerce LuxPipelineV2.process_one() return into a report dict.
    
    In this repo, process_one() returns the report dict directly.
    Some legacy wrappers may return {'report': report}.
    """
    if isinstance(obj, dict):
        if "report" in obj and isinstance(obj["report"], dict):
            return obj["report"]
        return obj
    return {}


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


def should_skip_scene_for_device(
    scene_name: str,
    input_path: Path,
    device_type: str,
    max_mp_mps: float = 30.0,
) -> Optional[str]:
    """
    Determine if scene should be skipped due to device limitations.
    
    Args:
        scene_name: Scene identifier
        input_path: Path to input image
        device_type: Device type (cpu, cuda, mps)
        max_mp_mps: Max megapixels for MPS (default 30.0)
        
    Returns:
        Skip reason string if should skip, None otherwise
    """
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
    """Run baseline and canary on one scene and compare pixel impact."""
    
    log.info(f"=== Processing {scene_name} ===")
    
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
    baseline_dir = output_root / f"{scene_name}_A_baseline"
    canary_dir = output_root / f"{scene_name}_B_glass_canary"
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
        log.info(f"Running glass canary APEX for {scene_name}...")
        
        # Select preset based on force_apply flag
        canary_preset = (
            Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS_VALIDATE
            if force_apply
            else Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS
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
    pixel_ops_report = canary_report.get("materials_v3_pixel_ops", {}) or {}
    response_plan = canary_report.get("materials_v3_response_plan", {}) or {}
    glass_plan = (response_plan.get("per_class", {}) or {}).get("glass", {}) or {}

    applied_to = pixel_ops_report.get("applied_to") or pixel_ops_report.get("applied_to_classes") or []
    applied = bool(applied_to) or bool(pixel_ops_report.get("applied", False))

    plan_should_refine = glass_plan.get("should_refine")
    plan_reason = (
        glass_plan.get("refine_reason")
        or glass_plan.get("skip_reason")
        or glass_plan.get("reason")
        or None
    )

    if not applied:
        # If the response plan explicitly says not to refine, that's a VALID skip.
        if plan_should_refine is False:
            return {
                "scene": scene_name,
                "status": "success_skipped",
                "reason": plan_reason or "plan_skip_no_reason",
                "pixel_ops_applied": False,
                "pixel_ops_expected": False,
                "preset_used": canary_preset.value,
                "forced_apply": force_apply,
                "glass_plan": {
                    "should_refine": False,
                    "refine_reason": plan_reason,
                    "mean_conf": glass_plan.get("mean_conf"),
                    "edge_conf": glass_plan.get("edge_conf"),
                    "coverage": glass_plan.get("coverage"),
                },
            }

        # Otherwise we expected ops (plan said refine), but nothing applied -> failure.
        return {
            "scene": scene_name,
            "status": "pixel_ops_not_applied",
            "reason": pixel_ops_report.get("reason") or plan_reason or "unknown",
            "pixel_ops_applied": False,
            "pixel_ops_expected": True if plan_should_refine is True else None,
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
    parser = argparse.ArgumentParser(
        description="PR-4B Glass Pixel Response Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Kitchen + Bedroom only (skip Bathroom on MPS)
  python scripts/pr4b_glass_pixel_validation.py
  
  # Explicit scene selection
  python scripts/pr4b_glass_pixel_validation.py --scenes kitchen bedroom
  
  # Include Bathroom (may OOM on MPS)
  python scripts/pr4b_glass_pixel_validation.py --include-bathroom
  
  # Force CPU device
  python scripts/pr4b_glass_pixel_validation.py --device cpu
        """,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["kitchen", "bedroom", "bathroom"],
        help="Scenes to test (default: kitchen + bedroom)",
    )
    parser.add_argument(
        "--include-bathroom",
        action="store_true",
        help="Include Bathroom scene (may OOM on MPS)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Force device (default: auto-detect)",
    )
    parser.add_argument(
        "--max-mp-mps",
        type=float,
        default=30.0,
        help="Max megapixels for MPS before skipping (default: 30.0)",
    )
    parser.add_argument(
        "--force-apply",
        action="store_true",
        help="Use VALIDATE preset that forces glass pixel ops (validation-only).",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Define validation set (scenes with glass)
    all_scenes = {
        "kitchen": repo_root / "assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif",
        "bedroom": repo_root / "assets/phase2_bench/750Picacho_PrimaryBedroom_Ultimate.tif",
        "bathroom": repo_root / "assets/phase2_bench/750Picacho_PrimaryBathroom_Ultimate.tif",
    }
    
    # Select scenes based on args
    if args.scenes:
        selected_scene_names = args.scenes
    elif args.include_bathroom:
        selected_scene_names = ["kitchen", "bedroom", "bathroom"]
    else:
        # Default: skip bathroom on MPS to avoid OOM
        selected_scene_names = ["kitchen", "bedroom"]
    
    benchmark_scenes = {
        name.capitalize(): all_scenes[name]
        for name in selected_scene_names
        if name in all_scenes
    }
    
    log.info(f"Selected scenes: {list(benchmark_scenes.keys())}")
    if args.device:
        log.info(f"Forced device: {args.device}")
    
    output_root = repo_root / "outputs/pr4b_glass_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = []
    for scene_name, input_path in benchmark_scenes.items():
        if not input_path.exists():
            log.warning(f"Missing input: {input_path}")
            result = {
                "scene": scene_name,
                "status": "input_missing",
                "input_path": str(input_path),
            }
        else:
            result = run_single_scene(scene_name, input_path, output_root, args.device, args.force_apply)
        
        results.append(result)
        
        # Write incremental results
        incremental_path = output_root / f"pr4b_{scene_name.lower()}_result.json"
        with open(incremental_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Scene result saved: {incremental_path}")
    
    # Aggregate and decide (only count successful runs)
    successful_results = [r for r in results if r.get("status") in ("success", "success_skipped")]
    applied_count = sum(1 for r in successful_results if r.get("pixel_ops_applied"))
    halo_risks = [r["halo_detection"]["halo_risk"] for r in successful_results if "halo_detection" in r]
    high_halo_count = sum(1 for risk in halo_risks if risk == "high")
    
    gradient_improvements = [
        r["edge_stats"]["gradient_delta"]
        for r in successful_results
        if "edge_stats" in r and r["edge_stats"]["gradient_delta"] > 0.0
    ]
    
    # Track skipped/failed scenes
    skipped_results = [r for r in results if r.get("status") == "skipped"]
    failed_results = [r for r in results if r.get("status") in ("baseline_failed", "canary_failed", "input_missing")]
    
    # Decision criteria (require at least 2 successful scenes with pixel ops)
    merge_recommended = (
        len(successful_results) >= 2  # At least 2 scenes completed
        and applied_count >= 2  # At least 2 scenes applied pixel ops
        and high_halo_count == 0  # No high halo risk
        and len(gradient_improvements) >= 2  # At least 2 scenes improved edges
    )
    
    summary = {
        "validation_date": "2025-12-14",
        "pr": "PR-4B",
        "feature": "glass_pixel_response",
        "scenes_total": len(results),
        "scenes_successful": len(successful_results),
        "scenes_skipped": len(skipped_results),
        "scenes_failed": len(failed_results),
        "scenes_applied": applied_count,
        "high_halo_risks": high_halo_count,
        "gradient_improvements": len(gradient_improvements),
        "merge_recommended": merge_recommended,  # Canary feature only
        "skipped_scenes": [r["scene"] for r in skipped_results],
        "failed_scenes": [r["scene"] for r in failed_results],
        "details": results,
    }
    
    summary_path = output_root / "pr4b_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"\n=== PR-4B Validation Summary ===")
    log.info(f"Scenes total: {len(results)}")
    log.info(f"Scenes successful: {len(successful_results)}")
    log.info(f"Scenes skipped: {len(skipped_results)} {skipped_results[0]['scene'] if skipped_results else ''}")
    log.info(f"Scenes failed: {len(failed_results)}")
    log.info(f"Pixel ops applied: {applied_count}/{len(successful_results)}")
    log.info(f"High halo risks: {high_halo_count}")
    log.info(f"Gradient improvements: {len(gradient_improvements)}")
    log.info(f"Merge recommended (canary): {merge_recommended}")
    log.info(f"\nFull report: {summary_path}")
    
    return 0 if merge_recommended else 1


if __name__ == "__main__":
    raise SystemExit(main())
