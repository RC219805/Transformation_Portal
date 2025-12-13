#!/usr/bin/env python3
"""
Stage 6 A/B Test with Boundary Metrics (PR-3C) - CORRECTED VERSION

Runs A/B comparison with EDGE-QUALITY scoring using in-memory mask extraction.

Key fixes:
1. Uses cfg.output_dir correctly (set before pipeline init)
2. Extracts masks from in-memory segmenter output (no disk I/O)
3. Aerial baseline uses APEX (not MAX) for apples-to-apples comparison
4. Edge alignment vs image gradients for "improvement" signal
5. BF1 used as regression guard, not improvement metric

Promotion gate:
- Promote FUSED to default APEX only if:
  * Edge alignment improves on ≥3/5 scenes (delta > +0.02)
  * BF1 regression guard passes (≥0.85, boundary_pixels > 0)
  * Visual diffs show no artifacts
  * Runtime delta acceptable
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

# Add lux_depth_v2 to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.material_segmentation import create_material_segmenter
from lux_depth_v2.metrics.boundary_metrics import compute_full_boundary_metrics
from lux_depth_v2 import io_utils, torch_ops


BENCHMARK_SET = {
    "interior_kitchen_750": {
        "path": "assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "interior",
        "target_classes": ["glass", "foliage"],
    },
    "exterior_pool_750": {
        "path": "assets/phase2_bench/750Picacho_Pool_Ultimate.tif",
        "baseline_preset": Preset.EXTERIOR_POOL_APEX_QUALITY,
        "canary_preset": Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "exterior",
        "target_classes": ["water", "foliage"],
    },
    "interior_bedroom": {
        "path": "assets/phase2_bench/750Picacho_PrimaryBedroom_Ultimate.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "interior",
        "target_classes": ["glass", "foliage"],
    },
    "interior_bathroom": {
        "path": "assets/phase2_bench/750Picacho_PrimaryBathroom_Ultimate.tif",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "interior",
        "target_classes": ["glass"],
    },
    "exterior_aerial": {
        "path": "assets/phase2_bench/750Picacho_Aerial_Ultimate.tif",
        # FIXED: use APEX for apples-to-apples comparison
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "exterior",
        "target_classes": ["foliage"],
    },
}


def compute_image_gradients(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude from RGB image at same resolution as masks.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image (HxWx3), float32 in [0,1]
    
    Returns
    -------
    np.ndarray
        Gradient magnitude (HxW), float32
    """
    if image.ndim == 3:
        # Convert to grayscale
        gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    else:
        gray = image.astype(np.float32)
    
    # Sobel gradients
    sx = sobel(gray, axis=0, mode='constant')
    sy = sobel(gray, axis=1, mode='constant')
    
    # Magnitude
    grad_mag = np.sqrt(sx**2 + sy**2).astype(np.float32)
    
    # Normalize to [0,1] for consistent thresholding
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    
    return grad_mag


def run_segmentation_only(
    input_path: Path,
    preset: Preset,
    target_classes: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    """Run segmentation only (no full pipeline) and extract in-memory masks.
    
    Parameters
    ----------
    input_path : Path
        Input image path
    preset : Preset
        Pipeline preset (determines segmentation config)
    target_classes : List[str]
        Classes to extract (e.g., ['glass', 'water', 'foliage'])
    
    Returns
    -------
    masks : Dict[str, np.ndarray]
        Mapping from class name to mask (HxW float32 in [0,1])
    rgb01 : np.ndarray
        Input image (HxWx3 float32 in [0,1])
    runtime_sec : float
        Segmentation runtime
    """
    import torch
    
    # Build config from preset
    cfg = PipelineConfig()
    cfg.apply_preset(preset)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load image
    rgb01, _ = io_utils.read_rgb_any(input_path)
    H, W = rgb01.shape[:2]
    
    # To tensor
    rgb_t = torch_ops.to_torch_rgb(rgb01, device)
    
    # Create segmenter (this respects backend_v3/fusion settings from preset)
    t0 = time.time()
    seg = create_material_segmenter(cfg.segmentation, device)
    
    # Run segmentation
    masks_dict_torch = seg.predict(rgb_t)  # dict[str, torch.Tensor] (1,1,H,W)
    runtime_sec = time.time() - t0
    
    # Extract target classes to numpy
    masks = {}
    for cls in target_classes:
        if cls in masks_dict_torch:
            mask_t = masks_dict_torch[cls]  # (1,1,H,W)
            mask_np = mask_t[0, 0].cpu().numpy().astype(np.float32)
            masks[cls] = mask_np
        else:
            masks[cls] = None
    
    return masks, rgb01, runtime_sec


def compute_per_class_metrics(
    baseline_mask: np.ndarray,
    canary_mask: np.ndarray,
    image_gradients: np.ndarray,
    class_name: str,
) -> Dict[str, float]:
    """Compute boundary metrics for a single class.
    
    Parameters
    ----------
    baseline_mask : np.ndarray
        Baseline mask (HxW float32 in [0,1])
    canary_mask : np.ndarray
        Canary mask (HxW float32 in [0,1])
    image_gradients : np.ndarray
        Image gradient magnitude (HxW float32 in [0,1])
    class_name : str
        Material class name
    
    Returns
    -------
    Dict[str, float]
        Metrics dict with:
        - bf1_canary_vs_baseline: boundary F1 (regression guard)
        - edge_align_baseline: baseline edge alignment to image gradients
        - edge_align_canary: canary edge alignment to image gradients
        - edge_align_delta: improvement signal (canary - baseline)
        - boundary_pixels_baseline: boundary pixel count (baseline)
        - boundary_pixels_canary: boundary pixel count (canary)
    """
    metrics = {}
    
    # Compute full boundary metrics using PR-3A module
    # Baseline vs gradients
    base_metrics = compute_full_boundary_metrics(
        pred_mask=baseline_mask,
        ref_mask=None,
        image_gradients=image_gradients,
        radius_px=2
    )
    
    # Canary vs gradients
    canary_metrics = compute_full_boundary_metrics(
        pred_mask=canary_mask,
        ref_mask=None,
        image_gradients=image_gradients,
        radius_px=2
    )
    
    # Canary vs baseline (regression guard)
    regression_metrics = compute_full_boundary_metrics(
        pred_mask=canary_mask,
        ref_mask=baseline_mask,
        image_gradients=None,  # not used when ref_mask provided
        radius_px=2
    )
    
    # Extract relevant values
    metrics["bf1_canary_vs_baseline"] = regression_metrics.get("boundary_f1", 0.0)
    metrics["edge_align_baseline"] = base_metrics.get("edge_alignment", 0.0)
    metrics["edge_align_canary"] = canary_metrics.get("edge_alignment", 0.0)
    metrics["edge_align_delta"] = metrics["edge_align_canary"] - metrics["edge_align_baseline"]
    metrics["boundary_pixels_baseline"] = base_metrics.get("boundary_pixels", 0)
    metrics["boundary_pixels_canary"] = canary_metrics.get("boundary_pixels", 0)
    
    return metrics


def run_single_scene(
    scene_name: str,
    scene_config: dict,
    output_root: Path,
) -> Dict[str, object]:
    """Run baseline + canary for a single scene and compute boundary metrics.
    
    Parameters
    ----------
    scene_name : str
        Scene identifier (e.g., 'interior_kitchen_750')
    scene_config : dict
        Scene configuration from BENCHMARK_SET
    output_root : Path
        Output root directory
    
    Returns
    -------
    Dict[str, object]
        Scene result with baseline/canary metrics
    """
    input_path = Path(scene_config["path"])
    baseline_preset = scene_config["baseline_preset"]
    canary_preset = scene_config["canary_preset"]
    target_classes = scene_config["target_classes"]
    
    print(f"\n{'='*60}")
    print(f"Scene: {scene_name}")
    print(f"Input: {input_path.name}")
    print(f"Target classes: {target_classes}")
    print(f"{'='*60}\n")
    
    if not input_path.exists():
        print(f"⚠️  SKIP: {input_path} not found")
        return {
            "scene": scene_name,
            "status": "skip_missing_input",
            "input_path": str(input_path),
        }
    
    # Run baseline segmentation
    print(f"[1/2] Baseline ({baseline_preset.name})...")
    try:
        baseline_masks, rgb01, baseline_runtime = run_segmentation_only(
            input_path, baseline_preset, target_classes
        )
        print(f"  ✅ Baseline runtime: {baseline_runtime:.2f}s")
        print(f"  ✅ Extracted classes: {[k for k, v in baseline_masks.items() if v is not None]}")
    except Exception as exc:
        print(f"  ❌ Baseline failed: {exc}")
        return {
            "scene": scene_name,
            "status": "baseline_failed",
            "error": str(exc),
        }
    
    # Run canary segmentation
    print(f"\n[2/2] Canary ({canary_preset.name})...")
    try:
        canary_masks, _, canary_runtime = run_segmentation_only(
            input_path, canary_preset, target_classes
        )
        print(f"  ✅ Canary runtime: {canary_runtime:.2f}s")
        print(f"  ✅ Extracted classes: {[k for k, v in canary_masks.items() if v is not None]}")
    except Exception as exc:
        print(f"  ❌ Canary failed: {exc}")
        return {
            "scene": scene_name,
            "status": "canary_failed",
            "error": str(exc),
        }
    
    # Compute image gradients at mask resolution
    print(f"\n[3/3] Computing boundary metrics...")
    image_gradients = compute_image_gradients(rgb01)
    
    # Compute per-class boundary metrics
    per_class_metrics = {}
    for cls in target_classes:
        base_mask = baseline_masks.get(cls)
        cana_mask = canary_masks.get(cls)
        
        if base_mask is None or cana_mask is None:
            print(f"  ⚠️  {cls}: mask missing (base={base_mask is not None}, canary={cana_mask is not None})")
            per_class_metrics[cls] = {
                "status": "masks_missing",
                "baseline_present": base_mask is not None,
                "canary_present": cana_mask is not None,
            }
            continue
        
        # Compute metrics
        metrics = compute_per_class_metrics(
            base_mask, cana_mask, image_gradients, cls
        )
        
        # Log key results
        print(f"  ✅ {cls}:")
        print(f"     BF1 (regression guard): {metrics['bf1_canary_vs_baseline']:.3f}")
        print(f"     Edge align baseline:     {metrics['edge_align_baseline']:.3f}")
        print(f"     Edge align canary:       {metrics['edge_align_canary']:.3f}")
        print(f"     Edge align Δ (improve):  {metrics['edge_align_delta']:+.3f}")
        
        per_class_metrics[cls] = metrics
    
    # Aggregate scene-level decision
    improvements = []
    regressions = []
    
    for cls, metrics in per_class_metrics.items():
        if "edge_align_delta" not in metrics:
            continue
        delta = metrics["edge_align_delta"]
        bf1 = metrics.get("bf1_canary_vs_baseline", 0.0)
        boundary_px = metrics.get("boundary_pixels_canary", 0)
        
        # Improvement criteria
        if delta > 0.02 and bf1 >= 0.85 and boundary_px > 0:
            improvements.append(cls)
        
        # Regression criteria
        if bf1 < 0.85 or delta < -0.05:
            regressions.append(cls)
    
    result = {
        "scene": scene_name,
        "status": "success",
        "input_path": str(input_path),
        "baseline_preset": baseline_preset.name,
        "canary_preset": canary_preset.name,
        "baseline_runtime_sec": baseline_runtime,
        "canary_runtime_sec": canary_runtime,
        "runtime_delta_sec": canary_runtime - baseline_runtime,
        "per_class_metrics": per_class_metrics,
        "improvements": improvements,
        "regressions": regressions,
        "scene_improved": len(improvements) > 0 and len(regressions) == 0,
    }
    
    print(f"\n  📊 Scene summary:")
    print(f"     Improvements: {improvements if improvements else 'none'}")
    print(f"     Regressions:  {regressions if regressions else 'none'}")
    print(f"     Overall:      {'✅ IMPROVED' if result['scene_improved'] else '⚠️  NO IMPROVEMENT'}")
    
    return result


def main() -> int:
    """Run Stage 6 A/B test with boundary metrics."""
    output_root = Path("outputs/stage6_ab_boundary_metrics")
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("STAGE 6 A/B TEST: BOUNDARY METRICS (PR-3C CORRECTED)")
    print("="*60)
    print(f"Output: {output_root}")
    print(f"Scenes: {len(BENCHMARK_SET)}")
    print()
    
    results = []
    
    for scene_name, scene_config in BENCHMARK_SET.items():
        result = run_single_scene(scene_name, scene_config, output_root)
        results.append(result)
    
    # Aggregate promotion decision
    print(f"\n{'='*60}")
    print("PROMOTION DECISION")
    print(f"{'='*60}\n")
    
    successful_scenes = [r for r in results if r["status"] == "success"]
    improved_scenes = [r for r in successful_scenes if r.get("scene_improved", False)]
    
    total_scenes = len(BENCHMARK_SET)
    success_count = len(successful_scenes)
    improved_count = len(improved_scenes)
    
    print(f"Successful runs:  {success_count}/{total_scenes}")
    print(f"Improved scenes:  {improved_count}/{total_scenes}")
    print()
    
    # Promotion gate: ≥3/5 improved
    promote = improved_count >= 3
    
    print(f"Promotion gate:   {improved_count} >= 3  →  {'✅ PASS' if promote else '❌ FAIL'}")
    print()
    
    if promote:
        print("✅ RECOMMENDATION: Promote FUSED to default APEX")
        print("   - Boundary metrics show consistent edge improvement")
        print("   - Regression guard passed")
        print("   - Proceed with visual diff validation")
    else:
        print("❌ RECOMMENDATION: Keep canary-only")
        print(f"   - Only {improved_count}/5 scenes improved")
        print("   - Insufficient evidence for default promotion")
        print("   - Continue Materials V3 PR-3B/PR-4 work")
    
    # Write summary JSON
    summary_path = output_root / "stage6_ab_summary.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenes": results,
        "aggregate": {
            "total_scenes": total_scenes,
            "successful_runs": success_count,
            "improved_scenes": improved_count,
            "promotion_gate_passed": promote,
            "recommendation": "promote" if promote else "keep_canary_only",
        },
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary written: {summary_path}")
    
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
