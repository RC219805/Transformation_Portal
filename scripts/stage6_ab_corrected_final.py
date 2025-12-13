#!/usr/bin/env python3
"""Stage 6 A/B test with boundary metrics (corrected final version).

This script addresses all three remaining issues:
1. BoundaryMetrics return type handling (.to_dict())
2. Material key canonicalization (water/pool_water/etc.)
3. Minimum boundary pixels guard (prevent degenerate cases)

Promotion gate:
- Edge alignment delta > +0.02 (real improvement)
- Boundary F1 >= 0.85 (no catastrophic regression)
- Boundary pixels >= 250 (no degenerate masks)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repo is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.material_segmentation import create_material_segmenter
from lux_depth_v2.metrics.boundary_metrics import compute_full_boundary_metrics
from lux_depth_v2.materials_v3_taxonomy import normalize_material_dict
from lux_depth_v2 import io_utils, torch_ops

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Constants
FORCE_DEVICE = "cpu"  # Keep constant for A/B stability
MIN_BOUNDARY_PX = 250  # Minimum boundary size to avoid noise
EDGE_ALIGN_IMPROVEMENT_THRESHOLD = 0.02  # Minimum delta for "improvement"
BF1_REGRESSION_THRESHOLD = 0.85  # Minimum BF1 to avoid "regression"

# Target classes for evaluation (canonical keys)
TARGET_CLASSES = ["glass", "water", "foliage"]


@dataclass
class SceneBenchmark:
    """Single scene benchmark config."""
    name: str
    input_path: Path
    baseline_preset: Preset
    canary_preset: Preset


# Benchmark scenes (using actual filenames from assets/phase2_bench/)
BENCHMARK_SCENES = [
    SceneBenchmark(
        name="kitchen",
        input_path=Path("assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif"),
        baseline_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
        canary_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
    ),
    SceneBenchmark(
        name="bedroom",
        input_path=Path("assets/phase2_bench/750Picacho_PrimaryBedroom_Ultimate.tif"),
        baseline_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
        canary_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
    ),
    SceneBenchmark(
        name="bathroom",
        input_path=Path("assets/phase2_bench/750Picacho_PrimaryBathroom_Ultimate.tif"),
        baseline_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
        canary_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
    ),
    SceneBenchmark(
        name="pool",
        input_path=Path("assets/phase2_bench/750Picacho_Pool_Ultimate.tif"),
        baseline_preset=Preset.EXTERIOR_POOL_APEX_QUALITY,
        canary_preset=Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM,
    ),
    SceneBenchmark(
        name="aerial",
        input_path=Path("assets/phase2_bench/750Picacho_Aerial_Ultimate.tif"),
        baseline_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,  # Fixed: was MAX
        canary_preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
    ),
]


def _as_dict(metrics_obj):
    """Convert metrics object to dict defensively."""
    if metrics_obj is None:
        return {}
    if isinstance(metrics_obj, dict):
        return metrics_obj
    if hasattr(metrics_obj, "to_dict"):
        return metrics_obj.to_dict()
    raise TypeError(f"Unexpected metrics type: {type(metrics_obj)}")


def compute_image_gradients(rgb: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude (Sobel) for edge alignment."""
    from scipy.ndimage import sobel
    
    if rgb.ndim == 3:
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        gray = rgb
    
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize to [0,1]
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    return grad_mag.astype(np.float32)


def extract_masks_from_segmenter(
    rgb01: np.ndarray,
    cfg: PipelineConfig,
    device: str,
) -> Dict[str, np.ndarray]:
    """Extract material masks directly from segmenter (no disk I/O).
    
    Returns canonicalized masks: {canonical_key: HxW float32 [0,1]}
    """
    rgb_t = torch_ops.to_torch_rgb(rgb01, device=device)
    
    seg = create_material_segmenter(cfg.segmentation, device=device)
    masks_dict_torch = seg.predict(rgb_t)
    
    # Convert to numpy dict
    raw_masks = {
        k: masks_dict_torch[k][0, 0].cpu().numpy().astype(np.float32)
        for k in masks_dict_torch.keys()
    }
    
    # Canonicalize keys (fixes water/pool_water/etc.)
    normalized_masks = normalize_material_dict(raw_masks)
    
    return normalized_masks


def run_segmentation_only(
    input_path: Path,
    preset: Preset,
    device: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    """Run segmentation and return masks + RGB + runtime.
    
    Returns
    -------
    masks : Dict[str, np.ndarray]
        Canonical material masks (HxW float32)
    rgb01 : np.ndarray
        RGB image [0,1] float32 (HxWx3)
    runtime : float
        Segmentation runtime in seconds
    """
    import time
    
    # Load image
    rgb01, _ = io_utils.read_rgb_any(input_path)
    
    # Configure for segmentation-only
    cfg = PipelineConfig(preset=preset)
    
    t0 = time.perf_counter()
    masks = extract_masks_from_segmenter(rgb01, cfg, device)
    runtime = time.perf_counter() - t0
    
    return masks, rgb01, runtime


def compute_class_boundary_metrics(
    base_mask: np.ndarray,
    canary_mask: np.ndarray,
    rgb01: np.ndarray,
    class_name: str,
) -> Dict:
    """Compute boundary metrics for one class.
    
    Returns dict with:
    - boundary_f1_vs_baseline
    - trimap_iou_*
    - edge_align_baseline
    - edge_align_canary
    - edge_align_delta
    - boundary_pixels
    - improvements/regressions flags
    """
    # Compute gradients at mask resolution
    if rgb01.shape[:2] != base_mask.shape:
        from skimage.transform import resize
        rgb_resized = resize(
            rgb01, base_mask.shape, order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    else:
        rgb_resized = rgb01
    
    gradients = compute_image_gradients(rgb_resized)
    
    # Metrics: canary vs baseline (regression guard)
    regression_metrics_obj = compute_full_boundary_metrics(
        pred_mask=canary_mask,
        ref_mask=base_mask,
        image_gradients=None,  # Not needed for regression check
        band_width_px=5,
    )
    regression_metrics = _as_dict(regression_metrics_obj)
    
    # Metrics: baseline vs gradients (improvement baseline)
    base_metrics_obj = compute_full_boundary_metrics(
        pred_mask=base_mask,
        ref_mask=base_mask,  # Not used for edge alignment
        image_gradients=gradients,
        band_width_px=5,
    )
    base_metrics = _as_dict(base_metrics_obj)
    
    # Metrics: canary vs gradients (improvement target)
    canary_metrics_obj = compute_full_boundary_metrics(
        pred_mask=canary_mask,
        ref_mask=base_mask,  # For BF1/trimap
        image_gradients=gradients,
        band_width_px=5,
    )
    canary_metrics = _as_dict(canary_metrics_obj)
    
    # Extract key values
    bf1 = regression_metrics.get("boundary_f1", 0.0)
    boundary_px = regression_metrics.get("boundary_pixels", 0)
    
    edge_align_base = base_metrics.get("edge_alignment", 0.0)
    edge_align_canary = canary_metrics.get("edge_alignment", 0.0)
    edge_align_delta = edge_align_canary - edge_align_base
    
    # Decision logic
    is_improvement = (
        edge_align_delta > EDGE_ALIGN_IMPROVEMENT_THRESHOLD
        and bf1 >= BF1_REGRESSION_THRESHOLD
        and boundary_px >= MIN_BOUNDARY_PX
    )
    
    is_regression = (
        bf1 < BF1_REGRESSION_THRESHOLD
        and boundary_px >= MIN_BOUNDARY_PX
    )
    
    return {
        "class": class_name,
        "boundary_f1_vs_baseline": float(bf1),
        "trimap_iou_core": canary_metrics.get("trimap_iou_core", 0.0),
        "trimap_iou_boundary": canary_metrics.get("trimap_iou_boundary", 0.0),
        "trimap_iou_background": canary_metrics.get("trimap_iou_background", 0.0),
        "edge_align_baseline": float(edge_align_base),
        "edge_align_canary": float(edge_align_canary),
        "edge_align_delta": float(edge_align_delta),
        "boundary_pixels": int(boundary_px),
        "is_improvement": bool(is_improvement),
        "is_regression": bool(is_regression),
    }


def run_scene_ab(scene: SceneBenchmark, device: str) -> Dict:
    """Run A/B test for one scene."""
    log.info(f"=== {scene.name.upper()} ===")
    
    if not scene.input_path.exists():
        log.warning(f"Input missing: {scene.input_path}")
        return {
            "scene": scene.name,
            "status": "input_missing",
        }
    
    try:
        # Baseline
        log.info(f"  Baseline: {scene.baseline_preset.name}")
        base_masks, rgb01, base_runtime = run_segmentation_only(
            scene.input_path, scene.baseline_preset, device
        )
        
        # Canary
        log.info(f"  Canary: {scene.canary_preset.name}")
        canary_masks, _, canary_runtime = run_segmentation_only(
            scene.input_path, scene.canary_preset, device
        )
        
    except Exception as exc:
        log.error(f"  ERROR: {exc}", exc_info=True)
        return {
            "scene": scene.name,
            "status": "error",
            "error": str(exc),
        }
    
    # Per-class metrics
    class_results = []
    improvements = []
    regressions = []
    
    for cls in TARGET_CLASSES:
        if cls not in base_masks or cls not in canary_masks:
            log.warning(f"  {cls}: missing in one or both runs")
            continue
        
        base_mask = base_masks[cls]
        canary_mask = canary_masks[cls]
        
        metrics = compute_class_boundary_metrics(
            base_mask, canary_mask, rgb01, cls
        )
        
        class_results.append(metrics)
        
        if metrics["is_improvement"]:
            improvements.append(cls)
            log.info(
                f"  {cls}: IMPROVEMENT "
                f"(edge_delta={metrics['edge_align_delta']:+.4f}, "
                f"bf1={metrics['boundary_f1_vs_baseline']:.3f})"
            )
        
        if metrics["is_regression"]:
            regressions.append(cls)
            log.warning(
                f"  {cls}: REGRESSION "
                f"(bf1={metrics['boundary_f1_vs_baseline']:.3f})"
            )
    
    # Scene decision
    scene_improved = len(improvements) > 0 and len(regressions) == 0
    
    return {
        "scene": scene.name,
        "status": "success",
        "baseline_preset": scene.baseline_preset.name,
        "canary_preset": scene.canary_preset.name,
        "baseline_runtime_sec": float(base_runtime),
        "canary_runtime_sec": float(canary_runtime),
        "runtime_delta_sec": float(canary_runtime - base_runtime),
        "improvements": improvements,
        "regressions": regressions,
        "scene_improved": scene_improved,
        "class_results": class_results,
    }


def run_full_benchmark(device: str) -> Dict:
    """Run full A/B benchmark across all scenes."""
    results = []
    
    for scene in BENCHMARK_SCENES:
        result = run_scene_ab(scene, device)
        results.append(result)
    
    # Global summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    improved_count = sum(1 for r in results if r.get("scene_improved", False))
    
    # Collect all improvements/regressions
    all_improvements = []
    all_regressions = []
    for r in results:
        all_improvements.extend(r.get("improvements", []))
        all_regressions.extend(r.get("regressions", []))
    
    # Promotion decision
    promote_to_apex = (
        success_count >= 4
        and improved_count >= 3
        and len(all_regressions) == 0
    )
    
    summary = {
        "device": device,
        "total_scenes": len(BENCHMARK_SCENES),
        "success_count": success_count,
        "improved_count": improved_count,
        "all_improvements": list(set(all_improvements)),
        "all_regressions": list(set(all_regressions)),
        "promote_to_apex": promote_to_apex,
        "decision": (
            "PROMOTE: Enable FUSED in default APEX presets"
            if promote_to_apex
            else "KEEP CANARY: Insufficient improvement or regressions present"
        ),
        "scene_results": results,
    }
    
    return summary


def main() -> int:
    """Run Stage 6 A/B boundary metrics test."""
    output_dir = Path("outputs/stage6_ab_boundary_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Stage 6 A/B Test: Boundary Metrics (Corrected Final)")
    log.info(f"Device: {FORCE_DEVICE}")
    log.info(f"Min boundary pixels: {MIN_BOUNDARY_PX}")
    log.info(f"Edge align threshold: +{EDGE_ALIGN_IMPROVEMENT_THRESHOLD}")
    log.info(f"BF1 regression threshold: {BF1_REGRESSION_THRESHOLD}")
    log.info("")
    
    summary = run_full_benchmark(FORCE_DEVICE)
    
    # Write results
    summary_json = output_dir / "stage6_ab_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    
    log.info("")
    log.info("=== SUMMARY ===")
    log.info(f"Success: {summary['success_count']}/{summary['total_scenes']}")
    log.info(f"Improved: {summary['improved_count']}/{summary['total_scenes']}")
    log.info(f"Improvements: {summary['all_improvements']}")
    log.info(f"Regressions: {summary['all_regressions']}")
    log.info("")
    log.info(f"DECISION: {summary['decision']}")
    log.info("")
    log.info(f"Full results: {summary_json}")
    
    return 0 if summary["promote_to_apex"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
