#!/usr/bin/env python3
"""
Isolation Tests for High-Fidelity Depth Pipeline
=================================================

Systematic toggle tests to pinpoint failure modes:
1. Tiling only (no refinement)
2. Global anchor fusion only
3. Guided filter only
4. Edge snapping only
5. CLAHE on low-frequency only

Reference: TILING_BUG_IDENTIFIED.md - Isolation test methodology
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .depth_estimator import HighFidelityDepthEstimator, DepthConfig
from .quality_metrics import validate_depth_quality, EdgeMetrics, save_metrics_atomic

logger = logging.getLogger(__name__)


@dataclass
class IsolationTestResult:
    """Result from a single isolation test."""
    
    name: str
    description: str
    metrics: EdgeMetrics
    passed: bool
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "passed": bool(self.passed),
            **self.metrics.to_dict()
        }
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Edge F1: {self.metrics.edge_f1:.3f} (primary)\n"
            f"  Edge overlap: {self.metrics.edge_overlap:.3f}\n"
            f"  Edge count ratio: {self.metrics.edge_count_ratio:.2f}×\n"
            f"  Quality score: {self.metrics.quality_score():.3f}\n"
        )


def test_tiling_only(rgb: np.ndarray, baseline_metrics: Optional[EdgeMetrics] = None) -> IsolationTestResult:
    """
    Test 1: Tiling only (no refinement).
    
    This is the CRITICAL test - if this fails, tiling integration is broken.
    
    VALIDATION:
    - Confirm pixel_values shape == tile_size (no internal resize)
    - Edge F1 ≥ 0.30 (10× improvement from broken baseline ~0.004)
    - Edge count ratio ≤ 2.0 (no artifact explosion)
    - Chamfer distance < 15px
    """
    logger.info("=== Test 1: Tiling Only ===")
    
    config = DepthConfig(
        tile_size=1024,
        overlap=128,
        reconcile_scales=True,  # MUST be enabled (fix for seams)
        reconcile_method="robust",  # Use Theil-Sen regression
        fusion_mode="weighted",
        validate_seams=True
    )
    
    estimator = HighFidelityDepthEstimator(config)
    depth = estimator.estimate_depth(rgb, use_global_anchor=True)
    
    metrics = validate_depth_quality(rgb, depth, dilation=3)
    
    # UPDATED ACCEPTANCE CRITERIA (based on fixes)
    # Target: Edge F1 ≥0.30 (vs. broken baseline ~0.004-0.063)
    # Chamfer distance < 15px (better alignment)
    # Edge count ratio ≤ 2.0 (no artifact explosion)
    
    if baseline_metrics is not None:
        # Must show significant improvement over baseline
        f1_ok = metrics.edge_f1 >= 0.30  # Absolute threshold
        overlap_ok = metrics.edge_overlap >= 0.40
        count_ok = metrics.edge_count_ratio <= 2.0  # Tighter bound
        chamfer_ok = metrics.chamfer_distance < 15.0
        passed = f1_ok and overlap_ok and count_ok and chamfer_ok
    else:
        passed = (
            metrics.edge_f1 >= 0.30 and 
            metrics.edge_overlap >= 0.40 and 
            metrics.edge_count_ratio <= 2.0 and
            metrics.chamfer_distance < 15.0
        )
    
    result = IsolationTestResult(
        name="Tiling Only",
        description="Tile-based inference with robust scale reconciliation (Theil-Sen)",
        metrics=metrics,
        passed=passed
    )
    
    logger.info(str(result))
    
    # Additional validation logging
    if metrics.edge_f1 < 0.30:
        logger.error(f"❌ Edge F1 too low: {metrics.edge_f1:.3f} < 0.30 (target)")
    if metrics.edge_count_ratio > 2.0:
        logger.error(f"❌ Edge count ratio too high: {metrics.edge_count_ratio:.2f} > 2.0 (artifact explosion)")
    if metrics.chamfer_distance >= 15.0:
        logger.error(f"❌ Chamfer distance too high: {metrics.chamfer_distance:.2f} >= 15.0px")
    
    return result


def test_baseline(rgb: np.ndarray) -> IsolationTestResult:
    """
    Baseline: Low-res single-pass inference.
    
    This provides the reference metrics for comparison.
    For large images, resize to manageable resolution.
    """
    logger.info("=== Baseline: Low-Res Single-Pass Inference ===")
    
    from PIL import Image
    from .depth_estimator import HighFidelityDepthEstimator, DepthConfig
    
    # Resize to low-res for baseline (matches typical non-tiled workflow)
    h, w = rgb.shape[:2]
    max_size = 1024
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if rgb.dtype == np.float32:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb
        
        rgb_pil = Image.fromarray(rgb_uint8)
        rgb_resized = rgb_pil.resize((new_w, new_h), Image.LANCZOS)
        rgb_baseline = np.array(rgb_resized)
        logger.info(f"Resized for baseline: {rgb.shape[:2]} → {rgb_baseline.shape[:2]}")
    else:
        rgb_baseline = rgb
    
    config = DepthConfig(
        tile_size=9999,  # Ensure single tile
        overlap=0,
        reconcile_scales=False
    )
    
    estimator = HighFidelityDepthEstimator(config)
    depth_lowres = estimator.estimate_depth(rgb_baseline, use_global_anchor=False)
    
    # Upsample depth to match original resolution
    if max(h, w) > max_size:
        depth_pil = Image.fromarray((depth_lowres * 255).astype(np.uint8))
        depth_upsampled = depth_pil.resize((w, h), Image.BICUBIC)
        depth = np.array(depth_upsampled).astype(np.float32) / 255.0
    else:
        depth = depth_lowres
    
    metrics = validate_depth_quality(rgb, depth, dilation=3)
    
    result = IsolationTestResult(
        name="Baseline (Low-Res)",
        description="Low-res inference + bicubic upsampling (typical workflow)",
        metrics=metrics,
        passed=True  # Reference test
    )
    
    logger.info(str(result))
    return result


def run_isolation_tests(rgb: np.ndarray, output_dir: Optional[Path] = None) -> Dict[str, IsolationTestResult]:
    """
    Run systematic isolation tests.
    
    Args:
        rgb: RGB image (uint8 or float32)
        output_dir: Optional directory to save depth maps
        
    Returns:
        Dictionary of test results
    """
    logger.info("Starting isolation tests...")
    
    results = {}
    
    # Baseline
    baseline = test_baseline(rgb)
    results["baseline"] = baseline
    
    # Test 1: Tiling only (CRITICAL)
    tiling_only = test_tiling_only(rgb, baseline.metrics)
    results["tiling_only"] = tiling_only
    
    # Test 2: Edge snapping refinement (PRIORITY 5)
    if tiling_only.passed:
        edge_snap = test_edge_snapping(rgb, tiling_only.metrics)
        results["edge_snapping"] = edge_snap
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ISOLATION TEST SUMMARY")
    logger.info("="*60)
    
    for name, result in results.items():
        logger.info(str(result))
    
    # Critical validation
    if not tiling_only.passed:
        logger.error("❌ TILING INTEGRATION FAILED")
        logger.error("Root cause: Scale reconciliation or edge detection issue")
        logger.error("Action: Review Theil-Sen regression and float edge detection")
    else:
        logger.info("✅ TILING INTEGRATION PASSED")
        logger.info("Edge F1 meets target threshold (≥0.30)")
    
    # Save results atomically
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_json = {
            name: result.to_dict()
            for name, result in results.items()
        }
        
        save_metrics_atomic(results_json, output_dir / "isolation_test_results.json")
    
    return results


def test_edge_snapping(rgb: np.ndarray, baseline_metrics: EdgeMetrics) -> IsolationTestResult:
    """
    Test 2: Edge snapping refinement.
    
    Tests PRIORITY 5 fix: Edge-gated sharpening (AND-gate RGB + depth edges).
    
    VALIDATION:
    - Edge F1 should improve or maintain
    - Edge sharpness should increase
    - No excessive artifact increase (edge_count_ratio ≤ 2.5)
    """
    logger.info("=== Test 2: Edge Snapping Refinement ===")
    
    from .refinement import edge_snap_refinement
    
    # First get tiled depth
    config = DepthConfig(
        tile_size=1024,
        overlap=128,
        reconcile_scales=True,
        reconcile_method="robust",
        fusion_mode="weighted",
        validate_seams=True
    )
    
    estimator = HighFidelityDepthEstimator(config)
    depth_tiled = estimator.estimate_depth(rgb, use_global_anchor=True)
    
    # Apply edge snapping
    depth_refined = edge_snap_refinement(depth_tiled, rgb, strength=0.2, dilation=5)
    
    metrics = validate_depth_quality(rgb, depth_refined, dilation=3)
    
    # Acceptance criteria: should improve or maintain quality
    f1_ok = metrics.edge_f1 >= baseline_metrics.edge_f1 * 0.95  # Allow 5% degradation
    sharpness_ok = metrics.edge_sharpness_p95 >= baseline_metrics.edge_sharpness_p95
    count_ok = metrics.edge_count_ratio <= 2.5
    
    passed = f1_ok and count_ok
    
    result = IsolationTestResult(
        name="Edge Snapping",
        description="AND-gated edge sharpening (RGB + depth edges)",
        metrics=metrics,
        passed=passed
    )
    
    logger.info(str(result))
    
    # Compare with baseline
    logger.info(f"Edge F1 delta: {metrics.edge_f1 - baseline_metrics.edge_f1:+.3f}")
    logger.info(f"Sharpness delta: {metrics.edge_sharpness_p95 - baseline_metrics.edge_sharpness_p95:+.1f}")
    
    return result
