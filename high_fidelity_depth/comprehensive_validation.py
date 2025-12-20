#!/usr/bin/env python3
"""
Comprehensive High-Fidelity Depth Validation
=============================================

Implements all critical fixes from the review:
1. Atomic JSON write + readback validation (prevents truncation)
2. Unified metric implementation (single source of truth)
3. Shift-tolerant primary metric (edge F1 instead of correlation)
4. Calibrated thresholds based on empirical data
5. Seam detection and validation
6. Edge artifact gates

This script runs the COMPLETE validation suite and produces:
- Validated JSON metrics (atomic write)
- Visual overlays (RGB + depth edges)
- Pass/fail with clear root cause analysis
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from .quality_metrics import (
    EdgeMetrics,
    validate_depth_quality,
    save_metrics_atomic,
    detect_edges
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def load_depth(path: Path) -> np.ndarray:
    """Load depth as float32 [0, 1]."""
    img = Image.open(path)
    
    if img.mode == 'I;16':
        # 16-bit depth
        arr = np.array(img, dtype=np.uint16)
        depth = arr.astype(np.float32) / 65535.0
    elif img.mode == 'I':
        # 32-bit depth
        arr = np.array(img, dtype=np.uint32)
        depth = arr.astype(np.float32) / arr.max() if arr.max() > 0 else arr.astype(np.float32)
    else:
        # 8-bit grayscale
        arr = np.array(img.convert('L'))
        depth = arr.astype(np.float32) / 255.0
    
    return depth


def validate_seams(
    depth: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 128,
    band: int = 2
) -> Tuple[bool, float]:
    """
    Detect seam artifacts at tile boundaries.
    
    Returns:
        (passed, boundary_energy_ratio)
    """
    h, w = depth.shape[:2]
    
    # Compute gradient magnitude
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Create boundary mask (tile edges with tolerance band)
    boundary_mask = np.zeros((h, w), dtype=bool)
    
    step = tile_size - overlap
    
    # Vertical boundaries
    for x in range(step, w, step):
        x_start = max(0, x - band)
        x_end = min(w, x + band)
        boundary_mask[:, x_start:x_end] = True
    
    # Horizontal boundaries
    for y in range(step, h, step):
        y_start = max(0, y - band)
        y_end = min(h, y + band)
        boundary_mask[y_start:y_end, :] = True
    
    # Compute gradient energy
    if boundary_mask.sum() == 0:
        return True, 0.0
    
    boundary_energy = grad_mag[boundary_mask].mean()
    global_energy = grad_mag[~boundary_mask].mean()
    
    ratio = boundary_energy / max(global_energy, 1e-6)
    
    # Threshold: boundary energy should not be >20% higher than global
    passed = ratio < 1.2
    
    logger.info(f"Seam validation: boundary/global ratio = {ratio:.3f} ({'✅ PASS' if passed else '❌ FAIL'})")
    
    return passed, ratio


def create_edge_overlay(
    rgb: np.ndarray,
    depth: np.ndarray,
    output_path: Path
) -> None:
    """
    PRIORITY 4 FIX: Create readable edge alignment overlay with thin colored lines.
    
    Color scheme:
    - RED: RGB edges only (depth missing edge)
    - BLUE: Depth edges only (hallucinated edge)
    - GREEN: Aligned edges (both agree)
    """
    # Detect edges
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = detect_edges(gray)
    
    # For depth, use float-aware detection
    depth_edges = detect_edges(depth)
    
    # Start with RGB as base
    overlay = rgb.copy()
    
    # Dilate edges slightly for visibility
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rgb_e = cv2.dilate(rgb_edges.astype(np.uint8), kernel, iterations=1)
    depth_e = cv2.dilate(depth_edges.astype(np.uint8), kernel, iterations=1)
    
    # Compute overlap
    overlap = (rgb_e > 0) & (depth_e > 0)
    rgb_only = (rgb_e > 0) & (depth_e == 0)
    depth_only = (rgb_e == 0) & (depth_e > 0)
    
    # Draw thin colored lines
    overlay[rgb_only] = [255, 0, 0]     # RED: RGB edges only
    overlay[depth_only] = [0, 0, 255]   # BLUE: depth edges only
    overlay[overlap] = [0, 255, 0]      # GREEN: aligned edges
    
    # Add legend
    legend_h = 60
    legend = np.ones((legend_h, overlay.shape[1], 3), dtype=np.uint8) * 240
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    cv2.putText(legend, "RED: RGB only", (10, 20), font, font_scale, (255, 0, 0), thickness)
    cv2.putText(legend, "BLUE: Depth only", (10, 45), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(legend, "GREEN: Aligned", (250, 20), font, font_scale, (0, 255, 0), thickness)
    
    # Add alignment stats to legend
    total_edges = rgb_e.sum() + depth_e.sum() - overlap.sum()
    if total_edges > 0:
        aligned_pct = 100.0 * overlap.sum() / total_edges
        cv2.putText(legend, f"Alignment: {aligned_pct:.1f}%", (250, 45), font, font_scale, (0, 0, 0), thickness)
    
    final = np.vstack([overlay, legend])
    
    # Save
    from PIL import Image
    Image.fromarray(final).save(output_path)
    logger.info(f"Edge overlay saved (PRIORITY 4 readable format): {output_path}")


def run_comprehensive_validation(
    rgb_path: Path,
    depth_path: Path,
    output_dir: Path,
    tile_size: int = 1024,
    overlap: int = 128
) -> Dict:
    """
    Run comprehensive validation with all critical checks.
    
    Returns:
        Complete validation report (dict)
    """
    logger.info("="*60)
    logger.info("COMPREHENSIVE HIGH-FIDELITY DEPTH VALIDATION")
    logger.info("="*60)
    
    # Load inputs
    logger.info(f"Loading RGB: {rgb_path}")
    rgb = load_image(rgb_path)
    
    logger.info(f"Loading depth: {depth_path}")
    depth = load_depth(depth_path)
    
    # Validate dimensions
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"Dimension mismatch: RGB {rgb.shape[:2]} vs depth {depth.shape[:2]}")
    
    logger.info(f"Image size: {rgb.shape[:2]}")
    
    # 1. Edge-based quality metrics (CANONICAL)
    logger.info("\n--- Edge-Based Quality Metrics ---")
    metrics = validate_depth_quality(rgb, depth, dilation=3)
    
    # 2. Seam detection
    logger.info("\n--- Seam Detection ---")
    seams_ok, seam_ratio = validate_seams(depth, tile_size, overlap)
    
    # 3. Edge count sanity check
    logger.info("\n--- Artifact Detection ---")
    edge_count_ok = metrics.edge_count_ratio <= 3.0
    
    if not edge_count_ok:
        logger.warning(f"⚠️  Edge count ratio {metrics.edge_count_ratio:.2f}× exceeds threshold (3.0×)")
        logger.warning("   This suggests artifact edges (not real boundaries)")
    
    # 4. Overall quality assessment
    logger.info("\n--- Quality Assessment ---")
    quality_score = metrics.quality_score()
    
    passed_lenient = metrics.passed(strict=False)
    passed_strict = metrics.passed(strict=True)
    
    logger.info(f"Quality score: {quality_score:.3f}")
    logger.info(f"Lenient pass: {passed_lenient}")
    logger.info(f"Strict pass: {passed_strict}")
    
    # 5. Root cause analysis
    logger.info("\n--- Root Cause Analysis ---")
    
    issues = []
    
    if not seams_ok:
        issues.append("SEAM_ARTIFACTS: Tile boundary energy elevated")
        logger.error("❌ Seam artifacts detected → scale reconciliation insufficient")
    
    if metrics.edge_count_ratio > 3.0:
        issues.append("EDGE_EXPLOSION: Excessive depth edges")
        logger.error("❌ Edge explosion → global sharpening or snapping mask inverted")
    
    if metrics.edge_f1 < 0.25:
        issues.append("LOW_ALIGNMENT: Poor edge F1 score")
        logger.error("❌ Low alignment → depth/RGB resolution mismatch or wrong guide")
    
    if metrics.overshoot_penalty > 0.5:
        issues.append("OVERSHOOT: Excessive ringing")
        logger.warning("⚠️  Overshoot penalty high → check unsharp mask / edge snapping")
    
    if not issues:
        logger.info("✅ No critical issues detected")
    
    # 6. Create visual overlay
    logger.info("\n--- Creating Visual Overlay ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "edge_overlay.png"
    create_edge_overlay(rgb, depth, overlay_path)
    
    # 7. Compile report
    report = {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "image_size": list(rgb.shape[:2]),
        "metrics": metrics.to_dict(),
        "seam_validation": {
            "passed": bool(seams_ok),
            "boundary_energy_ratio": float(seam_ratio)
        },
        "quality_score": float(quality_score),
        "passed_lenient": bool(passed_lenient),
        "passed_strict": bool(passed_strict),
        "issues": issues
    }
    
    # 8. Save report atomically
    logger.info("\n--- Saving Report ---")
    report_path = output_dir / "validation_report.json"
    save_metrics_atomic(report, report_path)
    
    # 9. Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Quality score: {quality_score:.3f}")
    logger.info(f"Edge F1 (primary): {metrics.edge_f1:.3f}")
    logger.info(f"Edge overlap: {metrics.edge_overlap:.3f}")
    logger.info(f"Edge count ratio: {metrics.edge_count_ratio:.2f}×")
    logger.info(f"Seam check: {'✅ PASS' if seams_ok else '❌ FAIL'}")
    logger.info(f"Overall: {'✅ PASS (strict)' if passed_strict else '✅ PASS (lenient)' if passed_lenient else '❌ FAIL'}")
    logger.info("="*60)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive high-fidelity depth validation"
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        required=True,
        help="Path to RGB image"
    )
    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Path to depth map"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comprehensive_validation"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for seam detection"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Tile overlap for seam detection"
    )
    
    args = parser.parse_args()
    
    try:
        report = run_comprehensive_validation(
            args.rgb,
            args.depth,
            args.output_dir,
            args.tile_size,
            args.overlap
        )
        
        # Exit code based on validation result
        if report["passed_lenient"]:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
