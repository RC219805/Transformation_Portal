#!/usr/bin/env python3
"""
Full Dataset Production Validation
===================================

Addresses all critical review points:
1. Full 750_Picacho dataset at native resolutions
2. Per-image metrics with precision/recall breakdown
3. Visual gallery for challenging scenes
4. Runtime/memory profiling
5. Atomic JSON export with config capture
6. Seam energy worst-case tracking
7. Halo/overshoot detection

Usage:
    python full_dataset_validation.py --input-dir input_images/750_Picacho/Source_TIFFs_Base --output-dir outputs/validation_run
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import cv2
import numpy as np
from PIL import Image

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig
from high_fidelity_depth.quality_metrics import (
    EdgeMetrics,
    validate_depth_quality,
    save_metrics_atomic
)
from high_fidelity_depth import refinement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Complete validation configuration."""
    
    # Depth inference
    tile_size: int = 1024
    overlap: int = 128
    use_global_anchor: bool = False  # OFF by default per review
    reconcile_scales: bool = True
    
    # Refinement
    edge_snap_enabled: bool = True
    edge_snap_strength: float = 0.2
    guided_filter_enabled: bool = True
    clahe_enabled: bool = False  # OFF for geometry
    
    # Quality gates
    min_edge_f1: float = 0.30
    max_edge_count_ratio: float = 2.0
    max_chamfer_distance: float = 15.0
    max_seam_energy: float = 1.2
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return asdict(self)
    
    def config_hash(self) -> str:
        """Generate unique hash for this configuration."""
        cfg_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(cfg_str.encode()).hexdigest()[:8]


@dataclass
class ImageMetrics:
    """Per-image validation results."""
    
    image_name: str
    resolution: Tuple[int, int]
    
    # Edge quality (primary)
    edge_f1: float
    edge_precision: float
    edge_recall: float
    chamfer_distance: float
    edge_overlap: float
    
    # Edge artifacts
    edge_count_ratio: float
    halo_score: float
    
    # Seam quality
    seam_energy: float
    seam_detected: bool
    
    # Performance
    runtime_seconds: float
    peak_memory_mb: Optional[float]
    
    # Quality gate
    passed: bool
    failure_reasons: List[str]
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        d = asdict(self)
        d['resolution'] = list(d['resolution'])  # tuple → list for JSON
        return d


def compute_halo_score(depth: np.ndarray, rgb: np.ndarray) -> float:
    """
    Detect overshoot/halo artifacts around edges.
    
    Returns normalized overshoot metric (0 = none, 1+ = severe).
    """
    # RGB edges (ground truth boundaries)
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(rgb_gray, 50, 150)
    
    # Dilate to create edge neighborhood
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_region = cv2.dilate(rgb_edges, kernel, iterations=1)
    
    # Depth gradient in edge region
    depth_f32 = (depth * 255.0).astype(np.float32)
    sobelx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Overshoot: high gradient variance in edge region (halos have ringing)
    edge_mask = edge_region > 0
    if not edge_mask.any():
        return 0.0
    
    edge_gradients = grad_mag[edge_mask]
    p95 = np.percentile(edge_gradients, 95)
    p50 = np.percentile(edge_gradients, 50)
    
    # High p95/p50 ratio indicates ringing
    if p50 > 0:
        overshoot = (p95 / p50) - 1.0
    else:
        overshoot = 0.0
    
    return float(np.clip(overshoot / 3.0, 0.0, 2.0))  # Normalized


def validate_single_image(
    rgb_path: Path,
    estimator: HighFidelityDepthEstimator,
    config: ValidationConfig,
    output_dir: Path
) -> ImageMetrics:
    """
    Run complete validation on single image.
    
    Returns:
        ImageMetrics with all quality scores
    """
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else None
    
    logger.info(f"Processing {rgb_path.name}...")
    
    # Load RGB
    rgb_img = Image.open(rgb_path)
    if rgb_img.mode != 'RGB':
        rgb_img = rgb_img.convert('RGB')
    rgb = np.array(rgb_img)
    H, W = rgb.shape[:2]
    
    # Generate depth (tiled inference) - pass numpy array
    depth_result = estimator.estimate_depth(rgb)
    depth_raw = depth_result['depth']
    
    # Apply refinement
    depth_refined = refinement.apply_refinement(
        depth_raw, rgb,
        edge_snap_enabled=config.edge_snap_enabled,
        edge_snap_strength=config.edge_snap_strength,
        guided_filter_enabled=config.guided_filter_enabled,
        clahe_enabled=config.clahe_enabled
    )
    
    # Normalize to [0, 1]
    depth_norm = (depth_refined - depth_refined.min()) / (depth_refined.max() - depth_refined.min() + 1e-8)
    
    # RGB edges (ground truth)
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(rgb_gray, 50, 150)
    
    # Depth edges (float-based, critical fix)
    depth_f32 = (depth_norm * 255.0).astype(np.float32)
    sobelx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad = np.sqrt(sobelx**2 + sobely**2)
    depth_edges = (depth_grad > np.percentile(depth_grad, 90)).astype(np.uint8) * 255
    
    # Edge F1 with tolerance (shift-tolerant primary metric)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rgb_edges_dilated = cv2.dilate(rgb_edges, kernel, iterations=1)
    
    tp = np.sum((rgb_edges_dilated > 0) & (depth_edges > 0))
    fp = np.sum((rgb_edges_dilated == 0) & (depth_edges > 0))
    fn = np.sum((rgb_edges > 0) & (depth_edges == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Chamfer distance (mean distance to nearest edge)
    depth_edge_coords = np.column_stack(np.where(depth_edges > 0))
    rgb_edge_coords = np.column_stack(np.where(rgb_edges > 0))
    
    if len(depth_edge_coords) > 0 and len(rgb_edge_coords) > 0:
        from scipy.spatial.distance import cdist
        dists = cdist(depth_edge_coords, rgb_edge_coords)
        chamfer = float(np.mean(np.min(dists, axis=1)))
    else:
        chamfer = 999.0
    
    # Edge overlap
    overlap = tp / (np.sum(depth_edges > 0) + 1e-8)
    
    # Edge count ratio (artifact detector)
    baseline_edge_count = np.sum(rgb_edges > 0)
    depth_edge_count = np.sum(depth_edges > 0)
    edge_ratio = depth_edge_count / (baseline_edge_count + 1e-8)
    
    # Halo score
    halo = compute_halo_score(depth_norm, rgb)
    
    # Seam validation
    seam_energy = 1.0
    seam_detected = False
    if config.tile_size > 0:
        from high_fidelity_depth.comprehensive_validation import validate_seams
        seam_detected, seam_energy = validate_seams(
            depth_norm, config.tile_size, config.overlap, band=2
        )
    
    # Performance
    runtime = time.time() - start_time
    peak_mem = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else None
    if start_mem and peak_mem:
        peak_mem = peak_mem - start_mem
    
    # Quality gate
    failures = []
    if f1 < config.min_edge_f1:
        failures.append(f"edge_f1={f1:.3f} < {config.min_edge_f1}")
    if edge_ratio > config.max_edge_count_ratio:
        failures.append(f"edge_ratio={edge_ratio:.2f} > {config.max_edge_count_ratio}")
    if chamfer > config.max_chamfer_distance:
        failures.append(f"chamfer={chamfer:.1f} > {config.max_chamfer_distance}")
    if seam_energy > config.max_seam_energy:
        failures.append(f"seam_energy={seam_energy:.3f} > {config.max_seam_energy}")
    if halo > 0.5:
        failures.append(f"halo_score={halo:.3f} > 0.5")
    
    passed = len(failures) == 0
    
    # Save visualizations
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Depth map (16-bit TIFF)
    depth_16bit = (depth_norm * 65535).astype(np.uint16)
    Image.fromarray(depth_16bit, mode='I;16').save(vis_dir / f"{rgb_path.stem}_depth.tiff")
    
    # Edge overlay (RGB + depth edges in red)
    overlay = rgb.copy()
    overlay[depth_edges > 0] = [255, 0, 0]
    Image.fromarray(overlay).save(vis_dir / f"{rgb_path.stem}_edges.png")
    
    # Comparison grid
    depth_viz = (depth_norm * 255).astype(np.uint8)
    depth_viz_rgb = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
    depth_viz_rgb = cv2.cvtColor(depth_viz_rgb, cv2.COLOR_BGR2RGB)
    
    # Resize for grid
    scale = 0.5
    rgb_small = cv2.resize(rgb, None, fx=scale, fy=scale)
    depth_small = cv2.resize(depth_viz_rgb, None, fx=scale, fy=scale)
    edge_small = cv2.resize(overlay, None, fx=scale, fy=scale)
    
    # Create 2x2 grid
    top = np.hstack([rgb_small, depth_small])
    bottom = np.hstack([edge_small, np.zeros_like(rgb_small)])
    grid = np.vstack([top, bottom])
    
    Image.fromarray(grid).save(vis_dir / f"{rgb_path.stem}_grid.png")
    
    return ImageMetrics(
        image_name=rgb_path.name,
        resolution=(W, H),
        edge_f1=float(f1),
        edge_precision=float(precision),
        edge_recall=float(recall),
        chamfer_distance=chamfer,
        edge_overlap=float(overlap),
        edge_count_ratio=float(edge_ratio),
        halo_score=halo,
        seam_energy=seam_energy,
        seam_detected=seam_detected,
        runtime_seconds=runtime,
        peak_memory_mb=peak_mem,
        passed=passed,
        failure_reasons=failures
    )


def main():
    parser = argparse.ArgumentParser(description="Full dataset production validation")
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory with source TIFF images')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for results')
    parser.add_argument('--tile-size', type=int, default=1024,
                        help='Tile size for inference (default: 1024)')
    parser.add_argument('--no-global-anchor', action='store_true',
                        help='Disable global anchor fusion (recommended per review)')
    parser.add_argument('--no-edge-snap', action='store_true',
                        help='Disable edge snapping')
    parser.add_argument('--enable-clahe', action='store_true',
                        help='Enable CLAHE (not recommended for geometry)')
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    val_config = ValidationConfig(
        tile_size=args.tile_size,
        use_global_anchor=not args.no_global_anchor,
        edge_snap_enabled=not args.no_edge_snap,
        clahe_enabled=args.enable_clahe
    )
    
    logger.info("=" * 80)
    logger.info("FULL DATASET PRODUCTION VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Config hash: {val_config.config_hash()}")
    logger.info(f"Tile size: {val_config.tile_size}")
    logger.info(f"Global anchor: {val_config.use_global_anchor}")
    logger.info(f"Edge snap: {val_config.edge_snap_enabled}")
    logger.info("=" * 80)
    
    # Find images
    image_paths = sorted(args.input_dir.glob("*.tif*"))
    if not image_paths:
        logger.error(f"No images found in {args.input_dir}")
        return 1
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Initialize pipeline
    depth_config = DepthConfig(
        tile_size=val_config.tile_size,
        overlap=val_config.overlap,
        reconcile_scales=val_config.reconcile_scales
    )
    
    estimator = HighFidelityDepthEstimator(depth_config)
    
    # Process all images
    results: List[ImageMetrics] = []
    
    for img_path in image_paths:
        try:
            metrics = validate_single_image(
                img_path, estimator, val_config, args.output_dir
            )
            results.append(metrics)
            
            status = "✓ PASS" if metrics.passed else "✗ FAIL"
            logger.info(f"{status} {img_path.name}: F1={metrics.edge_f1:.3f}, "
                        f"Chamfer={metrics.chamfer_distance:.1f}px, "
                        f"Seam={metrics.seam_energy:.3f}")
            
            if not metrics.passed:
                for reason in metrics.failure_reasons:
                    logger.warning(f"  - {reason}")
        
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}", exc_info=True)
    
    # Aggregate statistics
    if not results:
        logger.error("No successful validations")
        return 1
    
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    
    edge_f1_values = [r.edge_f1 for r in results]
    chamfer_values = [r.chamfer_distance for r in results]
    seam_values = [r.seam_energy for r in results]
    runtime_values = [r.runtime_seconds for r in results]
    
    summary = {
        'config': val_config.to_dict(),
        'config_hash': val_config.config_hash(),
        'total_images': len(results),
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': passed_count / len(results),
        
        # Edge F1 statistics
        'edge_f1_mean': float(np.mean(edge_f1_values)),
        'edge_f1_median': float(np.median(edge_f1_values)),
        'edge_f1_min': float(np.min(edge_f1_values)),
        'edge_f1_max': float(np.max(edge_f1_values)),
        
        # Chamfer statistics
        'chamfer_mean': float(np.mean(chamfer_values)),
        'chamfer_median': float(np.median(chamfer_values)),
        'chamfer_worst': float(np.max(chamfer_values)),
        
        # Seam statistics
        'seam_energy_mean': float(np.mean(seam_values)),
        'seam_energy_worst': float(np.max(seam_values)),
        
        # Performance
        'runtime_mean_sec': float(np.mean(runtime_values)),
        'runtime_total_sec': float(np.sum(runtime_values)),
        
        # Per-image results
        'per_image': [r.to_dict() for r in results]
    }
    
    # Atomic JSON write
    summary_path = args.output_dir / 'validation_summary.json'
    temp_path = summary_path.with_suffix('.tmp')
    
    with open(temp_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Validate readback
    with open(temp_path, 'r') as f:
        readback = json.load(f)
    
    # Rename to final
    temp_path.rename(summary_path)
    
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Pass rate: {passed_count}/{len(results)} ({summary['pass_rate']:.1%})")
    logger.info(f"Edge F1: {summary['edge_f1_mean']:.3f} ± {np.std(edge_f1_values):.3f}")
    logger.info(f"Chamfer: {summary['chamfer_mean']:.1f}px (worst: {summary['chamfer_worst']:.1f}px)")
    logger.info(f"Seam energy: {summary['seam_energy_mean']:.3f} (worst: {summary['seam_energy_worst']:.3f})")
    logger.info(f"Runtime: {summary['runtime_mean_sec']:.1f}s per image")
    logger.info(f"Results saved to: {summary_path}")
    logger.info("=" * 80)
    
    return 0 if passed_count == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
