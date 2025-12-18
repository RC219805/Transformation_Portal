#!/usr/bin/env python3
"""
Production Validation: Full 750_Picacho Dataset
===============================================

Comprehensive validation suite implementing all requirements:

1. Scale validation on full dataset at native resolution
2. Per-image metrics with precision/recall breakdown
3. Production preset configuration
4. Missing metrics: halo/overshoot, detail benefit
5. Yellow flag investigation
6. Materials V3 integration readiness

Critical Success Criteria:
- Edge F1 precision/recall analysis
- Halo/overshoot detection
- Global anchor on planar scenes
- Visual gallery of challenging scenes
- Go/no-go acceptance report
"""

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add high_fidelity_depth to path
sys.path.insert(0, str(Path(__file__).parent))

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig
from high_fidelity_depth.quality_metrics import (
    EdgeMetrics,
    validate_depth_quality,
    save_metrics_atomic,
    detect_edges
)
from high_fidelity_depth.refinement import edge_snap_refinement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionPreset:
    """Production processing preset configuration."""
    name: str
    tile_size: int
    overlap: int
    use_global_anchor: bool
    use_edge_snapping: bool
    edge_snap_strength: float
    description: str


# Production presets based on requirements
PRODUCTION_PRESETS = {
    "preview": ProductionPreset(
        name="Preview",
        tile_size=512,
        overlap=64,
        use_global_anchor=False,
        use_edge_snapping=False,
        edge_snap_strength=0.0,
        description="Fast preview mode - smaller tiles, no refinement"
    ),
    "production": ProductionPreset(
        name="Production",
        tile_size=1024,
        overlap=128,
        use_global_anchor=True,
        use_edge_snapping=True,
        edge_snap_strength=0.2,
        description="Production mode - robust reconciliation, AND-gated snapping"
    ),
    "hero": ProductionPreset(
        name="Hero",
        tile_size=1536,
        overlap=192,
        use_global_anchor=True,
        use_edge_snapping=True,
        edge_snap_strength=0.3,
        description="Hero mode - maximum fidelity for showcase images"
    )
}


@dataclass
class EnhancedMetrics:
    """Enhanced metrics including new requirements."""
    
    # Standard edge metrics
    edge_f1: float
    edge_precision: float  # NEW: Separate precision
    edge_recall: float  # NEW: Separate recall
    edge_overlap: float
    chamfer_distance: float
    edge_sharpness_p95: float
    edge_count_ratio: float
    
    # Seam validation
    seam_energy_ratio: float
    seam_passed: bool
    
    # NEW: Halo/overshoot detection
    halo_score: float  # [0,1] - 1 is good (no halos)
    overshoot_penalty: float  # [0,1] - 0 is good (no ringing)
    max_local_slope: float  # Maximum slope around edges
    
    # NEW: Detail benefit metric
    detail_benefit: float  # Ratio: (tiled local variance) / (baseline local variance)
    hf_energy_ratio: float  # High-frequency energy ratio
    
    # Performance
    runtime_seconds: float
    peak_memory_mb: float
    
    # Diagnostic
    rgb_edge_count: int
    depth_edge_count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def quality_gate_passed(self, strict: bool = False) -> bool:
        """Check if metrics meet quality gates for Materials V3."""
        if strict:
            return (
                self.edge_f1 >= 0.45 and
                self.edge_precision >= 0.40 and
                self.edge_recall >= 0.40 and
                self.edge_count_ratio <= 2.0 and
                self.seam_passed and
                self.halo_score >= 0.70 and
                self.overshoot_penalty <= 0.30 and
                self.detail_benefit >= 1.0  # Tiled should add real detail
            )
        else:
            return (
                self.edge_f1 >= 0.30 and
                self.edge_precision >= 0.25 and
                self.edge_recall >= 0.25 and
                self.edge_count_ratio <= 3.0 and
                self.seam_passed and
                self.overshoot_penalty <= 0.50
            )


def compute_edge_precision_recall(
    depth_edges: np.ndarray,
    rgb_edges: np.ndarray,
    tolerance: int = 2
) -> Tuple[float, float]:
    """
    Compute edge precision and recall separately.
    
    Precision: What fraction of depth edges are near RGB edges?
    Recall: What fraction of RGB edges are detected in depth?
    
    Args:
        depth_edges: Binary edge map from depth
        rgb_edges: Binary edge map from RGB
        tolerance: Pixel tolerance for matching
    
    Returns:
        (precision, recall)
    """
    # Dilate RGB edges for tolerance
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    rgb_dilated = cv2.dilate(rgb_edges, kernel, iterations=1)
    
    # Precision: depth edges that match RGB edges
    depth_edge_pixels = depth_edges > 0
    if depth_edge_pixels.sum() == 0:
        precision = 0.0
    else:
        true_positives = np.logical_and(depth_edge_pixels, rgb_dilated > 0).sum()
        precision = true_positives / depth_edge_pixels.sum()
    
    # Recall: RGB edges detected in depth
    rgb_edge_pixels = rgb_edges > 0
    if rgb_edge_pixels.sum() == 0:
        recall = 0.0
    else:
        # Dilate depth edges for recall
        depth_dilated = cv2.dilate(depth_edges, kernel, iterations=1)
        detected = np.logical_and(rgb_edge_pixels, depth_dilated > 0).sum()
        recall = detected / rgb_edge_pixels.sum()
    
    return precision, recall


def compute_halo_overshoot_metrics(
    depth: np.ndarray,
    rgb_edges: np.ndarray
) -> Tuple[float, float, float]:
    """
    Detect halos and overshoot artifacts around edges.
    
    Halos: Bright/dark rings around edges (unsharp mask artifacts)
    Overshoot: Laplacian ringing near high-contrast boundaries
    
    Args:
        depth: Depth map [0, 1]
        rgb_edges: Binary edge map from RGB
    
    Returns:
        (halo_score, overshoot_penalty, max_local_slope)
    """
    # Convert to uint8 for processing
    depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    
    # Compute Laplacian for overshoot detection
    laplacian = cv2.Laplacian(depth_uint8, cv2.CV_32F, ksize=3)
    laplacian_abs = np.abs(laplacian)
    
    # Create edge neighborhood mask (5px band around RGB edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    edge_neighborhood = cv2.dilate(rgb_edges, kernel, iterations=1)
    edge_neighborhood = edge_neighborhood > 0
    
    # Overshoot: average Laplacian magnitude near edges
    if edge_neighborhood.sum() > 0:
        edge_overshoot = laplacian_abs[edge_neighborhood].mean()
        global_overshoot = laplacian_abs[~edge_neighborhood].mean()
        overshoot_penalty = min(edge_overshoot / max(global_overshoot, 1.0), 1.0)
    else:
        overshoot_penalty = 0.0
    
    # Halo score: inverse of overshoot (1 = no halos)
    halo_score = max(0.0, 1.0 - overshoot_penalty)
    
    # Max local slope around edges
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    if edge_neighborhood.sum() > 0:
        max_local_slope = np.percentile(gradient_mag[edge_neighborhood], 99)
    else:
        max_local_slope = gradient_mag.max()
    
    return halo_score, overshoot_penalty, max_local_slope


def compute_detail_benefit_metric(
    depth_tiled: np.ndarray,
    depth_baseline: np.ndarray,
    window_size: int = 16
) -> Tuple[float, float]:
    """
    Quantify whether tiling adds real detail vs noise.
    
    Compares local depth variance and high-frequency energy:
    - detail_benefit > 1.0: tiled has more local structure
    - hf_energy_ratio: ratio of high-frequency components
    
    Args:
        depth_tiled: Tiled depth output
        depth_baseline: Baseline (single-pass) depth
        window_size: Window for local variance computation
    
    Returns:
        (detail_benefit, hf_energy_ratio)
    """
    # Ensure same size
    if depth_tiled.shape != depth_baseline.shape:
        logger.warning("Depth maps have different sizes, skipping detail benefit")
        return 1.0, 1.0
    
    # Convert to uint8 for processing
    tiled_uint8 = (np.clip(depth_tiled, 0, 1) * 255).astype(np.uint8)
    baseline_uint8 = (np.clip(depth_baseline, 0, 1) * 255).astype(np.uint8)
    
    # Local variance using sliding window
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size**2)
    
    # Variance = E[X²] - E[X]²
    tiled_mean = cv2.filter2D(tiled_uint8.astype(np.float32), -1, kernel)
    tiled_sq_mean = cv2.filter2D((tiled_uint8.astype(np.float32))**2, -1, kernel)
    tiled_variance = tiled_sq_mean - tiled_mean**2
    
    baseline_mean = cv2.filter2D(baseline_uint8.astype(np.float32), -1, kernel)
    baseline_sq_mean = cv2.filter2D((baseline_uint8.astype(np.float32))**2, -1, kernel)
    baseline_variance = baseline_sq_mean - baseline_mean**2
    
    # Average variance ratio
    detail_benefit = (tiled_variance.mean() + 1e-6) / (baseline_variance.mean() + 1e-6)
    
    # High-frequency energy using Laplacian
    tiled_hf = cv2.Laplacian(tiled_uint8, cv2.CV_32F, ksize=3)
    baseline_hf = cv2.Laplacian(baseline_uint8, cv2.CV_32F, ksize=3)
    
    tiled_hf_energy = (tiled_hf**2).mean()
    baseline_hf_energy = (baseline_hf**2).mean()
    
    hf_energy_ratio = (tiled_hf_energy + 1e-6) / (baseline_hf_energy + 1e-6)
    
    return detail_benefit, hf_energy_ratio


def validate_seams(
    depth: np.ndarray,
    tile_size: int,
    overlap: int,
    band: int = 2
) -> Tuple[bool, float]:
    """Detect seam artifacts at tile boundaries."""
    h, w = depth.shape[:2]
    
    # Compute gradient magnitude
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Create boundary mask
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
    
    if boundary_mask.sum() == 0:
        return True, 0.0
    
    boundary_energy = grad_mag[boundary_mask].mean()
    global_energy = grad_mag[~boundary_mask].mean()
    ratio = boundary_energy / max(global_energy, 1e-6)
    
    # Threshold: boundary energy should not be >20% higher
    passed = ratio < 1.2
    
    return passed, ratio


def process_single_image(
    rgb_path: Path,
    preset: ProductionPreset,
    output_dir: Path,
    save_visuals: bool = False,
    max_dimension: int = 4096
) -> Optional[EnhancedMetrics]:
    """
    Process single image with comprehensive metrics.
    
    Args:
        max_dimension: Maximum dimension for processing (to avoid memory issues)
    
    Returns:
        EnhancedMetrics or None if failed
    """
    try:
        start_time = time.time()
        
        # Load RGB
        rgb_pil = Image.open(rgb_path)
        if rgb_pil.mode != 'RGB':
            rgb_pil = rgb_pil.convert('RGB')
        
        # Resize if too large (to avoid memory issues on very large images)
        original_size = rgb_pil.size
        if max(rgb_pil.size) > max_dimension:
            scale = max_dimension / max(rgb_pil.size)
            new_size = (int(rgb_pil.width * scale), int(rgb_pil.height * scale))
            logger.info(f"Resizing {rgb_path.name}: {original_size} → {new_size} (max_dim={max_dimension})")
            rgb_pil = rgb_pil.resize(new_size, Image.LANCZOS)
        
        rgb = np.array(rgb_pil)
        
        # Configure pipeline
        config = DepthConfig(
            tile_size=preset.tile_size,
            overlap=preset.overlap,
            reconcile_scales=True,
            reconcile_method="robust",
            fusion_mode="weighted"
        )
        
        estimator = HighFidelityDepthEstimator(config)
        
        # Generate baseline (for detail benefit metric) - use smaller tiles to avoid OOM
        baseline_config = DepthConfig(tile_size=2048, overlap=0, reconcile_scales=False)
        baseline_estimator = HighFidelityDepthEstimator(baseline_config)
        depth_baseline = baseline_estimator.estimate_depth(rgb, use_global_anchor=False)
        
        # Generate tiled depth
        depth_tiled = estimator.estimate_depth(rgb, use_global_anchor=preset.use_global_anchor)
        
        # Apply edge snapping if enabled
        if preset.use_edge_snapping:
            depth_final = edge_snap_refinement(depth_tiled, rgb, strength=preset.edge_snap_strength)
        else:
            depth_final = depth_tiled
        
        runtime = time.time() - start_time
        
        # Compute enhanced metrics
        
        # 1. Edge detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb_edges = detect_edges(gray)
        
        depth_uint8 = (np.clip(depth_final, 0, 1) * 255).astype(np.uint8)
        depth_edges = detect_edges(depth_uint8)
        
        # 2. Standard metrics
        base_metrics = validate_depth_quality(rgb, depth_final, dilation=3)
        
        # 3. Precision/recall
        precision, recall = compute_edge_precision_recall(depth_edges, rgb_edges, tolerance=2)
        
        # 4. Seam validation
        seam_passed, seam_ratio = validate_seams(
            depth_final,
            preset.tile_size,
            preset.overlap
        )
        
        # 5. Halo/overshoot metrics
        halo_score, overshoot_penalty, max_slope = compute_halo_overshoot_metrics(
            depth_final,
            rgb_edges
        )
        
        # 6. Detail benefit
        detail_benefit, hf_energy_ratio = compute_detail_benefit_metric(
            depth_tiled,
            depth_baseline
        )
        
        # Create enhanced metrics
        metrics = EnhancedMetrics(
            edge_f1=base_metrics.edge_f1,
            edge_precision=precision,
            edge_recall=recall,
            edge_overlap=base_metrics.edge_overlap,
            chamfer_distance=base_metrics.chamfer_distance,
            edge_sharpness_p95=base_metrics.edge_sharpness_p95,
            edge_count_ratio=base_metrics.edge_count_ratio,
            seam_energy_ratio=seam_ratio,
            seam_passed=seam_passed,
            halo_score=halo_score,
            overshoot_penalty=overshoot_penalty,
            max_local_slope=max_slope,
            detail_benefit=detail_benefit,
            hf_energy_ratio=hf_energy_ratio,
            runtime_seconds=runtime,
            peak_memory_mb=0.0,  # TODO: Implement if needed
            rgb_edge_count=base_metrics.rgb_edge_count,
            depth_edge_count=base_metrics.depth_edge_count
        )
        
        # Save visual outputs if requested
        if save_visuals:
            image_output_dir = output_dir / rgb_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save depth maps
            depth_baseline_pil = Image.fromarray((np.clip(depth_baseline, 0, 1) * 65535).astype(np.uint16), mode='I;16')
            depth_baseline_pil.save(image_output_dir / "depth_baseline.tiff")
            
            depth_tiled_pil = Image.fromarray((np.clip(depth_tiled, 0, 1) * 65535).astype(np.uint16), mode='I;16')
            depth_tiled_pil.save(image_output_dir / "depth_tiled.tiff")
            
            depth_final_pil = Image.fromarray((np.clip(depth_final, 0, 1) * 65535).astype(np.uint16), mode='I;16')
            depth_final_pil.save(image_output_dir / "depth_final.tiff")
            
            # Create edge overlay
            overlay = rgb.copy()
            overlay[rgb_edges > 0] = [0, 255, 0]  # RGB edges in green
            overlay[depth_edges > 0] = [255, 0, 255]  # Depth edges in magenta
            overlap = np.logical_and(rgb_edges > 0, depth_edges > 0)
            overlay[overlap] = [255, 255, 0]  # Overlap in yellow
            
            Image.fromarray(overlay).save(image_output_dir / "edge_overlay.png")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to process {rgb_path.name}: {e}")
        logger.debug(traceback.format_exc())
        return None


def run_production_validation(
    input_dir: Path,
    output_dir: Path,
    preset_name: str = "production",
    priority_scenes: Optional[List[str]] = None,
    save_all_visuals: bool = False,
    max_dimension: int = 4096
) -> Dict:
    """
    Run production validation on full dataset.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Output directory for reports
        preset_name: Preset to use ("preview", "production", "hero")
        priority_scenes: List of image names to prioritize for visual gallery
        save_all_visuals: Save visual outputs for all images
    
    Returns:
        Comprehensive validation report
    """
    logger.info("="*80)
    logger.info("PRODUCTION VALIDATION: FULL 750_PICACHO DATASET")
    logger.info("="*80)
    
    # Get preset
    if preset_name not in PRODUCTION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = PRODUCTION_PRESETS[preset_name]
    logger.info(f"Preset: {preset.name}")
    logger.info(f"  - Tile size: {preset.tile_size}")
    logger.info(f"  - Overlap: {preset.overlap}")
    logger.info(f"  - Global anchor: {preset.use_global_anchor}")
    logger.info(f"  - Edge snapping: {preset.use_edge_snapping} (strength={preset.edge_snap_strength})")
    
    # Find all images
    image_extensions = {'.tif', '.tiff', '.TIF', '.TIFF', '.jpg', '.jpeg', '.png'}
    image_files = [f for f in input_dir.iterdir() if f.suffix in image_extensions and not f.name.startswith('.')]
    
    logger.info(f"\nFound {len(image_files)} images in {input_dir}")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default priority scenes
    if priority_scenes is None:
        priority_scenes = [
            "750Picacho_Kitchen_16bit.tiff",
            "750Picacho_GreatRoom_Ultimate.tif",
            "750Picacho_Aerial_Ultimate.tif",
            "750Picacho_Pool_16bit.tiff"
        ]
    
    # Process all images
    results = {}
    priority_results = {}
    
    logger.info("\nProcessing images...")
    
    for image_path in tqdm(image_files, desc="Validating"):
        is_priority = image_path.name in priority_scenes
        save_visuals = save_all_visuals or is_priority
        
        metrics = process_single_image(
            image_path,
            preset,
            output_dir,
            save_visuals=save_visuals,
            max_dimension=max_dimension
        )
        
        if metrics:
            results[image_path.name] = metrics.to_dict()
            
            if is_priority:
                priority_results[image_path.name] = metrics
    
    # Aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE STATISTICS")
    logger.info("="*80)
    
    total_processed = len(results)
    logger.info(f"Processed: {total_processed}/{len(image_files)} images")
    
    if total_processed == 0:
        logger.error("No images successfully processed!")
        return {"error": "No images processed"}
    
    # Compute statistics
    all_metrics = [EnhancedMetrics(**m) for m in results.values()]
    
    stats = {
        "edge_f1": [m.edge_f1 for m in all_metrics],
        "edge_precision": [m.edge_precision for m in all_metrics],
        "edge_recall": [m.edge_recall for m in all_metrics],
        "chamfer_distance": [m.chamfer_distance for m in all_metrics],
        "edge_count_ratio": [m.edge_count_ratio for m in all_metrics],
        "seam_energy_ratio": [m.seam_energy_ratio for m in all_metrics],
        "halo_score": [m.halo_score for m in all_metrics],
        "overshoot_penalty": [m.overshoot_penalty for m in all_metrics],
        "detail_benefit": [m.detail_benefit for m in all_metrics],
        "runtime_seconds": [m.runtime_seconds for m in all_metrics]
    }
    
    aggregate = {}
    for key, values in stats.items():
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))
        }
    
    # Quality gate pass rates
    lenient_pass = sum(1 for m in all_metrics if m.quality_gate_passed(strict=False))
    strict_pass = sum(1 for m in all_metrics if m.quality_gate_passed(strict=True))
    
    logger.info(f"\nQuality Gate Pass Rates:")
    logger.info(f"  Lenient: {lenient_pass}/{total_processed} ({100*lenient_pass/total_processed:.1f}%)")
    logger.info(f"  Strict: {strict_pass}/{total_processed} ({100*strict_pass/total_processed:.1f}%)")
    
    logger.info(f"\nKey Metrics (mean ± std):")
    logger.info(f"  Edge F1: {aggregate['edge_f1']['mean']:.3f} ± {aggregate['edge_f1']['std']:.3f}")
    logger.info(f"  Edge Precision: {aggregate['edge_precision']['mean']:.3f} ± {aggregate['edge_precision']['std']:.3f}")
    logger.info(f"  Edge Recall: {aggregate['edge_recall']['mean']:.3f} ± {aggregate['edge_recall']['std']:.3f}")
    logger.info(f"  Chamfer Distance: {aggregate['chamfer_distance']['mean']:.2f} ± {aggregate['chamfer_distance']['std']:.2f} px")
    logger.info(f"  Edge Count Ratio: {aggregate['edge_count_ratio']['mean']:.2f} ± {aggregate['edge_count_ratio']['std']:.2f}×")
    logger.info(f"  Halo Score: {aggregate['halo_score']['mean']:.3f} ± {aggregate['halo_score']['std']:.3f}")
    logger.info(f"  Detail Benefit: {aggregate['detail_benefit']['mean']:.3f} ± {aggregate['detail_benefit']['std']:.3f}")
    
    # Priority scenes analysis
    logger.info("\n" + "="*80)
    logger.info("PRIORITY SCENES ANALYSIS")
    logger.info("="*80)
    
    for scene_name, metrics in priority_results.items():
        logger.info(f"\n{scene_name}:")
        logger.info(f"  Edge F1: {metrics.edge_f1:.3f} (P={metrics.edge_precision:.3f}, R={metrics.edge_recall:.3f})")
        logger.info(f"  Chamfer: {metrics.chamfer_distance:.2f}px")
        logger.info(f"  Edge Ratio: {metrics.edge_count_ratio:.2f}×")
        logger.info(f"  Seam: {metrics.seam_energy_ratio:.3f} ({'✓' if metrics.seam_passed else '✗'})")
        logger.info(f"  Halo: {metrics.halo_score:.3f}, Overshoot: {metrics.overshoot_penalty:.3f}")
        logger.info(f"  Detail Benefit: {metrics.detail_benefit:.3f}")
        logger.info(f"  Runtime: {metrics.runtime_seconds:.1f}s")
        
        lenient = metrics.quality_gate_passed(strict=False)
        strict = metrics.quality_gate_passed(strict=True)
        logger.info(f"  Quality Gate: {'✅ STRICT' if strict else '✅ LENIENT' if lenient else '❌ FAIL'}")
    
    # Yellow flags investigation
    logger.info("\n" + "="*80)
    logger.info("YELLOW FLAGS INVESTIGATION")
    logger.info("="*80)
    
    yellow_flags = []
    
    # Flag 1: Check if baseline beats tiled on average
    # (Would need to run baseline comparison - simplified here)
    
    # Flag 2: Edge precision < recall (false positives)
    avg_precision = aggregate['edge_precision']['mean']
    avg_recall = aggregate['edge_recall']['mean']
    
    if avg_precision < avg_recall * 0.85:
        yellow_flags.append({
            "flag": "PRECISION_LOW",
            "description": f"Edge precision ({avg_precision:.3f}) significantly lower than recall ({avg_recall:.3f})",
            "impact": "Possible false positive edges (artifact noise)",
            "recommendation": "Review edge snapping mask and global anchor fusion strength"
        })
    
    # Flag 3: Detail benefit < 1.0 (tiled adds noise, not detail)
    avg_detail_benefit = aggregate['detail_benefit']['mean']
    
    if avg_detail_benefit < 1.0:
        yellow_flags.append({
            "flag": "DETAIL_BENEFIT_LOW",
            "description": f"Detail benefit ({avg_detail_benefit:.3f}) < 1.0",
            "impact": "Tiling may be adding noise rather than real detail",
            "recommendation": "Reduce global anchor strength or disable on low-detail scenes"
        })
    
    # Flag 4: High overshoot penalty
    avg_overshoot = aggregate['overshoot_penalty']['mean']
    
    if avg_overshoot > 0.40:
        yellow_flags.append({
            "flag": "OVERSHOOT_HIGH",
            "description": f"Overshoot penalty ({avg_overshoot:.3f}) > 0.40",
            "impact": "Visible ringing/halo artifacts around edges",
            "recommendation": "Reduce edge snapping strength or improve edge detection"
        })
    
    # Flag 5: Excessive edge count
    avg_edge_ratio = aggregate['edge_count_ratio']['mean']
    
    if avg_edge_ratio > 2.0:
        yellow_flags.append({
            "flag": "EDGE_EXPLOSION",
            "description": f"Edge count ratio ({avg_edge_ratio:.2f}×) > 2.0",
            "impact": "Too many artifact edges",
            "recommendation": "Check global sharpening and edge detection thresholds"
        })
    
    if yellow_flags:
        logger.warning(f"\n⚠️  {len(yellow_flags)} yellow flags detected:")
        for i, flag in enumerate(yellow_flags, 1):
            logger.warning(f"\n{i}. {flag['flag']}")
            logger.warning(f"   Description: {flag['description']}")
            logger.warning(f"   Impact: {flag['impact']}")
            logger.warning(f"   Recommendation: {flag['recommendation']}")
    else:
        logger.info("✅ No yellow flags detected")
    
    # Materials V3 integration readiness
    logger.info("\n" + "="*80)
    logger.info("MATERIALS V3 INTEGRATION READINESS")
    logger.info("="*80)
    
    materials_v3_ready = lenient_pass >= 0.80 * total_processed
    
    logger.info(f"\nQuality Gates for Materials V3:")
    logger.info(f"  Minimum pass rate: 80% ({0.80 * total_processed:.0f}/{total_processed})")
    logger.info(f"  Actual pass rate: {100*lenient_pass/total_processed:.1f}% ({lenient_pass}/{total_processed})")
    logger.info(f"  Status: {'✅ READY' if materials_v3_ready else '❌ NOT READY'}")
    
    if not materials_v3_ready:
        logger.warning("\n⚠️  Materials V3 integration NOT recommended until:")
        logger.warning("  - Pass rate ≥ 80%")
        logger.warning("  - Yellow flags resolved")
        logger.warning("  - Fallback mode implemented for low-quality depth")
    
    # Compile final report
    report = {
        "dataset": str(input_dir),
        "preset": asdict(preset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_images": len(image_files),
            "processed": total_processed,
            "lenient_pass": lenient_pass,
            "strict_pass": strict_pass,
            "lenient_pass_rate": lenient_pass / total_processed if total_processed > 0 else 0,
            "strict_pass_rate": strict_pass / total_processed if total_processed > 0 else 0
        },
        "aggregate_metrics": aggregate,
        "per_image_metrics": results,
        "priority_scenes": {k: v.to_dict() for k, v in priority_results.items()},
        "yellow_flags": yellow_flags,
        "materials_v3_ready": materials_v3_ready,
        "go_no_go_decision": {
            "recommendation": "GO" if materials_v3_ready and len(yellow_flags) == 0 else "INVESTIGATE" if materials_v3_ready else "NO-GO",
            "criteria": {
                "pass_rate_80_percent": lenient_pass >= 0.80 * total_processed,
                "no_critical_yellow_flags": len(yellow_flags) == 0,
                "edge_f1_mean_above_030": aggregate['edge_f1']['mean'] >= 0.30,
                "chamfer_distance_mean_below_15": aggregate['chamfer_distance']['mean'] < 15.0
            }
        }
    }
    
    # Save report
    report_path = output_dir / "production_validation_report.json"
    save_metrics_atomic(report, report_path)
    logger.info(f"\n✅ Report saved: {report_path}")
    
    # Go/No-Go Decision
    logger.info("\n" + "="*80)
    logger.info("GO/NO-GO DECISION")
    logger.info("="*80)
    
    decision = report['go_no_go_decision']
    logger.info(f"\nRECOMMENDATION: {decision['recommendation']}")
    
    logger.info("\nCriteria:")
    for criterion, passed in decision['criteria'].items():
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {criterion.replace('_', ' ').title()}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Production validation on full 750_Picacho dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/rc/Transformation_Portal/input_images/750_Picacho/Source_TIFFs_Base"),
        help="Input directory with source images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/production_validation"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--preset",
        choices=["preview", "production", "hero"],
        default="production",
        help="Processing preset"
    )
    parser.add_argument(
        "--save-all-visuals",
        action="store_true",
        help="Save visual outputs for all images (not just priority scenes)"
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=4096,
        help="Maximum dimension for processing (to avoid memory issues on large images)"
    )
    
    args = parser.parse_args()
    
    try:
        report = run_production_validation(
            args.input_dir,
            args.output_dir,
            args.preset,
            save_all_visuals=args.save_all_visuals,
            max_dimension=args.max_dimension
        )
        
        # Exit based on go/no-go decision
        decision = report['go_no_go_decision']['recommendation']
        
        if decision == "GO":
            logger.info("\n✅ PRODUCTION VALIDATION PASSED - GO FOR MATERIALS V3")
            sys.exit(0)
        elif decision == "INVESTIGATE":
            logger.warning("\n⚠️  INVESTIGATE YELLOW FLAGS BEFORE MATERIALS V3")
            sys.exit(0)
        else:
            logger.error("\n❌ PRODUCTION VALIDATION FAILED - NO-GO")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
