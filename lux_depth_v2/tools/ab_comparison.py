#!/usr/bin/env python3
"""
A/B Comparison: Baseline vs Enhanced Tiled Depth Pipeline
=========================================================

Compares:
- Baseline: HF pipeline (518px resize, no tiling)
- Enhanced: Tiled inference + global anchor + edge snapping

Metrics:
- Edge alignment score (correlation with RGB edges)
- Depth discontinuity sharpness
- Processing time
- Visual quality (side-by-side comparison)

Reference: User feedback 2025-12-18
"Test it empirically and report the measured improvement, not the expected one."
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - some metrics will be disabled")


@dataclass
class ComparisonResult:
    """Results from A/B comparison."""
    
    # Baseline metrics
    baseline_edge_alignment: float
    baseline_edge_sharpness: float
    baseline_time_ms: float
    
    # Enhanced metrics
    enhanced_edge_alignment: float
    enhanced_edge_sharpness: float
    enhanced_time_ms: float
    
    # Improvements
    edge_alignment_improvement: float
    edge_sharpness_improvement: float
    time_overhead_factor: float
    
    def __str__(self):
        return f"""
A/B Comparison Results
======================

Baseline (HF Pipeline, 518px):
  Edge alignment:  {self.baseline_edge_alignment:.3f}
  Edge sharpness:  {self.baseline_edge_sharpness:.3f}
  Processing time: {self.baseline_time_ms:.1f}ms

Enhanced (Tiled + Global + Snapping):
  Edge alignment:  {self.enhanced_edge_alignment:.3f} ({self.edge_alignment_improvement:+.1%})
  Edge sharpness:  {self.enhanced_edge_sharpness:.3f} ({self.edge_sharpness_improvement:+.1%})
  Processing time: {self.enhanced_time_ms:.1f}ms ({self.time_overhead_factor:.1f}x)

Verdict: {'IMPROVEMENT' if self.edge_alignment_improvement > 0 else 'NO IMPROVEMENT'}
"""


def create_synthetic_test_pattern(size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """
    Create synthetic test pattern with known depth structure.
    
    Pattern:
    - Horizontal gradient (simulates depth ramp)
    - Vertical stripes (sharp edges every 256px)
    - Circular regions (object boundaries)
    """
    h, w = size
    pattern = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background: horizontal gradient
    for i in range(h):
        intensity = int(128 + 127 * np.sin(np.pi * i / h))
        pattern[i, :] = intensity
    
    # Vertical stripes (sharp edges)
    for j in range(0, w, 256):
        pattern[:, j:j+32] = 255
    
    # Circular regions (object boundaries)
    center_y, center_x = h // 2, w // 2
    for radius in [200, 400, 600]:
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
        pattern[mask] = [200, 150, 100]
    
    logger.info(f"Created synthetic test pattern: {pattern.shape}")
    return pattern


def compute_edge_alignment(rgb: np.ndarray, depth: np.ndarray) -> float:
    """
    Compute edge alignment: correlation between RGB edges and depth edges.
    
    This is the primary quality metric for depth maps.
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV required for edge alignment metric")
        return 0.0
    
    # RGB edges
    if rgb.dtype == np.float32:
        rgb_uint8 = (rgb * 255).astype(np.uint8)
    else:
        rgb_uint8 = rgb
    
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY) if rgb_uint8.ndim == 3 else rgb_uint8
    rgb_edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    
    # Depth edges
    if depth.dtype != np.uint8:
        depth_uint8 = (depth * 255).astype(np.uint8)
    else:
        depth_uint8 = depth
    
    sobel_x = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
    depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    depth_edges = depth_edges / (depth_edges.max() + 1e-8)
    
    # Correlation
    correlation = np.corrcoef(rgb_edges.ravel(), depth_edges.ravel())[0, 1]
    
    return correlation


def compute_edge_sharpness(depth: np.ndarray) -> float:
    """
    Compute average edge sharpness (gradient magnitude at edges).
    
    Higher is sharper (luxury-grade DOF/masking requires sharp edges).
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV required for edge sharpness metric")
        return 0.0
    
    if depth.dtype != np.uint8:
        depth_uint8 = (depth * 255).astype(np.uint8)
    else:
        depth_uint8 = depth
    
    # Sobel gradients
    sobel_x = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Average gradient at top 10% strongest edges
    threshold = np.percentile(gradient_magnitude, 90)
    edge_gradients = gradient_magnitude[gradient_magnitude >= threshold]
    
    sharpness = edge_gradients.mean() if len(edge_gradients) > 0 else 0.0
    
    return sharpness


def run_baseline_inference(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run baseline inference (HF pipeline, 518px resize)."""
    try:
        from transformers import pipeline
    except ImportError:
        logger.error("transformers required")
        return None, 0.0
    
    logger.info("Running baseline inference (HF pipeline)...")
    
    # Create pipeline
    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=-1  # CPU
    )
    
    # Convert to PIL
    if rgb.dtype == np.float32:
        rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        rgb_pil = Image.fromarray(rgb)
    
    # Inference with timing
    start = time.time()
    result = pipe(rgb_pil)
    elapsed_ms = (time.time() - start) * 1000
    
    # Extract depth (handle dict or object)
    if isinstance(result, dict):
        # HF pipeline returns dict with 'predicted_depth' or 'depth' key
        if 'predicted_depth' in result:
            depth = np.array(result['predicted_depth'])
        elif 'depth' in result:
            depth = np.array(result['depth'])
        else:
            raise ValueError(f"Unknown result format: {result.keys()}")
    elif hasattr(result, "depth"):
        depth = np.array(result.depth)
    elif hasattr(result, "predicted_depth"):
        depth = np.array(result.predicted_depth)
    else:
        depth = np.array(result)
    
    # Normalize
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    
    logger.info(f"✓ Baseline inference: {elapsed_ms:.1f}ms, output shape: {depth.shape}")
    
    return depth, elapsed_ms


def run_enhanced_inference(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run enhanced inference (tiled + global + snapping)."""
    from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
    from lux_depth_v2.global_anchor import GlobalAnchorConfig
    from lux_depth_v2.edge_snapping import EdgeSnappingConfig
    
    logger.info("Running enhanced inference (tiled + global + snapping)...")
    
    # Configure with all enhancements
    # CRITICAL FIX: Disable standalone edge snapping to avoid double application
    # (production refinement already includes edge snapping)
    config = TiledInferenceConfig(
        tile_size=1024,
        overlap=128,
        bypass_image_processor=True,  # CRITICAL: No 518px resize
        use_global_anchor=True,
        global_anchor_config=GlobalAnchorConfig(),
        use_edge_snapping=False,  # Disabled to avoid double application
        use_production_refinement=True,  # Includes edge snapping
        refinement_use_edge_snap=True
    )
    
    estimator = TiledDepthEstimator(config)
    
    # Inference with timing
    start = time.time()
    depth = estimator.estimate_depth(rgb)
    elapsed_ms = (time.time() - start) * 1000
    
    logger.info(f"✓ Enhanced inference: {elapsed_ms:.1f}ms, output shape: {depth.shape}")
    
    return depth, elapsed_ms


def save_comparison_visualization(
    rgb: np.ndarray,
    baseline_depth: np.ndarray,
    enhanced_depth: np.ndarray,
    output_path: Path
):
    """Save side-by-side comparison visualization."""
    if not CV2_AVAILABLE:
        logger.warning("OpenCV required for visualization")
        return
    
    # Resize for display
    display_height = 512
    h, w = rgb.shape[:2]
    scale = display_height / h
    display_width = int(w * scale)
    
    rgb_small = cv2.resize(rgb, (display_width, display_height))
    baseline_small = cv2.resize(baseline_depth, (display_width, display_height))
    enhanced_small = cv2.resize(enhanced_depth, (display_width, display_height))
    
    # Convert depth to colormap
    baseline_color = cv2.applyColorMap((baseline_small * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    enhanced_color = cv2.applyColorMap((enhanced_small * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # Stack horizontally
    comparison = np.hstack([
        cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR),
        baseline_color,
        enhanced_color
    ])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "RGB", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Baseline (518px)", (display_width + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Enhanced (Tiled)", (display_width * 2 + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)
    logger.info(f"✓ Comparison saved: {output_path}")


def run_ab_comparison(
    rgb: Optional[np.ndarray] = None,
    output_dir: Path = Path("lux_depth_v2/ab_comparison_results")
) -> ComparisonResult:
    """
    Run A/B comparison between baseline and enhanced pipelines.
    
    Args:
        rgb: Test image (or None to use synthetic pattern)
        output_dir: Directory for outputs
        
    Returns:
        Comparison results with metrics
    """
    logger.info("=" * 60)
    logger.info("A/B COMPARISON: Baseline vs Enhanced")
    logger.info("=" * 60)
    
    # Create test image if needed
    if rgb is None:
        logger.info("No test image provided, creating synthetic pattern...")
        rgb = create_synthetic_test_pattern()
    
    # Run baseline
    baseline_depth, baseline_time = run_baseline_inference(rgb)
    baseline_edge_align = compute_edge_alignment(rgb, baseline_depth)
    baseline_edge_sharp = compute_edge_sharpness(baseline_depth)
    
    # Run enhanced
    enhanced_depth, enhanced_time = run_enhanced_inference(rgb)
    enhanced_edge_align = compute_edge_alignment(rgb, enhanced_depth)
    enhanced_edge_sharp = compute_edge_sharpness(enhanced_depth)
    
    # Compute improvements
    edge_align_improvement = (enhanced_edge_align - baseline_edge_align) / max(baseline_edge_align, 1e-8)
    edge_sharp_improvement = (enhanced_edge_sharp - baseline_edge_sharp) / max(baseline_edge_sharp, 1e-8)
    time_overhead = enhanced_time / max(baseline_time, 1e-8)
    
    # Create result
    result = ComparisonResult(
        baseline_edge_alignment=baseline_edge_align,
        baseline_edge_sharpness=baseline_edge_sharp,
        baseline_time_ms=baseline_time,
        enhanced_edge_alignment=enhanced_edge_align,
        enhanced_edge_sharpness=enhanced_edge_sharp,
        enhanced_time_ms=enhanced_time,
        edge_alignment_improvement=edge_align_improvement,
        edge_sharpness_improvement=edge_sharp_improvement,
        time_overhead_factor=time_overhead
    )
    
    # Save visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_visualization(
        rgb, baseline_depth, enhanced_depth,
        output_dir / "comparison.png"
    )
    
    # Save depth maps
    Image.fromarray((baseline_depth * 255).astype(np.uint8)).save(output_dir / "baseline_depth.png")
    Image.fromarray((enhanced_depth * 255).astype(np.uint8)).save(output_dir / "enhanced_depth.png")
    
    # Save report
    with open(output_dir / "report.txt", 'w') as f:
        f.write(str(result))
    
    logger.info(str(result))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A/B comparison of depth pipelines")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input RGB image (or None for synthetic pattern)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lux_depth_v2/ab_comparison_results"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load input if provided
    rgb = None
    if args.input and args.input.exists():
        logger.info(f"Loading test image: {args.input}")
        rgb = np.array(Image.open(args.input))
    
    # Run comparison
    result = run_ab_comparison(rgb, args.output_dir)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
