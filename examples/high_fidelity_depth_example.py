#!/usr/bin/env python3
"""
High-Fidelity Depth Pipeline Example
=====================================

Demonstrates the three critical fixes for luxury rendering quality:

1. Tiled high-resolution inference (vs low-res + bicubic)
2. Correct normal map generation (vs uniform purple/blue)
3. Proper quality metrics (vs misleading edge gradients)

Usage:
    python examples/high_fidelity_depth_example.py \
        --input sample.jpg \
        --output-dir output/ \
        --mode all

Modes:
    - tiled: High-resolution tiled depth inference
    - normals: Generate corrected normal maps
    - quality: Comprehensive quality analysis
    - all: Run complete pipeline
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(arr: np.ndarray, path: Path):
    """Save numpy array as image."""
    if arr.dtype == np.float32:
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        # For 16-bit, save as TIFF
        if path.suffix.lower() not in ['.tif', '.tiff']:
            path = path.with_suffix('.tif')
        Image.fromarray(arr).save(path)
        return
    
    Image.fromarray(arr).save(path)
    logger.info(f"Saved: {path}")


def demo_tiled_inference(rgb: np.ndarray, output_dir: Path):
    """Demonstrate tiled high-resolution depth inference."""
    logger.info("=" * 60)
    logger.info("Demo: Tiled High-Resolution Depth Inference")
    logger.info("=" * 60)
    
    try:
        from lux_depth_v2.depth_inference import create_tiled_estimator
    except ImportError as e:
        logger.error(f"Tiled inference requires PyTorch and transformers: {e}")
        return
    
    # Create estimator
    logger.info("Creating tiled depth estimator...")
    estimator = create_tiled_estimator(
        tile_size=1024,
        overlap=128,
        fusion_mode="median",  # Edge-preserving
        device="auto"
    )
    
    # Estimate depth
    logger.info(f"Running tiled inference on {rgb.shape[0]}x{rgb.shape[1]} image...")
    depth = estimator.estimate_depth(rgb)
    
    # Compute edge alignment
    edge_score = estimator.compute_edge_alignment(rgb, depth)
    logger.info(f"Edge alignment score: {edge_score:.3f}")
    
    # Save depth map (16-bit TIFF)
    depth_uint16 = (depth * 65535).astype(np.uint16)
    save_image(depth_uint16, output_dir / "depth_tiled_16bit.tif")
    
    # Save visualization (8-bit PNG)
    depth_viz = (depth * 255).astype(np.uint8)
    depth_viz = np.stack([depth_viz] * 3, axis=-1)  # Grayscale to RGB
    save_image(depth_viz, output_dir / "depth_tiled_viz.png")
    
    logger.info("✓ Tiled inference complete")
    return depth, depth_uint16


def demo_normal_map_generation(depth_uint16: np.ndarray, output_dir: Path):
    """Demonstrate corrected normal map generation."""
    logger.info("=" * 60)
    logger.info("Demo: Corrected Normal Map Generation")
    logger.info("=" * 60)
    
    from lux_depth_v2.normal_map import (
        NormalMapGenerator,
        PRESETS,
        generate_normal_map
    )
    
    # Method 1: Quick generation with preset
    logger.info("Generating normal map with 'architectural' preset...")
    normals = generate_normal_map(depth_uint16, preset="architectural")
    save_image(normals, output_dir / "normals_architectural.png")
    
    # Method 2: Advanced with custom config
    logger.info("Generating normal map with custom config...")
    generator = NormalMapGenerator(PRESETS["pronounced"])
    normals_pronounced = generator.generate(depth_uint16, strength=1.2)
    
    # Validate quality
    metrics = generator.validate_normal_map(normals_pronounced)
    logger.info(f"Normal map quality:")
    logger.info(f"  X std: {metrics['nx_std']:.3f}")
    logger.info(f"  Y std: {metrics['ny_std']:.3f}")
    logger.info(f"  Z mean: {metrics['nz_mean']:.3f}")
    logger.info(f"  Angle median: {metrics['angle_median_deg']:.1f}°")
    
    # Check for issues
    if metrics['nx_std'] < 0.05:
        logger.warning("⚠️  Low X variation - normals may be too flat")
    if metrics['nz_mean'] > 0.95:
        logger.warning("⚠️  High Z mean - normals mostly camera-facing")
    
    save_image(normals_pronounced, output_dir / "normals_pronounced.png")
    
    logger.info("✓ Normal map generation complete")
    return normals


def demo_quality_analysis(rgb: np.ndarray, depth: np.ndarray, depth_uint16: np.ndarray, output_dir: Path):
    """Demonstrate comprehensive quality analysis."""
    logger.info("=" * 60)
    logger.info("Demo: Correct Quality Metrics")
    logger.info("=" * 60)
    
    from lux_depth_v2.quality_metrics import (
        DepthQualityAnalyzer,
        quick_quality_check
    )
    
    # Quick quality check
    logger.info("Running quick quality check...")
    metrics = quick_quality_check(rgb, depth, depth_uint16)
    
    logger.info("\n" + str(metrics))
    
    # Detailed validation for luxury rendering
    logger.info("\nValidating for luxury rendering...")
    analyzer = DepthQualityAnalyzer(
        target_edge_alignment=0.6,
        target_edge_width_px=3.0,
        target_unique_levels=10000
    )
    
    passes, issues = analyzer.validate_for_luxury_rendering(metrics)
    
    if passes:
        logger.info("✅ PASS: Depth map meets luxury rendering quality bar")
    else:
        logger.warning(f"❌ FAIL: Found {len(issues)} quality issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Save quality report
    report_path = output_dir / "quality_report.txt"
    with open(report_path, 'w') as f:
        f.write("High-Fidelity Depth Quality Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(metrics) + "\n\n")
        
        if not passes:
            f.write("Quality Issues:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
    
    logger.info(f"Quality report saved: {report_path}")
    logger.info("✓ Quality analysis complete")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="High-fidelity depth pipeline example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input RGB image"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_highfid"),
        help="Output directory (default: output_highfid/)"
    )
    parser.add_argument(
        "--mode",
        choices=["tiled", "normals", "quality", "all"],
        default="all",
        help="Processing mode (default: all)"
    )
    parser.add_argument(
        "--depth",
        type=Path,
        help="Optional: Pre-computed depth map (for normals/quality only)"
    )
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load input
    logger.info(f"Loading input: {args.input}")
    rgb = load_image(args.input)
    logger.info(f"Image shape: {rgb.shape}")
    
    # Process based on mode
    depth = None
    depth_uint16 = None
    normals = None
    
    if args.mode in ["tiled", "all"]:
        depth, depth_uint16 = demo_tiled_inference(rgb, args.output_dir)
    elif args.depth:
        # Load pre-computed depth
        logger.info(f"Loading depth map: {args.depth}")
        depth_img = Image.open(args.depth)
        if depth_img.mode == 'I;16':
            depth_uint16 = np.array(depth_img)
            depth = depth_uint16.astype(np.float32) / 65535.0
        else:
            depth = np.array(depth_img).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            depth = depth / depth.max()
            depth_uint16 = (depth * 65535).astype(np.uint16)
    
    if args.mode in ["normals", "all"] and depth_uint16 is not None:
        normals = demo_normal_map_generation(depth_uint16, args.output_dir)
    
    if args.mode in ["quality", "all"] and depth is not None:
        metrics = demo_quality_analysis(rgb, depth, depth_uint16, args.output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
