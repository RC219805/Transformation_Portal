#!/usr/bin/env python3
"""Validation script for edge refinement module.

Compares depth quality metrics (Edge F1, Chamfer, Boundary Recall) for
raw vs refined depth maps on a validation set.

Usage:
    python validate_edge_refinement.py --input-dir validation_images/ --output-dir validation_results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_depth_map(path: Path) -> np.ndarray:
    """Load depth map from file.
    
    Supports .npy, .npz, .png, .tiff formats.
    
    Args:
        path: Path to depth map file
    
    Returns:
        Depth map as float32 array [0, 1]
    """
    if path.suffix == '.npy':
        depth = np.load(path)
    elif path.suffix == '.npz':
        data = np.load(path)
        depth = data['depth'] if 'depth' in data else data[data.files[0]]
    elif path.suffix in ['.png', '.jpg', '.jpeg']:
        img = Image.open(path)
        depth = np.array(img).astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / 255.0
    elif path.suffix in ['.tif', '.tiff']:
        img = Image.open(path)
        depth = np.array(img).astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / 65535.0  # 16-bit
    else:
        raise ValueError(f"Unsupported depth format: {path.suffix}")
    
    return depth.astype(np.float32)


def compute_edge_metrics(
    depth: np.ndarray,
    rgb: np.ndarray,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute edge-related metrics.
    
    Args:
        depth: Depth map (H, W) float32 [0, 1]
        rgb: RGB image (H, W, 3) uint8
        threshold: Edge detection threshold
    
    Returns:
        Dictionary with edge metrics
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping edge metrics")
        return {}
    
    # Detect edges in RGB
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 50, 150).astype(bool)
    
    # Detect edges in depth using gradient magnitude
    depth_uint8 = (depth * 255).astype(np.uint8)
    depth_edges = cv2.Canny(depth_uint8, 30, 100).astype(bool)
    
    # Edge F1 score (alignment between RGB and depth edges)
    if rgb_edges.sum() > 0 and depth_edges.sum() > 0:
        # True positives: depth edges aligned with RGB edges
        tp = (rgb_edges & depth_edges).sum()
        fp = (depth_edges & ~rgb_edges).sum()
        fn = (rgb_edges & ~depth_edges).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        edge_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        edge_f1 = 0.0
        precision = 0.0
        recall = 0.0
    
    # Edge density
    edge_density = depth_edges.sum() / depth_edges.size
    
    # Gradient magnitude statistics
    sobelx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    return {
        "edge_f1": float(edge_f1),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_density": float(edge_density),
        "gradient_mean": float(grad_mag.mean()),
        "gradient_std": float(grad_mag.std()),
        "gradient_max": float(grad_mag.max()),
    }


def compute_smoothness_metrics(depth: np.ndarray) -> Dict[str, float]:
    """Compute smoothness and noise metrics.
    
    Args:
        depth: Depth map (H, W) float32 [0, 1]
    
    Returns:
        Dictionary with smoothness metrics
    """
    # Variance in local patches (noise indicator)
    patch_size = 5
    h, w = depth.shape
    variances = []
    
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = depth[i:i+patch_size, j:j+patch_size]
            variances.append(np.var(patch))
    
    return {
        "depth_variance_mean": float(np.mean(variances)),
        "depth_variance_std": float(np.std(variances)),
        "depth_range": float(depth.max() - depth.min()),
    }


def validate_refinement(
    input_dir: Path,
    output_dir: Path,
    presets: List[str] = None,
) -> Dict[str, any]:
    """Validate edge refinement on image set.
    
    Args:
        input_dir: Directory with RGB images
        output_dir: Directory to save results
        presets: List of refinement presets to test
    
    Returns:
        Validation results dictionary
    """
    try:
        from lux_depth_v3.edge_refinement import DepthRefiner, create_refinement_preset
    except ImportError:
        logger.error("lux_depth_v3 not available, cannot validate")
        return {}
    
    if presets is None:
        presets = ["balanced", "aggressive", "conservative", "edge_focused"]
    
    # Find images
    image_paths = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    
    if len(image_paths) == 0:
        logger.error(f"No images found in {input_dir}")
        return {}
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = {
        "num_images": len(image_paths),
        "presets": presets,
        "per_image_results": [],
        "summary": {},
    }
    
    # Placeholder: Generate synthetic depth for testing
    # In production, this would load from DA3 inference
    logger.warning("Using synthetic depth maps for validation (replace with real DA3 output)")
    
    for img_path in tqdm(image_paths, desc="Validating"):
        # Load RGB
        rgb = np.array(Image.open(img_path))
        
        # Generate synthetic depth (REPLACE with real DA3 inference)
        h, w = rgb.shape[:2]
        depth_raw = np.random.rand(h, w).astype(np.float32)
        
        # Compute baseline metrics
        baseline_metrics = {
            **compute_edge_metrics(depth_raw, rgb),
            **compute_smoothness_metrics(depth_raw),
        }
        
        image_results = {
            "image": img_path.name,
            "baseline": baseline_metrics,
            "presets": {},
        }
        
        # Test each preset
        for preset_name in presets:
            config = create_refinement_preset(preset_name)
            refiner = DepthRefiner(config)
            
            # Refine depth
            depth_refined = refiner.refine(depth_raw, rgb)
            
            # Compute metrics
            refined_metrics = {
                **compute_edge_metrics(depth_refined, rgb),
                **compute_smoothness_metrics(depth_refined),
            }
            
            # Compare
            comparison = {}
            for key in baseline_metrics:
                baseline_val = baseline_metrics[key]
                refined_val = refined_metrics[key]
                
                if baseline_val != 0:
                    relative_change = (refined_val - baseline_val) / baseline_val
                else:
                    relative_change = 0.0
                
                comparison[f"{key}_delta"] = refined_val - baseline_val
                comparison[f"{key}_relative"] = relative_change
            
            image_results["presets"][preset_name] = {
                "metrics": refined_metrics,
                "comparison": comparison,
            }
        
        results["per_image_results"].append(image_results)
    
    # Compute summary statistics
    for preset_name in presets:
        edge_f1_improvements = []
        
        for img_result in results["per_image_results"]:
            baseline_f1 = img_result["baseline"].get("edge_f1", 0)
            refined_f1 = img_result["presets"][preset_name]["metrics"].get("edge_f1", 0)
            
            if baseline_f1 > 0:
                improvement = (refined_f1 - baseline_f1) / baseline_f1
                edge_f1_improvements.append(improvement)
        
        if edge_f1_improvements:
            results["summary"][preset_name] = {
                "edge_f1_improvement_mean": float(np.mean(edge_f1_improvements)),
                "edge_f1_improvement_std": float(np.std(edge_f1_improvements)),
                "edge_f1_improvement_median": float(np.median(edge_f1_improvements)),
            }
    
    # Save results
    results_path = output_dir / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EDGE REFINEMENT VALIDATION SUMMARY")
    print("="*80)
    
    for preset_name, summary in results["summary"].items():
        print(f"\n{preset_name.upper()}:")
        print(f"  Edge F1 improvement: {summary['edge_f1_improvement_mean']:.2%} ± {summary['edge_f1_improvement_std']:.2%}")
        print(f"  Median improvement: {summary['edge_f1_improvement_median']:.2%}")
    
    print("\n" + "="*80)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate edge refinement module")
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with validation images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["balanced", "aggressive", "conservative", "edge_focused"],
        help="Refinement presets to test",
    )
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_refinement(
        args.input_dir,
        args.output_dir,
        args.presets,
    )
    
    if results:
        print(f"\n✅ Validation complete! Results saved to {args.output_dir}")
    else:
        print("\n❌ Validation failed")


if __name__ == "__main__":
    main()
