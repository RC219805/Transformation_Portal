#!/usr/bin/env python3
"""
Generate diagnostic overlays for edge detection failures.
========================================================

Creates RGB edges | Depth edges | Confusion overlay visualization
to diagnose whether edge metric is measuring the right thing.

Usage:
    python scripts/validation/visualize_edge_failures.py \
        --validation-dir outputs/validation_sliver_quick_20251218_122536 \
        --output-dir outputs/edge_diagnostics/
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_depth_16bit(path: Path) -> np.ndarray:
    """Load 16-bit depth TIFF and normalize to [0, 1]."""
    depth_u16 = np.array(Image.open(path))
    return depth_u16.astype(np.float32) / 65535.0


def detect_edges_canny(image: np.ndarray, threshold_low: float = 50, threshold_high: float = 150) -> np.ndarray:
    """Detect edges using Canny edge detector."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # For float depth: use gradient-based detection
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize and threshold based on percentiles
        p_low = np.percentile(grad_mag, threshold_low)
        p_high = np.percentile(grad_mag, threshold_high)

        edges = ((grad_mag >= p_low) & (grad_mag <= p_high * 10)).astype(np.uint8) * 255
        return edges
    else:
        # For uint8: use standard Canny
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, int(threshold_low), int(threshold_high))


def classify_edge_type(rgb: np.ndarray, rgb_edges: np.ndarray) -> str:
    """
    Classify whether RGB edges are texture-dominated or structure-dominated.

    Returns:
        "texture" or "structure"
    """
    # Convert to grayscale if needed
    if len(rgb.shape) == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb

    # Compute texture variance
    # High variance in non-edge regions → texture-dominated
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)

    # Get variance in edge vs non-edge regions
    edge_mask = rgb_edges > 0
    non_edge_mask = ~edge_mask

    if non_edge_mask.sum() == 0:
        return "structure"  # All edges → likely architectural

    non_edge_variance = local_var[non_edge_mask].mean()
    edge_variance = local_var[edge_mask].mean() if edge_mask.sum() > 0 else 0

    # If non-edge variance is high → texture-dominated (e.g., water ripples)
    texture_ratio = non_edge_variance / (edge_variance + 1e-6)

    if texture_ratio > 0.5:
        return "texture"
    else:
        return "structure"


def visualize_edge_comparison(rgb_path: Path, depth_path: Path, metrics_path: Path, output_path: Path) -> dict:
    """
    Create RGB edges | Depth edges | Overlay visualization.

    Returns:
        Dict with classification and statistics
    """
    # Load inputs
    rgb = np.array(Image.open(rgb_path))
    depth = load_depth_16bit(depth_path)

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Extract edges
    rgb_edges = detect_edges_canny(rgb, threshold_low=50, threshold_high=150)

    # Normalize depth for edge detection
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    depth_edges = cv2.Canny(depth_norm, 50, 150)

    # Classify edge type
    edge_classification = classify_edge_type(rgb, rgb_edges)

    # Create confusion overlay
    # Green = TP (both), Red = FP (depth only), Blue = FN (RGB only), Gray = background
    h, w = rgb.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Background = dimmed RGB
    overlay = (rgb * 0.3).astype(np.uint8)

    # True positives (both detect edge)
    tp_mask = (rgb_edges > 0) & (depth_edges > 0)
    overlay[tp_mask] = [0, 255, 0]  # Green

    # False positives (depth only, no RGB edge)
    fp_mask = (depth_edges > 0) & (rgb_edges == 0)
    overlay[fp_mask] = [255, 0, 0]  # Red

    # False negatives (RGB only, no depth edge)
    fn_mask = (rgb_edges > 0) & (depth_edges == 0)
    overlay[fn_mask] = [0, 0, 255]  # Blue

    # Compute statistics
    stats = {
        "tp_pixels": int(tp_mask.sum()),
        "fp_pixels": int(fp_mask.sum()),
        "fn_pixels": int(fn_mask.sum()),
        "tp_ratio": float(tp_mask.sum() / max(rgb_edges.sum() / 255, 1)),
        "fp_ratio": float(fp_mask.sum() / max(depth_edges.sum() / 255, 1)),
        "edge_classification": edge_classification,
    }

    # Composite visualization: RGB edges | Depth edges | Overlay
    rgb_edges_rgb = cv2.cvtColor(rgb_edges, cv2.COLOR_GRAY2BGR)
    depth_edges_rgb = cv2.cvtColor(depth_edges, cv2.COLOR_GRAY2BGR)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb_edges_rgb, "RGB Edges", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(depth_edges_rgb, "Depth Edges", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(overlay, "Confusion (G=TP, R=FP, B=FN)", (10, 30), font, 0.8, (255, 255, 255), 2)

    # Add metrics
    metrics_text = [
        f"Edge F1: {metrics.get('metrics', metrics).get('edge_f1', 0):.3f}",
        f"Chamfer: {metrics.get('metrics', metrics).get('chamfer_distance', 0):.1f}px",
        f"Type: {edge_classification.upper()}",
    ]

    y_offset = 70
    for text in metrics_text:
        cv2.putText(overlay, text, (10, y_offset), font, 0.7, (255, 255, 0), 2)
        y_offset += 35

    # Stack horizontally (may need to resize for large images)
    max_width = 2400  # Limit visualization width
    if w > max_width // 3:
        scale = (max_width // 3) / w
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_edges_rgb = cv2.resize(rgb_edges_rgb, (new_w, new_h))
        depth_edges_rgb = cv2.resize(depth_edges_rgb, (new_w, new_h))
        overlay = cv2.resize(overlay, (new_w, new_h))

    vis = np.hstack([rgb_edges_rgb, depth_edges_rgb, overlay])

    # Save
    Image.fromarray(vis).save(output_path)
    logger.info(f"✓ Saved: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Visualize edge detection failures")
    parser.add_argument("--validation-dir", type=Path, required=True, help="Validation output directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for diagnostics",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all metrics files
    metrics_files = sorted(args.validation_dir.glob("*_metrics.json"))

    if not metrics_files:
        logger.error(f"No metrics files found in {args.validation_dir}")
        return 1

    logger.info(f"Found {len(metrics_files)} images to visualize")

    # Process each image
    all_stats = {}

    for metrics_path in metrics_files:
        image_name = metrics_path.stem.replace("_metrics", "")

        # Find corresponding RGB and depth files
        rgb_candidates = [
            args.validation_dir.parent / "validation_quick" / f"{image_name}.jpg",
            args.validation_dir.parent / "validation_quick" / f"{image_name}.jpeg",
            args.validation_dir.parent / "validation_quick" / f"{image_name}.png",
            Path("data/validation_quick") / f"{image_name}.jpg",
        ]

        rgb_path = next((p for p in rgb_candidates if p.exists()), None)
        depth_path = args.validation_dir / f"{image_name}_depth.tiff"

        if not rgb_path or not rgb_path.exists():
            logger.warning(f"⚠️  RGB not found for {image_name}, skipping")
            continue

        if not depth_path.exists():
            logger.warning(f"⚠️  Depth not found for {image_name}, skipping")
            continue

        output_path = args.output_dir / f"{image_name}_edge_diagnostic.png"

        try:
            stats = visualize_edge_comparison(rgb_path, depth_path, metrics_path, output_path)
            all_stats[image_name] = stats
        except Exception as e:
            logger.error(f"✗ Failed to process {image_name}: {e}")

    # Generate summary report
    summary_path = args.output_dir / "edge_diagnostic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\n✓ Summary saved: {summary_path}")

    # Print classification summary
    texture_count = sum(1 for s in all_stats.values() if s["edge_classification"] == "texture")
    structure_count = sum(1 for s in all_stats.values() if s["edge_classification"] == "structure")

    print("\n" + "=" * 80)
    print(" EDGE CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"Texture-dominated:  {texture_count}/{len(all_stats)}")
    print(f"Structure-dominated: {structure_count}/{len(all_stats)}")
    print("\nNext steps:")
    if texture_count > structure_count:
        print("  → Metric is misaligned: suppress texture edges in RGB before comparison")
        print("  → Add Gaussian blur to RGB before edge detection to focus on structure")
    else:
        print("  → Metric seems aligned: depth genuinely missing structural edges")
        print("  → Consider increasing inference resolution or preprocessing small images")

    return 0


if __name__ == "__main__":
    exit(main())
