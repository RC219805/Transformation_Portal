#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate material assignment visualization for MBAR (Material-Based Aerial Renderer).

This script creates a color-coded visualization showing which materials are assigned
to different regions of an aerial image based on k-means clustering.

Usage:
    python scripts/utilities/visualize_material_assignments.py INPUT_IMAGE [OPTIONS]

Examples:
    # Basic usage
    python scripts/utilities/visualize_material_assignments.py aerial.jpg

    # Load existing palette
    python scripts/utilities/visualize_material_assignments.py aerial.jpg --palette palette.json

    # Save palette for reuse
    python scripts/utilities/visualize_material_assignments.py aerial.jpg --save-palette my_palette.json

    # Custom output path
    python scripts/utilities/visualize_material_assignments.py aerial.jpg --output result.jpg

Author: Transformation Portal Team
License: Attribution (see LICENSE)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from transformation_portal.enhancers.board_material_aerial_enhancer import (
        DEFAULT_TEXTURES,
        auto_assign_materials_by_stats,
        build_material_rules,
        compute_cluster_stats,
        load_palette_assignments,
        save_palette_assignments,
    )
except ImportError as exc:
    print(f"Error: transformation_portal enhancer module not found: {exc}", file=sys.stderr)
    print("This script requires the Transformation Portal repository", file=sys.stderr)
    sys.exit(1)


# Default color palette for visualization (8 distinct colors)
DEFAULT_COLORS = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (100, 100, 255),  # Blue
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Cyan
    (255, 200, 100),  # Orange
    (200, 100, 255),  # Purple
]


def perform_clustering(image: Image.Image, n_clusters: int = 8) -> np.ndarray:
    """
    Perform k-means clustering on image to identify material regions.

    Args:
        image: Input PIL Image
        n_clusters: Number of clusters (default: 8)

    Returns:
        numpy array of cluster labels (same size as image)
    """
    from sklearn.cluster import KMeans

    # Convert to array and normalize
    img_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # Downsample for faster clustering if image is large
    max_dim = 1280
    if max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        analysis_image = image.resize(new_size, Image.Resampling.LANCZOS)
        analysis_array = np.asarray(analysis_image, dtype=np.float32) / 255.0
    else:
        analysis_array = img_array
        analysis_image = image

    # Reshape for k-means
    pixels = analysis_array.reshape(-1, 3)

    # Sample pixels if too many (for performance)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    sample_size = min(len(pixels), 200_000)
    if sample_size < len(pixels):
        indices = rng.choice(len(pixels), size=sample_size, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(sample)

    # Assign all pixels to clusters
    all_labels = kmeans.predict(analysis_array.reshape(-1, 3))
    labels_small = all_labels.reshape(analysis_array.shape[:2])

    # Upsample labels back to original size if needed
    if analysis_image.size != image.size:
        labels_img = Image.fromarray(labels_small.astype("uint8"), mode="L")
        labels_full = labels_img.resize(image.size, Image.Resampling.NEAREST)
        labels = np.asarray(labels_full, dtype=np.uint8)
    else:
        labels = labels_small.astype(np.uint8)

    return labels


def create_visualization(image: Image.Image, labels: np.ndarray, assignments: Dict, colors: List[tuple] = None) -> Image.Image:
    """
    Create color-coded visualization with legend.

    Args:
        image: Original image
        labels: Cluster labels array
        assignments: Dictionary mapping cluster labels to material rules
        colors: Optional custom color palette

    Returns:
        PIL Image with visualization and legend
    """
    if colors is None:
        colors = DEFAULT_COLORS

    # Create color-coded visualization
    viz_array = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in range(labels.max() + 1):
        mask = labels == label
        viz_array[mask] = colors[label % len(colors)]

    viz_img = Image.fromarray(viz_array)

    # Add legend
    legend_height = min(400, len(assignments) * 50 + 100)
    legend_img = Image.new("RGB", (viz_img.width, viz_img.height + legend_height), (255, 255, 255))
    legend_img.paste(viz_img, (0, 0))

    draw = ImageDraw.Draw(legend_img)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw legend
    y_offset = viz_img.height + 20
    x_offset = 40
    box_size = 30

    draw.text((x_offset, y_offset), "MATERIAL ASSIGNMENTS:", fill=(0, 0, 0), font=font)
    y_offset += 40

    for label, rule in sorted(assignments.items(), key=lambda x: x[0]):
        # Draw color box
        draw.rectangle(
            [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
            fill=colors[label % len(colors)],
            outline=(0, 0, 0),
        )

        # Calculate percentage
        cluster_pixels = (labels == label).sum()
        percentage = (cluster_pixels / labels.size) * 100

        # Draw material name and percentage
        text = f"{rule.name.upper()} - Cluster {label} ({percentage:.1f}%)"
        draw.text((x_offset + box_size + 15, y_offset + 5), text, fill=(0, 0, 0), font=font)
        y_offset += 45

    # Add unassigned clusters
    all_labels = set(range(labels.max() + 1))
    unassigned = all_labels - set(assignments.keys())
    if unassigned:
        draw.text((x_offset, y_offset), "UNASSIGNED CLUSTERS:", fill=(128, 128, 128), font=font)
        y_offset += 35
        for label in sorted(unassigned):
            draw.rectangle(
                [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
                fill=colors[label % len(colors)],
                outline=(0, 0, 0),
            )
            cluster_pixels = (labels == label).sum()
            percentage = (cluster_pixels / labels.size) * 100
            text = f"Cluster {label} ({percentage:.1f}% - no material match)"
            draw.text((x_offset + box_size + 15, y_offset + 5), text, fill=(128, 128, 128), font=font)
            y_offset += 45

    return legend_img


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate material assignment visualization for aerial images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].split("Author:")[0].strip(),
    )

    parser.add_argument("input_image", type=Path, help="Input aerial image (TIFF, JPEG, PNG)")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output visualization path (default: input_name + _material_map.jpg)"
    )
    parser.add_argument("--palette", type=Path, help="Load existing material palette from JSON file")
    parser.add_argument("--save-palette", type=Path, help="Save computed material palette to JSON file for reuse")
    parser.add_argument("--clusters", "-k", type=int, default=8, help="Number of k-means clusters (default: 8)")

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate input
    if not args.input_image.exists():
        print(f"Error: Input image not found: {args.input_image}", file=sys.stderr)
        return 1

    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_image.parent / f"{args.input_image.stem}_material_map.jpg"

    print(f"Processing: {args.input_image}")

    # Load image
    try:
        image = Image.open(args.input_image).convert("RGB")
    except Exception as exc:
        print(f"Error loading image: {exc}", file=sys.stderr)
        return 1

    print(f"Image size: {image.width}x{image.height}")

    # Perform clustering
    print(f"Performing k-means clustering (k={args.clusters})...")
    labels = perform_clustering(image, n_clusters=args.clusters)

    # Convert to numpy array for stats
    base_array = np.asarray(image, dtype=np.float32) / 255.0

    # Get material assignments
    rules = build_material_rules(DEFAULT_TEXTURES)

    if args.palette and args.palette.exists():
        print(f"Loading palette from: {args.palette}")
        assignments = load_palette_assignments(args.palette, rules)
    else:
        print("Computing material assignments...")
        stats = compute_cluster_stats(base_array, labels)
        assignments = auto_assign_materials_by_stats(stats, rules)

    # Save palette if requested
    if args.save_palette:
        print(f"Saving palette to: {args.save_palette}")
        save_palette_assignments(assignments, args.save_palette)

    # Create visualization
    print("Creating visualization...")
    viz_img = create_visualization(image, labels, assignments)

    # Save result
    viz_img.save(output_path, quality=95)
    print(f"✅ Material assignment map saved to: {output_path}")

    # Print summary
    print("\nMaterial Assignments:")
    for label, rule in sorted(assignments.items(), key=lambda x: x[0]):
        cluster_pixels = (labels == label).sum()
        percentage = (cluster_pixels / labels.size) * 100
        print(f"  • {rule.name.upper()}: Cluster {label} ({percentage:.1f}% of image)")

    all_labels = set(range(labels.max() + 1))
    unassigned = all_labels - set(assignments.keys())
    if unassigned:
        print(f"\nUnassigned Clusters: {len(unassigned)}")
        for label in sorted(unassigned):
            cluster_pixels = (labels == label).sum()
            percentage = (cluster_pixels / labels.size) * 100
            print(f"  • Cluster {label}: {percentage:.1f}% of image (below threshold)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
