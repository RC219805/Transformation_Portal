#!/usr/bin/env python3
"""
Confidence Scoring Demo for Materials V3 Segmentation

This script demonstrates the new confidence scoring feature that provides
transparency and trust in material classification results.

Features:
- CLIP similarity scores for each detected material
- Area-weighted confidence aggregation
- Confidence-based filtering
- Heuristic fallback with 0.5 confidence marker

Usage:
    python examples/confidence_scoring_demo.py path/to/image.jpg
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Setup logging to see confidence scores in output
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from transformation_portal.lux_depth_v3.segmentation_backend import EfficientSAMBackend


def demo_confidence_scoring(image_path: str):
    """Demonstrate confidence scoring with a real image."""

    print(f"\n{'='*70}")
    print("Materials V3 Confidence Scoring Demo")
    print(f"{'='*70}\n")

    # Load image
    print(f"Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert("RGB"))
    print(f"Image size: {image.shape[:2]}")

    # Initialize backend
    print("\nInitializing EfficientSAM backend...")
    backend = EfficientSAMBackend()
    backend.load(device="cpu")  # Use CPU for demo (or "mps" for Apple Silicon)

    # Run segmentation with confidence scoring
    print("\nRunning segmentation with confidence scoring...")
    results = backend.segment(image)

    print(f"\n{'='*70}")
    print("Segmentation Results with Confidence Scores")
    print(f"{'='*70}\n")

    if not results:
        print("No materials detected.")
        return

    # Display results sorted by confidence (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)  # Sort by confidence

    for material, (mask, confidence) in sorted_results:
        coverage_pct = (mask.sum() / mask.size) * 100

        # Confidence indicator
        if confidence > 0.7:
            indicator = "🟢 HIGH"
        elif confidence > 0.4:
            indicator = "🟡 MEDIUM"
        else:
            indicator = "🔴 LOW"

        print(f"{material:12s} │ {indicator:12s} │ Confidence: {confidence:5.1%} │ Coverage: {coverage_pct:5.1f}%")

    # Demonstrate confidence-based filtering
    print(f"\n{'='*70}")
    print("Confidence Filtering Examples")
    print(f"{'='*70}\n")

    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        filtered = {material: (mask, conf) for material, (mask, conf) in results.items() if conf >= threshold}
        print(
            f"Threshold {threshold:.0%}: {len(filtered)} materials pass ({', '.join(filtered.keys()) if filtered else 'none'})"
        )

    # Show confidence interpretation
    print(f"\n{'='*70}")
    print("Confidence Score Interpretation")
    print(f"{'='*70}\n")
    print("• 0.8-1.0: Very high confidence (strong CLIP similarity)")
    print("• 0.6-0.8: High confidence (good CLIP match)")
    print("• 0.4-0.6: Medium confidence (moderate match or heuristic)")
    print("• 0.2-0.4: Low confidence (weak match, consider filtering)")
    print("• 0.5:     Heuristic fallback (color-based, not ML)")

    print(f"\n{'='*70}\n")


def demo_with_synthetic_image():
    """Demo with a synthetic test image (no file needed)."""

    print(f"\n{'='*70}")
    print("Synthetic Image Demo (No File Needed)")
    print(f"{'='*70}\n")

    # Create synthetic image with distinct material colors
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Blue region (water/glass)
    image[10:100, 10:100] = [50, 120, 200]

    # Green region (foliage)
    image[150:240, 10:100] = [80, 150, 90]

    # Gray region (stone)
    image[10:100, 150:240] = [120, 125, 120]

    print("Created synthetic test image with 3 material regions")

    # Run segmentation
    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    print("\nRunning segmentation...")
    results = backend.segment(image)

    print(f"\nDetected {len(results)} materials:")
    for material, (mask, confidence) in results.items():
        coverage = (mask.sum() / mask.size) * 100
        print(f"  • {material}: {confidence:.1%} confidence, {coverage:.1f}% coverage")

    print("\nNote: Heuristic mode returns 0.5 confidence (color-based)")
    print("      CLIP mode would show varied confidence scores (0.2-1.0)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided an image path
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
        demo_confidence_scoring(image_path)
    else:
        # No image provided, use synthetic demo
        print("\nNo image provided. Running synthetic image demo...")
        print("(To test with your own image: python confidence_scoring_demo.py path/to/image.jpg)")
        demo_with_synthetic_image()
