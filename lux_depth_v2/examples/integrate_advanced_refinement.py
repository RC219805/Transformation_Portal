#!/usr/bin/env python3
"""
Example: Integrate Advanced Refinement into Depth Pipeline
===========================================================

Demonstrates how to integrate advanced edge-aware refinement
into the existing lux_depth_v2 pipeline for production use.

Author: Transformation Portal Specialist
Date: 2025-12-20
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# Import existing pipeline components
from lux_depth_v2.pipeline import DepthPipeline
from lux_depth_v2.config import PipelineConfig, Preset

# Import advanced refinement
from lux_depth_v2.advanced_refinement import (
    DepthRefiner,
    AdvancedRefinementConfig,
    compute_edge_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDepthPipeline:
    """
    Enhanced depth pipeline with integrated advanced refinement.

    Extends the base DepthPipeline with edge-aware post-processing
    for improved structure quality in architectural scenes.
    """

    def __init__(
        self,
        preset: str = "interior_luxury",
        use_advanced_refinement: bool = True,
        refinement_technique: str = "hybrid",
        refinement_config: Optional[AdvancedRefinementConfig] = None,
    ):
        """
        Initialize enhanced pipeline.

        Args:
            preset: Pipeline preset (e.g., "interior_luxury")
            use_advanced_refinement: Enable advanced refinement
            refinement_technique: Refinement technique to use
            refinement_config: Optional custom refinement config
        """
        # Initialize base pipeline
        self.pipeline_config = PipelineConfig()
        self.pipeline_config.apply_preset(Preset(preset))

        # Initialize advanced refinement
        self.use_advanced_refinement = use_advanced_refinement
        self.refinement_technique = refinement_technique

        if refinement_config is None:
            refinement_config = AdvancedRefinementConfig()

        self.refiner = DepthRefiner(refinement_config)

        logger.info(
            f"Enhanced pipeline initialized: preset={preset}, refinement={use_advanced_refinement} ({refinement_technique})"
        )

    def process_image(self, rgb_path: Path, output_dir: Path, compute_metrics: bool = True) -> dict:
        """
        Process single image through enhanced pipeline.

        Args:
            rgb_path: Path to input RGB image
            output_dir: Output directory for results
            compute_metrics: Whether to compute edge quality metrics

        Returns:
            Dict with processing results and metrics
        """
        # Load RGB image
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise ValueError(f"Failed to load image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        logger.info(f"Processing: {rgb_path.name}")

        # Step 1: Depth inference (placeholder - replace with actual model)
        # In production, use: depth_raw = self.pipeline.infer_depth(rgb)
        depth_raw = self._placeholder_depth_inference(rgb)

        # Step 2: Advanced refinement (optional)
        if self.use_advanced_refinement:
            logger.info(f"Applying {self.refinement_technique} refinement")
            depth_refined = self.refiner.refine(depth_raw, rgb, technique=self.refinement_technique)
        else:
            depth_refined = depth_raw

        # Step 3: Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_path = output_dir / f"{rgb_path.stem}_depth.tif"
        self._save_depth(depth_refined, depth_path)

        # Step 4: Compute metrics (optional)
        metrics = {}
        if compute_metrics:
            if self.use_advanced_refinement:
                metrics_before = compute_edge_metrics(depth_raw, rgb, "comprehensive")
                metrics_after = compute_edge_metrics(depth_refined, rgb, "comprehensive")

                metrics = {
                    "before": metrics_before,
                    "after": metrics_after,
                    "improvements": {
                        k: metrics_after[k] - metrics_before[k]
                        for k in metrics_before.keys()
                        if isinstance(metrics_before[k], (int, float))
                    },
                }

                logger.info(
                    f"Edge F1: {metrics_before['edge_f1']:.3f} → "
                    f"{metrics_after['edge_f1']:.3f} "
                    f"({metrics['improvements']['edge_f1']:+.3f})"
                )
            else:
                metrics = {"after": compute_edge_metrics(depth_refined, rgb, "comprehensive")}

        return {"rgb_path": rgb_path, "depth_path": depth_path, "metrics": metrics}

    def _placeholder_depth_inference(self, rgb: np.ndarray) -> np.ndarray:
        """
        Placeholder depth inference.

        Replace with actual Depth Anything V2 model inference.
        """
        # Simple gradient-based placeholder
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        depth = cv2.GaussianBlur(gray, (15, 15), 5)
        return depth.astype(np.float32) / 255.0

    def _save_depth(self, depth: np.ndarray, path: Path) -> None:
        """Save depth map as 16-bit TIFF."""
        if depth.dtype == np.float32:
            depth_uint16 = (depth * 65535).astype(np.uint16)
        else:
            depth_uint16 = depth.astype(np.uint16)

        cv2.imwrite(str(path), depth_uint16)
        logger.info(f"Saved: {path}")


def example_single_image():
    """Example: Process single image."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Image Processing")
    print("=" * 60 + "\n")

    # Initialize enhanced pipeline
    pipeline = EnhancedDepthPipeline(
        preset="interior_luxury",
        use_advanced_refinement=True,
        refinement_technique="hybrid",
    )

    # Process image
    rgb_path = Path("input_images/interior_001.jpg")
    output_dir = Path("output/refined/")

    # Note: This will fail if input image doesn't exist
    # Replace with actual image path
    if not rgb_path.exists():
        logger.warning(f"Example image not found: {rgb_path}")
        logger.info("Skipping single image example")
        return

    result = pipeline.process_image(rgb_path, output_dir, compute_metrics=True)

    print(f"\nProcessed: {result['rgb_path'].name}")
    print(f"Output: {result['depth_path']}")

    if result["metrics"]:
        metrics = result["metrics"]["after"]
        print(f"\nEdge Metrics:")
        print(f"  Edge F1: {metrics['edge_f1']:.3f}")
        print(f"  Edge alignment: {metrics['edge_alignment']:.3f}")


def example_batch_processing():
    """Example: Batch process directory."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 60 + "\n")

    # Initialize enhanced pipeline with fast config
    refinement_config = AdvancedRefinementConfig(
        use_bilateral_first=True,
        use_gradient_alignment=False,  # Skip for speed
        use_edge_preservation=True,
    )

    pipeline = EnhancedDepthPipeline(
        preset="interior_luxury",
        use_advanced_refinement=True,
        refinement_technique="hybrid",
        refinement_config=refinement_config,
    )

    # Process all images in directory
    input_dir = Path("input_images/")
    output_dir = Path("output/batch_refined/")

    if not input_dir.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        logger.info("Skipping batch processing example")
        return

    image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    results = []
    for img_path in image_paths:
        result = pipeline.process_image(img_path, output_dir, compute_metrics=True)
        results.append(result)

    # Print summary
    print(f"\nProcessed {len(results)} images")

    if results and results[0]["metrics"]:
        avg_f1_before = np.mean([r["metrics"]["before"]["edge_f1"] for r in results])
        avg_f1_after = np.mean([r["metrics"]["after"]["edge_f1"] for r in results])

        print(f"Average Edge F1: {avg_f1_before:.3f} → {avg_f1_after:.3f}")


def example_custom_refinement_config():
    """Example: Custom refinement configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Refinement Configuration")
    print("=" * 60 + "\n")

    # Create custom refinement config for architectural scenes
    # Emphasize edge preservation, minimal smoothing
    custom_config = AdvancedRefinementConfig(
        # Bilateral parameters (pre-smoothing)
        bilateral_d=7,
        bilateral_sigma_color=60.0,
        bilateral_sigma_space=60.0,
        # Guided filter parameters (edge-aware smoothing)
        guided_radius=6,
        guided_eps=0.005,  # Lower = sharper edges
        # Edge-guided enhancement
        edge_canny_low=40,
        edge_canny_high=120,
        edge_blur_sigma=0.8,  # Less smoothing in uniform regions
        # Gradient consistency
        gradient_smooth_sigma=1.0,
        gradient_threshold_percentile=60.0,
        # Pipeline stages
        use_bilateral_first=True,
        use_gradient_alignment=True,
        use_edge_preservation=True,
        # Quality settings
        preserve_16bit=True,
        normalize_output=True,
    )

    pipeline = EnhancedDepthPipeline(
        preset="architectural",
        use_advanced_refinement=True,
        refinement_technique="hybrid",
        refinement_config=custom_config,
    )

    print("Custom pipeline initialized with architectural-optimized config")
    print(f"  Guided eps: {custom_config.guided_eps}")
    print(f"  Edge blur sigma: {custom_config.edge_blur_sigma}")


def example_compare_techniques():
    """Example: Compare different refinement techniques."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Compare Refinement Techniques")
    print("=" * 60 + "\n")

    techniques = [
        "bilateral",
        "guided",
        "edge_guided",
        "gradient_consistency",
        "hybrid",
    ]

    rgb_path = Path("input_images/structure_test.jpg")

    if not rgb_path.exists():
        logger.warning(f"Test image not found: {rgb_path}")
        logger.info("Skipping technique comparison example")
        return

    print(f"Comparing techniques on: {rgb_path.name}\n")

    for technique in techniques:
        pipeline = EnhancedDepthPipeline(
            preset="interior_luxury",
            use_advanced_refinement=True,
            refinement_technique=technique,
        )

        output_dir = Path(f"output/compare/{technique}/")
        result = pipeline.process_image(rgb_path, output_dir, compute_metrics=True)

        if result["metrics"]:
            metrics = result["metrics"]["after"]
            print(f"{technique:25s} Edge F1: {metrics['edge_f1']:.3f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ADVANCED REFINEMENT INTEGRATION EXAMPLES")
    print("=" * 60)

    # Example 1: Single image
    example_single_image()

    # Example 2: Batch processing
    example_batch_processing()

    # Example 3: Custom config
    example_custom_refinement_config()

    # Example 4: Compare techniques
    example_compare_techniques()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
