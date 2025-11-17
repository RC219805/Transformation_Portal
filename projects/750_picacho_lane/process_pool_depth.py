#!/usr/bin/env python3
"""
Process luxury pool image with depth-aware enhancements.

Optimized for luxury real estate exterior/pool photography using:
- Depth Anything V2 model
- Zone-based tone mapping (foreground pool, midground landscape, background sky)
- Atmospheric depth effects
- Depth-guided clarity enhancement
"""

import logging
import sys
from pathlib import Path

# Add src (at repository root) to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Process luxury pool image with depth pipeline."""

    # Input and output paths (relative to project directory)
    project_dir = Path(__file__).parent
    input_image = project_dir / "input_images" / "V2_V2_750Picacho_Pool_Luxury_Enhanced.tiff"
    output_dir = project_dir / "output_images" / "depth_processed"

    # Verify input exists
    if not input_image.exists():
        logger.error(f"Input image not found: {input_image}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LUXURY POOL DEPTH PROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_image.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Load exterior preset optimized for pool scenes
    config_path = Path(__file__).parent / "config" / "exterior_preset.yaml"

    logger.info(f"Loading configuration: {config_path.name}")
    logger.info("Preset: Exterior/Aerial (optimized for outdoor luxury scenes)")
    logger.info("")

    try:
        # Initialize pipeline
        logger.info("Initializing Architectural Depth Pipeline...")
        pipeline = ArchitecturalDepthPipeline.from_config(config_path)
        logger.info("✓ Pipeline initialized successfully")
        logger.info("")

        # Process the image
        logger.info("Processing luxury pool image...")
        logger.info("Steps:")
        logger.info("  1. Estimating depth map (Depth Anything V2)")
        logger.info("  2. Applying depth-aware denoising")
        logger.info("  3. Zone-based tone mapping (pool/landscape/sky)")
        logger.info("  4. Atmospheric depth effects")
        logger.info("  5. Depth-guided clarity enhancement")
        logger.info("")

        result = pipeline.process_render(input_image)

        logger.info("✓ Processing complete!")
        logger.info("")

        # Save results
        logger.info("Saving outputs...")
        pipeline.save_result(
            result,
            output_dir,
            save_depth=True,
            save_visualization=True
        )

        # Print results summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)

        metadata = result['metadata']
        logger.info(f"Processing Time:      {metadata['processing_time_sec']:.2f}s")
        logger.info(f"Depth Inference:      {metadata['depth_inference_time_ms']:.1f}ms")
        logger.info(f"Input Resolution:     {metadata['input_shape'][1]}x{metadata['input_shape'][0]}")
        logger.info(f"Processors Applied:   {', '.join(metadata['processors_applied'])}")
        logger.info("")

        # Depth statistics
        depth_stats = metadata['depth_stats']
        logger.info("Depth Map Statistics:")
        logger.info(f"  Min Depth:          {depth_stats['min']:.3f}")
        logger.info(f"  Max Depth:          {depth_stats['max']:.3f}")
        logger.info(f"  Mean Depth:         {depth_stats['mean']:.3f}")
        logger.info(f"  Std Dev:            {depth_stats['std']:.3f}")
        logger.info(f"  Median Depth:       {depth_stats['median']:.3f}")
        logger.info("")

        # Output files
        stem = input_image.stem
        logger.info("Output Files:")
        logger.info(f"  ✓ Enhanced Image:   {stem}_enhanced.png")
        logger.info(f"  ✓ Depth Map:        {stem}_depth.npy")
        logger.info(f"  ✓ Depth Viz:        {stem}_depth_viz.png")
        logger.info("")

        logger.info("=" * 80)
        logger.info("DEPTH PROCESSING COMPLETE - LUXURY POOL ENHANCED")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
