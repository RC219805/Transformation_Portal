#!/usr/bin/env python3
"""Run APEX Ultra Research Spatial AI pipeline on raw input files.

This script processes raw camera files (CR2, CRW, TIFF) through the complete
spatial AI pipeline with APEX Ultra Research preset:
- Linear ingest (preserves full dynamic range)
- Depth estimation (multi-model ensemble)
- SAM2 segmentation
- PBR materials (optional)
- 3D reconstruction (optional, requires research tier)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from transformation_portal.spatial_ai.orchestration.pipeline import SpatialAIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="APEX Ultra Research Spatial AI Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input image or directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_apex_ultra_spatial_ai"),
        help="Output directory (default: output_apex_ultra_spatial_ai)",
    )
    parser.add_argument(
        "--preset",
        default="apex_research_ultra",
        help="Preset to use (default: apex_research_ultra)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["ingest", "segment", "materials", "reconstruction"],
        help="Pipeline stages to run (default: ingest + segment)",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        default=True,
        help="Save intermediate outputs (default: True)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("APEX Ultra Research Spatial AI Pipeline")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Preset: {args.preset}")
    if args.stages:
        logger.info(f"Stages: {', '.join(args.stages)}")
    logger.info("")

    # Initialize pipeline from preset
    try:
        logger.info(f"Loading configuration: {args.preset}")

        # Check if it's a file path
        preset_path = Path(args.preset)
        if preset_path.exists() and preset_path.is_file():
            logger.info(f"Loading from file: {preset_path}")
            pipeline = SpatialAIPipeline(str(preset_path))
        else:
            # Try as preset name
            pipeline = SpatialAIPipeline.from_preset(args.preset)

        # Override stages if specified (must update both config and progress tracker)
        if args.stages:
            pipeline.config.stages = args.stages
            # Reinitialize progress tracker with correct stage count
            from transformation_portal.spatial_ai.orchestration.progress_tracker import ProgressTracker

            pipeline.progress_tracker = ProgressTracker(total_stages=len(args.stages))
            logger.info(f"Configured stages: {', '.join(args.stages)}")

        logger.info("✓ Pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to load configuration '{args.preset}': {e}")
        logger.info("Available presets: spatial_ai_standard, spatial_ai_research")
        logger.info("Or provide a path to a YAML config file")
        return 1

    # Process input
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        # Find all image files
        extensions = {".cr2", ".crw", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        files_to_process = [f for f in input_path.iterdir() if f.suffix.lower() in extensions and not f.name.startswith(".")]
        if not files_to_process:
            logger.error(f"No image files found in {input_path}")
            return 1
        files_to_process.sort()
    else:
        logger.error(f"Input path must be file or directory: {input_path}")
        return 1

    logger.info(f"Found {len(files_to_process)} file(s) to process\n")

    # Process each file
    results = []
    start_time = time.time()

    for i, file_path in enumerate(files_to_process, 1):
        logger.info(f"[{i}/{len(files_to_process)}] Processing: {file_path.name}")
        logger.info("-" * 80)

        try:
            result = pipeline.process(
                input_path=file_path,
                output_dir=output_dir,
                save_intermediates=args.save_intermediates,
            )

            logger.info(f"✓ Success!")
            logger.info(f"  Stages completed: {', '.join(result.stages_completed)}")
            logger.info(f"  Execution time: {result.execution_time:.1f}s")
            logger.info(f"  Peak memory: {result.peak_memory_mb:.1f} MB")

            # Show stage results
            stage_info = []
            if result.linear_image:
                h, w = result.linear_image.linear_rgb.shape[:2]
                stage_info.append(f"Ingest: {w}×{h}")
            if result.segmentation:
                stage_info.append(f"Segmentation: {len(result.segmentation.masks)} masks")
            if result.materials:
                stage_info.append(f"Materials: {len(result.materials)} segments")
            if result.scene_3d:
                stage_info.append(f"3D: {len(result.scene_3d.gaussians)} Gaussians")

            if stage_info:
                logger.info(f"  Results:")
                for info in stage_info:
                    logger.info(f"    - {info}")

            results.append({"file": file_path.name, "success": True, "result": result})

        except Exception as e:
            logger.error(f"✗ Failed: {e}", exc_info=args.verbose)
            results.append({"file": file_path.name, "success": False, "error": str(e)})

        logger.info("")

    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r["success"])

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(results) - successful}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    if successful < len(results):
        logger.warning("Some files failed to process:")
        for r in results:
            if not r["success"]:
                logger.warning(f"  - {r['file']}: {r['error']}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
