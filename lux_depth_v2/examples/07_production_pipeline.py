#!/usr/bin/env python3
"""
Production Pipeline Example
============================

Complete production workflow with:
- Robust error handling
- Detailed logging
- Progress tracking
- Output validation
- Summary reporting
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure production logging."""
    logger = logging.getLogger("production_pipeline")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def validate_outputs(output_dir: Path, stem: str) -> Dict[str, bool]:
    """Validate expected output files exist."""
    checks = {
        "master": (output_dir / f"{stem}_master16.tif").exists(),
        "upscaled": (output_dir / f"{stem}_upscaled16.tif").exists(),
        "marketing": (output_dir / f"{stem}_marketing.png").exists(),
        "report": (output_dir / f"{stem}_report.json").exists(),
    }
    return checks


def generate_summary_report(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate human-readable summary report."""
    report_path = output_dir / "PROCESSING_SUMMARY.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LUX DEPTH V2 - PRODUCTION PROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Statistics
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'ok')
        failed = sum(1 for r in results if r['status'] == 'error')
        skipped = sum(1 for r in results if r['status'] == 'skipped')

        f.write(f"Total Images: {total}\n")
        f.write(f"  Successful: {success}\n")
        f.write(f"  Failed: {failed}\n")
        f.write(f"  Skipped: {skipped}\n\n")

        # Timing
        ok_results = [r for r in results if r['status'] == 'ok']
        if ok_results:
            times = [r['timing_s'] for r in ok_results]
            f.write("Processing Times:\n")
            f.write(f"  Average: {sum(times)/len(times):.2f}s\n")
            f.write(f"  Minimum: {min(times):.2f}s\n")
            f.write(f"  Maximum: {max(times):.2f}s\n")
            f.write(f"  Total: {sum(times):.2f}s ({sum(times)/60:.1f} min)\n\n")

        # Successful images
        if success > 0:
            f.write("Successfully Processed:\n")
            f.write("-" * 70 + "\n")
            for r in results:
                if r['status'] == 'ok':
                    name = Path(r['image']).name
                    f.write(f"  {name}\n")
                    f.write(f"    Time: {r['timing_s']:.2f}s\n")
                    f.write(f"    Weights: {r['zone_weights']}\n")
                    if r.get('material_mods'):
                        f.write(f"    Materials: {r['material_mods']}\n")
                    if r.get('ai_color_diff'):
                        f.write(f"    AI drift: RGB={r['ai_color_diff']:.4f}, "
                               f"Luma={r['ai_luma_diff']:.4f}\n")
            f.write("\n")

        # Failed images
        if failed > 0:
            f.write("Failed Images:\n")
            f.write("-" * 70 + "\n")
            for r in results:
                if r['status'] == 'error':
                    name = Path(r['image']).name
                    error = r.get('error', 'Unknown error')
                    f.write(f"  {name}: {error}\n")
            f.write("\n")

        # Quality warnings
        warnings = []
        for r in results:
            if r['status'] == 'ok':
                if r.get('ai_color_diff', 0) > 0.08:
                    warnings.append((Path(r['image']).name, 'High AI color drift'))
                if r.get('ai_luma_diff', 0) > 0.08:
                    warnings.append((Path(r['image']).name, 'High AI luma drift'))

        if warnings:
            f.write("Quality Warnings:\n")
            f.write("-" * 70 + "\n")
            for name, warning in warnings:
                f.write(f"  {name}: {warning}\n")

    return report_path


def main():
    # Production configuration
    config = PipelineConfig(
        preset=Preset.INTERIOR_LUXURY,

        # Paths
        input_dir=Path("input"),
        output_dir=Path("output_production"),
        depth_dir=Path("depth_maps"),

        # Processing
        device="auto",
        upscale=4,
        upscaler_backend="realesrgan",  # Or "none" for testing
        model_path=Path("models/RealESRGAN_x4plus.pth") if Path("models/RealESRGAN_x4plus.pth").exists() else None,

        # Material enhancement
        enable_material=True,
        material_strength=0.85,

        # Production options
        skip_existing=True,  # Resume capability
        overwrite=False,
        save_master=True,
        save_upscaled=True,
        save_marketing_png=True,
        save_preview_jpg=True,

        # Quality control
        validate_ai=True,
        ai_color_warn=0.06,
        ai_color_fail=0.12,
        ai_luma_warn=0.06,
        ai_luma_fail=0.12,

        # Memory safety
        warn_float_gb=6.0,
        post_tile=512,  # Enable tiling for large images
        post_overlap=32,
    )

    # Setup logging
    log_file = config.output_dir / "processing.log"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("PRODUCTION PIPELINE START")
    logger.info("=" * 70)
    logger.info(f"Input: {config.input_dir}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Preset: {config.preset.value}")
    logger.info(f"Device: {config.device}")

    # Validate input directory
    if not config.input_dir.exists():
        logger.error(f"Input directory not found: {config.input_dir}")
        sys.exit(1)

    # Initialize pipeline
    try:
        logger.info("Initializing pipeline...")
        pipeline = LuxPipelineV2(config, logger=logger)
        logger.info(f"Pipeline ready: device={pipeline.device}, autocast={pipeline.autocast}")
    except Exception as e:
        logger.exception(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Process directory
    logger.info("Starting batch processing...")
    try:
        results = pipeline.process_directory()
    except Exception as e:
        logger.exception(f"Batch processing failed: {e}")
        sys.exit(1)

    # Validate outputs
    logger.info("Validating outputs...")
    for result in results:
        if result['status'] == 'ok':
            stem = Path(result['image']).stem
            checks = validate_outputs(config.output_dir, stem)
            missing = [k for k, v in checks.items() if not v]
            if missing:
                logger.warning(f"{stem}: Missing outputs: {', '.join(missing)}")

    # Generate summary report
    logger.info("Generating summary report...")
    report_path = generate_summary_report(results, config.output_dir)
    logger.info(f"Summary report: {report_path}")

    # Final statistics
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'ok')
    failed = sum(1 for r in results if r['status'] == 'error')

    logger.info("=" * 70)
    logger.info("PRODUCTION PIPELINE COMPLETE")
    logger.info(f"Success: {success}/{total} images")
    if failed > 0:
        logger.warning(f"Failed: {failed} images - see log for details")
    logger.info("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
