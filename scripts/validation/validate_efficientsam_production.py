#!/usr/bin/env python3
"""
Production Validation: EfficientSAM Segmentation Backend

Tests the EfficientSAM backend integration on 800 Picacho luxury real estate images.

Expected outcomes:
- Material masks generated for each image
- Pixel operations applied based on detected materials
- Enhanced images with surface-aware finishing
- Manifest contains segmentation telemetry
- Performance metrics within expected bounds
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_validation():
    """Execute production validation with EfficientSAM backend."""

    print("=" * 80)
    print("EfficientSAM Production Validation")
    print("=" * 80)
    print()

    # Configuration
    input_dir = Path("input_images/800 Picacho")
    output_dir = Path(f"output_800_picacho_efficientsam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Count input images
    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG"))
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Images found: {len(input_images)}")
    print()

    # Import pipeline components
    logger.info("Importing pipeline components...")
    try:
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
    except ImportError as e:
        logger.error(f"Failed to import pipeline components: {e}")
        return 1

    # Create configuration with EfficientSAM enabled
    logger.info("Creating configuration...")
    config = EnhanceConfig(
        # Quality tier
        quality_tier="apex",
        # Depth backend (commercial-safe)
        depth_backend="da3",
        depth_device="auto",  # Auto-detect MPS/CUDA/CPU
        # Materials V3 with EfficientSAM
        enable_materials_v3=True,
        apply_pixel_ops=True,
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,  # Fail if EfficientSAM unavailable
        # Materials V3 parameters
        min_coverage_px=500,
        min_mean_conf=0.2,
        glass_response_enabled=True,
        # V2 enhancement
        enable_v2=True,
        v2_preset="default",
        # PBR
        generate_pbr=True,
        # Caching
        enable_depth_cache=True,
        # Performance
        enable_parallel_processing=True,
        max_parallel_workers=1,  # Serial for controlled validation
        # Emit flags
        output_bit_depth=16,
        emit_run_card=True,
    )

    print()
    print("Configuration:")
    print(f"  Quality tier: {config.quality_tier}")
    print(f"  Depth backend: {config.depth_backend}")
    print(f"  Depth device: {config.depth_device}")
    print(f"  Materials V3: {config.enable_materials_v3}")
    print(f"  Segmentation backend: {config.material_segmentation_backend}")
    print(f"  Strict backend: {config.strict_backend}")
    print(f"  V2 enhancement: {config.enable_v2}")
    print(f"  PBR generation: {config.generate_pbr}")
    print()

    # Create orchestrator
    logger.info("Initializing orchestrator...")
    try:
        prepared = prepare_lux_execution(config, input_dir, sorted(input_images))
        orchestrator = EnhanceOrchestrator.from_prepared(prepared, output_dir)
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        return 1

    # Process images
    logger.info("Starting pipeline...")
    print()
    print("-" * 80)
    print()

    start_time = time.time()
    results: List[Dict] = []
    errors: List[Dict] = []

    for idx, image_path in enumerate(sorted(input_images), 1):
        logger.info(f"[{idx}/{len(input_images)}] Processing: {image_path.name}")
        image_start = time.time()

        try:
            # Process single image
            result = orchestrator.enhance_image(ImageInput(path=image_path))

            image_elapsed = time.time() - image_start

            # Collect metrics
            results.append(
                {
                    "image": image_path.name,
                    "status": "success",
                    "elapsed_sec": round(image_elapsed, 2),
                    "result": result,
                }
            )

            logger.info(f"  ✓ Complete in {image_elapsed:.2f}s")

        except Exception as e:
            image_elapsed = time.time() - image_start
            logger.error(f"  ✗ Failed after {image_elapsed:.2f}s: {e}")

            errors.append(
                {
                    "image": image_path.name,
                    "error": str(e),
                    "elapsed_sec": round(image_elapsed, 2),
                }
            )

        print()

    total_elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 80)
    print("Validation Complete")
    print("=" * 80)
    print()
    print(f"Total images: {len(input_images)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")
    print(f"Total time: {total_elapsed:.2f}s")

    if results:
        avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
        print(f"Average time per image: {avg_time:.2f}s")

    print()

    # Analyze outputs
    if output_dir.exists():
        logger.info("Analyzing outputs...")

        # Count files
        all_files = list(output_dir.rglob("*"))
        total_files = len([f for f in all_files if f.is_file()])
        total_size_mb = sum(f.stat().st_size for f in all_files if f.is_file()) / (1024 * 1024)

        print(f"Output directory: {output_dir}")
        print(f"  Total files: {total_files}")
        print(f"  Total size: {total_size_mb:.1f} MB")
        print()

        # Check for Materials V3 artifacts
        materials_dir = output_dir / "materials_v3"
        if materials_dir.exists():
            masks = list(materials_dir.glob("*_mask_*.png"))
            enhanced = list(materials_dir.glob("*_enhanced.png"))
            print(f"Materials V3 outputs:")
            print(f"  Mask files: {len(masks)}")
            print(f"  Enhanced images: {len(enhanced)}")
            print()

        # Check manifests for segmentation metadata
        manifests = list(output_dir.rglob("*_manifest.json"))
        print(f"Manifests: {len(manifests)}")

        if manifests:
            # Analyze first manifest
            first_manifest = manifests[0]
            logger.info(f"Analyzing manifest: {first_manifest.name}")

            try:
                with open(first_manifest, "r") as f:
                    manifest_data = json.load(f)

                # Check for materials_v3 section
                if "materials_v3" in manifest_data:
                    mat_v3 = manifest_data["materials_v3"]
                    print()
                    print("Sample manifest (materials_v3 section):")
                    print(json.dumps(mat_v3, indent=2)[:500])
                    print()

                    # Check segmentation backend
                    if "segmentation_backend" in mat_v3:
                        backend = mat_v3["segmentation_backend"]
                        print(f"✓ Segmentation backend recorded: {backend}")

                        if backend == "efficientsam":
                            logger.info("✓ EfficientSAM backend confirmed in manifest")
                        else:
                            logger.warning(f"⚠ Expected 'efficientsam', got '{backend}'")
                    else:
                        logger.warning("⚠ No segmentation_backend field in manifest")
                else:
                    logger.warning("⚠ No materials_v3 section in manifest")

            except Exception as e:
                logger.error(f"Failed to parse manifest: {e}")

        print()

    # Report errors
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  {err['image']}: {err['error']}")
        print()

    # Exit code
    if len(results) == len(input_images):
        logger.info("✓ All images processed successfully")
        return 0
    elif results:
        logger.warning(f"⚠ Partial success: {len(results)}/{len(input_images)} images")
        return 2
    else:
        logger.error("✗ All images failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
