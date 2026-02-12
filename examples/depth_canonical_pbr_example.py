#!/usr/bin/env python3
"""Example: PBR Map Generation with Depth Canonical Pipeline.

This example demonstrates the Phase 1 implementation of the canonical depth
processing pipeline with integrated PBR (Physically Based Rendering) map generation.

Phase 1 Features:
- Unified configuration with PBRConfig
- Pipeline orchestration with DepthPipeline
- Atomic PBR map writing (normal, roughness, AO)
- Batch processing support

Phase 2 (Coming Soon):
- Automatic depth estimation from images
- Full integration with Depth Anything V2/V3
- Advanced postprocessing and effects
"""

import tempfile
from pathlib import Path

import numpy as np

from transformation_portal.depth_canonical import (
    DepthPipeline,
    DeviceType,
    ModelConfig,
    ModelVariant,
    PBRConfig,
    ProcessingConfig,
    UnifiedDepthConfig,
)


def example_single_image_processing():
    """Example: Process single image with PBR generation."""
    print("=" * 60)
    print("Example 1: Single Image PBR Generation")
    print("=" * 60)

    # Configure pipeline with PBR enabled
    config = UnifiedDepthConfig(
        model=ModelConfig(
            variant=ModelVariant.DA3_METRIC_LARGE,
            device=DeviceType.CPU,
        ),
        processing=ProcessingConfig(
            pbr=PBRConfig(
                enabled=True,
                normal_strength=1.2,  # Slightly stronger normals
                roughness_blur_radius=5,  # Smoother roughness
                ao_bias=0.6,  # Prevent overly dark occlusion
            )
        ),
    )

    # Create pipeline
    pipeline = DepthPipeline(config)

    # Simulate depth map (in Phase 2, this will be auto-generated)
    # For this example, create a synthetic depth gradient
    depth_map = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Process depth map to generate PBR maps
        result = pipeline.process(depth_map=depth_map, output_dir=output_dir, basename="example_render")

        # Results
        print(f"\nPBR Maps Generated:")
        print(f"  Normal map: {result.pbr_paths['normal']}")
        print(f"  Roughness map: {result.pbr_paths['roughness']}")
        print(f"  Ambient Occlusion: {result.pbr_paths['ao']}")

        print(f"\nMap Shapes:")
        print(f"  Normal: {result.pbr_maps['normal'].shape}")
        print(f"  Roughness: {result.pbr_maps['roughness'].shape}")
        print(f"  AO: {result.pbr_maps['ao'].shape}")


def example_batch_processing():
    """Example: Batch process multiple images."""
    print("\n" + "=" * 60)
    print("Example 2: Batch PBR Generation")
    print("=" * 60)

    # Configure pipeline
    config = UnifiedDepthConfig(processing=ProcessingConfig(pbr=PBRConfig(enabled=True)))

    pipeline = DepthPipeline(config)

    # Simulate multiple depth maps
    depth_maps = [
        np.random.rand(128, 128).astype(np.float32),
        np.random.rand(128, 128).astype(np.float32),
        np.random.rand(128, 128).astype(np.float32),
    ]

    image_paths = [
        Path("render_001.jpg"),
        Path("render_002.jpg"),
        Path("render_003.jpg"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Batch process
        results = pipeline.process_batch(image_paths=image_paths, output_dir=output_dir, depth_maps=depth_maps)

        print(f"\nProcessed {len(results)} images:")
        for i, result in enumerate(results):
            print(f"\nImage {i + 1}:")
            print(f"  Normal: {result.pbr_paths['normal'].name}")
            print(f"  Roughness: {result.pbr_paths['roughness'].name}")
            print(f"  AO: {result.pbr_paths['ao'].name}")


def example_custom_pbr_parameters():
    """Example: Custom PBR parameters for different materials."""
    print("\n" + "=" * 60)
    print("Example 3: Custom PBR Parameters")
    print("=" * 60)

    # Example: High-gloss metal surface
    metal_config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(
                enabled=True,
                normal_strength=1.5,  # Strong surface detail
                roughness_strength=0.5,  # Low roughness (glossy)
                roughness_blur_radius=7,  # Smooth variation
                ao_strength=0.8,  # Subtle occlusion
                ao_bias=0.7,  # Bright overall
            )
        )
    )

    # Example: Matte wood surface
    wood_config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(
                enabled=True,
                normal_strength=1.0,  # Natural surface detail
                roughness_strength=1.2,  # Higher roughness (matte)
                roughness_blur_radius=3,  # Fine texture
                ao_strength=1.2,  # Strong occlusion in grain
                ao_bias=0.4,  # Allow darker areas
            )
        )
    )

    depth_map = np.random.rand(256, 256).astype(np.float32)

    # Process with metal parameters
    metal_pipeline = DepthPipeline(metal_config)
    metal_result = metal_pipeline.process(depth_map=depth_map)

    # Process with wood parameters
    wood_pipeline = DepthPipeline(wood_config)
    wood_result = wood_pipeline.process(depth_map=depth_map)

    print("\nMetal Surface PBR:")
    print(f"  Roughness mean: {metal_result.pbr_maps['roughness'].mean():.2f}")
    print(f"  AO mean: {metal_result.pbr_maps['ao'].mean():.2f}")

    print("\nWood Surface PBR:")
    print(f"  Roughness mean: {wood_result.pbr_maps['roughness'].mean():.2f}")
    print(f"  AO mean: {wood_result.pbr_maps['ao'].mean():.2f}")


def example_pbr_disabled():
    """Example: Pipeline without PBR generation."""
    print("\n" + "=" * 60)
    print("Example 4: Depth Processing Without PBR")
    print("=" * 60)

    # PBR disabled (default)
    config = UnifiedDepthConfig(processing=ProcessingConfig(pbr=PBRConfig(enabled=False)))

    pipeline = DepthPipeline(config)
    depth_map = np.random.rand(256, 256).astype(np.float32)

    result = pipeline.process(depth_map=depth_map)

    print(f"\nDepth map stored: {result.depth_map is not None}")
    print(f"PBR maps generated: {result.pbr_maps is not None}")
    print("(PBR generation is opt-in via config)")


if __name__ == "__main__":
    print("\nDepth Canonical Pipeline - PBR Integration Examples")
    print("Phase 1: Foundation Module with PBR Generation\n")

    example_single_image_processing()
    example_batch_processing()
    example_custom_pbr_parameters()
    example_pbr_disabled()

    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)
    print("\nNext Steps (Phase 2):")
    print("  - Automatic depth estimation from RGB images")
    print("  - Integration with Depth Anything V2/V3 models")
    print("  - Zone-based tone mapping and atmospheric effects")
    print("  - LRU caching for iterative workflows")
    print("\nAll examples completed successfully! ✓")
