#!/usr/bin/env python3
"""Example: Phase 2 Depth Estimation Demo

This script demonstrates the Phase 2 capabilities of the canonical depth pipeline:
- Automatic depth estimation from images
- Model caching for fast repeated processing
- PBR map generation from estimated depth
- Batch processing support

Usage:
    python examples/phase2_depth_demo.py

Requirements:
    pip install transformers torch pillow numpy scikit-image
"""

from pathlib import Path
import numpy as np
from PIL import Image

from transformation_portal.depth_canonical import DepthPipeline
from transformation_portal.depth_canonical.config import (
    UnifiedDepthConfig,
    ModelConfig,
    ModelVariant,
    DeviceType,
    ProcessingConfig,
    PBRConfig,
    IOConfig,
)


def create_demo_image(size=(512, 512), name="demo"):
    """Create a demo gradient image for testing."""
    img = np.zeros((*size, 3), dtype=np.uint8)

    # Create gradient pattern
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j, 0] = int((i / size[0]) * 255)  # Red gradient
            img[i, j, 1] = int((j / size[1]) * 255)  # Green gradient
            img[i, j, 2] = int(((i + j) / (size[0] + size[1])) * 255)  # Blue gradient

    return Image.fromarray(img)


def demo_basic_depth_estimation():
    """Demo 1: Basic depth estimation without PBR."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Depth Estimation")
    print("="*70)

    # Create config - use small model for speed
    config = UnifiedDepthConfig(
        model=ModelConfig(
            variant=ModelVariant.DA3_METRIC_SMALL,
            device=None,  # Auto-detect
        ),
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=False)
        ),
        io=IOConfig(
            cache_enabled=True
        )
    )

    # Initialize pipeline
    print("\n📦 Initializing pipeline...")
    pipeline = DepthPipeline(config)
    print(f"✓ Device selected: {pipeline.model_registry._auto_detect_device().value}")

    # Create demo image
    print("\n🎨 Creating demo image...")
    demo_img = create_demo_image(size=(512, 512))

    # First run - will download model and compute depth
    print("\n🔍 First run (model loading + depth estimation)...")
    result1 = pipeline.process(image=demo_img)
    print(f"✓ Depth map shape: {result1.depth_map.shape}")
    print(f"✓ Depth range: [{result1.depth_map.min():.3f}, {result1.depth_map.max():.3f}]")

    # Second run - should use cache
    print("\n⚡ Second run (cached)...")
    result2 = pipeline.process(image=demo_img)
    print(f"✓ Depth map shape: {result2.depth_map.shape}")
    print(f"✓ Identical to first run: {np.allclose(result1.depth_map, result2.depth_map)}")


def demo_depth_with_pbr():
    """Demo 2: Depth estimation + PBR generation."""
    print("\n" + "="*70)
    print("DEMO 2: Depth Estimation + PBR Generation")
    print("="*70)

    # Create config with PBR enabled
    config = UnifiedDepthConfig(
        model=ModelConfig(
            variant=ModelVariant.DA3_METRIC_SMALL,
        ),
        processing=ProcessingConfig(
            pbr=PBRConfig(
                enabled=True,
                normal_strength=1.2,
                roughness_strength=1.0,
                ao_strength=0.8,
            )
        )
    )

    # Initialize pipeline
    print("\n📦 Initializing pipeline with PBR...")
    pipeline = DepthPipeline(config)

    # Create demo image
    demo_img = create_demo_image(size=(512, 512))

    # Create output directory
    output_dir = Path("output/phase2_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process with PBR
    print(f"\n🔍 Processing image with PBR...")
    result = pipeline.process(
        image=demo_img,
        output_dir=output_dir,
        basename="demo"
    )

    print(f"\n✓ Depth map: {result.depth_map.shape}")
    print(f"✓ PBR maps generated:")
    for map_name, map_data in result.pbr_maps.items():
        print(f"  - {map_name}: {map_data.shape}")

    print(f"\n💾 Files saved:")
    for map_name, map_path in result.pbr_paths.items():
        print(f"  - {map_path.name}")

    print(f"\n📂 Output directory: {output_dir.absolute()}")


def demo_batch_processing():
    """Demo 3: Batch processing multiple images."""
    print("\n" + "="*70)
    print("DEMO 3: Batch Processing")
    print("="*70)

    # Create config
    config = UnifiedDepthConfig(
        model=ModelConfig(
            variant=ModelVariant.DA3_METRIC_SMALL,
        ),
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=True)
        )
    )

    # Initialize pipeline
    print("\n📦 Initializing pipeline for batch processing...")
    pipeline = DepthPipeline(config)

    # Create multiple demo images with different patterns
    print("\n🎨 Creating batch of demo images...")
    images = []
    temp_dir = Path("output/phase2_demo/batch_input")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        img = create_demo_image(size=(256, 256), name=f"batch_{i}")
        img_path = temp_dir / f"image_{i}.png"
        img.save(img_path)
        images.append(img_path)
        print(f"  ✓ Created {img_path.name}")

    # Batch process
    output_dir = Path("output/phase2_demo/batch_output")

    print(f"\n🔄 Processing {len(images)} images...")
    results = pipeline.batch_process(
        images=images,
        output_dir=output_dir
    )

    print(f"\n✓ Processed {len(results)} images:")
    for i, result in enumerate(results):
        print(f"  Image {i+1}:")
        print(f"    - Depth: {result.depth_map.shape}")
        print(f"    - PBR maps: {len(result.pbr_maps)} generated")
        if result.pbr_paths:
            print(f"    - Files: {len(result.pbr_paths)} saved")


def demo_backward_compatibility():
    """Demo 4: Backward compatibility with Phase 1 API."""
    print("\n" + "="*70)
    print("DEMO 4: Backward Compatibility")
    print("="*70)

    config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=True)
        )
    )

    pipeline = DepthPipeline(config)

    # Old API: Provide pre-computed depth map
    print("\n📊 Using Phase 1 API (pre-computed depth)...")
    depth_map = np.random.rand(512, 512).astype(np.float32)

    result = pipeline.process(
        image_path=Path("dummy.jpg"),  # Old parameter name
        depth_map=depth_map,
        output_dir=Path("output/phase2_demo/backward_compat"),
        basename="old_api"
    )

    print(f"✓ Depth map used: {result.depth_map.shape}")
    print(f"✓ Identical to input: {np.array_equal(result.depth_map, depth_map)}")
    print(f"✓ PBR generated: {result.pbr_maps is not None}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🚀 PHASE 2 DEPTH ESTIMATION DEMO")
    print("="*70)
    print("\nThis demo showcases Phase 2 capabilities:")
    print("  1. Basic depth estimation with caching")
    print("  2. Depth estimation + PBR generation")
    print("  3. Batch processing")
    print("  4. Backward compatibility")

    try:
        # Run demos
        demo_basic_depth_estimation()
        demo_depth_with_pbr()
        demo_batch_processing()
        demo_backward_compatibility()

        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📂 Check output/phase2_demo/ for generated files")

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install transformers torch pillow numpy scikit-image")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
