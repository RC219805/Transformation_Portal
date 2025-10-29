#!/usr/bin/env python3
"""
Custom pipeline example.

Demonstrates building a custom processing pipeline with specific parameters.

Usage:
    python examples/custom_pipeline.py input.jpg output.jpg
"""

from depth_pipeline.utils import (
    load_image,
    save_image,
    visualize_depth,
    depth_statistics,
)
from depth_pipeline.processors import (
    DepthAwareDenoise,
    ZoneToneMapping,
    AtmosphericEffects,
    DepthGuidedFilters,
)
from depth_pipeline.models import DepthAnythingV2Model, ModelVariant
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    if len(sys.argv) < 3:
        print("Usage: python custom_pipeline.py <input_image> <output_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("=" * 60)
    print("CUSTOM DEPTH PIPELINE")
    print("=" * 60)

    # 1. Initialize depth model
    print("\n[1/6] Initializing depth model...")
    model = DepthAnythingV2Model(
        variant=ModelVariant.SMALL,
        backend="pytorch_mps",  # or "coreml" for ANE
    )
    print(f"      Model: {model}")

    # 2. Load image
    print(f"\n[2/6] Loading image: {input_path}")
    image = load_image(input_path, normalize=True)
    print(f"      Shape: {image.shape}, dtype: {image.dtype}")

    # 3. Estimate depth
    print("\n[3/6] Estimating depth...")
    depth_result = model.estimate_depth(image)
    depth = depth_result['depth']
    print(f"      Inference time: {depth_result['metadata']['inference_time_ms']:.1f}ms")

    # Print depth statistics
    stats = depth_statistics(depth)
    print(f"      Depth range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")

    # 4. Custom processing
    print("\n[4/6] Applying custom processing...")

    # Initialize processors with custom parameters
    denoiser = DepthAwareDenoise(
        sigma_spatial=2.5,
        edge_threshold=0.04,
        preserve_strength=0.85,
    )

    tone_mapper = ZoneToneMapping(
        num_zones=4,
        zone_params=[
            {'contrast': 1.3, 'saturation': 1.15, 'exposure': 0.0},
            {'contrast': 1.15, 'saturation': 1.08, 'exposure': 0.0},
            {'contrast': 1.0, 'saturation': 1.0, 'exposure': 0.0},
            {'contrast': 0.9, 'saturation': 0.85, 'exposure': -0.1},
        ],
        method='agx'
    )

    atmosphere = AtmosphericEffects(
        haze_density=0.012,
        haze_color=(0.72, 0.82, 0.92),
        desaturation_strength=0.25,
    )

    filters = DepthGuidedFilters(
        clarity_strength=0.55,
        scale_count=3,
    )

    # Apply processing chain
    result = image.copy()

    print("      - Depth-aware denoising...")
    result = denoiser(result, depth)

    print("      - Zone tone mapping (4 zones)...")
    result = tone_mapper(result, depth)

    print("      - Atmospheric effects...")
    result = atmosphere(result, depth)

    print("      - Depth-guided filters...")
    result = filters(result, depth)

    # 5. Save results
    print("\n[5/6] Saving results...")
    output_dir = Path(output_path).parent
    stem = Path(output_path).stem

    # Save enhanced image
    save_image(result, output_path)
    print(f"      Enhanced: {output_path}")

    # Save depth visualization
    depth_viz_path = output_dir / f"{stem}_depth.png"
    visualize_depth(depth, colormap='turbo', save_path=str(depth_viz_path))
    print(f"      Depth viz: {depth_viz_path}")

    # 6. Done
    print("\n[6/6] Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
