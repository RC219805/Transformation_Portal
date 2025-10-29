#!/usr/bin/env python3
"""
Simple single-image processing example.

Usage:
    python examples/simple_process.py input.jpg output/
"""

from depth_pipeline import ArchitecturalDepthPipeline
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_process.py <input_image> <output_dir>")
        print("Example: python simple_process.py render.jpg output/")
        sys.exit(1)

    input_image = sys.argv[1]
    output_dir = sys.argv[2]

    # Initialize pipeline with default configuration
    print("Loading pipeline...")
    pipeline = ArchitecturalDepthPipeline.from_config('config/default_config.yaml')

    # Process image
    print(f"Processing: {input_image}")
    result = pipeline.process_render(input_image)

    # Save results
    print(f"Saving to: {output_dir}")
    pipeline.save_result(
        result,
        output_dir,
        save_depth=True,
        save_visualization=True
    )

    # Print statistics
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"  Processing time: {result['metadata']['processing_time_sec']:.2f}s")
    print(f"  Depth inference: {result['metadata']['depth_inference_time_ms']:.1f}ms")
    print(f"  Input shape: {result['metadata']['input_shape']}")
    print("=" * 50)


if __name__ == '__main__':
    main()
