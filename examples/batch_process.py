#!/usr/bin/env python3
"""
Batch processing example.

Processes all images in a directory with progress tracking.

Usage:
    python examples/batch_process.py input_dir/ output_dir/ [--preset interior|exterior]
"""

from depth_pipeline import ArchitecturalDepthPipeline
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Batch process architectural renders')
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument(
        '--preset',
        type=str,
        choices=['default', 'interior', 'exterior'],
        default='default',
        help='Configuration preset'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='File pattern (e.g., "*.png", "render_*.jpg")'
    )
    parser.add_argument(
        '--no-depth',
        action='store_true',
        help='Skip saving depth maps'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip depth visualizations'
    )

    args = parser.parse_args()

    # Get input images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    image_paths = list(input_dir.glob(args.pattern))

    if not image_paths:
        print(f"Error: No images found matching pattern '{args.pattern}' in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Load pipeline
    config_map = {
        'default': 'config/default_config.yaml',
        'interior': 'config/interior_preset.yaml',
        'exterior': 'config/exterior_preset.yaml',
    }
    config_path = config_map[args.preset]

    print(f"Loading pipeline (preset: {args.preset})...")
    pipeline = ArchitecturalDepthPipeline.from_config(config_path)

    # Process batch
    _results = pipeline.batch_process(  # noqa: F841
        image_paths,
        args.output_dir,
        save_depth=not args.no_depth,
        save_visualization=not args.no_viz,
    )

    # Print final statistics
    stats = pipeline.get_stats()
    print("\nFinal Statistics:")
    print(f"  Total images: {stats['images_processed']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")


if __name__ == '__main__':
    main()
