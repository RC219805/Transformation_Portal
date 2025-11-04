#!/usr/bin/env python3
"""
Example script demonstrating format validation utilities.

This script shows how to use the format_utils module to validate
and get information about image and video file formats before
processing them with Transformation Portal pipelines.

NOTE: Requires package installation: pip install -e .

Usage:
    python examples/validate_file_formats.py path/to/file.jpg
    python examples/validate_file_formats.py --scan directory/
"""

import sys
from pathlib import Path

# pylint: disable=wrong-import-position
from format_utils import (  # noqa: E402
    validate_format,
    get_format_info,
    suggest_output_format,
    get_supported_formats_summary,
    is_supported_tiff_format,
    UnsupportedFormatError,
)


def validate_single_file(filepath: Path) -> None:
    """Validate and display information for a single file.

    Args:
        filepath: Path to file to validate
    """
    print(f"\n{'=' * 70}")
    print(f"File: {filepath.name}")
    print(f"Path: {filepath}")
    print('=' * 70)

    # Check if file exists
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return

    # Validate format
    try:
        validate_format(filepath, 'both', raise_error=True)
        print("✅ Format is supported")
    except UnsupportedFormatError as e:
        print(f"❌ {e}")
        return

    # Get detailed format information
    info = get_format_info(filepath)

    print("\nFormat Details:")
    print(f"  Extension: {info['extension']}")
    print("  Type: ", end='')

    if info['is_image']:
        print("Image", end='')
        if info['is_tiff']:
            print(" (TIFF)", end='')
        if info['is_luxury']:
            print(" [LUXURY GRADE]", end='')

    if info['is_video']:
        print("Video", end='')

    print()

    # Show recommendations
    if info['recommendations']:
        print("\n📋 Recommendations:")
        for rec in info['recommendations']:
            print(f"   • {rec}")

    # Suggest output format
    if info['is_image']:
        quality_output = suggest_output_format(filepath, preserve_quality=True)
        web_output = suggest_output_format(filepath, preserve_quality=False)

        print("\n💡 Output Format Suggestions:")
        print(f"   • For maximum quality: {quality_output}")
        print(f"   • For web/delivery: {web_output}")

    # Suggest appropriate pipeline
    print("\n🔧 Recommended Pipeline:")
    if info['is_tiff']:
        print("   → Luxury TIFF Batch Processor")
        print("     python luxury_tiff_batch_processor.py input/ output/ --preset signature")
    elif info['is_image']:
        print("   → Depth Pipeline or Lux Render Pipeline")
        print("     python depth_pipeline/pipeline.py --input", filepath.name, "--output enhanced/")
        print("     python lux_render_pipeline.py --input", filepath.name, "--out enhanced/")
    elif info['is_video']:
        print("   → Luxury Video Master Grader")
        print("     python luxury_video_master_grader.py --input", filepath.name, "--output graded.mov")


def scan_directory(directory: Path) -> None:
    """Scan directory and validate all files.

    Args:
        directory: Directory to scan
    """
    print(f"\n{'=' * 70}")
    print(f"Scanning Directory: {directory}")
    print('=' * 70)

    if not directory.is_dir():
        print(f"❌ Not a directory: {directory}")
        return

    # Collect all files
    all_files = list(directory.rglob('*'))
    files = [f for f in all_files if f.is_file()]

    print(f"\nFound {len(files)} files")

    # Categorize by format
    supported_images = []
    supported_videos = []
    unsupported = []

    for file in files:
        info = get_format_info(file)
        if info['is_image']:
            supported_images.append(file)
        elif info['is_video']:
            supported_videos.append(file)
        else:
            unsupported.append(file)

    # Display summary
    print("\n📊 Format Summary:")
    print(f"   ✅ Supported Images: {len(supported_images)}")
    print(f"   ✅ Supported Videos: {len(supported_videos)}")
    print(f"   ❌ Unsupported: {len(unsupported)}")

    # List supported images
    if supported_images:
        print(f"\n🖼️  Supported Image Files ({len(supported_images)}):")
        for file in sorted(supported_images)[:10]:  # Show first 10
            info = get_format_info(file)
            luxury_badge = " [LUXURY]" if info['is_luxury'] else ""
            print(f"   • {file.name}{luxury_badge}")

        if len(supported_images) > 10:
            print(f"   ... and {len(supported_images) - 10} more")

    # List supported videos
    if supported_videos:
        print(f"\n🎬 Supported Video Files ({len(supported_videos)}):")
        for file in sorted(supported_videos)[:10]:
            print(f"   • {file.name}")

        if len(supported_videos) > 10:
            print(f"   ... and {len(supported_videos) - 10} more")

    # List unsupported files
    if unsupported:
        print(f"\n⚠️  Unsupported Files ({len(unsupported)}):")
        # Group by extension
        by_ext = {}
        for file in unsupported:
            ext = file.suffix.lower()
            by_ext.setdefault(ext, []).append(file)

        for ext in sorted(by_ext.keys())[:5]:  # Show first 5 types
            files_with_ext = by_ext[ext]
            print(f"   • {ext}: {len(files_with_ext)} files")
            for file in files_with_ext[:3]:  # Show 3 examples
                print(f"       - {file.name}")

    # Batch processing recommendations
    if supported_images:
        print("\n🚀 Batch Processing Recommendations:")

        tiff_files = [f for f in supported_images if is_supported_tiff_format(f)]
        if tiff_files:
            print(f"   • {len(tiff_files)} TIFF files → Luxury TIFF Batch Processor")
            print(f"     python luxury_tiff_batch_processor.py {directory}/ output/ --recursive")

        other_images = [f for f in supported_images if not is_supported_tiff_format(f)]
        if other_images:
            print(f"   • {len(other_images)} other images → Depth Pipeline / Lux Render")
            print(f"     python depth_pipeline/pipeline.py --input {directory}/ --output enhanced/")

        if supported_videos:
            print(f"   • {len(supported_videos)} videos → Luxury Video Master Grader")
            print(f"     python luxury_video_master_grader.py --input {directory}/ --output graded/")


def show_format_summary() -> None:
    """Display summary of all supported formats."""
    summary = get_supported_formats_summary()

    print("\n" + "=" * 70)
    print("SUPPORTED FILE FORMATS")
    print("=" * 70)

    print("\n🖼️  Image Formats:")
    for ext in summary['image']:
        print(f"   • {ext}", end='')
        if ext in summary['luxury']:
            print(" [LUXURY GRADE]", end='')
        if ext in summary['tiff']:
            print(" [16-BIT CAPABLE]", end='')
        print()

    print("\n🎬 Video Formats:")
    for ext in summary['video']:
        print(f"   • {ext}")

    print("\n💎 Luxury Formats (Recommended for Professional Work):")
    for ext in summary['luxury']:
        print(f"   • {ext}")

    print("\n📖 For detailed format documentation, see:")
    print("   SUPPORTED_FILE_FORMATS.md")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate file formats for Transformation Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python examples/validate_file_formats.py render.jpg

  # Scan directory
  python examples/validate_file_formats.py --scan images/

  # Show all supported formats
  python examples/validate_file_formats.py --formats
        """
    )

    parser.add_argument(
        'path',
        nargs='?',
        type=Path,
        help='File or directory to validate'
    )

    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan directory and validate all files'
    )

    parser.add_argument(
        '--formats',
        action='store_true',
        help='Show all supported formats'
    )

    args = parser.parse_args()

    # Show formats if requested
    if args.formats:
        show_format_summary()
        return 0

    # Validate path is provided
    if not args.path:
        parser.print_help()
        print("\n❌ Error: Please provide a file or directory path")
        return 1

    # Resolve path
    path = args.path.resolve()

    # Scan directory or validate single file
    if args.scan or path.is_dir():
        scan_directory(path)
    else:
        validate_single_file(path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
