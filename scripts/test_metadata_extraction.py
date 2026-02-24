#!/usr/bin/env python3
"""Test Phase 3.7 metadata extraction capabilities on image directories.

This script provides commands for testing the Phase 3.7 metadata extraction
capabilities, including:
- Provenance capture (complete EXIF + toolchain + environment metadata)
- Sidecar generation (deterministic JSON output)
- Schema validation (schema compliance verification)
- Batch processing (multiple images)

Requirements:
- exiftool must be installed (brew install exiftool / apt-get install libimage-exiftool-perl)
- pydantic >= 2.0

Usage:
    # Single image extraction
    python scripts/test_metadata_extraction.py extract /path/to/image.tif

    # Batch extraction (entire directory)
    python scripts/test_metadata_extraction.py extract-batch /path/to/images/ --output ./output_sidecars/

    # Validate existing sidecar
    python scripts/test_metadata_extraction.py validate /path/to/provenance.json

    # Check system readiness
    python scripts/test_metadata_extraction.py check-system

    # Summary of extraction results
    python scripts/test_metadata_extraction.py summarize /path/to/sidecars/

Exit Codes:
    0: Success
    1: Schema validation failed
    2: 8-bit conversion detected
    3: Gamma correction detected
    4: Schema drift detected
    5: Other failure (e.g., exiftool not found)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Exit Codes (matching ingest contract)
# =============================================================================

EXIT_SUCCESS = 0
EXIT_SCHEMA_VALIDATION_FAILED = 1
EXIT_8BIT_CONVERSION = 2
EXIT_GAMMA_VIOLATION = 3
EXIT_SCHEMA_DRIFT = 4
EXIT_OTHER_FAILURE = 5


# =============================================================================
# Supported Image Extensions
# =============================================================================

SUPPORTED_EXTENSIONS = {
    # RAW formats
    ".cr2", ".cr3",   # Canon
    ".nef", ".nrw",   # Nikon
    ".arw", ".srf",   # Sony
    ".dng",           # Adobe DNG
    ".raf",           # Fujifilm
    ".orf",           # Olympus
    ".rw2",           # Panasonic
    ".pef",           # Pentax
    ".srw",           # Samsung

    # TIFF formats
    ".tif", ".tiff",

    # Common formats
    ".jpg", ".jpeg",
    ".png",
    ".heic", ".heif",
}


# =============================================================================
# Helper Functions
# =============================================================================

def find_images(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all supported image files in directory.

    Args:
        directory: Directory to search
        recursive: If True, search subdirectories

    Returns:
        List of image file paths
    """
    images = []
    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(file_path)

    return sorted(images)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# Command: check-system
# =============================================================================

def cmd_check_system(args: argparse.Namespace) -> int:
    """Check system readiness for metadata extraction.

    Verifies:
    - exiftool availability and version
    - pydantic installation
    - git availability (optional)
    - rawpy availability (optional)
    """
    print("=" * 70)
    print("Phase 3.7 Metadata Extraction - System Check")
    print("=" * 70)
    print()

    all_ok = True

    # Check exiftool
    print("Checking exiftool...")
    try:
        from transformation_portal.ingest.provenance import (
            _check_exiftool_available,
            _get_exiftool_version,
        )

        if _check_exiftool_available():
            version = _get_exiftool_version()
            print(f"  ✅ exiftool found: version {version}")
        else:
            print("  ❌ exiftool not found")
            print("     Install with: brew install exiftool (macOS)")
            print("                   apt-get install libimage-exiftool-perl (Linux)")
            all_ok = False
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        all_ok = False

    # Check pydantic
    print("\nChecking pydantic...")
    try:
        import pydantic
        print(f"  ✅ pydantic found: version {pydantic.__version__}")
    except ImportError:
        print("  ❌ pydantic not found")
        print("     Install with: pip install pydantic>=2.0")
        all_ok = False

    # Check git (optional)
    print("\nChecking git (optional)...")
    try:
        import subprocess
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ git found: {version}")
        else:
            print("  ⚠️  git not available (git commit SHA will not be captured)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠️  git not available (git commit SHA will not be captured)")

    # Check rawpy (optional)
    print("\nChecking rawpy (optional, for RAW file support)...")
    try:
        import rawpy
        print(f"  ✅ rawpy found: version {rawpy.version.version}")
        if hasattr(rawpy, "libraw_version"):
            print(f"      libraw version: {rawpy.libraw_version}")
    except ImportError:
        print("  ⚠️  rawpy not found (RAW file reading may be limited)")
        print("     Install with: pip install rawpy")

    # Check ingest module
    print("\nChecking transformation_portal.ingest module...")
    try:
        from transformation_portal.ingest import (  # noqa: F401
            capture_provenance,
            validate_schema,
            write_sidecar,
        )
        print("  ✅ ingest module available")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        all_ok = False

    # Summary
    print()
    print("=" * 70)
    if all_ok:
        print("✅ System is ready for Phase 3.7 metadata extraction")
        return EXIT_SUCCESS
    else:
        print("❌ System has missing dependencies")
        return EXIT_OTHER_FAILURE


# =============================================================================
# Command: extract
# =============================================================================

def cmd_extract(args: argparse.Namespace) -> int:
    """Extract metadata from a single image file.

    Captures:
    - Complete EXIF metadata via exiftool
    - File integrity (SHA256, size, MIME type)
    - Toolchain versions
    - Host environment
    - Pipeline configuration fingerprint
    """
    from transformation_portal.ingest import capture_provenance, write_sidecar

    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return EXIT_OTHER_FAILURE

    print(f"📷 Extracting metadata from: {image_path.name}")
    print(f"   Path: {image_path}")
    print(f"   Size: {format_size(image_path.stat().st_size)}")
    print()

    try:
        start_time = time.time()

        # Capture provenance
        sidecar = capture_provenance(
            input_path=image_path,
            cli_args=sys.argv[1:],
            config_dict={"mode": "test_extraction", "phase": "3.7"},
            preset=args.preset if hasattr(args, "preset") else None,
        )

        elapsed = time.time() - start_time

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = image_path.with_name(f"{image_path.stem}_provenance.json")

        # Write sidecar
        write_sidecar(sidecar, output_path, fsync=args.fsync)

        # Print summary
        print("✅ Metadata extraction complete")
        print()
        print("📊 Extraction Summary:")
        print(f"   Schema version:    {sidecar.schema_version}")
        print(f"   File SHA256:       {sidecar.file_integrity.sha256[:16]}...")
        print(f"   File size:         {format_size(sidecar.file_integrity.size_bytes)}")
        print(f"   MIME type:         {sidecar.file_integrity.mime_type or 'unknown'}")
        print()
        print("📸 EXIF Summary:")
        exif = sidecar.exif
        if exif.camera_make:
            print(f"   Camera:            {exif.camera_make} {exif.camera_model or ''}")
        if exif.lens_model:
            print(f"   Lens:              {exif.lens_model}")
        if exif.iso:
            print(f"   ISO:               {exif.iso}")
        if exif.aperture:
            print(f"   Aperture:          f/{exif.aperture}")
        if exif.shutter_speed:
            print(f"   Shutter:           {exif.shutter_speed}")
        if exif.focal_length:
            print(f"   Focal length:      {exif.focal_length}mm")
        if exif.width and exif.height:
            print(f"   Dimensions:        {exif.width} x {exif.height}")
        if exif.bit_depth:
            print(f"   Bit depth:         {exif.bit_depth} bits")
        if exif.datetime_original:
            print(f"   Date taken:        {exif.datetime_original}")
        if exif.gps_latitude and exif.gps_longitude:
            print("   GPS present:       yes")
        print(f"   Total EXIF tags:   {len(exif.all_tags)}")
        print()
        print("🔧 Toolchain:")
        for tool in sidecar.toolchain:
            print(f"   {tool.name}: {tool.version}")
        print()
        print("💻 Host Environment:")
        print(f"   Hostname:          {sidecar.host.hostname}")
        print(f"   OS:                {sidecar.host.os} {sidecar.host.os_version}")
        print(f"   Architecture:      {sidecar.host.arch}")
        print(f"   Python:            {sidecar.host.python_version}")
        if sidecar.git_commit:
            print(f"   Git commit:        {sidecar.git_commit[:12]}")
        print()
        print(f"⏱️  Extraction time:   {elapsed:.2f}s")
        print(f"📄 Sidecar written:   {output_path}")

        return EXIT_SUCCESS

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return EXIT_OTHER_FAILURE


# =============================================================================
# Command: extract-batch
# =============================================================================

def cmd_extract_batch(args: argparse.Namespace) -> int:
    """Extract metadata from all images in a directory."""
    from transformation_portal.ingest import capture_provenance, write_sidecar
    from transformation_portal.ingest.provenance import ExiftoolNotFoundError

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return EXIT_OTHER_FAILURE

    if not input_dir.is_dir():
        print(f"❌ Not a directory: {input_dir}")
        return EXIT_OTHER_FAILURE

    # Determine output directory
    output_dir = Path(args.output) if args.output else input_dir / "provenance_sidecars"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(input_dir, recursive=args.recursive)

    if not images:
        print(f"⚠️  No supported images found in: {input_dir}")
        print(f"   Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return EXIT_SUCCESS

    print("=" * 70)
    print("Phase 3.7 Metadata Extraction - Batch Mode")
    print("=" * 70)
    print()
    print(f"📁 Input directory:  {input_dir}")
    print(f"📄 Output directory: {output_dir}")
    print(f"🖼️  Images found:     {len(images)}")
    print()

    # Process images
    results: Dict[str, Tuple[bool, str, float]] = {}
    total_start = time.time()

    for i, image_path in enumerate(images, 1):
        relative_path = image_path.relative_to(input_dir)
        print(f"[{i}/{len(images)}] Processing: {relative_path}")

        try:
            start_time = time.time()

            # Capture provenance
            sidecar = capture_provenance(
                input_path=image_path,
                cli_args=sys.argv[1:],
                config_dict={"mode": "batch_extraction", "phase": "3.7"},
            )

            # Create output path preserving directory structure
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_path = output_subdir / f"{image_path.stem}_provenance.json"

            # Write sidecar
            write_sidecar(sidecar, output_path, fsync=args.fsync)

            elapsed = time.time() - start_time
            results[str(relative_path)] = (True, str(output_path), elapsed)
            print(f"         ✅ Done ({elapsed:.2f}s)")

        except ExiftoolNotFoundError as e:
            results[str(relative_path)] = (False, str(e), 0)
            print("         ❌ exiftool not found")
            if args.fail_fast:
                break

        except Exception as e:
            results[str(relative_path)] = (False, str(e), 0)
            print(f"         ❌ Error: {e}")
            if args.fail_fast:
                break

    total_elapsed = time.time() - total_start

    # Summary
    print()
    print("=" * 70)
    print("Batch Extraction Summary")
    print("=" * 70)

    success_count = sum(1 for success, _, _ in results.values() if success)
    failure_count = len(results) - success_count

    print(f"Total images:     {len(images)}")
    print(f"Processed:        {len(results)}")
    print(f"Successful:       {success_count}")
    print(f"Failed:           {failure_count}")
    print(f"Total time:       {total_elapsed:.2f}s")
    if success_count > 0:
        avg_time = sum(t for _, _, t in results.values() if t > 0) / success_count
        print(f"Average per image: {avg_time:.2f}s")
    print(f"Output directory: {output_dir}")

    if failure_count > 0:
        print()
        print("Failed images:")
        for path, (success, error, _) in results.items():
            if not success:
                print(f"  ❌ {path}: {error}")
        return EXIT_OTHER_FAILURE

    return EXIT_SUCCESS


# =============================================================================
# Command: validate
# =============================================================================

def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a provenance sidecar JSON file."""
    from transformation_portal.ingest import validate_schema
    from transformation_portal.ingest.validator import SchemaValidationError

    sidecar_path = Path(args.sidecar_path)

    if not sidecar_path.exists():
        print(f"❌ Sidecar not found: {sidecar_path}")
        return EXIT_OTHER_FAILURE

    print(f"🔍 Validating: {sidecar_path}")
    print()

    try:
        errors = validate_schema(
            sidecar_path,
            schema_type="provenance",
            strict_mode=args.strict,
        )

        if errors:
            print("❌ Validation failed:")
            for error in errors:
                print(f"   - {error}")

            # Determine exit code based on error type
            for error in errors:
                if "drift" in error.lower():
                    return EXIT_SCHEMA_DRIFT
                elif "schema version" in error.lower():
                    return EXIT_SCHEMA_VALIDATION_FAILED

            return EXIT_SCHEMA_VALIDATION_FAILED
        else:
            print("✅ Sidecar is valid")

            # Optionally show summary
            if args.verbose:
                with open(sidecar_path) as f:
                    data = json.load(f)
                print()
                print("📊 Sidecar Summary:")
                print(f"   Schema version: {data.get('schema_version')}")
                print(f"   Run ID:         {data.get('run_id')}")
                fi = data.get("file_integrity", {})
                print(f"   File SHA256:    {fi.get('sha256', 'N/A')[:16]}...")
                print(f"   File size:      {format_size(fi.get('size_bytes', 0))}")

            return EXIT_SUCCESS

    except SchemaValidationError as e:
        print("❌ Validation failed:")
        for error in e.errors:
            print(f"   - {error}")
        return EXIT_SCHEMA_VALIDATION_FAILED

    except Exception as e:
        print(f"❌ Validation error: {e}")
        return EXIT_OTHER_FAILURE


# =============================================================================
# Command: summarize
# =============================================================================

def cmd_summarize(args: argparse.Namespace) -> int:
    """Summarize metadata from multiple sidecar files."""
    sidecar_dir = Path(args.sidecar_dir)

    if not sidecar_dir.exists():
        print(f"❌ Directory not found: {sidecar_dir}")
        return EXIT_OTHER_FAILURE

    # Find sidecar files
    sidecars = list(sidecar_dir.rglob("*_provenance.json"))

    if not sidecars:
        print(f"⚠️  No sidecar files found in: {sidecar_dir}")
        return EXIT_SUCCESS

    print("=" * 70)
    print("Phase 3.7 Metadata Extraction - Summary")
    print("=" * 70)
    print()
    print(f"📁 Directory: {sidecar_dir}")
    print(f"📄 Sidecars found: {len(sidecars)}")
    print()

    # Aggregate statistics
    cameras: Dict[str, int] = {}
    total_size = 0
    total_tags = 0
    dimensions: List[Tuple[int, int]] = []
    gps_count = 0

    for sidecar_path in sidecars:
        try:
            with open(sidecar_path) as f:
                data = json.load(f)

            # File integrity
            fi = data.get("file_integrity", {})
            total_size += fi.get("size_bytes", 0)

            # EXIF
            exif = data.get("exif", {})
            all_tags = exif.get("all_tags", {})
            total_tags += len(all_tags)

            # Camera
            make = exif.get("camera_make", "Unknown")
            model = exif.get("camera_model", "")
            camera = f"{make} {model}".strip()
            cameras[camera] = cameras.get(camera, 0) + 1

            # Dimensions
            width = exif.get("width")
            height = exif.get("height")
            if width and height:
                dimensions.append((width, height))

            # GPS
            if exif.get("gps_latitude") and exif.get("gps_longitude"):
                gps_count += 1

        except Exception as e:
            print(f"⚠️  Error reading {sidecar_path.name}: {e}")

    # Print summary
    print("📊 Aggregate Statistics:")
    print(f"   Total file size:    {format_size(total_size)}")
    print(f"   Total EXIF tags:    {total_tags}")
    print(f"   Average tags/image: {total_tags / len(sidecars):.0f}")
    print(f"   Images with GPS:    {gps_count} ({gps_count/len(sidecars)*100:.1f}%)")
    print()

    print("📷 Cameras:")
    for camera, count in sorted(cameras.items(), key=lambda x: -x[1]):
        print(f"   {camera}: {count}")
    print()

    if dimensions:
        print("📐 Dimensions:")
        unique_dims = set(dimensions)
        for w, h in sorted(unique_dims, key=lambda x: -x[0]*x[1]):
            count = dimensions.count((w, h))
            mp = (w * h) / 1_000_000
            print(f"   {w} x {h} ({mp:.1f} MP): {count}")

    return EXIT_SUCCESS


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Phase 3.7 metadata extraction capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check-system command
    parser_check = subparsers.add_parser(
        "check-system",
        help="Check system readiness for metadata extraction",
    )
    parser_check.set_defaults(func=cmd_check_system)

    # extract command
    parser_extract = subparsers.add_parser(
        "extract",
        help="Extract metadata from a single image",
    )
    parser_extract.add_argument(
        "image_path",
        help="Path to image file",
    )
    parser_extract.add_argument(
        "-o", "--output",
        help="Output path for sidecar JSON (default: <image>_provenance.json)",
    )
    parser_extract.add_argument(
        "--preset",
        help="Preset name to record in provenance",
    )
    parser_extract.add_argument(
        "--fsync",
        action="store_true",
        help="Use fsync for durable writes",
    )
    parser_extract.set_defaults(func=cmd_extract)

    # extract-batch command
    parser_batch = subparsers.add_parser(
        "extract-batch",
        help="Extract metadata from all images in a directory",
    )
    parser_batch.add_argument(
        "input_dir",
        help="Directory containing images",
    )
    parser_batch.add_argument(
        "-o", "--output",
        help="Output directory for sidecars (default: <input_dir>/provenance_sidecars/)",
    )
    parser_batch.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search subdirectories (default: True)",
    )
    parser_batch.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only process images in the top-level directory",
    )
    parser_batch.add_argument(
        "--fsync",
        action="store_true",
        help="Use fsync for durable writes",
    )
    parser_batch.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error",
    )
    parser_batch.set_defaults(func=cmd_extract_batch)

    # validate command
    parser_validate = subparsers.add_parser(
        "validate",
        help="Validate a provenance sidecar file",
    )
    parser_validate.add_argument(
        "sidecar_path",
        help="Path to sidecar JSON file",
    )
    parser_validate.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enable strict mode (fail on unknown fields)",
    )
    parser_validate.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Disable strict mode",
    )
    parser_validate.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show sidecar summary after validation",
    )
    parser_validate.set_defaults(func=cmd_validate)

    # summarize command
    parser_summary = subparsers.add_parser(
        "summarize",
        help="Summarize metadata from multiple sidecar files",
    )
    parser_summary.add_argument(
        "sidecar_dir",
        help="Directory containing sidecar JSON files",
    )
    parser_summary.set_defaults(func=cmd_summarize)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
