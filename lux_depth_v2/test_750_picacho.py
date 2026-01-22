#!/usr/bin/env python3
"""
750 Picacho TIFF Processing Test Script
Lux Depth V2 Pipeline
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test Lux Depth V2 pipeline on 750 Picacho TIFF files")
    parser.add_argument(
        "--preset",
        default="interior_luxury",
        help="Processing preset (default: interior_luxury)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: lux_depth_v2/test_outputs/750_picacho)",
    )
    parser.add_argument(
        "--edge-refinement",
        action="store_true",
        help="Enable edge refinement (opt-in feature)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - check dependencies and files only",
    )

    args = parser.parse_args()

    # Configuration
    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / "projects" / "750_picacho_lane" / "Final_Production_UltraQuality"
    output_dir = args.output_dir or (repo_root / "lux_depth_v2" / "test_outputs" / "750_picacho")

    print("=" * 60)
    print("750 Picacho TIFF Processing Test")
    print("Lux Depth V2 Pipeline")
    print("=" * 60)
    print()

    # Step 1: Pre-flight checks
    print("Step 1: Pre-flight Checks")
    print("-" * 40)

    if not repo_root.exists():
        print(f"✗ Repository root not found: {repo_root}")
        return 1
    print(f"✓ Repository root: {repo_root}")

    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        return 1
    print(f"✓ Input directory exists: {input_dir}")

    tiff_files = list(input_dir.glob("*.tif"))
    print(f"✓ Found {len(tiff_files)} TIFF files")

    if len(tiff_files) != 6:
        print(f"⚠ Warning: Expected 6 TIFF files, found {len(tiff_files)}")

    for tiff_file in tiff_files:
        size_mb = tiff_file.stat().st_size / (1024 * 1024)
        print(f"  - {tiff_file.name} ({size_mb:.1f} MB)")

    # Step 2: Dependency check
    print()
    print("Step 2: Dependency Check")
    print("-" * 40)

    deps_ok = True
    required_deps = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "tifffile": "tifffile",
        "torch": "torch",
        "tqdm": "tqdm",
    }

    for module_name, package_name in required_deps.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            deps_ok = False

    if not deps_ok:
        print()
        print("Missing dependencies detected!")
        print("Install with:")
        print("  pip install numpy opencv-python tifffile torch tqdm")
        print("Or:")
        print("  pip install -r lux_depth_v2/requirements-repo.txt")
        return 1

    # Check if lux_depth_v2 module is importable
    print()
    try:
        from lux_depth_v2.pipeline import LuxPipelineV2
        from lux_depth_v2.config import PipelineConfig

        print("✓ lux_depth_v2 module importable")
    except ImportError as e:
        print(f"✗ lux_depth_v2 module import failed: {e}")
        return 1

    # Step 3: Prepare output directory
    print()
    print("Step 3: Prepare Output Directory")
    print("-" * 40)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    if args.dry_run:
        print()
        print("Dry run complete - all checks passed!")
        print("Remove --dry-run to execute processing")
        return 0

    # Step 4: Process TIFF files
    print()
    print("Step 4: Process TIFF Files")
    print("-" * 40)
    print(f"Configuration:")
    print(f"  Preset: {args.preset}")
    print(f"  Device: {args.device}")
    print(f"  Edge Refinement: {'Enabled' if args.edge_refinement else 'Disabled'}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print()
    print("Starting processing...")
    print()

    start_time = time.time()

    # Build CLI command
    cli_args = [
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--preset",
        args.preset,
        "--device",
        args.device,
        "--upscaler-backend",
        "torch",
        "--file-pattern",
        "*.tif",
    ]

    if args.edge_refinement:
        cli_args.extend(["--edge-refinement", "--refinement-preset", "balanced"])

    # Import and run CLI
    try:
        from lux_depth_v2.cli import main as cli_main

        # Save original sys.argv
        orig_argv = sys.argv
        # Set new argv for CLI
        sys.argv = ["lux_depth_v2.cli"] + cli_args
        # Run CLI
        cli_main()
        # Restore original argv
        sys.argv = orig_argv
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    end_time = time.time()
    duration = end_time - start_time

    print()
    print(f"✓ Processing complete")
    print(f"Total time: {duration:.1f} seconds ({duration / 60:.2f} minutes)")

    # Step 5: Validate outputs
    print()
    print("Step 5: Validate Outputs")
    print("-" * 40)

    master_tiffs = list(output_dir.glob("*_master16.tif"))
    print(f"✓ Generated {len(master_tiffs)} master TIFF files")

    if len(master_tiffs) != len(tiff_files):
        print(f"⚠ Warning: Expected {len(tiff_files)} outputs, got {len(master_tiffs)}")

    # List all output files
    print()
    print("Output files:")
    output_files = list(output_dir.glob("*"))
    output_files.sort()

    for file in output_files:
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {size_mb:6.1f} MB  {file.name}")

    # Verify 16-bit TIFF format
    print()
    print("Verifying 16-bit TIFF format...")

    import tifffile

    all_ok = True

    for tiff_file in master_tiffs:
        try:
            img = tifffile.imread(tiff_file)
            is_16bit = str(img.dtype) == "uint16"
            status = "✓" if is_16bit else "✗"
            print(f"  {status} {tiff_file.name}: {img.dtype} {img.shape}")
            if not is_16bit:
                all_ok = False
        except Exception as e:
            print(f"  ✗ {tiff_file.name}: Error - {e}")
            all_ok = False

    if not all_ok:
        print("✗ 16-bit verification failed")
        return 1

    # Step 6: Generate summary report
    print()
    print("Step 6: Generate Summary Report")
    print("-" * 40)

    summary_data = {
        "date": datetime.now().isoformat(),
        "pipeline": "Lux Depth V2",
        "preset": args.preset,
        "device": args.device,
        "edge_refinement": args.edge_refinement,
        "input_files": len(tiff_files),
        "output_files": len(master_tiffs),
        "processing_time_seconds": duration,
        "processing_time_minutes": duration / 60,
        "average_time_per_file": duration / len(tiff_files) if tiff_files else 0,
        "output_directory": str(output_dir),
        "input_files_list": [f.name for f in tiff_files],
        "output_files_list": [f.name for f in output_files if f.is_file()],
    }

    summary_file = output_dir / "TEST_SUMMARY.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"✓ Summary report saved to: {summary_file}")

    # Also create text summary
    summary_txt = output_dir / "TEST_SUMMARY.txt"
    with open(summary_txt, "w") as f:
        f.write("750 Picacho TIFF Processing Test - Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {summary_data['date']}\n")
        f.write(f"Pipeline: {summary_data['pipeline']}\n")
        f.write(f"Preset: {summary_data['preset']}\n")
        f.write(f"Device: {summary_data['device']}\n")
        f.write(f"Edge Refinement: {summary_data['edge_refinement']}\n\n")
        f.write("Results\n")
        f.write("-" * 40 + "\n")
        f.write(f"Input Files: {summary_data['input_files']}\n")
        f.write(f"Output Files: {summary_data['output_files']}\n")
        f.write(f"Processing Time: {duration:.1f} seconds ({duration / 60:.2f} minutes)\n")
        f.write(f"Average Time: {summary_data['average_time_per_file']:.1f} seconds/file\n\n")
        f.write(f"Output Directory: {summary_data['output_directory']}\n\n")
        f.write("Files Processed:\n")
        for fname in summary_data["input_files_list"]:
            f.write(f"  - {fname}\n")
        f.write("\nOutput Files Generated:\n")
        for fname in summary_data["output_files_list"]:
            file_path = output_dir / fname
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                f.write(f"  - {size_mb:6.1f} MB  {fname}\n")

    # Final summary
    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Input files:     {len(tiff_files)}")
    print(f"  Output files:    {len(master_tiffs)}")
    print(f"  Processing time: {duration:.1f} seconds")
    print(f"  Output location: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Review output files in: {output_dir}")
    print(f"  2. Check summary reports:")
    print(f"     - {summary_file}")
    print(f"     - {summary_txt}")
    print("  3. Visually inspect output quality")
    print("  4. Compare input vs output side-by-side")
    print()
    print("✓ All tests passed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
