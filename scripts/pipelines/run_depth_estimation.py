#!/usr/bin/env python3
"""
CLI wrapper for depth estimation compatible with APEX V2 batch processing.
Delegates to depth_pro_export.py and normalizes output naming.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Depth estimation CLI wrapper for APEX V2")
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output depth map path")
    parser.add_argument("--backend", default="depth_pro", help="Backend (only depth_pro supported)")
    parser.add_argument("--device", default="mps", help="Device (mps, cpu)")

    args = parser.parse_args()

    if args.backend != "depth_pro":
        print(f"Warning: Backend '{args.backend}' not supported, using depth_pro", file=sys.stderr)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build command for depth_pro_export.py (resolve path to be CWD-independent)
    script_dir = Path(__file__).resolve().parent
    export_script = script_dir / "depth_pro_export.py"
    cmd = [sys.executable, str(export_script), str(args.input)]

    # Add device flag
    if args.device.lower() == "cpu":
        cmd.append("--cpu")

    # Run depth estimation
    print(f"Running depth estimation: {args.input.name}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Error: Depth estimation failed for {args.input}", file=sys.stderr)
        sys.exit(1)

    # Find the generated depth map (depth_pro_export saves as *_depthpro_depth16.png)
    stem = args.input.stem
    generated_depth = args.input.with_name(f"{stem}_depthpro_depth16.png")

    if not generated_depth.exists():
        print(f"Error: Expected depth map not found: {generated_depth}", file=sys.stderr)
        sys.exit(1)

    # Move/rename to expected location
    shutil.move(str(generated_depth), str(args.output))

    # Also move provenance and npy files if they exist
    npy_src = args.input.with_name(f"{stem}_depthpro_depth.npy")
    json_src = args.input.with_name(f"{stem}_depthpro_provenance.json")

    if npy_src.exists():
        npy_dst = args.output.with_suffix(".npy")
        shutil.move(str(npy_src), str(npy_dst))

    if json_src.exists():
        json_dst = args.output.with_suffix(".json")
        shutil.move(str(json_src), str(json_dst))

    print(f"✓ Depth map saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
