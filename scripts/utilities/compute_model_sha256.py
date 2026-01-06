#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute SHA256 hashes for downloaded models and generate registry updates.

This utility helps verify model integrity by computing SHA256 checksums
for ONNX models and generating code snippets to update the model registry.

Usage:
    # Compute SHA256 for a single model
    python scripts/utilities/compute_model_sha256.py weights/efficientsam/efficientsam_ti_vit_s.onnx

    # Compute for all models in a directory
    python scripts/utilities/compute_model_sha256.py weights/efficientsam/

    # Generate registry update code
    python scripts/utilities/compute_model_sha256.py weights/efficientsam/ --generate-registry

Example Output:
    ✓ efficientsam_ti_vit_s.onnx
      SHA256: a1b2c3d4...
      Size: 41.2 MB

    Registry update:
    "efficientsam_ti_vit_s": {
        ...
        "sha256": "a1b2c3d4...",
    }

Author: Transformation Portal Team
License: MIT
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Dict


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Parameters
    ----------
    file_path : Path
        Path to file.

    Returns
    -------
    str
        SHA256 hash as hexadecimal string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 8KB chunks for memory efficiency
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_model_files(path: Path) -> List[Path]:
    """
    Find all ONNX model files in path.

    Parameters
    ----------
    path : Path
        File or directory to search.

    Returns
    -------
    List[Path]
        List of ONNX model file paths.
    """
    if path.is_file():
        if path.suffix == ".onnx":
            return [path]
        else:
            print(f"Warning: {path} is not an ONNX file", file=sys.stderr)
            return []
    elif path.is_dir():
        return sorted(path.glob("*.onnx"))
    else:
        print(f"Error: {path} does not exist", file=sys.stderr)
        return []


def generate_registry_snippet(model_name: str, sha256: str, size_mb: float, filename: str) -> str:
    """
    Generate Python code snippet for registry update.

    Parameters
    ----------
    model_name : str
        Model name (without .onnx extension) - used as registry key.
    sha256 : str
        SHA256 hash.
    size_mb : float
        File size in MB.
    filename : str
        Actual filename (with .onnx extension).

    Returns
    -------
    str
        Python code snippet.

    Notes
    -----
    This function generates only the sha256 and size_mb fields.
    The URL should be preserved from the existing registry or set manually,
    as the registry key (model_name) often differs from the filename.
    """
    return f'''    "{model_name}": {{
        # URL: Preserve existing URL or set manually - filename is {filename}
        "sha256": "{sha256}",
        "size_mb": {size_mb:.1f},
    }},'''


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute SHA256 hashes for ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to ONNX file or directory containing models",
    )
    parser.add_argument(
        "--generate-registry",
        action="store_true",
        help="Generate Python code snippets for registry updates",
    )
    parser.add_argument(
        "--verify",
        type=str,
        metavar="SHA256",
        help="Verify file matches expected SHA256 hash",
    )

    args = parser.parse_args()

    # Find model files
    model_files = find_model_files(args.path)

    if not model_files:
        print("No ONNX files found", file=sys.stderr)
        return 1

    # Compute hashes
    results: Dict[str, Dict] = {}

    print(f"Computing SHA256 for {len(model_files)} model(s)...\n")

    for model_file in model_files:
        try:
            sha256 = compute_sha256(model_file)
            size_mb = model_file.stat().st_size / (1024 * 1024)
            model_name = model_file.stem  # Filename without extension

            results[model_name] = {
                "sha256": sha256,
                "size_mb": size_mb,
                "path": model_file,
            }

            # Display result
            print(f"✓ {model_file.name}")
            print(f"  SHA256: {sha256}")
            print(f"  Size: {size_mb:.1f} MB")

            # Verify if requested
            if args.verify:
                if sha256 == args.verify:
                    print(f"  ✓ Hash verified!")
                else:
                    print(f"  ✗ Hash mismatch! Expected: {args.verify}")
                    return 1

            print()

        except Exception as e:
            print(f"✗ Error processing {model_file}: {e}", file=sys.stderr)
            return 1

    # Generate registry updates if requested
    if args.generate_registry:
        print("\n" + "=" * 70)
        print("Registry Update Code (copy to lux_depth_v2/backends/model_cache.py):")
        print("=" * 70 + "\n")

        for model_name, info in results.items():
            print(generate_registry_snippet(model_name, info["sha256"], info["size_mb"], info["path"].name))

        print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
