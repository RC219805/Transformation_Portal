#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for example file paths.

This script validates that all file paths referenced in example files
actually exist in the repository after restructuring.
"""

from pathlib import Path


def verify_example_paths():
    """Verify all paths referenced in examples exist."""
    repo_root = Path(__file__).resolve().parents[2]

    # Paths referenced in examples/vfx_extension_example.py
    paths_to_verify = [
        "assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube",
        "assets/luts/location_aesthetic/Mediterranean/Spanish_Colonial_Warm_HDR.cube",
        "assets/luts/film_emulation/Kodak/Kodak_2393_D55_HDR.cube",
        "assets/luts/film_emulation/FilmConvert/FilmConvert_Nitrate_HDR.cube",
    ]

    all_valid = True
    print("Verifying example file paths...")
    print("=" * 60)

    for path_str in paths_to_verify:
        full_path = repo_root / path_str
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {path_str}")
        if exists:
            size_kb = full_path.stat().st_size / 1024
            print(f"   Size: {size_kb:.1f} KB")
        else:
            all_valid = False

    print("=" * 60)
    if all_valid:
        print("✓ All example paths verified successfully!")
        return 0

    print("✗ Some example paths are invalid!")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(verify_example_paths())
