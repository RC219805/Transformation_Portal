#!/usr/bin/env python3
"""Stage 6 A/B sanity check: Kitchen scene only (quick validation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.stage6_ab_corrected_final import (
    BENCHMARK_SCENES,
    run_scene_ab,
    FORCE_DEVICE,
)

def main():
    """Run kitchen scene only as a sanity check."""
    kitchen = BENCHMARK_SCENES[0]
    
    print(f"Sanity check: {kitchen.name}")
    print(f"Input: {kitchen.input_path}")
    print(f"Exists: {kitchen.input_path.exists()}")
    print(f"Device: {FORCE_DEVICE}")
    print("")
    
    if not kitchen.input_path.exists():
        print("ERROR: Input file not found")
        return 1
    
    result = run_scene_ab(kitchen, FORCE_DEVICE)
    
    print("")
    print("=== RESULT ===")
    print(f"Status: {result.get('status')}")
    print(f"Improved: {result.get('scene_improved')}")
    print(f"Improvements: {result.get('improvements')}")
    print(f"Regressions: {result.get('regressions')}")
    print("")
    
    if result.get("status") == "success":
        print("✓ Sanity check PASSED")
        return 0
    else:
        print("✗ Sanity check FAILED")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
