#!/usr/bin/env python3
"""
Pixel-Level Validation: RAW Baseline vs Source
Bypasses viewer interpretation to confirm pixel-perfect preservation.
"""

import sys
from pathlib import Path
import numpy as np
from tifffile import imread
import hashlib


def compute_sha256(array):
    """Compute SHA256 hash of numpy array."""
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_tiffs(source_path, baseline_path):
    """
    Compare source and baseline TIFFs at pixel level.
    
    Returns:
        dict: Comprehensive comparison results
    """
    print("=" * 80)
    print("PIXEL-LEVEL VALIDATION: RAW Baseline vs Source")
    print("=" * 80)
    print()
    
    # Load TIFFs
    print(f"Loading source: {source_path}")
    source = imread(source_path)
    print(f"  Shape: {source.shape}, dtype: {source.dtype}")
    
    print(f"Loading baseline: {baseline_path}")
    baseline = imread(baseline_path)
    print(f"  Shape: {baseline.shape}, dtype: {baseline.dtype}")
    print()
    
    # Geometry check
    print("1. GEOMETRY CHECK")
    print("-" * 80)
    geometry_match = source.shape == baseline.shape
    print(f"  Source shape:   {source.shape}")
    print(f"  Baseline shape: {baseline.shape}")
    print(f"  Match: {'✅ YES' if geometry_match else '❌ NO'}")
    print()
    
    if not geometry_match:
        print("❌ GEOMETRY MISMATCH - baseline is invalid")
        return {"status": "FAILED", "reason": "geometry_mismatch"}
    
    # Pixel-perfect comparison
    print("2. PIXEL-PERFECT COMPARISON")
    print("-" * 80)
    pixel_match = np.array_equal(source, baseline)
    print(f"  Arrays identical: {'✅ YES' if pixel_match else '❌ NO'}")
    
    if pixel_match:
        print("  ✅ PIXEL-PERFECT MATCH - baseline is valid")
        print()
        return {"status": "PASSED", "pixel_perfect": True}
    
    # Detailed difference analysis
    print("  ⚠️  Pixels differ - analyzing...")
    print()
    
    diff = np.abs(source.astype(np.float32) - baseline.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    nonzero_diff = np.count_nonzero(diff)
    total_pixels = diff.size
    
    print("3. DIFFERENCE ANALYSIS")
    print("-" * 80)
    print(f"  Max difference:  {max_diff}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Different pixels: {nonzero_diff:,} / {total_pixels:,} ({100*nonzero_diff/total_pixels:.4f}%)")
    print()
    
    # Sample pixel comparison
    print("4. SAMPLE PIXEL VALUES (5 locations)")
    print("-" * 80)
    h, w = source.shape[:2]
    coords = [
        (0, 0, "Top-left"),
        (h//2, w//2, "Center"),
        (h-1, w-1, "Bottom-right"),
        (h//4, w//4, "Quarter"),
        (3*h//4, 3*w//4, "Three-quarter"),
    ]
    
    for y, x, label in coords:
        src_val = source[y, x]
        base_val = baseline[y, x]
        match = np.array_equal(src_val, base_val)
        print(f"  [{y:4d}, {x:4d}] {label:15s}: ", end="")
        print(f"Source={src_val} Baseline={base_val} {'✅' if match else '❌'}")
    print()
    
    # Hash comparison
    print("5. SHA256 HASH COMPARISON")
    print("-" * 80)
    source_hash = compute_sha256(source)
    baseline_hash = compute_sha256(baseline)
    hash_match = source_hash == baseline_hash
    print(f"  Source hash:   {source_hash[:16]}...")
    print(f"  Baseline hash: {baseline_hash[:16]}...")
    print(f"  Match: {'✅ YES' if hash_match else '❌ NO'}")
    print()
    
    # Channel statistics
    print("6. CHANNEL STATISTICS")
    print("-" * 80)
    if source.ndim == 3 and source.shape[2] >= 3:
        for ch, name in enumerate(['Red', 'Green', 'Blue'][:source.shape[2]]):
            src_mean = source[:, :, ch].mean()
            base_mean = baseline[:, :, ch].mean()
            src_std = source[:, :, ch].std()
            base_std = baseline[:, :, ch].std()
            print(f"  {name:6s}: Source mean={src_mean:8.2f} std={src_std:8.2f}")
            print(f"         Baseline mean={base_mean:8.2f} std={base_std:8.2f}")
            print()
    
    # Final verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if pixel_match or (max_diff == 0):
        print("✅ BASELINE VALID: Pixel-perfect match")
        status = "PASSED"
    elif max_diff < 1.0 and mean_diff < 0.01:
        print("⚠️  BASELINE ACCEPTABLE: Negligible differences (< 1 LSB)")
        status = "ACCEPTABLE"
    else:
        print("❌ BASELINE INVALID: Significant pixel differences detected")
        status = "FAILED"
    
    print()
    
    return {
        "status": status,
        "pixel_perfect": pixel_match,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "different_pixels": int(nonzero_diff),
        "total_pixels": int(total_pixels),
        "hash_match": hash_match,
    }


def main():
    source_path = Path("projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Kitchen_UltraQuality.tif")
    baseline_path = Path("forensics/raw_baseline/750Picacho_Kitchen_UltraQuality_master16.tif")
    
    if not source_path.exists():
        print(f"❌ Source file not found: {source_path}")
        return 1
    
    if not baseline_path.exists():
        print(f"❌ Baseline file not found: {baseline_path}")
        return 1
    
    result = compare_tiffs(source_path, baseline_path)
    
    if result["status"] == "PASSED":
        print("✅ RAW BASELINE VALIDATION: PASSED")
        print("   Baseline is pixel-perfect and can be used as canonical reference.")
        return 0
    elif result["status"] == "ACCEPTABLE":
        print("⚠️  RAW BASELINE VALIDATION: ACCEPTABLE")
        print("   Minor differences detected but within tolerance.")
        return 0
    else:
        print("❌ RAW BASELINE VALIDATION: FAILED")
        print("   Baseline cannot be trusted as reference.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
