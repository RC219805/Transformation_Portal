#!/usr/bin/env python3
"""
Calculate processing requirements for Transformation Portal batch jobs.

Based on empirical analysis from 750 Picacho processing runs (Dec 2025).
"""

import sys
import argparse
from pathlib import Path


def calculate_mps_memory(megapixels: float, upscale_factor: int = 4) -> float:
    """
    Calculate MPS memory requirements.
    
    Formula derived from 750 Picacho analysis:
    - Base processing: ~0.5GB per MP
    - 4x upscaling: ~0.75GB per MP
    - Total: ~1.25GB per MP
    """
    base_memory = megapixels * 0.5
    upscale_memory = megapixels * (0.75 if upscale_factor == 4 else 0.2)
    return base_memory + upscale_memory


def calculate_disk_space(file_size_mb: float, include_upscale: bool = True) -> dict:
    """
    Calculate disk space requirements.
    
    Based on observed multipliers:
    - Master: 0.75x source
    - Upscaled 4x: 12x source
    - Marketing PNG: 2.5x source
    """
    master = file_size_mb * 0.75
    upscaled = file_size_mb * 12 if include_upscale else 0
    marketing = file_size_mb * 2.5
    total = master + upscaled + marketing
    
    return {
        "master_mb": master,
        "upscaled_mb": upscaled,
        "marketing_mb": marketing,
        "total_mb": total,
        "total_gb": total / 1024,
        "with_safety_margin_gb": total / 1024 * 1.2,
    }


def estimate_processing_time(megapixels: float, device: str = "mps") -> float:
    """
    Estimate processing time in minutes.
    
    Based on 750 Picacho analysis:
    - MPS: 6 seconds per megapixel
    - CPU: 60-120 seconds per megapixel (assume 90s average)
    """
    seconds_per_mp = 6 if device == "mps" else 90
    return (megapixels * seconds_per_mp) / 60


def check_safety(megapixels: float, file_size_mb: float, device: str = "mps") -> dict:
    """Check if processing is safe based on empirical thresholds."""
    if device == "mps":
        if megapixels <= 24 and file_size_mb <= 163:
            status = "✅ SAFE"
            risk = "low"
        elif megapixels <= 35 and file_size_mb <= 240:
            status = "⚠️ RISKY"
            risk = "medium"
        else:
            status = "❌ UNSAFE"
            risk = "high"
    else:  # CPU
        if megapixels <= 48:
            status = "✅ SAFE"
            risk = "low"
        else:
            status = "⚠️ USE TILING"
            risk = "medium"
    
    return {"status": status, "risk": risk}


def main():
    parser = argparse.ArgumentParser(
        description="Calculate processing requirements for image batch jobs"
    )
    parser.add_argument(
        "--megapixels", "-mp", type=float, help="Image resolution in megapixels"
    )
    parser.add_argument(
        "--file-size", "-fs", type=float, help="File size in MB (16-bit TIFF)"
    )
    parser.add_argument(
        "--num-images", "-n", type=int, default=1, help="Number of images in batch"
    )
    parser.add_argument(
        "--upscale", type=int, default=4, choices=[2, 4], help="Upscale factor"
    )
    parser.add_argument(
        "--device", choices=["mps", "cpu"], default="mps", help="Processing device"
    )
    parser.add_argument(
        "--no-upscale-output",
        action="store_true",
        help="Skip upscaled output (saves disk space)",
    )
    
    args = parser.parse_args()
    
    # Calculate from either MP or file size
    if args.megapixels:
        mp = args.megapixels
        # 16-bit TIFF: ~6.77MB per MP average
        file_size = mp * 6.77
    elif args.file_size:
        file_size = args.file_size
        mp = file_size / 6.77
    else:
        parser.error("Must specify either --megapixels or --file-size")
    
    # Calculate requirements
    mps_memory_gb = calculate_mps_memory(mp, args.upscale)
    disk = calculate_disk_space(file_size, not args.no_upscale_output)
    time_min = estimate_processing_time(mp, args.device)
    safety = check_safety(mp, file_size, args.device)
    
    # Print results
    print("=" * 60)
    print("TRANSFORMATION PORTAL PROCESSING CALCULATOR")
    print("=" * 60)
    print(f"\nInput Specifications:")
    print(f"  Resolution: {mp:.1f} megapixels")
    print(f"  File size: {file_size:.0f} MB (16-bit TIFF)")
    print(f"  Batch size: {args.num_images} image(s)")
    print(f"  Device: {args.device.upper()}")
    print(f"  Upscale: {args.upscale}x")
    
    print(f"\nSafety Assessment: {safety['status']}")
    
    if args.device == "mps":
        print(f"\nMPS Memory Requirements:")
        print(f"  Per image: {mps_memory_gb:.1f} GB")
        print(f"  Total batch: {mps_memory_gb * args.num_images:.1f} GB")
        
        if mps_memory_gb * args.num_images > 50:
            print(f"  ⚠️ WARNING: May exceed 64GB unified memory")
            print(f"  Recommendation: Process {int(50 / mps_memory_gb)} images at a time")
    
    print(f"\nDisk Space Requirements:")
    print(f"  Master TIFF: {disk['master_mb']:.0f} MB")
    if not args.no_upscale_output:
        print(f"  Upscaled {args.upscale}x TIFF: {disk['upscaled_mb']:.0f} MB")
    print(f"  Marketing PNG: {disk['marketing_mb']:.0f} MB")
    print(f"  Total per image: {disk['total_gb']:.2f} GB")
    print(f"  With 20% safety margin: {disk['with_safety_margin_gb']:.2f} GB")
    print(f"  Total batch: {disk['with_safety_margin_gb'] * args.num_images:.2f} GB")
    
    print(f"\nProcessing Time Estimate:")
    print(f"  Per image: {time_min:.1f} minutes")
    print(f"  Total batch: {time_min * args.num_images:.1f} minutes")
    
    # Recommendations
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if safety["risk"] == "high":
        if args.device == "mps":
            print("❌ This resolution is too large for MPS processing")
            print("   Recommendation 1: Use CPU mode (--device cpu)")
            print("   Recommendation 2: Reduce upscale to 2x (--upscale 2)")
            print("   Recommendation 3: Use tiling (--tile 256)")
    elif safety["risk"] == "medium":
        print("⚠️ Processing may succeed but monitor for issues")
        if args.device == "mps":
            print("   Consider CPU fallback if MPS OOM occurs")
    else:
        print("✅ Processing should complete successfully")
    
    # Disk space check
    total_disk_needed = disk['with_safety_margin_gb'] * args.num_images
    if total_disk_needed > 10:
        print(f"\n💾 Ensure {total_disk_needed:.1f}GB free disk space")
        print(f"   Keep disk usage below 85% for optimal performance")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
