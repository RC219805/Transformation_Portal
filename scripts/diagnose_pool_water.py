#!/usr/bin/env python3
"""Diagnostic script for pool water class detection issues.

This script addresses the Stage 6 finding that pool water is consistently
missing (`status=missing_mask`) by:

1. Running segmentation on the pool image
2. Printing all emitted class keys
3. Showing which classes map to "water" via canonicalization
4. Displaying coverage for any water-related classes
5. Recommending taxonomy fixes if needed

Usage:
    python scripts/diagnose_pool_water.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lux_depth_v2 import io_utils, torch_ops
from lux_depth_v2.config import SegmentationConfig, SegmentationBackend
from lux_depth_v2.material_segmentation import create_material_segmenter
from lux_depth_v2.materials_v3_taxonomy import (
    normalize_material_name,
    normalize_material_dict,
    SEMANTIC_TO_CANONICAL,
)


def main() -> int:
    """Run pool water diagnostic."""
    
    # Pool image from actual location
    pool_image = repo_root / "input_images" / "750_Picacho" / "Pool.tif"
    
    if not pool_image.exists():
        print(f"❌ Pool image not found: {pool_image}")
        print("   This diagnostic requires the Phase 2 benchmark set.")
        return 1
    
    print("=" * 80)
    print("POOL WATER CLASS DETECTION DIAGNOSTIC")
    print("=" * 80)
    print(f"\n📸 Input: {pool_image.name}")
    
    # Load image
    print("\n1️⃣  Loading image...")
    rgb01, _ = io_utils.read_rgb_any(pool_image)
    H, W = rgb01.shape[:2]
    print(f"   Resolution: {W}×{H} ({W*H/1e6:.1f} MP)")
    
    # Create segmenter (baseline: SegFormer only)
    print("\n2️⃣  Running SegFormer segmentation...")
    device_t = torch.device("cpu")  # Use CPU for reproducibility
    rgb_t = torch_ops.to_torch_rgb(rgb01, device_t)
    
    seg_cfg = SegmentationConfig(
        backend="segformer",  # Baseline, no EfficientSAM
        backend_v3=SegmentationBackend.SEGFORMER,
    )
    
    segmenter = create_material_segmenter(seg_cfg, device_t)
    masks_dict_torch = segmenter.predict(rgb_t)
    
    # Convert to numpy for analysis
    masks_np = {
        k: v[0, 0].cpu().numpy().astype(np.float32)
        for k, v in masks_dict_torch.items()
    }
    
    print(f"   Emitted {len(masks_np)} material classes")
    
    # Report all emitted classes
    print("\n3️⃣  Emitted classes (raw output from segmenter):")
    print("   " + "-" * 70)
    
    total_px = H * W
    for cls_name in sorted(masks_np.keys()):
        mask = masks_np[cls_name]
        coverage_px = int((mask > 0.5).sum())
        coverage_pct = 100.0 * coverage_px / total_px
        
        # Highlight water-related classes
        is_water_related = "water" in cls_name.lower() or "pool" in cls_name.lower()
        marker = "💧" if is_water_related else "  "
        
        print(f"   {marker} {cls_name:30s} → {coverage_px:8d} px ({coverage_pct:5.2f}%)")
    
    # Check canonical mapping
    print("\n4️⃣  Canonical name mapping (via taxonomy normalizer):")
    print("   " + "-" * 70)
    
    canonical_dict = normalize_material_dict(masks_np)
    
    for raw_name in sorted(masks_np.keys()):
        canonical = normalize_material_name(raw_name)
        arrow = "→" if canonical != raw_name else "="
        
        is_water = canonical == "water"
        marker = "💧" if is_water else "  "
        
        print(f"   {marker} {raw_name:30s} {arrow} {canonical}")
    
    # Check if water is present in canonical dict
    print("\n5️⃣  Water class status:")
    print("   " + "-" * 70)
    
    if "water" in canonical_dict:
        water_mask = canonical_dict["water"]
        coverage_px = int((water_mask > 0.5).sum())
        coverage_pct = 100.0 * coverage_px / total_px
        
        print(f"   ✅ FOUND: 'water' is present in canonical materials")
        print(f"      Coverage: {coverage_px:,} pixels ({coverage_pct:.2f}%)")
        
        if coverage_px == 0:
            print(f"      ⚠️  WARNING: Water mask exists but has ZERO coverage")
            print(f"         This suggests the segmenter emitted 'water' but no pixels classified")
    else:
        print(f"   ❌ MISSING: 'water' is NOT in canonical materials")
        
        # Check if any emitted class should have mapped to water
        water_candidates = [
            k for k in masks_np.keys()
            if normalize_material_name(k) == "water"
        ]
        
        if water_candidates:
            print(f"      ℹ️  The following raw classes should have mapped to 'water':")
            for c in water_candidates:
                print(f"         - {c}")
            print(f"      This suggests a normalize_material_dict() bug.")
        else:
            print(f"      ℹ️  No emitted classes map to 'water' via SEMANTIC_TO_CANONICAL")
            print(f"         The segmenter did not detect water in this image.")
    
    # Recommendations
    print("\n6️⃣  Recommendations:")
    print("   " + "-" * 70)
    
    if "water" not in canonical_dict:
        # Check if there are any pool/ocean/lake classes we're missing
        water_related = [
            k for k in masks_np.keys()
            if any(w in k.lower() for w in ["water", "pool", "ocean", "lake", "pond", "sea"])
        ]
        
        if water_related:
            print(f"   💡 Water-related classes detected but not mapped:")
            for cls in water_related:
                print(f"      - {cls}")
            print(f"\n   ACTION: Add these to SEMANTIC_TO_CANONICAL in materials_v3_taxonomy.py")
        else:
            print(f"   💡 No water-like classes detected by segmenter.")
            print(f"      Possible causes:")
            print(f"      - Pool image does not have visible water surface")
            print(f"      - SegFormer model does not include 'pool/water' in its vocabulary")
            print(f"      - Water region is too small (< min_coverage threshold)")
            print(f"\n   ACTION: Inspect the pool image to confirm water is visible")
            print(f"           Consider using a segmenter with explicit water/pool classes")
    else:
        water_mask = canonical_dict["water"]
        coverage_px = int((water_mask > 0.5).sum())
        
        if coverage_px == 0:
            print(f"   💡 Water class exists but zero coverage — likely a threshold issue")
            print(f"      Try lowering MaterialsV3Config.confidence_semantics.material_thresholds['water']")
        elif coverage_px < 500:
            print(f"   💡 Water coverage very low ({coverage_px} px) — may be filtered out")
            print(f"      Check min_coverage settings in Materials V3 gating")
        else:
            print(f"   ✅ Water detection appears correct ({coverage_px:,} pixels)")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
