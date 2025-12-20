#!/usr/bin/env python3
"""Run A/B comparison with FIXED configuration."""
import sys
sys.path.insert(0, '.')

import numpy as np
from PIL import Image
from pathlib import Path
from lux_depth_v2.tools.ab_comparison import run_ab_comparison

# Load small test image for quick validation
print("Loading test image...")
img_path = Path("input_images/750_Picacho/Source_TIFFs/V2_750Picacho_GreatRoom.tiff")
img = Image.open(img_path)

# Downsample for speed
img_small = img.resize((2000, 1500), Image.LANCZOS)
rgb = np.array(img_small)
print(f"Image: {rgb.shape}")

# Run A/B comparison (now uses fixed config)
output_dir = Path("outputs/ab_validation_fixed")
output_dir.mkdir(parents=True, exist_ok=True)

print("\nRunning A/B comparison with FIXED configuration...")
print("=" * 60)

result = run_ab_comparison(rgb, output_dir)

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(result)

# Check if improved
if result.edge_alignment_improvement > 0:
    print("\n✅ SUCCESS: Edge alignment IMPROVED")
else:
    print("\n❌ FAILED: Edge alignment still degraded")
    
print(f"\nEdge overlap metrics:")
print(f"  Baseline alignment: {result.baseline_edge_alignment:.4f}")
print(f"  Enhanced alignment: {result.enhanced_edge_alignment:.4f}")
print(f"  Improvement: {result.edge_alignment_improvement:.2%}")
