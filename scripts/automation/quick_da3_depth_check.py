#!/usr/bin/env python3
"""Quick DA3 depth value check."""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "lux_depth_v3"))

from lux_depth_v3.config import DA3Config, ModelVariant, InferenceMode
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput

# Test image
test_img = Path("data/validation_full/800-picacho-12.jpg")
if not test_img.exists():
    # Try alternative
    test_img = list(Path("data/validation_full").glob("*.jpg"))[0]

print(f"Testing with: {test_img}")

# Initialize DA3
config = DA3Config(
    model_variant=ModelVariant.DA3_LARGE_V1_1,
    inference_mode=InferenceMode.MONOCULAR,
)

print("Initializing DA3 engine...")
engine = DA3InferenceEngine(config, commercial_use=False)

# Run inference
image_input = ImageInput(path=test_img)
result = engine.infer([image_input])

# Extract depth
depth = result.depth[0]  # Remove batch dim

print(f"\n📊 RAW DA3 DEPTH STATS:")
print(f"  Shape: {depth.shape}")
print(f"  Dtype: {depth.dtype}")
print(f"  Min: {depth.min():.6f}")
print(f"  Max: {depth.max():.6f}")
print(f"  Mean: {depth.mean():.6f}")
print(f"  Median: {np.median(depth):.6f}")
print(f"  Std: {depth.std():.6f}")
print(f"  P2: {np.percentile(depth, 2):.6f}")
print(f"  P98: {np.percentile(depth, 98):.6f}")
print(f"  Range: {depth.ptp():.6f}")

# Check if values suggest inverse depth or metric depth
if depth.max() < 100:
    print("\n✓ Depth appears to be in RELATIVE scale (0-1 or 0-100 range)")
else:
    print("\n⚠ Depth appears to be in METRIC scale (meters or mm)")

# Test different normalizations
print(f"\n📊 NORMALIZATION COMPARISON:")

# Current method (min-max)
norm_minmax = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
print(f"\n1. Min-Max Normalization:")
print(f"   Range: [{norm_minmax.min():.6f}, {norm_minmax.max():.6f}]")
print(f"   Mean: {norm_minmax.mean():.6f}, Std: {norm_minmax.std():.6f}")

# Percentile-based (robust)
p2, p98 = np.percentile(depth, [2, 98])
norm_percentile = np.clip((depth - p2) / (p98 - p2 + 1e-8), 0, 1)
print(f"\n2. Percentile Normalization (P2-P98):")
print(f"   Range: [{norm_percentile.min():.6f}, {norm_percentile.max():.6f}]")
print(f"   Mean: {norm_percentile.mean():.6f}, Std: {norm_percentile.std():.6f}")

# Inverse depth
depth_inv = 1.0 / (depth + 1e-6)
norm_inverse = (depth_inv - depth_inv.min()) / (depth_inv.max() - depth_inv.min() + 1e-8)
print(f"\n3. Inverse Depth Normalization:")
print(f"   Range: [{norm_inverse.min():.6f}, {norm_inverse.max():.6f}]")
print(f"   Mean: {norm_inverse.mean():.6f}, Std: {norm_inverse.std():.6f}")

# Compare to DA2 if available
da2_baseline = Path("validation_v1_baseline_pack/46img_validation_results")
baseline_files = list(da2_baseline.glob("*_depth.tiff"))
if baseline_files:
    da2_depth = np.array(Image.open(baseline_files[0]))
    if da2_depth.ndim == 3:
        da2_depth = da2_depth[:, :, 0]
    
    print(f"\n📊 DA2 BASELINE DEPTH (for comparison):")
    print(f"  Shape: {da2_depth.shape}")
    print(f"  Min: {da2_depth.min():.6f}")
    print(f"  Max: {da2_depth.max():.6f}")
    print(f"  Mean: {da2_depth.mean():.6f}")
    print(f"  Std: {da2_depth.std():.6f}")
    print(f"  Range: {da2_depth.ptp():.6f}")
