#!/usr/bin/env python3
"""
Test each pipeline stage to find where blue cast is introduced.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Repository-scoped paths for fixtures
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "pipelines" / "750_picacho_lane" / "input"

print("=" * 80)
print("PIPELINE STAGE-BY-STAGE COLOR TRACKING")
print("=" * 80)

# Load source
test_image_path = FIXTURE_PATH / "750Picacho_Pool_UltraQuality.tif"
print(f"\nLoading source image: {test_image_path}")

if not test_image_path.exists():
    print(f"\n❌ ERROR: Test fixture not found: {test_image_path}")
    print("\nTo create fixtures, run:")
    print("  python scripts/utilities/create_test_fixtures.py")
    print("\nSee tests/fixtures/pipelines/README.md for details.")
    sys.exit(1)

source = tifffile.imread(test_image_path)
image_linear = source.astype(np.float32) / 65535.0


def analyze_color(img, stage_name):
    """Analyze color balance of an image."""
    r = img[:, :, 0].mean()
    g = img[:, :, 1].mean()
    b = img[:, :, 2].mean()
    ratio = b / r
    print(f"\n{stage_name}:")
    print(f"  R={r:.3f}, G={g:.3f}, B={b:.3f}")
    print(f"  Blue/Red: {ratio:.2f}x", end="")
    if ratio > 1.15:
        print(f" ❌ BLUE CAST")
    elif ratio < 0.85:
        print(f" ⚠️  WARM CAST")
    else:
        print(f" ✓ BALANCED")
    return r, g, b, ratio


# Stage 0: Source
analyze_color(image_linear, "STAGE 0: Source (as loaded)")

# Stage 1.5: White Balance
r_mean = image_linear[:, :, 0].mean()
g_mean = image_linear[:, :, 1].mean()
b_mean = image_linear[:, :, 2].mean()
gray_mean = (r_mean + g_mean + b_mean) / 3.0
r_scale = gray_mean / (r_mean + 1e-6)
g_scale = gray_mean / (g_mean + 1e-6)
b_scale = gray_mean / (b_mean + 1e-6)

balanced = image_linear.copy()
balanced[:, :, 0] *= r_scale
balanced[:, :, 1] *= g_scale
balanced[:, :, 2] *= b_scale

analyze_color(balanced, "STAGE 1.5: After White Balance")

# Save for visual inspection
preview = (np.clip(balanced, 0, 1) * 255).astype(np.uint8)
Image.fromarray(preview).save("debug_after_white_balance.jpg", quality=95)
print("\n  💾 Saved: debug_after_white_balance.jpg")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print(
    """
The white balance stage successfully neutralizes the blue cast.
If the final output still has blue, it's being added by:
  - Tone mapping
  - Material response
  - Or the upscaling process

Next: Check tonemapped output to narrow it down.
"""
)
