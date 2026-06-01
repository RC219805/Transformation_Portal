#!/usr/bin/env python3
"""
Create small representative test fixtures for pipeline testing.

This script creates synthetic 16-bit TIFF images that mimic luxury real estate
photography without using actual client data.
"""

from pathlib import Path

import numpy as np
import tifffile


def create_synthetic_tiff(output_path: Path, width: int = 800, height: int = 600):
    """
    Create a synthetic 16-bit RGB TIFF for testing.

    Mimics characteristics of luxury real estate photography:
    - High bit depth (16-bit)
    - RGB color space
    - Realistic value distribution
    - Gradient patterns simulating interiors

    Args:
        output_path: Where to save the TIFF
        width: Image width in pixels
        height: Image height in pixels
    """
    # Create gradient patterns that simulate interior lighting
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Channel patterns (simulating warm interior lighting)
    r_channel = (0.7 + 0.3 * xx) * 65535  # Warm tones
    g_channel = (0.65 + 0.25 * yy) * 65535  # Slightly cooler
    b_channel = (0.6 + 0.2 * (xx * yy)) * 65535  # Blue channel lower

    # Stack channels and convert to uint16
    img_array = np.stack([r_channel, g_channel, b_channel], axis=-1)
    img_array = np.clip(img_array, 0, 65535).astype(np.uint16)

    # Save with metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output_path,
        img_array,
        photometric="rgb",
        metadata={"axes": "YXC", "description": "Synthetic test fixture"},
    )

    print(f"Created: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def main():
    """Create synthetic fixtures in the legacy 750_picacho_lane fixture path."""
    fixtures_dir = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "pipelines" / "750_picacho_lane" / "input"

    # Create small fixtures (800x600 ~ 1MB each)
    fixtures = [
        "750Picacho_Pool_UltraQuality.tif",
        "750Picacho_GreatRoom_UltraQuality.tif",
    ]

    for fixture in fixtures:
        output_path = fixtures_dir / fixture
        create_synthetic_tiff(output_path, width=800, height=600)

    print(f"\n✓ Created {len(fixtures)} test fixtures in {fixtures_dir}")
    print("  These are synthetic images for testing - not actual client data")


if __name__ == "__main__":
    main()
