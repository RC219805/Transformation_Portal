#!/usr/bin/env python3
"""
Quick DA3 depth estimation test.

Usage:
    python test_da3_quick.py path/to/image.jpg

Example:
    python test_da3_quick.py test_output/da3_basic/test_image.png
"""

import os
import sys
from pathlib import Path
import pytest

# Skip if torch not available (ML dependency)
torch = pytest.importorskip("torch")

# Set OpenMP workaround before importing torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from lux_depth_v3.config import ModelVariant, DA3Config
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput
import numpy as np
from PIL import Image


def estimate_depth(image_path: str, output_dir: str = "output"):
    """Estimate depth for a single image."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 DA3 Depth Estimation")
    print(f"📸 Input: {image_path}")
    print(f"📁 Output: {output_dir}")
    
    # Configure DA3
    config = DA3Config(model_variant=ModelVariant.DA3_LARGE_V1_1)
    
    # Initialize engine
    print("\n⚙️  Loading DA3-LARGE-1.1 model...")
    engine = DA3InferenceEngine(config, commercial_use=False)
    print("✓ Model loaded")
    
    # Run inference
    print("\n🚀 Running depth estimation...")
    image_input = ImageInput(path=image_path)
    result = engine.infer([image_input])
    
    # Results
    print(f"\n✓ Depth estimation complete!")
    print(f"  - Shape: {result.depth.shape}")
    print(f"  - Range: [{result.depth.min():.3f}, {result.depth.max():.3f}]")
    print(f"  - Confidence available: {result.conf is not None}")
    
    # Save depth visualization
    depth_vis_path = output_dir / f"{image_path.stem}_depth.png"
    depth = result.depth[0] if result.depth.ndim == 3 else result.depth
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    Image.fromarray(depth_normalized).save(depth_vis_path)
    print(f"\n💾 Saved depth map: {depth_vis_path}")
    
    # Save NPZ
    npz_path = output_dir / f"{image_path.stem}_depth.npz"
    np.savez_compressed(
        npz_path,
        depth=result.depth,
        conf=result.conf if result.conf is not None else np.array([]),
    )
    print(f"💾 Saved NPZ: {npz_path}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_da3_quick.py <image_path>")
        print("\nExample:")
        print("  python test_da3_quick.py test_output/da3_basic/test_image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    try:
        estimate_depth(image_path, output_dir)
        print("\n✅ Success!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
