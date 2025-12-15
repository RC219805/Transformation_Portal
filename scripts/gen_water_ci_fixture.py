#!/usr/bin/env python3
"""Generate deterministic synthetic water images for CI testing.

Creates synthetic pool, ocean, and hard-negative images matching the ci_subset.txt
specification. Output is deterministic (seed-based) for reproducible CI tests.

Usage:
    python scripts/gen_water_ci_fixture.py --seed 42 --output data/water_v0/images/

Generates:
    - 5 pool scenes (clear blue, various lighting)
    - 4 ocean scenes (blue-green gradient, wave textures)
    - 3 hard negatives (blue wall, specular glass, sky patch)
    - 2 edge cases (low saturation pool, mixed scene)
    - ground_truth.json with metadata
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_pool_scene(width: int, height: int, scene_type: str = "standard") -> np.ndarray:
    """Generate synthetic pool scene.
    
    Args:
        width: Image width
        height: Image height
        scene_type: "standard", "bright", "dark", "low_sat"
    
    Returns:
        RGB image array (uint8)
    """
    # Base pool color (cyan-blue)
    if scene_type == "low_sat":
        base_color = np.array([100, 110, 115], dtype=np.float32)  # Desaturated
    else:
        base_color = np.array([30, 120, 200], dtype=np.float32)  # Rich blue
    
    # Create gradient (deeper blue at bottom)
    y_grad = np.linspace(0, 1, height)[:, None]
    gradient = base_color + y_grad * np.array([0, -30, -40])
    gradient = np.clip(gradient, 0, 255)
    
    # Add water texture (noise + ripples)
    noise = np.random.randn(height, width, 3) * 8
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 3 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    ripples = 10 * (np.sin(X) * np.cos(Y))[:, :, None]
    
    img = gradient + noise + ripples
    
    # Adjust brightness
    if scene_type == "bright":
        img = img * 1.3
    elif scene_type == "dark":
        img = img * 0.6
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Broadcast to full image
    img = np.broadcast_to(img, (height, width, 3)).copy()
    
    # Add pool edge/tile (top 10% of image)
    edge_height = int(height * 0.1)
    tile_color = [220, 200, 180]  # Beige tile
    img[:edge_height, :] = tile_color
    
    return img


def generate_ocean_scene(width: int, height: int, scene_type: str = "standard") -> np.ndarray:
    """Generate synthetic ocean scene.
    
    Args:
        width: Image width
        height: Image height
        scene_type: "standard", "waves", "calm", "green"
    
    Returns:
        RGB image array (uint8)
    """
    # Base ocean color (blue-green)
    if scene_type == "green":
        base_color = np.array([20, 100, 90], dtype=np.float32)  # More green
    else:
        base_color = np.array([10, 80, 140], dtype=np.float32)  # Blue
    
    # Create horizon gradient
    y_grad = np.linspace(0, 1, height)[:, None]
    gradient = base_color + y_grad * np.array([40, 60, 80])  # Lighter at horizon
    gradient = np.clip(gradient, 0, 255)
    
    # Add wave texture
    x = np.linspace(0, 6 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    if scene_type == "waves":
        waves = 15 * (np.sin(X + Y * 0.5) + 0.5 * np.sin(2 * X - Y))[:, :, None]
    else:
        waves = 8 * np.sin(X + Y * 0.3)[:, :, None]
    
    noise = np.random.randn(height, width, 3) * 5
    
    img = gradient + waves + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Broadcast to full image
    img = np.broadcast_to(img, (height, width, 3)).copy()
    
    # Add sky (top 20%)
    sky_height = int(height * 0.2)
    sky_color = np.array([180, 200, 230], dtype=np.float32)  # Light blue sky
    for i in range(sky_height):
        blend = i / sky_height
        img[i, :] = (1 - blend) * sky_color + blend * img[i, :]
    
    return img


def generate_hard_negative(width: int, height: int, neg_type: str) -> np.ndarray:
    """Generate hard negative (blue non-water scene).
    
    Args:
        width: Image width
        height: Image height
        neg_type: "blue_wall", "glass", "sky_patch"
    
    Returns:
        RGB image array (uint8)
    """
    if neg_type == "blue_wall":
        # Solid blue painted wall
        base_color = np.array([40, 100, 180], dtype=np.float32)
        noise = np.random.randn(height, width, 3) * 3  # Slight texture
        img = base_color + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.broadcast_to(img, (height, width, 3)).copy()
        
    elif neg_type == "glass":
        # Specular reflective glass
        base_color = np.array([60, 140, 200], dtype=np.float32)
        # Add specular highlights
        y_grad = np.linspace(0, 1, height)[:, None]
        x_grad = np.linspace(0, 1, width)[None, :]
        highlight = 80 * np.exp(-((y_grad - 0.3)**2 + (x_grad - 0.5)**2) / 0.1)
        img = base_color + highlight[:, :, None]
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.broadcast_to(img, (height, width, 3)).copy()
        
    else:  # sky_patch
        # Blue sky through window
        base_color = np.array([100, 150, 220], dtype=np.float32)
        # Add cloud-like texture
        noise = np.random.randn(height, width, 3) * 20
        img = base_color + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.broadcast_to(img, (height, width, 3)).copy()
    
    return img


def generate_fixture_images(
    output_dir: Path,
    width: int = 512,
    height: int = 512,
    seed: int = 42
) -> Dict[str, dict]:
    """Generate all fixture images matching ci_subset.txt.
    
    Returns:
        Dictionary mapping image paths to ground truth metadata
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pool").mkdir(exist_ok=True)
    (output_dir / "ocean").mkdir(exist_ok=True)
    
    ground_truth = {}
    
    # Pool scenes (5 total from ci_subset.txt: 0001, 0003, 0005, 0007, 0008, 0009)
    # We'll generate based on actual ci_subset
    pool_configs = [
        ("pool_0001.jpg", "standard", "easy"),
        ("pool_0003.jpg", "bright", "easy"),
        ("pool_0005.jpg", "standard", "medium"),
        ("pool_0007.jpg", "dark", "medium"),
        ("pool_0008.jpg", "low_sat", "hard"),  # Edge case: low saturation
    ]
    
    # Add pool_0009 from ci_subset
    pool_configs.append(("pool_0009.jpg", "standard", "easy"))
    
    for filename, scene_type, difficulty in pool_configs:
        img_array = generate_pool_scene(width, height, scene_type)
        img = Image.fromarray(img_array)
        
        # Apply slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        output_path = output_dir / "pool" / filename
        img.save(output_path, quality=95)
        
        ground_truth[f"pool/{filename}"] = {
            "label": "pool",
            "should_detect": True,
            "difficulty": difficulty,
            "tags": ["synthetic", scene_type] if scene_type != "standard" else ["synthetic"]
        }
    
    # Ocean scenes (4 total from ci_subset.txt: 0001, 0003, 0004, 0005, 0007, 0009)
    # Pick 4 representative ones
    ocean_configs = [
        ("ocean_0001.jpg", "standard", "easy"),
        ("ocean_0003.jpg", "waves", "medium"),
        ("ocean_0004.jpg", "calm", "easy"),
        ("ocean_0005.jpg", "green", "medium"),
    ]
    
    # Add more from ci_subset
    ocean_configs.extend([
        ("ocean_0007.jpg", "standard", "easy"),
        ("ocean_0009.jpg", "waves", "medium"),
    ])
    
    for filename, scene_type, difficulty in ocean_configs:
        img_array = generate_ocean_scene(width, height, scene_type)
        img = Image.fromarray(img_array)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        output_path = output_dir / "ocean" / filename
        img.save(output_path, quality=95)
        
        ground_truth[f"ocean/{filename}"] = {
            "label": "ocean",
            "should_detect": True,
            "difficulty": difficulty,
            "tags": ["synthetic", scene_type] if scene_type != "standard" else ["synthetic"]
        }
    
    # Hard negatives (2 total from ci_subset.txt)
    hard_neg_configs = [
        ("pool", "neg_blue_wall_0001.jpg", "blue_wall"),
        ("ocean", "neg_glass_building_0001.jpg", "glass"),
    ]
    
    for folder, filename, neg_type in hard_neg_configs:
        img_array = generate_hard_negative(width, height, neg_type)
        img = Image.fromarray(img_array)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        output_path = output_dir / folder / filename
        img.save(output_path, quality=95)
        
        label = folder  # "pool" or "ocean"
        ground_truth[f"{folder}/{filename}"] = {
            "label": label,
            "should_detect": False,  # Hard negative
            "difficulty": "hard",
            "tags": ["synthetic", "hard_negative", neg_type]
        }
    
    return ground_truth


def save_ground_truth(ground_truth: Dict[str, dict], output_path: Path):
    """Save ground truth JSON."""
    data = {
        "root": "images",
        "images": ground_truth
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Ground truth saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic water images for CI testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/water_v0/images"),
        help="Output directory for images (default: data/water_v0/images/)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (default: 512)"
    )
    parser.add_argument(
        "--ground-truth-output",
        type=Path,
        help="Path to save ground_truth.json (default: <output>/../ground_truth.json)"
    )
    
    args = parser.parse_args()
    
    # Generate images
    print(f"🎨 Generating synthetic water images (seed={args.seed})...")
    ground_truth = generate_fixture_images(
        output_dir=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    
    print(f"✅ Generated {len(ground_truth)} images in {args.output}")
    
    # Save ground truth
    if args.ground_truth_output:
        gt_path = args.ground_truth_output
    else:
        gt_path = args.output.parent / "ground_truth.json"
    
    save_ground_truth(ground_truth, gt_path)
    
    # Print summary
    pool_count = sum(1 for v in ground_truth.values() if v["label"] == "pool" and v["should_detect"])
    ocean_count = sum(1 for v in ground_truth.values() if v["label"] == "ocean" and v["should_detect"])
    neg_count = sum(1 for v in ground_truth.values() if not v["should_detect"])
    
    print(f"\n📊 Summary:")
    print(f"  • Pool scenes: {pool_count}")
    print(f"  • Ocean scenes: {ocean_count}")
    print(f"  • Hard negatives: {neg_count}")
    print(f"  • Total: {len(ground_truth)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
