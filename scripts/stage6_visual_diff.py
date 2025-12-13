#!/usr/bin/env python3
"""
Stage 6 Visual Diff Generator

Automatically finds regions of highest change between baseline and canary,
then generates side-by-side crops for objective comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image


def load_image_as_float(path: Path) -> np.ndarray:
    """Load image as HxWx3 float32 in [0,1]."""
    # Increase PIL limit for large architectural images
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_luma(rgb: np.ndarray) -> np.ndarray:
    """Compute perceptual luma (Rec. 709)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def compute_diff_map(baseline: np.ndarray, canary: np.ndarray) -> np.ndarray:
    """Compute absolute difference map (luma-weighted)."""
    luma_base = compute_luma(baseline)
    luma_canary = compute_luma(canary)
    diff = np.abs(luma_canary - luma_base)
    return diff


def find_top_change_regions(
    diff: np.ndarray,
    *,
    crop_size: int = 256,
    num_regions: int = 6,
    min_spacing: int = 128,
) -> List[Tuple[int, int]]:
    """
    Find top N regions with highest change.
    
    Returns list of (y, x) top-left corners for crops.
    """
    H, W = diff.shape
    pad = crop_size // 2
    
    # Apply Gaussian blur to average locally
    from scipy.ndimage import gaussian_filter
    diff_smooth = gaussian_filter(diff, sigma=crop_size / 6)
    
    regions = []
    used_mask = np.zeros_like(diff, dtype=bool)
    
    for _ in range(num_regions):
        # Find max in remaining area
        masked_diff = diff_smooth.copy()
        masked_diff[used_mask] = 0
        
        if masked_diff.max() < 1e-6:
            break
        
        # Find peak
        y, x = np.unravel_index(np.argmax(masked_diff), masked_diff.shape)
        
        # Center crop
        y0 = max(0, y - crop_size // 2)
        x0 = max(0, x - crop_size // 2)
        y1 = min(H, y0 + crop_size)
        x1 = min(W, x0 + crop_size)
        
        # Adjust if at edge
        if y1 - y0 < crop_size:
            y0 = max(0, y1 - crop_size)
        if x1 - x0 < crop_size:
            x0 = max(0, x1 - crop_size)
        
        regions.append((y0, x0))
        
        # Mark area as used
        y_used_0 = max(0, y - min_spacing)
        y_used_1 = min(H, y + min_spacing)
        x_used_0 = max(0, x - min_spacing)
        x_used_1 = min(W, x + min_spacing)
        used_mask[y_used_0:y_used_1, x_used_0:x_used_1] = True
    
    return regions


def create_triptych_crop(
    baseline: np.ndarray,
    canary: np.ndarray,
    diff: np.ndarray,
    y: int,
    x: int,
    crop_size: int,
) -> Image.Image:
    """Create side-by-side: baseline | canary | diff."""
    H, W, _ = baseline.shape
    y1 = min(H, y + crop_size)
    x1 = min(W, x + crop_size)
    
    crop_base = baseline[y:y1, x:x1]
    crop_canary = canary[y:y1, x:x1]
    crop_diff = diff[y:y1, x:x1]
    
    # Convert diff to heatmap (red = high change)
    diff_rgb = np.stack([crop_diff, crop_diff * 0.2, crop_diff * 0.2], axis=-1)
    diff_rgb = np.clip(diff_rgb * 5.0, 0, 1)  # amplify for visibility
    
    # Convert to uint8
    crop_base_u8 = (crop_base * 255).astype(np.uint8)
    crop_canary_u8 = (crop_canary * 255).astype(np.uint8)
    diff_u8 = (diff_rgb * 255).astype(np.uint8)
    
    # Concatenate horizontally
    triptych = np.concatenate([crop_base_u8, crop_canary_u8, diff_u8], axis=1)
    
    return Image.fromarray(triptych)


def process_scene(
    scene_name: str,
    baseline_dir: Path,
    canary_dir: Path,
    output_dir: Path,
    *,
    crop_size: int = 256,
    num_crops: int = 6,
) -> dict:
    """Process one scene and generate visual diffs."""
    print(f"\n=== Processing {scene_name} ===")
    
    # Find the main output image (look for Ultimate marketing PNG or master TIFF)
    candidates_base = (list(baseline_dir.glob("*_Ultimate_marketing.png")) + 
                      list(baseline_dir.glob("*_Ultimate_master16.tif")) +
                      list(baseline_dir.glob("*_Ultimate.png")) + 
                      list(baseline_dir.glob("*_Ultimate.tif")))
    candidates_canary = (list(canary_dir.glob("*_Ultimate_marketing.png")) + 
                        list(canary_dir.glob("*_Ultimate_master16.tif")) +
                        list(canary_dir.glob("*_Ultimate.png")) + 
                        list(canary_dir.glob("*_Ultimate.tif")))
    
    if not candidates_base or not candidates_canary:
        print(f"  ERROR: No Ultimate outputs found for {scene_name}")
        return {"status": "error", "reason": "missing_outputs"}
    
    baseline_img_path = candidates_base[0]
    canary_img_path = candidates_canary[0]
    
    print(f"  Baseline: {baseline_img_path.name}")
    print(f"  Canary:   {canary_img_path.name}")
    
    # Load images
    baseline = load_image_as_float(baseline_img_path)
    canary = load_image_as_float(canary_img_path)
    
    if baseline.shape != canary.shape:
        print(f"  ERROR: Shape mismatch {baseline.shape} vs {canary.shape}")
        return {"status": "error", "reason": "shape_mismatch"}
    
    # Compute diff
    diff = compute_diff_map(baseline, canary)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())
    
    print(f"  Diff: mean={mean_diff:.6f}, max={max_diff:.6f}")
    
    # Find top change regions
    regions = find_top_change_regions(diff, crop_size=crop_size, num_regions=num_crops)
    print(f"  Found {len(regions)} change regions")
    
    # Create output directory
    scene_out = output_dir / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)
    
    # Generate crops
    for i, (y, x) in enumerate(regions, 1):
        triptych = create_triptych_crop(baseline, canary, diff, y, x, crop_size)
        crop_path = scene_out / f"crop_{i:02d}_y{y}_x{x}.png"
        triptych.save(crop_path)
        print(f"    {crop_path.name}")
    
    # Also save full diff heatmap (downscaled for practical file size)
    H, W = diff.shape
    scale = min(1.0, 2048 / max(H, W))
    if scale < 1.0:
        from PIL import Image as PILImage
        diff_u8 = (np.clip(diff * 5.0, 0, 1) * 255).astype(np.uint8)
        diff_img = PILImage.fromarray(diff_u8, mode='L')
        new_size = (int(W * scale), int(H * scale))
        diff_img = diff_img.resize(new_size, PILImage.Resampling.LANCZOS)
        diff_img.save(scene_out / "full_diff_heatmap.png")
    
    return {
        "status": "success",
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "num_crops": len(regions),
        "crops": [{"y": int(y), "x": int(x)} for y, x in regions],
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Generate visual diff crops for Stage 6 A/B test")
    parser.add_argument("--ab-dir", type=Path, default=Path("outputs/stage6_ab"),
                        help="Stage 6 A/B output directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/stage6_visual_diffs"),
                        help="Output directory for diff crops")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size in pixels")
    parser.add_argument("--num-crops", type=int, default=6, help="Number of crops per scene")
    args = parser.parse_args()
    
    ab_dir = args.ab_dir
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenes of interest (the two wins from summary)
    scenes = [
        {
            "name": "bedroom_glass",
            "baseline": "interior_bedroom_A_baseline",
            "canary": "interior_bedroom_B_efficientsam",
        },
        {
            "name": "aerial_foliage",
            "baseline": "exterior_aerial_A_baseline",
            "canary": "exterior_aerial_B_efficientsam",
        },
    ]
    
    results = {}
    for scene_def in scenes:
        name = scene_def["name"]
        baseline_dir = ab_dir / scene_def["baseline"]
        canary_dir = ab_dir / scene_def["canary"]
        
        result = process_scene(
            name,
            baseline_dir,
            canary_dir,
            output_dir,
            crop_size=args.crop_size,
            num_crops=args.num_crops,
        )
        results[name] = result
    
    # Write summary
    summary_path = output_dir / "visual_diff_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Visual diffs written to: {output_dir}")
    print(f"✅ Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
