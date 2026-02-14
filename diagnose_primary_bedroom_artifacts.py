#!/usr/bin/env python3
"""
Diagnose edge artifacts in Primary Bedroom image.
Focus on:
1. Foliage edge halos (blue/white artifacts)
2. Sky/ocean boundary contamination
"""
import json
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    import subprocess

    subprocess.run(["pip", "install", "tifffile"], check=True)
    import tifffile

from PIL import Image, ImageDraw, ImageFont


def detect_edge_artifacts(input_arr, output_arr, threshold=0.05):
    """Detect edge artifacts by finding regions with large color differences."""
    # Crop to common size
    min_h = min(input_arr.shape[0], output_arr.shape[0])
    min_w = min(input_arr.shape[1], output_arr.shape[1])

    input_arr = input_arr[:min_h, :min_w, :]
    output_arr = output_arr[:min_h, :min_w, :]

    # Normalize to [0,1]
    if input_arr.dtype == np.uint16:
        input_norm = input_arr.astype(np.float32) / 65535.0
    else:
        input_norm = input_arr.astype(np.float32) / 255.0

    if output_arr.dtype == np.uint16:
        output_norm = output_arr.astype(np.float32) / 65535.0
    else:
        output_norm = output_arr.astype(np.float32) / 255.0

    # Compute per-pixel delta
    delta = np.abs(output_norm - input_norm)
    delta_magnitude = np.sqrt(np.sum(delta**2, axis=2))  # Euclidean distance in RGB space

    # Find edges using Sobel-like gradient
    from scipy import ndimage

    edges = ndimage.sobel(input_norm.mean(axis=2))
    edges = np.abs(edges)
    edges_norm = edges / edges.max() if edges.max() > 0 else edges

    # Artifacts are high delta at edges
    artifact_map = delta_magnitude * (edges_norm > 0.1)

    return delta, delta_magnitude, edges_norm, artifact_map


def analyze_sky_ocean_boundary(input_arr, output_arr):
    """Analyze the sky/ocean boundary region (top 30% of image)."""
    # Crop to common size
    min_h = min(input_arr.shape[0], output_arr.shape[0])
    min_w = min(input_arr.shape[1], output_arr.shape[1])

    input_arr = input_arr[:min_h, :min_w, :]
    output_arr = output_arr[:min_h, :min_w, :]

    h = input_arr.shape[0]
    boundary_region_in = input_arr[: int(h * 0.3), :, :]
    boundary_region_out = output_arr[: int(h * 0.3), :, :]

    # Normalize
    if input_arr.dtype == np.uint16:
        in_norm = boundary_region_in.astype(np.float32) / 65535.0
        out_norm = boundary_region_out.astype(np.float32) / 65535.0
    else:
        in_norm = boundary_region_in.astype(np.float32) / 255.0
        out_norm = boundary_region_out.astype(np.float32) / 255.0

    delta = np.abs(out_norm - in_norm)

    # Look for blue contamination (blue channel changed more than red/green)
    blue_excess = delta[:, :, 2] - (delta[:, :, 0] + delta[:, :, 1]) / 2

    # Stats
    stats = {
        "region_size": boundary_region_in.shape,
        "mean_delta_r": float(delta[:, :, 0].mean()),
        "mean_delta_g": float(delta[:, :, 1].mean()),
        "mean_delta_b": float(delta[:, :, 2].mean()),
        "max_delta_r": float(delta[:, :, 0].max()),
        "max_delta_g": float(delta[:, :, 1].max()),
        "max_delta_b": float(delta[:, :, 2].max()),
        "blue_contamination_mean": float(blue_excess.mean()),
        "blue_contamination_max": float(blue_excess.max()),
        "blue_contamination_pixels_significant": int((blue_excess > 0.05).sum()),
    }

    return stats, delta, blue_excess


def create_artifact_visualization(input_arr, output_arr, delta_magnitude, artifact_map, output_path):
    """Create a visualization showing artifacts."""
    # Crop to common size
    min_h = min(input_arr.shape[0], output_arr.shape[0])
    min_w = min(input_arr.shape[1], output_arr.shape[1])

    input_arr = input_arr[:min_h, :min_w, :]
    output_arr = output_arr[:min_h, :min_w, :]

    # Convert to uint8 for display
    if input_arr.dtype == np.uint16:
        input_display = (input_arr.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        output_display = (output_arr.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
    else:
        input_display = input_arr
        output_display = output_arr

    # Create heatmap of artifacts
    artifact_heatmap = (
        np.clip(artifact_map / artifact_map.max() if artifact_map.max() > 0 else artifact_map, 0, 1) * 255
    ).astype(np.uint8)
    artifact_rgb = np.zeros((*artifact_heatmap.shape, 3), dtype=np.uint8)
    artifact_rgb[:, :, 0] = artifact_heatmap  # Red channel for artifacts

    # Stack: input | output | artifact heatmap
    h, w = input_display.shape[:2]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w, :] = input_display
    combined[:, w : 2 * w, :] = output_display
    combined[:, 2 * w :, :] = artifact_rgb

    img = Image.fromarray(combined)
    img.save(output_path, quality=95)

    return output_path


def main():
    input_path = Path(
        "/Users/rc/Projects/Transformation_Portal/input_images/750Picacho_16-bit_TIFFs/750Picacho_PrimaryBedroom_master16.tif"
    )
    output_path = Path(
        "/Users/rc/Projects/Transformation_Portal/output_bugfix_validation_final/v2/750Picacho_PrimaryBedroom_master16_tif_032ef607_materials_v3_enhanced.tif"
    )
    manifest_path = Path(
        "/Users/rc/Projects/Transformation_Portal/output_bugfix_validation_final/manifests/750Picacho_PrimaryBedroom_master16_tif_032ef607_combined.json"
    )

    print("=" * 80)
    print("PRIMARY BEDROOM EDGE ARTIFACT DIAGNOSIS")
    print("=" * 80)

    # Load images
    print("\n📥 Loading images...")
    input_arr = tifffile.imread(str(input_path))
    output_arr = tifffile.imread(str(output_path))

    print(f"   Input:  {input_arr.shape}, {input_arr.dtype}, range [{input_arr.min()}, {input_arr.max()}]")
    print(f"   Output: {output_arr.shape}, {output_arr.dtype}, range [{output_arr.min()}, {output_arr.max()}]")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check Materials V3 operations
    print("\n🔧 Materials V3 Operations Applied:")
    mat_v3 = manifest.get("materials_v3", {})
    pixel_ops = mat_v3.get("pixel_ops", {})
    applied = pixel_ops.get("applied", [])

    for op in applied:
        mat = op.get("material")
        ops_list = op.get("ops", [])
        delta_stats = op.get("delta_stats", {})
        print(f"   {mat}: {ops_list}")
        print(f"      Mean delta: {delta_stats.get('_debug_mean_delta_all_pixels', 0):.6f}")
        print(f"      Max delta: {delta_stats.get('_debug_max_delta', 0):.6f}")

    # Detect edge artifacts
    print("\n🔍 Detecting Edge Artifacts...")
    try:
        from scipy import ndimage

        delta, delta_mag, edges, artifact_map = detect_edge_artifacts(input_arr, output_arr)

        artifact_pixels = (artifact_map > 0.05).sum()
        artifact_pct = (artifact_pixels / artifact_map.size) * 100

        print(f"   Total pixels with artifacts (>5% change at edges): {artifact_pixels:,} ({artifact_pct:.2f}%)")
        print(f"   Max artifact magnitude: {artifact_map.max():.4f}")
        print(f"   Mean artifact magnitude: {artifact_map[artifact_map > 0.05].mean() if artifact_pixels > 0 else 0:.4f}")

        # Find worst artifact regions
        if artifact_pixels > 0:
            artifact_coords = np.argwhere(artifact_map > 0.1)  # Significant artifacts
            if len(artifact_coords) > 0:
                print(f"\n   🔴 Found {len(artifact_coords):,} pixels with SIGNIFICANT artifacts (>10% change at edges)")

                # Sample some locations
                sample_size = min(5, len(artifact_coords))
                sample_idx = np.random.choice(len(artifact_coords), sample_size, replace=False)

                print(f"\n   Sample artifact locations:")
                for idx in sample_idx:
                    y, x = artifact_coords[idx]
                    artifact_val = artifact_map[y, x]
                    delta_rgb = delta[y, x]
                    print(
                        f"      ({y}, {x}): magnitude={artifact_val:.4f}, RGB delta=({delta_rgb[0]:.4f}, {delta_rgb[1]:.4f}, {delta_rgb[2]:.4f})"
                    )

        # Create visualization
        print("\n🎨 Creating artifact visualization...")
        viz_path = create_artifact_visualization(
            input_arr, output_arr, delta_mag, artifact_map, "primary_bedroom_artifacts_visualization.jpg"
        )
        print(f"   Saved to: {viz_path}")
        print(f"   (Left: INPUT | Middle: OUTPUT | Right: ARTIFACT HEATMAP)")

    except ImportError:
        print("   ⚠️  scipy not available, skipping edge detection")

    # Analyze sky/ocean boundary
    print("\n🌊 Analyzing Sky/Ocean Boundary...")
    boundary_stats, boundary_delta, blue_excess = analyze_sky_ocean_boundary(input_arr, output_arr)

    print(f"   Region analyzed: top {boundary_stats['region_size'][0]} rows")
    print(f"\n   Mean delta by channel:")
    print(f"      Red:   {boundary_stats['mean_delta_r']:.6f}")
    print(f"      Green: {boundary_stats['mean_delta_g']:.6f}")
    print(f"      Blue:  {boundary_stats['mean_delta_b']:.6f}")

    print(f"\n   Max delta by channel:")
    print(f"      Red:   {boundary_stats['max_delta_r']:.4f}")
    print(f"      Green: {boundary_stats['max_delta_g']:.4f}")
    print(f"      Blue:  {boundary_stats['max_delta_b']:.4f}")

    print(f"\n   Blue contamination:")
    print(f"      Mean: {boundary_stats['blue_contamination_mean']:.6f}")
    print(f"      Max: {boundary_stats['blue_contamination_max']:.4f}")
    print(f"      Pixels with >5% blue excess: {boundary_stats['blue_contamination_pixels_significant']:,}")

    if boundary_stats["blue_contamination_pixels_significant"] > 1000:
        print(f"      🔴 SIGNIFICANT blue contamination detected!")
    elif boundary_stats["blue_contamination_pixels_significant"] > 100:
        print(f"      ⚠️  Moderate blue contamination detected")
    else:
        print(f"      ✅ Minimal blue contamination")

    # Check for white halos (all channels increase together at edges)
    white_halo = np.all(delta > 0.05, axis=2)
    white_halo_pixels = white_halo.sum()
    white_halo_pct = (white_halo_pixels / white_halo.size) * 100

    print(f"\n🔳 White Halo Detection:")
    print(f"   Pixels with all RGB >5% increase: {white_halo_pixels:,} ({white_halo_pct:.2f}%)")

    if white_halo_pct > 1:
        print(f"   🔴 SIGNIFICANT white halos detected!")
    elif white_halo_pct > 0.1:
        print(f"   ⚠️  Moderate white halos detected")
    else:
        print(f"   ✅ Minimal white halos")

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    issues = []

    if "artifact_pixels" in locals() and artifact_pct > 1:
        issues.append(f"Edge artifacts: {artifact_pct:.2f}% of pixels affected")

    if boundary_stats["blue_contamination_pixels_significant"] > 1000:
        issues.append(f"Sky/ocean blue contamination: {boundary_stats['blue_contamination_pixels_significant']:,} pixels")

    if white_halo_pct > 1:
        issues.append(f"White halos: {white_halo_pct:.2f}% of pixels")

    if issues:
        print("\n🔴 ISSUES DETECTED:")
        for issue in issues:
            print(f"   • {issue}")

        print("\n💡 LIKELY CAUSES:")
        if "foliage" in [op.get("material") for op in applied]:
            print("   • Foliage vibrance_boost may be bleeding at mask edges")
        if "glass" in [op.get("material") for op in applied]:
            print("   • Glass brightness_boost may be creating halos")
        print("   • SAM2 mask boundaries may not be smooth enough")
        print("   • V2 enhancement may be amplifying Materials V3 edge artifacts")

        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Apply edge feathering/blur to Materials V3 masks before pixel ops")
        print("   2. Reduce pixel op strength near mask boundaries")
        print("   3. Use dilate/erode on masks to create smoother boundaries")
        print("   4. Consider Gaussian blur on masks (sigma=2-5 pixels)")
    else:
        print("\n✅ NO SIGNIFICANT ARTIFACTS DETECTED")
        print("   All changes within normal enhancement thresholds")

    # Save full report
    report = {
        "image": "Primary Bedroom",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "materials_v3_operations": applied,
        "boundary_analysis": boundary_stats,
        "edge_artifacts": (
            {
                "pixels_affected": int(artifact_pixels) if "artifact_pixels" in locals() else 0,
                "percentage": float(artifact_pct) if "artifact_pct" in locals() else 0,
            }
            if "artifact_pixels" in locals()
            else None
        ),
        "white_halos": {
            "pixels": int(white_halo_pixels),
            "percentage": float(white_halo_pct),
        },
        "issues": issues,
    }

    with open("primary_bedroom_artifact_diagnosis.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n💾 Full report saved to: primary_bedroom_artifact_diagnosis.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
