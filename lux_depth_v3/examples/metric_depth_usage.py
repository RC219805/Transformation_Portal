"""Examples of metric depth conversion for real-world measurements.

This script demonstrates various use cases for converting DA3 depth outputs
to metric depth in meters, enabling architectural measurements, spatial planning,
and material estimation.
"""

from pathlib import Path
import numpy as np
from typing import Optional

# Import metric depth utilities
try:
    from lux_depth_v3.metric_depth import (
        convert_to_metric_depth,
        get_depth_statistics,
        MetricDepthConverter,
        depth_to_meters
    )
except ImportError:
    print("Error: lux_depth_v3 package not found")
    print("Install with: pip install -e .")
    exit(1)


def example_1_architectural_measurements():
    """Example 1: Architectural measurements from depth map."""
    print("\n" + "="*60)
    print("Example 1: Architectural Measurements")
    print("="*60)
    
    # Simulate depth map from luxury real estate image
    # In practice, this would come from DA3 inference
    depth_output = np.random.uniform(2.0, 20.0, (1080, 1920))
    
    # Camera intrinsics (from EXIF or calibration)
    intrinsics = np.array([
        [2000.0, 0.0, 1920.0],  # fx, 0, cx
        [0.0, 2000.0, 1080.0],  # 0, fy, cy
        [0.0, 0.0, 1.0]
    ])
    
    # Convert to metric depth
    result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3METRIC-LARGE",
        intrinsics=intrinsics
    )
    
    metric_depth = result.depth_meters
    
    print(f"✓ Converted depth to meters")
    print(f"  Focal length: {result.focal_length_px:.2f}px")
    print(f"  Scale factor: {result.scale_factor:.4f}")
    
    # Get statistics
    stats = get_depth_statistics(metric_depth)
    
    print(f"\n📊 Depth Statistics:")
    print(f"  Min depth:    {stats['min_m']:.2f}m")
    print(f"  Max depth:    {stats['max_m']:.2f}m")
    print(f"  Mean depth:   {stats['mean_m']:.2f}m")
    print(f"  Median depth: {stats['median_m']:.2f}m")
    print(f"  Range:        {stats['range_m']:.2f}m")
    
    # Measure specific architectural features
    h, w = metric_depth.shape
    
    # Ceiling height (top center vs bottom center)
    ceiling_point = metric_depth[h // 4, w // 2]
    floor_point = metric_depth[3 * h // 4, w // 2]
    room_height = abs(ceiling_point - floor_point)
    
    # Wall distance (center of image)
    wall_distance = metric_depth[h // 2, w // 2]
    
    print(f"\n🏠 Architectural Features:")
    print(f"  Estimated room height: {room_height:.2f}m")
    print(f"  Wall distance: {wall_distance:.2f}m")


def example_2_room_dimensions():
    """Example 2: Room dimension estimation from interior depth."""
    print("\n" + "="*60)
    print("Example 2: Room Dimension Estimation")
    print("="*60)
    
    # Simulate interior depth map (typical room: 3-6m depth)
    depth_output = np.random.uniform(1.0, 8.0, (720, 1280))
    
    # Use FOV estimation (when intrinsics unknown)
    result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3METRIC-LARGE",
        image_width=1280,
        fov_degrees=65.0  # Typical wide-angle interior lens
    )
    
    print(f"✓ Using FOV-based focal estimation")
    print(f"  Estimated focal: {result.focal_length_px:.2f}px")
    print(f"  (Note: Less accurate than intrinsics)")
    
    metric_depth = result.depth_meters
    
    # Measure room dimensions
    h, w = metric_depth.shape
    
    # Horizontal span (left wall to right wall)
    left_wall = metric_depth[h // 2, w // 4]
    right_wall = metric_depth[h // 2, 3 * w // 4]
    horizontal_span = abs(right_wall - left_wall)
    
    # Vertical span (ceiling to floor)
    ceiling = metric_depth[h // 4, w // 2]
    floor = metric_depth[3 * h // 4, w // 2]
    vertical_span = abs(ceiling - floor)
    
    # Back wall distance
    back_wall = metric_depth[h // 2, w // 2]
    
    print(f"\n📐 Room Dimensions:")
    print(f"  Width (est):  {horizontal_span:.2f}m")
    print(f"  Height (est): {vertical_span:.2f}m")
    print(f"  Depth (est):  {back_wall:.2f}m")
    print(f"  Volume (est): {horizontal_span * vertical_span * back_wall:.2f}m³")


def example_3_material_estimation():
    """Example 3: Material quantity estimation from depth."""
    print("\n" + "="*60)
    print("Example 3: Material Quantity Estimation")
    print("="*60)
    
    # Simulate downward-looking depth map (floor measurement)
    depth_output = np.random.uniform(2.5, 3.5, (1080, 1920))
    focal_px = 500.0
    
    result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3METRIC-LARGE",
        focal_length_px=focal_px
    )
    
    depth_meters = result.depth_meters
    
    # Calculate floor area
    # Each pixel represents area = (depth / focal) ^ 2
    focal_m = focal_px / 1000.0  # Rough conversion to meters
    pixel_area_m2 = (depth_meters / focal_m) ** 2
    total_area_m2 = np.sum(pixel_area_m2)
    
    # Material estimates
    tile_coverage = total_area_m2 * 1.10  # +10% for waste
    paint_area = total_area_m2  # Wall area (simplified)
    paint_liters = paint_area / 10.0  # ~10m²/L coverage
    
    print(f"📦 Material Estimates:")
    print(f"  Floor area: {total_area_m2:.2f}m²")
    print(f"  Tiles needed (with waste): {tile_coverage:.2f}m²")
    print(f"  Paint required: {paint_liters:.1f}L")
    
    # Cost estimation (example prices)
    tile_cost = tile_coverage * 45.0  # $45/m²
    paint_cost = paint_liters * 35.0  # $35/L
    
    print(f"\n💰 Cost Estimates:")
    print(f"  Flooring: ${tile_cost:,.2f}")
    print(f"  Paint: ${paint_cost:,.2f}")
    print(f"  Total: ${tile_cost + paint_cost:,.2f}")


def example_4_model_comparison():
    """Example 4: Compare DA3METRIC vs DA3NESTED models."""
    print("\n" + "="*60)
    print("Example 4: Model Comparison")
    print("="*60)
    
    # Simulate depth output
    depth_output = np.random.uniform(1.0, 20.0, (480, 640))
    focal_px = 500.0
    
    # DA3METRIC-LARGE (requires conversion)
    print("\n1️⃣  DA3METRIC-LARGE:")
    metric_result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3METRIC-LARGE",
        focal_length_px=focal_px
    )
    print(f"  Already metric: {metric_result.already_metric}")
    print(f"  Scale factor: {metric_result.scale_factor:.4f}")
    print(f"  Depth range: {metric_result.depth_meters.min():.2f}m - {metric_result.depth_meters.max():.2f}m")
    
    # DA3NESTED-GIANT-LARGE-1.1 (already metric)
    print("\n2️⃣  DA3NESTED-GIANT-LARGE-1.1:")
    nested_result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3NESTED-GIANT-LARGE-1.1"
    )
    print(f"  Already metric: {nested_result.already_metric}")
    print(f"  Scale factor: {nested_result.scale_factor:.4f}")
    print(f"  Depth range: {nested_result.depth_meters.min():.2f}m - {nested_result.depth_meters.max():.2f}m")
    
    print("\n📝 Key Differences:")
    print("  • DA3METRIC requires focal length for conversion")
    print("  • DA3NESTED outputs metric depth directly")
    print("  • Both can provide accurate measurements")


def example_5_depth_zones():
    """Example 5: Depth-based zone analysis."""
    print("\n" + "="*60)
    print("Example 5: Depth Zone Analysis")
    print("="*60)
    
    # Simulate depth map
    depth_output = np.random.uniform(0.5, 25.0, (720, 1280))
    
    result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3METRIC-LARGE",
        focal_length_px=600.0
    )
    
    depth_meters = result.depth_meters
    
    # Define depth zones
    near_zone = depth_meters < 3.0       # < 3m (foreground)
    mid_zone = (3.0 <= depth_meters) & (depth_meters < 10.0)  # 3-10m
    far_zone = depth_meters >= 10.0      # > 10m (background)
    
    # Calculate zone statistics
    total_pixels = depth_meters.size
    near_pct = 100 * np.sum(near_zone) / total_pixels
    mid_pct = 100 * np.sum(mid_zone) / total_pixels
    far_pct = 100 * np.sum(far_zone) / total_pixels
    
    print(f"🎯 Depth Zones:")
    print(f"  Near (< 3m):    {near_pct:.1f}%")
    print(f"  Mid (3-10m):    {mid_pct:.1f}%")
    print(f"  Far (> 10m):    {far_pct:.1f}%")
    
    # Average depth per zone
    if np.any(near_zone):
        near_avg = np.mean(depth_meters[near_zone])
        print(f"\n  Near zone avg: {near_avg:.2f}m")
    
    if np.any(mid_zone):
        mid_avg = np.mean(depth_meters[mid_zone])
        print(f"  Mid zone avg: {mid_avg:.2f}m")
    
    if np.any(far_zone):
        far_avg = np.mean(depth_meters[far_zone])
        print(f"  Far zone avg: {far_avg:.2f}m")


def example_6_save_load():
    """Example 6: Save and load metric depth results."""
    print("\n" + "="*60)
    print("Example 6: Save/Load Metric Depth")
    print("="*60)
    
    # Create metric depth
    depth_output = np.random.uniform(1.0, 15.0, (480, 640))
    
    result = convert_to_metric_depth(
        depth=depth_output,
        model_name="DA3METRIC-LARGE",
        focal_length_px=550.0
    )
    
    # Save to file
    output_dir = Path("output/metric_depth_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "depth_metric.npz"
    result.save(save_path)
    
    print(f"✓ Saved metric depth to: {save_path}")
    print(f"  File size: {save_path.stat().st_size / 1024:.1f} KB")
    
    # Load from file
    from lux_depth_v3.metric_depth import MetricDepthResult
    
    loaded = MetricDepthResult.load(save_path)
    
    print(f"\n✓ Loaded metric depth")
    print(f"  Focal length: {loaded.focal_length_px:.2f}px")
    print(f"  Scale factor: {loaded.scale_factor:.4f}")
    print(f"  Source model: {loaded.source_model}")
    print(f"  Depth shape: {loaded.depth_meters.shape}")
    
    # Verify data integrity
    assert np.array_equal(result.depth_meters, loaded.depth_meters)
    print(f"\n✓ Data integrity verified")


def example_7_quick_conversion():
    """Example 7: Quick conversion with helper function."""
    print("\n" + "="*60)
    print("Example 7: Quick Conversion")
    print("="*60)
    
    # Simulate depth output
    depth_output = np.random.uniform(2.0, 12.0, (100, 100))
    
    # Quick conversion (single function call)
    depth_meters = depth_to_meters(
        depth=depth_output,
        focal_length_px=500.0
    )
    
    print(f"✓ Quick conversion completed")
    print(f"  Input range: {depth_output.min():.2f} - {depth_output.max():.2f}")
    print(f"  Output range: {depth_meters.min():.2f}m - {depth_meters.max():.2f}m")
    print(f"  Scale factor: {500.0 / 300.0:.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Metric Depth Conversion Examples")
    print("Transformation Portal - Lux Depth V3")
    print("="*60)
    
    try:
        example_1_architectural_measurements()
        example_2_room_dimensions()
        example_3_material_estimation()
        example_4_model_comparison()
        example_5_depth_zones()
        example_6_save_load()
        example_7_quick_conversion()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  • Review the metric depth guide: lux_depth_v3/docs/METRIC_DEPTH_GUIDE.md")
        print("  • Run tests: pytest tests/test_metric_depth.py -v")
        print("  • Try CLI: lux-depth-v3 api-process --help")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
