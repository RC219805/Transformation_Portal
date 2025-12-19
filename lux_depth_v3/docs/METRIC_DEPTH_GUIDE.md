# Metric Depth Conversion Guide

## Overview

This guide explains how to convert Depth Anything 3 (DA3) depth outputs to metric depth in meters, enabling real-world measurements for architectural visualization, spatial planning, and material estimation.

## Table of Contents

- [What is Metric Depth?](#what-is-metric-depth)
- [When to Use Metric Depth Conversion](#when-to-use-metric-depth-conversion)
- [Model Types](#model-types)
- [Conversion Formula](#conversion-formula)
- [Usage Examples](#usage-examples)
- [Focal Length Determination](#focal-length-determination)
- [Accuracy Considerations](#accuracy-considerations)
- [Troubleshooting](#troubleshooting)

## What is Metric Depth?

**Metric depth** refers to depth values measured in real-world units (meters), as opposed to relative depth values that only preserve ordinal relationships between points.

- **Relative Depth**: Values are proportional but not in real units (e.g., 0.5, 0.8, 1.2)
- **Metric Depth**: Values represent actual distances (e.g., 2.5m, 5.0m, 10.3m)

Metric depth enables:
- Real-world measurements (room dimensions, object sizes)
- Spatial planning and layout design
- Material quantity estimation
- Integration with CAD/BIM workflows
- Accurate 3D reconstruction

## When to Use Metric Depth Conversion

### Required for DA3METRIC-LARGE

The **DA3METRIC-LARGE** model outputs disparity-like values that require conversion to obtain metric depth:

```python
from lux_depth_v3.metric_depth import convert_to_metric_depth

# DA3METRIC-LARGE requires conversion
result = convert_to_metric_depth(
    depth,
    model_name="DA3METRIC-LARGE",
    focal_length_px=500.0
)
depth_meters = result.depth_meters  # Now in meters
```

### Not Required for Nested Models

The **DA3NESTED-GIANT-LARGE** models (v1.0 and v1.1) already output metric depth:

```python
# DA3NESTED-GIANT-LARGE-1.1 already outputs metric depth
result = convert_to_metric_depth(
    depth,
    model_name="DA3NESTED-GIANT-LARGE-1.1"
)

print(result.already_metric)  # True
print(result.depth_meters)    # Same as input (already in meters)
```

## Model Types

### DA3METRIC-LARGE (Requires Conversion)

- **License**: Apache 2.0 (commercial-friendly)
- **Output**: Disparity-like values
- **Scale Constant**: 300.0
- **Conversion**: `metric_depth = focal * output / 300.0`

**Pros**:
- Commercial license
- Smaller model size (0.35B params)
- Good balance of speed and accuracy

**Cons**:
- Requires focal length for conversion
- Extra processing step

### DA3NESTED-GIANT-LARGE (Already Metric)

- **License**: CC-BY-NC-4.0 (non-commercial)
- **Output**: Metric depth in meters
- **Conversion**: Not required
- **Versions**: 1.0 (deprecated), 1.1 (recommended)

**Pros**:
- Direct metric depth output
- No conversion needed
- Higher accuracy for metric estimation

**Cons**:
- Non-commercial license
- Larger model (1.40B params)
- Slower inference

## Conversion Formula

For **DA3METRIC-LARGE**, the conversion formula is:

```
metric_depth = focal_length_px * model_output / 300.0
```

Where:
- `focal_length_px`: Focal length in pixels (average of fx and fy from intrinsics)
- `model_output`: Raw output from DA3METRIC-LARGE
- `300.0`: Model-specific scale constant

### Example

```python
# Model outputs disparity-like value
model_output = 8.5

# Camera has focal length of 600px
focal_length_px = 600.0

# Convert to metric depth
metric_depth = (600.0 * 8.5) / 300.0
# Result: 17.0 meters
```

## Usage Examples

### 1. Using Camera Intrinsics (Recommended)

The most accurate method uses the actual camera intrinsics matrix:

```python
from lux_depth_v3.metric_depth import convert_to_metric_depth
import numpy as np

# Camera intrinsics (from EXIF or calibration)
intrinsics = np.array([
    [2000.0, 0.0, 1920.0],  # fx, 0, cx
    [0.0, 2000.0, 1080.0],  # 0, fy, cy
    [0.0, 0.0, 1.0]
])

# Convert depth
result = convert_to_metric_depth(
    depth=depth_output,
    model_name="DA3METRIC-LARGE",
    intrinsics=intrinsics
)

print(f"Focal length: {result.focal_length_px:.2f}px")
print(f"Scale factor: {result.scale_factor:.4f}")
print(f"Depth range: {result.depth_meters.min():.2f}m - {result.depth_meters.max():.2f}m")
```

### 2. Using Explicit Focal Length

If you know the focal length directly:

```python
result = convert_to_metric_depth(
    depth=depth_output,
    model_name="DA3METRIC-LARGE",
    focal_length_px=500.0
)

depth_meters = result.depth_meters
```

### 3. Using Field of View (Approximation)

When intrinsics are unknown, estimate from image width and FOV:

```python
result = convert_to_metric_depth(
    depth=depth_output,
    model_name="DA3METRIC-LARGE",
    image_width=1920,      # Image width in pixels
    fov_degrees=60.0       # Horizontal field of view
)

# Warning: This is an approximation, use intrinsics for accuracy
```

### 4. Quick Conversion

For simple cases, use the convenience function:

```python
from lux_depth_v3.metric_depth import depth_to_meters

# Quick conversion with focal length
depth_meters = depth_to_meters(
    depth=depth_output,
    focal_length_px=500.0
)
```

### 5. Depth Statistics

Get depth statistics in meters:

```python
from lux_depth_v3.metric_depth import get_depth_statistics

stats = get_depth_statistics(depth_meters)

print(f"Min depth:    {stats['min_m']:.2f}m")
print(f"Max depth:    {stats['max_m']:.2f}m")
print(f"Mean depth:   {stats['mean_m']:.2f}m")
print(f"Median depth: {stats['median_m']:.2f}m")
print(f"Range:        {stats['range_m']:.2f}m")
print(f"Std dev:      {stats['std_m']:.2f}m")
```

### 6. CLI Usage

Convert depth via command line:

```bash
# With focal length
lux-depth-v3 api-process image.jpg -o output/ \
  --model metric-large \
  --metric \
  --focal-length 500.0 \
  --depth-stats

# With FOV estimation
lux-depth-v3 api-process image.jpg -o output/ \
  --model metric-large \
  --metric \
  --fov 60.0 \
  --depth-stats
```

## Focal Length Determination

### Priority Order

The converter determines focal length in this priority order:

1. **Explicit `focal_length_px` parameter** (highest priority)
2. **Extract from `intrinsics` matrix** (recommended)
3. **Estimate from `image_width` + `fov_degrees`** (approximation)

### Method 1: From Camera Intrinsics (Best)

Camera intrinsics matrix K:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Where:
- `fx`, `fy`: Focal lengths in pixels (x and y axes)
- `cx`, `cy`: Principal point (image center)

The converter uses: `focal = (fx + fy) / 2`

**Sources**:
- EXIF data (for calibrated cameras)
- Camera calibration (Zhang's method, chessboard)
- Manufacturer specifications

### Method 2: From Field of View (Approximation)

Formula:
```
focal_px = (image_width / 2) / tan(fov / 2)
```

**Example**:
- Image width: 1920px
- Horizontal FOV: 60°
- Calculated focal: ~554px

**Accuracy**: Less accurate than intrinsics, but acceptable for estimation.

### Method 3: Typical Values by Lens Type

Common focal lengths (35mm full-frame equivalent → pixel focal length):

| Lens Type       | FOV (degrees) | 4K (3840px) focal | 1080p (1920px) focal |
|-----------------|---------------|-------------------|----------------------|
| Ultra-wide 14mm | 114°          | ~800px            | ~400px               |
| Wide 24mm       | 84°           | ~1400px           | ~700px               |
| Normal 35mm     | 63°           | ~2000px           | ~1000px              |
| Normal 50mm     | 47°           | ~2800px           | ~1400px              |
| Portrait 85mm   | 28°           | ~4800px           | ~2400px              |

## Accuracy Considerations

### Factors Affecting Accuracy

1. **Focal Length Precision**
   - Intrinsics: ±1-2% error
   - FOV estimation: ±5-10% error
   - Manual entry: Depends on source

2. **Model Limitations**
   - DA3METRIC-LARGE: Designed for metric estimation
   - Scale ambiguity in monocular depth
   - Performance varies by scene type

3. **Scene Characteristics**
   - Indoor scenes: Better accuracy (2-20m range)
   - Outdoor scenes: More challenging (scale ambiguity)
   - Textureless areas: Lower confidence

### Best Practices

1. **Use Actual Intrinsics**: Always prefer calibrated camera intrinsics
2. **Validate Measurements**: Compare with known distances when possible
3. **Check Range**: Verify depth values are reasonable for scene
4. **Use Masks**: Filter low-confidence regions
5. **Average Multiple Views**: Reduce noise with multi-view fusion

### Expected Accuracy

| Scenario                  | Typical Accuracy |
|---------------------------|------------------|
| Indoor, intrinsics        | ±5-10%           |
| Indoor, FOV estimation    | ±10-20%          |
| Outdoor, intrinsics       | ±15-30%          |
| Outdoor, FOV estimation   | ±30-50%          |

## Real-World Applications

### 1. Architectural Measurements

```python
# Measure room dimensions
depth_meters = result.depth_meters

# Ceiling height (center top vs center bottom)
ceiling_height = abs(depth_meters[100, 1920] - depth_meters[900, 1920])
print(f"Room height: {ceiling_height:.2f}m")

# Wall distance
wall_distance = depth_meters[540, 1920]  # Center pixel
print(f"Wall distance: {wall_distance:.2f}m")
```

### 2. Material Estimation

```python
# Calculate floor area from depth
focal_m = result.focal_length_px / 1000.0
pixel_area = (depth_meters / focal_m) ** 2
total_area_m2 = np.sum(pixel_area)

print(f"Estimated floor area: {total_area_m2:.2f} m²")
```

### 3. Spatial Planning

```python
# Identify furniture placement zones
near_zone = depth_meters < 2.0      # < 2m
mid_zone = (2.0 <= depth_meters) & (depth_meters < 5.0)
far_zone = depth_meters >= 5.0

print(f"Near zone: {np.sum(near_zone)} pixels")
print(f"Mid zone: {np.sum(mid_zone)} pixels")
print(f"Far zone: {np.sum(far_zone)} pixels")
```

## Troubleshooting

### Problem: ValueError: No focal length information

**Cause**: No focal length source provided

**Solution**: Provide one of:
- `intrinsics` matrix
- `focal_length_px` value
- `image_width` + `fov_degrees`

```python
# Add focal length
result = convert_to_metric_depth(
    depth,
    focal_length_px=500.0  # Add this
)
```

### Problem: Unrealistic depth values

**Cause**: Incorrect focal length or scale

**Solution**: Verify focal length calculation

```python
# Check focal length
print(f"Focal length: {result.focal_length_px}px")
print(f"Scale factor: {result.scale_factor}")

# Typical range: 200-3000px for normal lenses
# If outside this range, check intrinsics
```

### Problem: Inconsistent measurements

**Cause**: Using FOV estimation instead of intrinsics

**Solution**: Use actual camera intrinsics

```python
# Get intrinsics from EXIF or calibration
intrinsics = get_camera_intrinsics(image_path)

result = convert_to_metric_depth(
    depth,
    intrinsics=intrinsics  # More accurate
)
```

### Problem: Model already outputs metric depth

**Symptom**: `result.already_metric == True`

**Cause**: Using DA3NESTED model (already metric)

**Solution**: No conversion needed, use depth directly

```python
if result.already_metric:
    print("Depth is already in meters")
    depth_meters = result.depth_meters
else:
    print(f"Converted with scale factor: {result.scale_factor}")
```

## API Reference

See full API documentation in the module docstrings:

```python
from lux_depth_v3.metric_depth import (
    MetricDepthConverter,      # Main converter class
    MetricDepthResult,         # Result dataclass
    convert_to_metric_depth,   # Convenience function
    depth_to_meters,           # Quick conversion
    get_depth_statistics,      # Statistics computation
)

# View docstrings
help(MetricDepthConverter)
help(convert_to_metric_depth)
```

## Further Reading

- [DA3 Official Documentation](https://github.com/depth-anything/depth-anything-3)
- [Camera Calibration Guide](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- [Understanding Camera Intrinsics](https://ksimek.github.io/2013/08/13/intrinsic/)
- [Depth Estimation Best Practices](https://www.depth-estimation-guide.com/)

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases in `tests/test_metric_depth.py`
