# DA3 Integration for Transformation Portal

## Overview

This module provides a clean Python wrapper around **Depth Anything 3 (DA3)** for integration into the Transformation Portal luxury real estate rendering pipeline.

**Key Features:**
- ✅ Simple Python API wrapping DA3 CLI
- ✅ Depth array loading from NPZ files
- ✅ Support for all DA3 models and export formats
- ✅ Batch processing for directories
- ✅ Video frame extraction and depth estimation
- ✅ Metric depth conversion utilities

## Quick Start

### Basic Usage

```python
from lux_depth_v3.da3_integration import estimate_depth

# Quick depth estimation
result = estimate_depth(
    "renders/kitchen.jpg",
    "output/kitchen_depth",
    model="large-1.1",
    device="cpu"
)

if result.success:
    print(f"✅ Depth map saved to {result.output_dir}")
    depth = result.depth_array  # Load depth as numpy array
    print(f"Depth shape: {depth.shape}")
```

### Advanced Usage

```python
from lux_depth_v3.da3_integration import DA3DepthEstimator

# Create estimator with custom settings
estimator = DA3DepthEstimator(
    model="nested-giant-large-1.1",  # Best quality + metric scale
    device="cpu",
    auto_cleanup=True,
    verbose=True
)

# Process single image with multiple export formats
result = estimator.process_image(
    input_path="renders/estate.jpg",
    output_dir="output/estate_depth",
    export_format="glb-depth_vis-mini_npz",  # 3D model + visualizations + data
    process_res=1024  # Higher resolution
)

# Access results
if result.success:
    # View 3D point cloud
    glb_file = result.glb_path  # scene.glb
    
    # Access depth array
    depth = result.depth_array  # numpy array (1, H, W)
    confidence = result.confidence_array  # confidence map
    
    # Depth visualizations
    depth_vis = result.depth_vis_dir  # folder with colorized depth images
```

### Batch Processing

```python
# Process entire directory
result = estimator.process_directory(
    input_dir="shoots/750_Picacho",
    output_dir="output/750_Picacho_depth",
    extensions=["jpg", "png"],
    export_format="mini_npz-depth_vis"
)
```

### Video Processing

```python
# Extract frames and estimate depth
result = estimator.process_video(
    input_path="walkthroughs/estate_tour.mp4",
    output_dir="output/estate_tour_depth",
    fps=2.0,  # Extract 2 frames per second
    export_format="glb-depth_vis"
)
```

## Available Models

| Model Name | Size | Best For | Features |
|------------|------|----------|----------|
| `large-1.1` | 0.35B | **General use** (recommended) | Fast, high quality |
| `giant-1.1` | 1.15B | Maximum quality | Best depth accuracy |
| `nested-giant-large-1.1` | 1.40B | Multi-view + metric | Full geometry reconstruction |
| `metric-large` | 0.35B | Metric depth only | Real-world scale (meters) |
| `mono-large` | 0.35B | Monocular depth | High-quality relative depth |
| `base` | 0.12B | Lightweight | Fast inference |
| `small` | 0.08B | Minimal resources | Mobile/edge deployment |

## Export Formats

Combine multiple formats with `-`:

| Format | Description | Output Files |
|--------|-------------|--------------|
| `glb` | 3D point cloud visualization | `scene.glb` |
| `depth_vis` | Colorized depth images | `depth_vis/0000.jpg` |
| `mini_npz` | Compact depth + confidence | `exports/mini_npz/results.npz` |
| `npz` | Full data export | `exports/npz/results.npz` |
| `gs_ply` | 3D Gaussian Splatting | `scene.ply` |
| `gs_video` | Rendered 3DGS video | `gs_video.mp4` |
| `feat_vis` | Feature visualization | `feat_vis.mp4` |

**Example:** `export_format="glb-depth_vis-mini_npz"` creates all three outputs.

## Metric Depth Conversion

For DA3METRIC models, convert to real-world meters:

```python
from lux_depth_v3.da3_integration import convert_to_metric_depth
import numpy as np

# Load depth from DA3METRIC model
estimator = DA3DepthEstimator(model="metric-large")
result = estimator.process_image("image.jpg", "output/")

depth_raw = result.depth_array
focal_length_px = 1000.0  # Camera focal length in pixels

# Convert to meters
depth_meters = convert_to_metric_depth(
    depth_raw,
    focal_length_px,
    model_type="metric"
)

print(f"Depth range: {depth_meters.min():.2f}m to {depth_meters.max():.2f}m")
```

## Integration with Existing Pipelines

### Example: Add Depth to Material Response

```python
from lux_depth_v3.da3_integration import DA3DepthEstimator
from material_response import MaterialResponse

# Estimate depth
estimator = DA3DepthEstimator()
depth_result = estimator.process_image("render.jpg", "temp/")
depth = depth_result.depth_array

# Use depth for zone-based material enhancement
mr = MaterialResponse()
enhanced = mr.enhance_with_depth(
    image="render.jpg",
    depth=depth,
    foreground_strength=0.8,
    background_strength=0.3
)
```

### Example: Batch Process Property Shoot

```python
from pathlib import Path
from lux_depth_v3.da3_integration import DA3DepthEstimator

properties = [
    "750_Picacho",
    "1200_Estates",
    "450_Sunset"
]

estimator = DA3DepthEstimator(device="cpu")

for prop in properties:
    print(f"Processing {prop}...")
    result = estimator.process_directory(
        input_dir=f"shoots/{prop}",
        output_dir=f"output/depth/{prop}",
        export_format="glb-depth_vis"
    )
    
    if result.success:
        print(f"✅ {prop} complete")
    else:
        print(f"❌ {prop} failed: {result.stderr}")
```

## CLI Usage (Direct DA3)

For one-off tasks, you can use the DA3 CLI directly:

```bash
# Single image
export KMP_DUPLICATE_LIB_OK=TRUE
da3 auto renders/kitchen.jpg \
    --export-format glb-depth_vis \
    --device cpu \
    --export-dir output/kitchen

# Batch directory
da3 images shoots/estate/ \
    --export-dir output/estate_depth \
    --device cpu

# Video processing
da3 video walkthrough.mp4 \
    --fps 2.0 \
    --export-dir output/video_depth \
    --device cpu
```

## Performance Notes

**Device Selection:**
- `cpu`: Works on all systems (Mac, Linux, Windows)
- `cuda`: Requires NVIDIA GPU (much faster)
- `mps`: Apple Silicon GPU acceleration (macOS only)

**Processing Times (DA3-LARGE-1.1 on M4 Max):**
- Single image (1024px): ~2-3 seconds (CPU)
- Batch processing: ~20-30 images/minute (CPU)
- With GPU: 3-5x faster

**Memory Usage:**
- DA3-LARGE: ~4GB RAM
- DA3-GIANT: ~8GB RAM
- Processing 4K images: +2-4GB RAM

## Troubleshooting

### OpenMP Error on Mac

If you see `OMP: Error #15: Initializing libiomp5.dylib`:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

This is automatically set by the DA3DepthEstimator class.

### CUDA Not Available

On Mac or systems without NVIDIA GPU, use `device="cpu"`:

```python
estimator = DA3DepthEstimator(device="cpu")
```

### Missing Models

Models are automatically downloaded from HuggingFace on first use. For offline usage, pre-download:

```bash
# Download model once
python -c "from depth_anything_3.api import DepthAnything3; DepthAnything3.from_pretrained('depth-anything/DA3-LARGE-1.1')"
```

## Testing

Run the integration tests:

```bash
# All tests
pytest tests/test_da3_integration.py -v

# Specific test
pytest tests/test_da3_integration.py::TestDA3DepthEstimator::test_process_image_success -v

# Integration tests only
pytest tests/test_da3_integration.py -v -m integration
```

## API Reference

See inline documentation in `lux_depth_v3/da3_integration.py` for complete API details.

### Main Classes

- `DA3DepthEstimator`: Main estimator class
- `DA3Result`: Result dataclass with output paths and arrays
- `estimate_depth()`: Convenience function for quick usage

### Key Methods

- `process_image()`: Process single image
- `process_directory()`: Batch process directory
- `process_video()`: Extract and process video frames
- `depth_array` property: Load depth as numpy array
- `confidence_array` property: Load confidence map

## Examples

See `examples/da3_integration_demo.py` for complete examples.

## License

This integration module: Apache 2.0
DA3 models: See individual model licenses on HuggingFace

## Support

For issues with this integration:
- Create issue in Transformation Portal repository

For DA3 model issues:
- Visit https://github.com/ByteDance-Seed/Depth-Anything-3
- HuggingFace: https://huggingface.co/depth-anything

---

**Last Updated:** 2025-12-19
**Version:** 1.0.0
