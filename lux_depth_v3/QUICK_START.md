# Lux Depth V3 - Quick Start Guide

## DA3 Integration Overview

This module provides production-ready integration of **Depth Anything 3 (DA3)** into the Transformation Portal pipeline.

### ✅ What's Implemented

1. **Core Integration** (`da3_integration.py`)
   - `estimate_depth()` - Simple function for quick depth estimation
   - `DA3DepthEstimator` - Full-featured class with batch processing
   - `convert_to_metric_depth()` - Metric depth conversion

2. **Model Cache Management** (`model_cache.py`)
   - Download and cache DA3 models
   - Offline operation after initial download
   - CLI commands: `cache-download`, `cache-list`, `cache-stats`

3. **Metric Depth Conversion** (`metric_depth.py`)
   - DA3METRIC-LARGE → meters conversion
   - Formula: `metric_depth = focal_length_px * net_output / 300.0`
   - Support for camera intrinsics and FOV estimation

4. **CLI Interface** (`cli.py`)
   - Batch processing with presets
   - Multiple export formats (NPZ, GLB, PLY, etc.)
   - Backend service acceleration (10-20x speedup)

## Quick Test

### User's Example

```python
from lux_depth_v3.da3_integration import estimate_depth

# Your test case (works immediately!)
result = estimate_depth(
    "input_images/750_Picacho/Kitchen_2K_test.png",
    "output/depth/",
    model="large-1.1"
)

if result.success:
    depth = result.depth_array
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
```

### Run Quick Test Script

```bash
# Test the implementation
python test_da3_integration.py
```

## Available Models

```python
from lux_depth_v3.da3_integration import DA3DepthEstimator

# All available models
DA3DepthEstimator.AVAILABLE_MODELS = {
    "giant-1.1": "depth-anything/DA3-GIANT-1.1",
    "large-1.1": "depth-anything/DA3-LARGE-1.1",  # Your choice
    "base": "depth-anything/DA3-BASE",
    "small": "depth-anything/DA3-SMALL",
    "metric-large": "depth-anything/DA3METRIC-LARGE",
    "mono-large": "depth-anything/DA3MONO-LARGE",
    "nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}
```

## Export Formats

- **mini_npz**: Lightweight NPZ with depth + confidence
- **npz**: Full NPZ with all data
- **glb**: 3D mesh export
- **depth_vis**: Depth visualization images
- **gs_ply**: Gaussian Splatting PLY
- **feat_vis**: Feature visualization

Combine with `-`: `"glb-depth_vis-mini_npz"`

## Usage Examples

### 1. Basic Depth Estimation

```python
from lux_depth_v3.da3_integration import estimate_depth

result = estimate_depth(
    image_path="render.jpg",
    output_dir="output/",
    model="large-1.1",
    device="cpu"  # or "cuda", "mps"
)

# Access results
if result.success:
    depth = result.depth_array  # (H, W) numpy array
    conf = result.confidence_array  # Confidence map
```

### 2. Using DA3DepthEstimator Class

```python
from lux_depth_v3.da3_integration import DA3DepthEstimator

estimator = DA3DepthEstimator(
    model="large-1.1",
    device="cuda",
    auto_cleanup=True,
    verbose=True
)

# Single image
result = estimator.process_image(
    input_path="image.jpg",
    output_dir="output/",
    export_format="mini_npz-depth_vis",
    process_res=504
)

# Batch directory
result = estimator.process_directory(
    input_dir="renders/",
    output_dir="depth_output/",
    extensions=["jpg", "png"],
    export_format="mini_npz"
)
```

### 3. Metric Depth Conversion

```python
from lux_depth_v3.da3_integration import estimate_depth
from lux_depth_v3.metric_depth import convert_to_metric_depth

# Get depth from DA3METRIC-LARGE
result = estimate_depth("image.jpg", "output/", model="metric-large")
depth = result.depth_array

# Convert to meters
metric_result = convert_to_metric_depth(
    depth,
    model_name="DA3METRIC-LARGE",
    focal_length_px=500.0  # Focal length in pixels
)

depth_meters = metric_result.depth_meters
print(f"Depth range: {depth_meters.min():.2f}m - {depth_meters.max():.2f}m")
```

### 4. Using Camera Intrinsics

```python
import numpy as np

# Camera intrinsics matrix
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

metric_result = convert_to_metric_depth(
    depth,
    intrinsics=K  # More accurate than focal_length_px
)
```

### 5. Using Field of View Estimation

```python
# Estimate focal length from FOV (less accurate)
metric_result = convert_to_metric_depth(
    depth,
    image_width=1920,
    fov_degrees=60.0  # Horizontal FOV
)
```

## CLI Commands

### Cache Management

```bash
# Download essential models
lux-depth-v3 cache-download --set essential

# Download specific models
lux-depth-v3 cache-download --models "large-1.1,metric-large"

# List cached models
lux-depth-v3 cache-list

# Show cache statistics
lux-depth-v3 cache-stats
```

### Batch Processing

```bash
# Process directory with preset
lux-depth-v3 process -i renders/ -o output/ --preset interior_luxury

# Use specific model
lux-depth-v3 process -i renders/ -o output/ --model metric-large

# Export multiple formats
lux-depth-v3 process -i renders/ -o output/ -f png -f npz -f ply
```

### Backend Service (10-20x speedup)

```bash
# Start backend
lux-depth-v3 backend-start --model-dir ~/.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1

# Use backend for processing
lux-depth-v3 process -i renders/ -o output/ --use-cli --use-backend

# Stop backend
lux-depth-v3 backend-stop --port 8008
```

## Integration with Existing Pipeline

### With lux_depth_v2

```python
from lux_depth_v3.da3_integration import estimate_depth
from lux_depth_v2.pipeline import LuxDepthPipeline

# Get depth from DA3
result = estimate_depth("render.jpg", "temp/", model="large-1.1")

# Use in lux_depth_v2 pipeline
pipeline = LuxDepthPipeline()
enhanced = pipeline.process(
    image="render.jpg",
    depth_map=result.depth_array
)
```

### With Validation Pipeline

```python
from lux_depth_v3.validation import DepthValidator

validator = DepthValidator(ground_truth_dir="ground_truth/")
metrics = validator.validate(result)

print(f"RMSE: {metrics.rmse:.4f}")
print(f"δ1: {metrics.delta_1:.3f}")
```

## Device Selection

```python
# Auto-detect best device
estimator = DA3DepthEstimator(device="auto")

# Force specific device
estimator = DA3DepthEstimator(device="cuda")  # NVIDIA GPU
estimator = DA3DepthEstimator(device="mps")   # Apple Silicon
estimator = DA3DepthEstimator(device="cpu")   # CPU fallback
```

## Security Notes 🔒

- ✅ No vulnerable dependencies (no basicsr/realesrgan)
- ✅ Uses official DA3 CLI via subprocess (safe)
- ✅ Input validation in service mode
- ✅ Rate limiting and file size limits
- 📚 See `lux_depth_v3/SECURITY.md` for details

## Troubleshooting

### OpenMP Error on Mac

If you see: `OMP: Error #15: Initializing libomp.dylib`

**Fix**: Set environment variable
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

This is already handled in `da3_integration.py` automatically.

### Model Not Found

1. Check if DA3 CLI is installed:
   ```bash
   which da3
   ```

2. Install DA3 if missing:
   ```bash
   cd depth_anything_3_official
   pip install -e .
   ```

### DA3-LARGE-1.1 Still Downloading?

While model downloads, you can:
- Use smaller models: `"base"` or `"small"`
- Wait for download to complete (~1.4GB)
- Check download progress in HuggingFace cache

## Performance

- **DA3-LARGE-1.1**: ~1.4GB, recommended quality/speed balance
- **DA3-GIANT-1.1**: ~4GB, highest quality
- **DA3-BASE**: ~400MB, faster processing
- **DA3-SMALL**: ~200MB, fastest processing

Backend service provides 10-20x speedup by keeping model in GPU memory.

## Next Steps

1. ✅ Test basic example: `python test_da3_integration.py`
2. ✅ Download models: `lux-depth-v3 cache-download --set essential`
3. ✅ Process batch: `lux-depth-v3 process -i renders/ -o output/`
4. ✅ Integrate with existing pipeline

## Documentation

- `README.md` - Module overview
- `SECURITY.md` - Security guidelines
- `INTEGRATION_GUIDE.md` - Detailed integration guide
- `METRIC_DEPTH_IMPLEMENTATION.md` - Metric depth conversion details

## Support

For issues or questions:
1. Check existing documentation in `lux_depth_v3/docs/`
2. Review test scripts in `lux_depth_v3/tests/`
3. See examples in `lux_depth_v3/examples/`
