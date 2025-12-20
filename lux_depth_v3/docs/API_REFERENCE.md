# Depth Anything 3 API Reference

Complete reference for the DA3 Python API integration in Lux Depth V3.

## Table of Contents

1. [Overview](#overview)
2. [Model Variants](#model-variants)
3. [API Classes](#api-classes)
4. [Parameters Reference](#parameters-reference)
5. [Export Formats](#export-formats)
6. [Gaussian Splatting](#gaussian-splatting)
7. [Feature Extraction](#feature-extraction)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Overview

The DA3 Python API provides comprehensive depth estimation capabilities with support for:

- **Monocular depth estimation**: Single-image depth prediction
- **Multi-view depth estimation**: Pose-conditioned depth from multiple views
- **Gaussian Splatting (3DGS)**: 3D reconstruction with neural rendering
- **Feature extraction**: Access intermediate layer representations
- **Multiple export formats**: NPZ, GLB, PLY, videos, and visualizations

### Quick Start

```python
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper
from pathlib import Path

# Initialize wrapper
wrapper = DepthAnything3Wrapper(model_name="da3-large", device="cuda")

# Run inference
prediction = wrapper.inference(
    image=["path/to/image.jpg"],
    export_dir="output",
    export_format="mini_npz-glb"
)

print(f"Depth shape: {prediction.depth.shape}")
```

## Model Variants

### Available Models

| Model Name | Parameters | Capabilities | Use Case |
|------------|-----------|--------------|----------|
| `da3nested-giant-large` | 1.40B | All features (any-view + metric + GS) | Maximum capability |
| `da3-giant` | 1.15B | Any-view with GS support | Multi-view + GS workflows |
| `da3-large` | 0.35B | Recommended general use | Best balance |
| `da3-base` | 0.12B | Balanced performance | Resource-constrained |
| `da3-small` | 0.08B | Lightweight | Fast inference |
| `da3metric-large` | 0.35B | Metric depth + sky segmentation | Absolute scale needed |
| `da3mono-large` | 0.35B | Monocular only | High-quality single-image |

### Model Selection Guidelines

**Choose `da3-large` (default)** for:
- General-purpose depth estimation
- Good balance of quality and speed
- Most production use cases

**Choose `da3-giant` for**:
- Multi-view reconstruction
- Gaussian Splatting workflows
- Maximum quality requirements

**Choose `da3metric-large` for**:
- Applications requiring absolute depth scale
- Outdoor scenes with sky segmentation
- Architectural measurements

**Choose `da3nested-giant-large` for**:
- Research and experimentation
- Maximum feature set
- When computational resources are abundant

## API Classes

### DepthAnything3Wrapper

Main wrapper class for DA3 Python API.

```python
class DepthAnything3Wrapper:
    def __init__(
        self,
        model_name: str = "da3-large",
        device: str = "cuda"
    )
```

**Parameters:**
- `model_name`: Model variant name (see Model Variants)
- `device`: Device to use (`cuda`, `cpu`, `mps`)

**Methods:**

#### `inference()`

Run depth estimation with full API capabilities.

```python
def inference(
    # Input parameters
    image: List[Union[np.ndarray, Image.Image, str, Path]],
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    
    # Pose alignment parameters
    align_to_input_ext_scale: bool = True,
    infer_gs: bool = False,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    
    # Rendering parameters (for gs_video)
    render_exts: Optional[np.ndarray] = None,
    render_ixts: Optional[np.ndarray] = None,
    render_hw: Optional[Tuple[int, int]] = None,
    
    # Processing parameters
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    
    # Export parameters
    export_dir: Optional[Union[str, Path]] = None,
    export_format: str = "mini_npz",
    export_feat_layers: List[int] = [],
    
    # GLB export parameters
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    
    # Feature visualization parameters
    feat_vis_fps: int = 15,
    
    # Additional export kwargs
    export_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> DA3Prediction
```

### DA3Prediction

Result dataclass containing all prediction outputs.

```python
@dataclass
class DA3Prediction:
    depth: np.ndarray              # (N, H, W) depth maps
    conf: Optional[np.ndarray]     # (N, H, W) confidence maps
    extrinsics: Optional[np.ndarray]  # (N, 4, 4) camera extrinsics
    intrinsics: Optional[np.ndarray]  # (N, 3, 3) camera intrinsics
    processed_images: Optional[np.ndarray]  # (N, H, W, 3) processed RGB
    aux: Optional[Dict[str, Any]]  # Auxiliary outputs (GS, features, etc.)
```

**Fields:**
- `depth`: Primary depth prediction (normalized 0-1 or metric scale)
- `conf`: Per-pixel confidence scores (higher = more confident)
- `extrinsics`: Estimated or refined camera poses (world-to-camera)
- `intrinsics`: Estimated or refined camera intrinsics
- `processed_images`: Preprocessed input images
- `aux`: Additional outputs based on `export_format` and flags

## Parameters Reference

### Input Parameters

#### `image`
- **Type**: `List[Union[np.ndarray, Image.Image, str, Path]]`
- **Description**: Input images as arrays, PIL Images, or file paths
- **Examples**:
  ```python
  # Single image (path)
  image=["image.jpg"]
  
  # Multiple images (paths)
  image=["img1.jpg", "img2.jpg", "img3.jpg"]
  
  # NumPy arrays
  image=[np.array(...), np.array(...)]
  
  # PIL Images
  image=[Image.open("img1.jpg"), Image.open("img2.jpg")]
  ```

#### `extrinsics`
- **Type**: `Optional[np.ndarray]` shape `(N, 4, 4)`
- **Description**: Camera extrinsics (world-to-camera transformation matrices)
- **Default**: `None` (will be estimated if multi-view)
- **Format**: 4x4 homogeneous transformation matrices
- **Example**:
  ```python
  # Provide known camera poses
  extrinsics = np.array([
      [[R11, R12, R13, tx],
       [R21, R22, R23, ty],
       [R31, R32, R33, tz],
       [  0,   0,   0,  1]],
      # ... for each view
  ])
  ```

#### `intrinsics`
- **Type**: `Optional[np.ndarray]` shape `(N, 3, 3)`
- **Description**: Camera intrinsics (projection matrices)
- **Default**: `None` (will be estimated)
- **Format**: 3x3 camera matrix
- **Example**:
  ```python
  # Provide known intrinsics
  intrinsics = np.array([
      [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]],
      # ... for each view
  ])
  ```

### Pose Alignment Parameters

#### `align_to_input_ext_scale`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Align predicted poses to input extrinsics scale
- **When to use**: Keep `True` when providing known camera poses

#### `use_ray_pose`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use ray-based pose estimation instead of default method
- **When to use**: Experimental alternative pose estimation

#### `ref_view_strategy`
- **Type**: `str`
- **Options**: `"first"`, `"middle"`, `"saddle_balanced"`, `"saddle_sim_range"`
- **Default**: `"saddle_balanced"`
- **Description**: Strategy for selecting reference view in multi-view scenarios
  - `"first"`: Use first image as reference
  - `"middle"`: Use middle image as reference
  - `"saddle_balanced"`: Balanced saddle point strategy (recommended)
  - `"saddle_sim_range"`: Saddle point with similarity range
- **Example**:
  ```python
  # For sequential captures, middle view often works well
  ref_view_strategy="middle"
  
  # For diverse viewpoints, saddle_balanced is recommended
  ref_view_strategy="saddle_balanced"
  ```

### Gaussian Splatting Parameters

#### `infer_gs`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable Gaussian Splatting branch
- **Requirements**: Requires `da3-giant` or `da3nested-giant-large` model
- **Output**: Adds Gaussian Splatting data to `aux` field
- **Example**:
  ```python
  wrapper = DepthAnything3Wrapper(model_name="da3-giant")
  prediction = wrapper.inference(
      image=image_list,
      infer_gs=True,
      export_format="gs_ply-gs_video"
  )
  ```

#### Rendering Parameters (for gs_video)

When `infer_gs=True` and `export_format` includes `"gs_video"`:

#### `render_exts`
- **Type**: `Optional[np.ndarray]` shape `(M, 4, 4)`
- **Description**: Rendering camera extrinsics (trajectory for video)
- **Example**:
  ```python
  # Create circular camera trajectory
  render_exts = create_circular_trajectory(
      radius=5.0,
      num_frames=120,
      height=2.0
  )
  ```

#### `render_ixts`
- **Type**: `Optional[np.ndarray]` shape `(M, 3, 3)`
- **Description**: Rendering camera intrinsics
- **Default**: Uses input intrinsics if not provided

#### `render_hw`
- **Type**: `Optional[Tuple[int, int]]`
- **Description**: Rendering resolution `(height, width)`
- **Example**: `render_hw=(1080, 1920)` for Full HD

### Processing Parameters

#### `process_res`
- **Type**: `int`
- **Default**: `504`
- **Description**: Processing resolution (affects quality/speed tradeoff)
- **Recommended values**:
  - `378`: Fast, lower quality
  - `504`: Default, good balance
  - `672`: Higher quality, slower
  - `1008`: Maximum quality, slowest

#### `process_res_method`
- **Type**: `str`
- **Options**: `"upper_bound_resize"`, `"lower_bound_resize"`
- **Default**: `"upper_bound_resize"`
- **Description**: How to resize images to `process_res`
  - `"upper_bound_resize"`: Resize so maximum dimension ≤ process_res
  - `"lower_bound_resize"`: Resize so minimum dimension ≥ process_res

### Export Parameters

#### `export_dir`
- **Type**: `Optional[Union[str, Path]]`
- **Description**: Directory to save exports (created if doesn't exist)
- **Example**: `export_dir="output/depth_results"`

#### `export_format`
- **Type**: `str`
- **Default**: `"mini_npz"`
- **Description**: Export format(s), separated by `-` for multiple
- **Available formats**: See [Export Formats](#export-formats) section
- **Examples**:
  ```python
  export_format="mini_npz"  # Minimal NPZ only
  export_format="mini_npz-glb"  # NPZ + 3D mesh
  export_format="mini_npz-glb-depth_vis"  # NPZ + mesh + video
  export_format="gs_ply-gs_video"  # Gaussian Splatting outputs
  ```

#### `export_feat_layers`
- **Type**: `List[int]`
- **Default**: `[]`
- **Description**: Layer indices to extract features from
- **Example**: `export_feat_layers=[0, 3, 6, 9]` for 4 layers

### GLB Export Parameters

Used when `export_format` includes `"glb"`.

#### `conf_thresh_percentile`
- **Type**: `float`
- **Default**: `40.0`
- **Range**: `0.0` - `100.0`
- **Description**: Confidence threshold percentile for point cloud filtering
- **Lower values**: More points (noisier)
- **Higher values**: Fewer points (cleaner, may lose detail)

#### `num_max_points`
- **Type**: `int`
- **Default**: `1_000_000`
- **Description**: Maximum number of points in exported mesh
- **Recommendations**:
  - `100_000`: Fast preview
  - `1_000_000`: Good balance
  - `5_000_000`: High detail

#### `show_cameras`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Include camera frustums in GLB visualization

### Feature Visualization Parameters

Used when `export_format` includes `"feat_vis"`.

#### `feat_vis_fps`
- **Type**: `int`
- **Default**: `15`
- **Description**: Frame rate for feature visualization video

## Export Formats

### Core Formats

#### `mini_npz`
Minimal NumPy compressed archive with depth and confidence.

**Contents:**
- `depth`: (N, H, W) float32 depth maps
- `conf`: (N, H, W) float32 confidence maps

**Use case**: Lightweight storage, further processing in Python

**Example:**
```python
prediction = wrapper.inference(
    image=images,
    export_format="mini_npz",
    export_dir="output"
)

# Load results
data = np.load("output/result.npz")
depth = data['depth']
conf = data['conf']
```

#### `full_npz`
Complete NumPy archive with all data.

**Contents:**
- `depth`: Depth maps
- `conf`: Confidence maps
- `extrinsics`: Camera poses
- `intrinsics`: Camera intrinsics
- `images`: Processed input images

**Use case**: Complete data archival, reproducibility

#### `glb`
GLTF binary 3D mesh with texture.

**Contents:**
- 3D point cloud mesh
- RGB texture mapping
- Camera frustums (if `show_cameras=True`)

**Use case**: 3D visualization, Blender/Unity import

**Parameters:**
- `conf_thresh_percentile`: Quality threshold
- `num_max_points`: Point count limit
- `show_cameras`: Include camera visualization

**Example:**
```python
prediction = wrapper.inference(
    image=images,
    export_format="glb",
    export_dir="output",
    conf_thresh_percentile=50.0,  # Higher quality
    num_max_points=2_000_000,     # More points
    show_cameras=True
)
```

#### `gs_ply`
Gaussian Splatting PLY point cloud.

**Requirements:** `infer_gs=True` and GS-capable model

**Contents:**
- 3D Gaussian splat parameters
- Per-splat opacity, scale, rotation

**Use case**: 3D Gaussian Splatting viewers, research

#### `gs_video`
Gaussian Splatting rendered video.

**Requirements:** 
- `infer_gs=True`
- `render_exts` (camera trajectory)

**Contents:**
- MP4 video of novel view synthesis

**Parameters:**
- `render_exts`: Rendering camera poses
- `render_ixts`: Rendering intrinsics (optional)
- `render_hw`: Output resolution

**Example:**
```python
# Create camera trajectory
num_frames = 120
render_exts = create_orbit_trajectory(num_frames)

prediction = wrapper.inference(
    image=training_images,
    infer_gs=True,
    export_format="gs_ply-gs_video",
    render_exts=render_exts,
    render_hw=(1080, 1920)
)
```

#### `depth_vis`
Depth visualization video.

**Contents:**
- MP4 video with colorized depth maps
- Per-frame depth visualization

**Use case**: Qualitative assessment, presentations

#### `feat_vis`
Feature visualization video.

**Requirements:** `export_feat_layers` not empty

**Contents:**
- MP4 video showing intermediate feature maps

**Parameters:**
- `export_feat_layers`: Which layers to visualize
- `feat_vis_fps`: Video frame rate

**Example:**
```python
prediction = wrapper.inference(
    image=images,
    export_format="feat_vis",
    export_feat_layers=[0, 3, 6, 9, 12],
    feat_vis_fps=10
)
```

### Format Combinations

Combine multiple formats with `-` separator:

```python
# Basic: depth + 3D mesh
export_format="mini_npz-glb"

# Complete: all standard outputs
export_format="full_npz-glb-depth_vis"

# GS workflow: all GS outputs
export_format="mini_npz-gs_ply-gs_video"

# Research: everything
export_format="full_npz-glb-gs_ply-gs_video-depth_vis-feat_vis"
```

## Gaussian Splatting

### Overview

Gaussian Splatting (3DGS) enables high-quality novel view synthesis from multi-view depth estimations.

**Requirements:**
- Model: `da3-giant` or `da3nested-giant-large`
- Flag: `infer_gs=True`
- Multiple input views (typically 10-50 images)

### Basic GS Workflow

```python
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper
import numpy as np

# 1. Initialize with GS-capable model
wrapper = DepthAnything3Wrapper(
    model_name="da3-giant",
    device="cuda"
)

# 2. Prepare input images
images = [f"scene/image_{i:03d}.jpg" for i in range(30)]

# 3. Run inference with GS enabled
prediction = wrapper.inference(
    image=images,
    infer_gs=True,
    export_format="gs_ply",
    export_dir="output/gs_scene"
)

# 4. Access GS data
gs_data = prediction.aux['gaussian_splatting']
print(f"Generated {len(gs_data['splats'])} Gaussian splats")
```

### Rendering Novel Views

```python
# Define rendering camera trajectory
def create_circular_trajectory(
    center=(0, 0, 0),
    radius=5.0,
    height=2.0,
    num_frames=120
):
    """Create circular camera trajectory."""
    angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    
    extrinsics = []
    for angle in angles:
        # Camera position
        cam_x = center[0] + radius * np.cos(angle)
        cam_y = center[1] + height
        cam_z = center[2] + radius * np.sin(angle)
        
        # Look at center
        forward = np.array(center) - np.array([cam_x, cam_y, cam_z])
        forward = forward / np.linalg.norm(forward)
        
        # Compute camera rotation
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build extrinsic matrix
        R = np.column_stack([right, up, -forward])
        t = np.array([cam_x, cam_y, cam_z])
        
        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = t
        extrinsics.append(ext)
    
    return np.array(extrinsics)

# Render video
render_exts = create_circular_trajectory(num_frames=240)

prediction = wrapper.inference(
    image=training_images,
    infer_gs=True,
    export_format="gs_video",
    render_exts=render_exts,
    render_hw=(1080, 1920),
    export_dir="output/rendered"
)
```

### GS Quality Optimization

**Image Coverage:**
- Use 20-50 images for best quality
- Ensure good spatial coverage of the scene
- Avoid large gaps between viewpoints

**Processing Resolution:**
- Higher `process_res` improves GS quality
- Recommended: `process_res=672` or `process_res=1008`

**Reference View Strategy:**
- Use `ref_view_strategy="saddle_balanced"` for diverse views
- Use `ref_view_strategy="middle"` for sequential captures

```python
# High-quality GS reconstruction
prediction = wrapper.inference(
    image=images,
    infer_gs=True,
    process_res=672,  # Higher quality
    ref_view_strategy="saddle_balanced",
    export_format="gs_ply-gs_video",
    export_dir="output/high_quality"
)
```

## Feature Extraction

Extract intermediate layer features for analysis or downstream tasks.

### Basic Feature Extraction

```python
# Extract features from 4 layers
prediction = wrapper.inference(
    image=images,
    export_feat_layers=[0, 3, 6, 9],
    export_format="feat_vis",
    export_dir="output/features"
)

# Access feature maps
features = prediction.aux['features']
print(f"Extracted features: {features.keys()}")
```

### Layer Selection Guidelines

DA3 architecture has multiple stages. Common layer choices:

- **Layer 0**: Early features (edges, textures)
- **Layers 3-6**: Mid-level features (object parts, patterns)
- **Layers 9-12**: High-level features (semantic understanding)

**Example - Semantic features:**
```python
export_feat_layers=[9, 10, 11, 12]
```

**Example - Multi-scale features:**
```python
export_feat_layers=[0, 3, 6, 9, 12]
```

### Feature Visualization

```python
# Visualize features as video
prediction = wrapper.inference(
    image=images,
    export_feat_layers=list(range(13)),  # All layers
    export_format="feat_vis",
    feat_vis_fps=15,
    export_dir="output/feat_viz"
)
```

## Examples

### Example 1: Basic Monocular Depth

```python
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper

wrapper = DepthAnything3Wrapper(model_name="da3-large")

prediction = wrapper.inference(
    image=["interior.jpg"],
    export_dir="output",
    export_format="mini_npz"
)

import matplotlib.pyplot as plt
plt.imshow(prediction.depth[0])
plt.colorbar()
plt.show()
```

### Example 2: Multi-View with Known Poses

```python
import numpy as np

# Load camera calibration
extrinsics = np.load("camera_poses.npy")  # (N, 4, 4)
intrinsics = np.load("camera_intrinsics.npy")  # (N, 3, 3)

wrapper = DepthAnything3Wrapper(model_name="da3-large")

prediction = wrapper.inference(
    image=["view_01.jpg", "view_02.jpg", "view_03.jpg"],
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_format="full_npz-glb",
    export_dir="output/multiview"
)

print(f"Refined poses: {prediction.extrinsics.shape}")
```

### Example 3: Gaussian Splatting Reconstruction

```python
from pathlib import Path

# Collect scene images
scene_dir = Path("scene_capture")
images = sorted(scene_dir.glob("*.jpg"))

wrapper = DepthAnything3Wrapper(model_name="da3-giant")

# Reconstruct scene
prediction = wrapper.inference(
    image=images,
    infer_gs=True,
    process_res=672,  # Higher quality
    export_format="gs_ply-depth_vis",
    export_dir="output/gs_reconstruction"
)

# Render novel view video
from utils import create_orbit_trajectory

render_exts = create_orbit_trajectory(radius=10, num_frames=240)

video_prediction = wrapper.inference(
    image=images,
    infer_gs=True,
    export_format="gs_video",
    render_exts=render_exts,
    render_hw=(1080, 1920),
    export_dir="output/rendered_video"
)
```

### Example 4: Metric Depth with Sky Segmentation

```python
wrapper = DepthAnything3Wrapper(model_name="da3metric-large")

prediction = wrapper.inference(
    image=["exterior_scene.jpg"],
    export_format="full_npz",
    export_dir="output/metric_depth"
)

# Access metric depth (absolute scale)
depth_meters = prediction.depth[0]
print(f"Depth range: {depth_meters.min():.2f}m - {depth_meters.max():.2f}m")

# Sky segmentation available in aux
if prediction.aux and 'sky_mask' in prediction.aux:
    sky_mask = prediction.aux['sky_mask']
    print(f"Sky pixels: {sky_mask.sum() / sky_mask.size * 100:.1f}%")
```

### Example 5: Feature Extraction for Analysis

```python
wrapper = DepthAnything3Wrapper(model_name="da3-large")

# Extract multi-scale features
prediction = wrapper.inference(
    image=["sample.jpg"],
    export_feat_layers=[0, 3, 6, 9, 12],
    export_format="full_npz-feat_vis",
    export_dir="output/features"
)

# Analyze features
features = prediction.aux['features']
for layer_idx, feat_map in features.items():
    print(f"Layer {layer_idx}: shape {feat_map.shape}")
    
# Features saved as NPZ and video
```

## Troubleshooting

### Import Error: depth_anything_3 not found

**Problem**: Official DA3 API not installed

**Solution**:
```bash
pip install depth-anything-3
```

### Model Download Issues

**Problem**: Model fails to download or load

**Solutions**:
1. Check internet connection
2. Set cache directory with write permissions
3. Manually download model:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id="depth-anything/Depth-Anything-V3",
       local_dir="models/da3"
   )
   ```

### CUDA Out of Memory

**Problem**: GPU memory exhausted

**Solutions**:
1. Reduce `process_res`:
   ```python
   process_res=378  # Lower resolution
   ```

2. Process fewer images at once:
   ```python
   # Batch processing
   for batch in image_batches:
       prediction = wrapper.inference(image=batch, ...)
   ```

3. Use CPU (slower):
   ```python
   wrapper = DepthAnything3Wrapper(device="cpu")
   ```

### Gaussian Splatting Error: Model Not Compatible

**Problem**: `infer_gs=True` with non-GS model

**Solution**: Use GS-capable model:
```python
wrapper = DepthAnything3Wrapper(model_name="da3-giant")
# or
wrapper = DepthAnything3Wrapper(model_name="da3nested-giant-large")
```

### GLB Export Too Large

**Problem**: Exported GLB file is excessively large

**Solutions**:
1. Reduce point count:
   ```python
   num_max_points=500_000  # Fewer points
   ```

2. Increase confidence threshold:
   ```python
   conf_thresh_percentile=60.0  # Higher threshold
   ```

### Poor Multi-View Quality

**Problem**: Multi-view reconstruction has artifacts

**Improvements**:
1. Add more views with better coverage
2. Ensure images are well-focused and properly exposed
3. Use higher processing resolution:
   ```python
   process_res=672
   ```
4. Try different reference view strategies:
   ```python
   ref_view_strategy="saddle_balanced"
   ```

### Feature Visualization Empty

**Problem**: No features in `feat_vis` export

**Check**:
1. Ensure `export_feat_layers` is not empty
2. Verify layer indices are valid (typically 0-12)
3. Check export directory for output files

---

**For additional support:**
- GitHub Issues: https://github.com/DepthAnything/Depth-Anything-V3/issues
- Lux Depth V3 Issues: https://github.com/RC219805/Transformation_Portal/issues
