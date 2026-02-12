# 3D Reconstruction Guide (Phase 2.3)

## Overview

The **reconstruction module** provides depth-guided 3D scene reconstruction using **3D Gaussian Splatting** (3DGS), a state-of-the-art technique for novel view synthesis and 3D representation.

### Key Features

- **Multi-view 3D reconstruction** from 2+ camera views
- **Depth-guided initialization** using Phase 1 depth maps
- **Segmentation-aware** reconstruction with Phase 2.1 masks
- **Material integration** with Phase 2.2 PBR textures
- **Novel view synthesis** for arbitrary camera positions
- **Mesh export** (PLY, OBJ formats)
- **Geometric validation** (RMSE < 2% target)

### License Notice

⚠️ **Important:** This module uses **Inria 3D Gaussian Splatting** which requires a **research license (non-commercial)**.

- **Tier restriction:** `apex_research`, `apex_research_ultra`, or `experimental`
- **Commercial use:** Requires separate license from Inria GraphDeco team
- **Citation:** Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023

See: https://github.com/graphdeco-inria/gaussian-splatting

---

## Architecture

### 3D Gaussian Splatting Overview

3D Gaussian Splatting represents scenes as collections of oriented 3D Gaussians:

- **Position:** 3D location in world space
- **Color:** RGB appearance (with optional view-dependent spherical harmonics)
- **Scale:** Anisotropic covariance (3D ellipsoid shape)
- **Rotation:** Quaternion orientation
- **Opacity:** Alpha transparency

**Advantages:**
- Real-time rendering (differentiable rasterization)
- High-quality novel view synthesis
- Compact scene representation
- Fast optimization (30k iterations in <30s on GPU)

**vs. NeRF:**
- 100x faster rendering
- Better reconstruction quality
- Explicit 3D representation (exportable to meshes)

---

## Quick Start

### Basic Multi-View Reconstruction

```python
from transformation_portal.spatial_ai.reconstruction import (
    CameraParams,
    SceneBuilder,
    MeshExporter,
    GeometricValidator,
)
import numpy as np

# 1. Prepare multi-view images (linear RGB, gamma=1.0)
images = [
    np.array(Image.open("view1.png")).astype(np.float32) / 255.0,
    np.array(Image.open("view2.png")).astype(np.float32) / 255.0,
    np.array(Image.open("view3.png")).astype(np.float32) / 255.0,
]

# 2. Define camera parameters
intrinsics = np.array([
    [525.0, 0, 320.0],
    [0, 525.0, 240.0],
    [0, 0, 1.0]
], dtype=np.float32)

cameras = [
    CameraParams(intrinsics, extrinsics_view1, 640, 480),
    CameraParams(intrinsics, extrinsics_view2, 640, 480),
    CameraParams(intrinsics, extrinsics_view3, 640, 480),
]

# 3. Build 3D scene
builder = SceneBuilder(tier="apex_research")
scene = builder.build_from_arrays(
    images=images,
    cameras=cameras,
    gamma=1.0,
    iterations=30000,
)

# 4. Validate quality
validator = GeometricValidator()
results = validator.validate_scene(scene)
print(f"Quality Grade: {results['quality_grade']}")
print(f"RMSE: {results['rmse']:.4f}")
print(f"Pass: {results['rmse_pass']}")

# 5. Export mesh
exporter = MeshExporter()
exporter.export_ply(scene, "scene.ply", include_attributes=True)
```

---

## Camera Calibration

### Camera Intrinsics

The intrinsic matrix `K` encodes internal camera parameters:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

- **fx, fy:** Focal lengths in pixels
- **cx, cy:** Principal point (optical center)

**Common values:**
- Smartphone: fx=fy ≈ 1000-1500 pixels (for 4K images)
- DSLR: fx=fy ≈ 2000-3000 pixels
- Wide-angle: fx=fy ≈ 500-800 pixels

**Calculate from FOV:**
```python
import math

fov_deg = 60  # horizontal field of view
width = 1920
fx = width / (2 * math.tan(math.radians(fov_deg / 2)))
```

### Camera Extrinsics

The extrinsic matrix `[R|t]` encodes camera pose (position + orientation):

```
Extrinsics = [[r11, r12, r13, tx],
              [r21, r22, r23, ty],
              [r31, r32, r33, tz],
              [  0,   0,   0,  1]]
```

- **R (3x3):** Rotation matrix (world → camera)
- **t (3x1):** Translation vector (camera position in world)

**Example: Camera looking down +Z axis**
```python
extrinsics = np.eye(4, dtype=np.float32)
extrinsics[0, 3] = 0.0  # X position
extrinsics[1, 3] = 1.0  # Y position (1m above origin)
extrinsics[2, 3] = 5.0  # Z position (5m forward)
```

### Multi-View Requirements

For successful reconstruction:

- **Minimum views:** 3 (more is better)
- **Baseline:** 10-50% of scene depth
  - Too small: Poor depth accuracy
  - Too large: Matching failures
- **Overlap:** >50% between adjacent views
- **Coverage:** All scene regions visible in 2+ views

---

## Integration with Previous Phases

### Phase 1: Depth Priors

Use depth maps from `LinearDecoder` for better initialization:

```python
from transformation_portal.spatial_ai.ingest import LinearDecoder

# Estimate depth for each view
decoder = LinearDecoder(device="mps")
depth_maps = []

for img in images:
    depth = decoder.estimate(img, gamma=1.0)
    depth_maps.append(depth)

# Reconstruct with depth priors
scene = builder.build_from_arrays(
    images=images,
    cameras=cameras,
    depth_maps=depth_maps,  # ← Phase 1 integration
    gamma=1.0,
)
```

**Benefits:**
- Faster convergence (fewer iterations needed)
- Better geometry in texture-less regions
- Improved scale consistency

### Phase 2.1: Segmentation Masks

Use SAM2 masks to focus reconstruction on foreground objects:

```python
from transformation_portal.spatial_ai.segmentation import SAM2Backend

# Segment each view
sam2 = SAM2Backend()
masks = []

for img in images:
    result = sam2.segment(
        SegmentationInput(image=img, gamma=1.0, mode="auto")
    )
    # Use largest mask
    mask = result.masks[0] if result.masks else None
    masks.append(mask)

# Reconstruct foreground only
scene = builder.build_from_arrays(
    images=images,
    cameras=cameras,
    masks=masks,  # ← Phase 2.1 integration
    gamma=1.0,
)
```

**Benefits:**
- Reduced background clutter
- Faster optimization (fewer points)
- Focus on objects of interest

### Phase 2.2: PBR Materials

Integrate material maps for physically-based appearance:

```python
from transformation_portal.spatial_ai.materials import MaterialBackend

# Generate PBR textures for each view
material_backend = MaterialBackend()
material_maps = []

for img, mask in zip(images, masks):
    pbr = material_backend.generate(
        MaterialInput(image=img, gamma=1.0, mask=mask)
    )
    material_maps.append({
        "albedo": pbr.albedo,
        "roughness": pbr.roughness,
        "metallic": pbr.metallic,
        "normal": pbr.normal,
    })

# Reconstruct with materials
scene = builder.build_from_arrays(
    images=images,
    cameras=cameras,
    material_maps=material_maps,  # ← Phase 2.2 integration
    gamma=1.0,
)
```

**Benefits:**
- Material-aware Gaussian colors
- Improved rendering realism
- Physical plausibility

---

## Novel View Synthesis

### Render New Viewpoint

```python
# Define novel camera viewpoint
novel_intrinsics = np.eye(3, dtype=np.float32)
novel_intrinsics[0, 0] = novel_intrinsics[1, 1] = 525.0
novel_intrinsics[0, 2] = 320.0
novel_intrinsics[1, 2] = 240.0

novel_extrinsics = np.eye(4, dtype=np.float32)
novel_extrinsics[0, 3] = 1.0  # Shift 1m to the right

novel_camera = CameraParams(novel_intrinsics, novel_extrinsics, 640, 480)

# Render
rendered_image = builder.render_novel_view(scene, novel_camera)

# Save
Image.fromarray((rendered_image * 255).astype(np.uint8)).save("novel_view.png")
```

### Camera Path Animation

Generate smooth camera trajectory for video:

```python
# Extract 100-frame path between first and last camera
camera_path = builder.extract_camera_path(scene, num_frames=100)

# Render all frames
frames = []
for cam in camera_path:
    frame = builder.render_novel_view(scene, cam)
    frames.append((frame * 255).astype(np.uint8))

# Save as video (using OpenCV or similar)
import cv2
out = cv2.VideoWriter('flythrough.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
```

---

## Mesh Export

### PLY Format (Preferred)

PLY format preserves full Gaussian attributes:

```python
exporter = MeshExporter()

# Binary PLY with all attributes
exporter.export_ply(
    scene,
    "scene.ply",
    include_attributes=True,  # Scales, rotations, opacities
    binary=True,              # Smaller file size
)

# ASCII PLY (human-readable, for debugging)
exporter.export_ply(
    scene,
    "scene_ascii.ply",
    include_attributes=True,
    binary=False,
)
```

**PLY Attributes:**
- `x, y, z`: Position
- `red, green, blue`: Color
- `scale_x, scale_y, scale_z`: Gaussian scale
- `rot_w, rot_x, rot_y, rot_z`: Rotation quaternion
- `opacity`: Alpha transparency

### OBJ Format (Limited)

OBJ has limited Gaussian support (point cloud representation):

```python
exporter.export_obj(
    scene,
    "scene.obj",
    vertex_colors=True,      # Export colors to MTL file
    subsample_factor=2,      # Downsample by 2x (optional)
)
```

**Note:** OBJ does not support Gaussian attributes (scales, rotations, opacities).

### Camera Parameters

Export camera calibration for external tools:

```python
exporter.export_cameras(scene, "cameras.json")
```

**JSON format:**
```json
{
  "num_cameras": 3,
  "cameras": [
    {
      "camera_id": "cam_001",
      "width": 640,
      "height": 480,
      "intrinsics": [[...], [...], [...]],
      "extrinsics": [[...], [...], [...], [...]]
    },
    ...
  ]
}
```

---

## Quality Validation

### RMSE (Root Mean Square Error)

Primary quality metric for reconstruction accuracy:

```python
validator = GeometricValidator()

# Compute RMSE against reference images
rmse = validator.compute_rmse(scene, reference_images)

# Quality thresholds
if rmse < 0.01:
    print("Excellent quality (< 1% error)")
elif rmse < 0.02:
    print("Good quality (< 2% error)")      # Target
elif rmse < 0.05:
    print("Acceptable quality (< 5% error)")
else:
    print("Poor quality (>= 5% error)")
```

### Comprehensive Validation

```python
results = validator.validate_scene(
    scene,
    reference_images=images,  # Ground truth
    depth_maps=depth_maps,    # Depth consistency check
)

print(f"RMSE: {results['rmse']:.4f}")
print(f"Pass: {results['rmse_pass']}")
print(f"Quality Grade: {results['quality_grade']}")
print(f"Depth Consistency: {results['depth_consistency']:.2f}")
print(f"Coverage: {results['coverage']}")
```

### Coverage Analysis

Check how well cameras cover the scene:

```python
coverage = validator.compute_coverage(scene)

print(f"Mean points per view: {coverage['mean_points_per_view']:.0f}")
print(f"Min points per view: {coverage['min_points_per_view']}")
print(f"Max points per view: {coverage['max_points_per_view']}")
print(f"Coverage std: {coverage['coverage_std']:.1f}")
```

**Interpretation:**
- High std → uneven coverage (add more views)
- Low min → blind spots (adjust camera poses)

---

## Performance Optimization

### Iteration Count

Balance quality vs. speed:

```python
# Quick preview (5k iterations, <5s)
scene_preview = builder.build_from_arrays(images, cameras, iterations=5000)

# Standard quality (30k iterations, ~30s on GPU)
scene_standard = builder.build_from_arrays(images, cameras, iterations=30000)

# High quality (50k+ iterations, >60s)
scene_hq = builder.build_from_arrays(images, cameras, iterations=50000)
```

### GPU Acceleration

Automatically uses available GPU:

```python
# Auto-detect (cuda > mps > cpu)
backend = GaussianBackend(tier="apex_research", device=None)

# Force specific device
backend_gpu = GaussianBackend(tier="apex_research", device="cuda")
backend_mps = GaussianBackend(tier="apex_research", device="mps")  # Apple Silicon
backend_cpu = GaussianBackend(tier="apex_research", device="cpu")
```

### Memory Management

For large scenes (>1M Gaussians):

```python
# Check Gaussian count
print(f"Gaussians: {scene.splats.num_gaussians:,}")

# Subsample before export
exporter.export_obj(scene, "scene.obj", subsample_factor=4)  # 4x reduction
```

**Typical VRAM usage:**
- 100k Gaussians: ~500MB
- 500k Gaussians: ~2GB
- 1M Gaussians: ~4GB

---

## Troubleshooting

### License Errors

**Error:** `LicenseRestrictionError: 3D Gaussian Splatting requires research tier`

**Solution:** Use research-only tier:
```python
builder = SceneBuilder(tier="apex_research")  # or apex_research_ultra, experimental
```

### Gamma Errors

**Error:** `ValueError: Reconstruction requires gamma=1.0 (linear RGB)`

**Solution:** Linearize images before reconstruction:
```python
# If images are sRGB (gamma ≈ 2.2)
linear_images = [np.power(img, 2.2) for img in srgb_images]
```

### Poor Reconstruction Quality

**Symptoms:**
- High RMSE (> 0.05)
- Noisy geometry
- Missing surfaces

**Solutions:**

1. **Add more views** (minimum 3, ideal 5-10)
2. **Improve camera calibration** (accurate intrinsics/extrinsics)
3. **Use depth priors** (Phase 1 integration)
4. **Increase iterations** (30k → 50k)
5. **Check camera baseline** (not too small or too large)
6. **Ensure sufficient overlap** (>50% between views)

### Slow Performance

**GPU not detected:**
```python
# Check device
print(backend.device)  # Should be "cuda" or "mps", not "cpu"

# Install CUDA (NVIDIA) or verify MPS (Apple Silicon)
```

**Too many Gaussians:**
```python
# Check count
if scene.splats.num_gaussians > 1_000_000:
    # Use segmentation masks to reduce count
    scene = builder.build_from_arrays(images, cameras, masks=masks)
```

---

## Advanced Topics

### Spherical Harmonics (Future)

View-dependent appearance using SH coefficients:

```python
# Check if SH coefficients are present
if scene.splats.sh_coefficients is not None:
    print(f"SH degree: {scene.splats.sh_coefficients.shape[1]}")
```

**Current status:** Not implemented (DC term only)
**Future:** 3rd order SH for realistic view-dependent effects

### Custom Optimization

Override default optimization parameters:

```python
backend = GaussianBackend(
    tier="apex_research",
    model_repo="graphdeco-inria/gaussian-splatting",
    model_revision="<verified_commit_hash>",  # Replace placeholder
)

# See config/presets/experimental/gaussian_splat_3d.yaml for parameters
```

### Multi-Material Scenes

Combine material-aware reconstruction with segmentation:

```python
# Segment by material
from transformation_portal.spatial_ai.segmentation import MaterialClassifier

classifier = MaterialClassifier()
masks_by_material = []

for img in images:
    # Segment and classify
    seg_result = sam2.segment(SegmentationInput(image=img, gamma=1.0, mode="auto"))
    classified = classifier.classify(img, seg_result.masks)

    # Group by material
    masks_by_material.append(classified)

# Reconstruct each material separately
# (Implementation varies by use case)
```

---

## Citation

If you use this module in research, please cite:

```bibtex
@inproceedings{kerbl20233dgaussians,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  booktitle={ACM Transactions on Graphics},
  year={2023}
}
```

---

## References

- **Inria 3DGS:** https://github.com/graphdeco-inria/gaussian-splatting
- **Paper:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **License:** Non-commercial research license
- **Commercial licensing:** Contact Inria GraphDeco team

---

## API Reference

See module docstrings for detailed API documentation:

```python
from transformation_portal.spatial_ai.reconstruction import (
    # Data contracts
    CameraParams,
    GaussianSplat,
    ReconstructionInput,
    Scene3D,

    # Backend and builders
    GaussianBackend,
    SceneBuilder,

    # Export and validation
    MeshExporter,
    GeometricValidator,

    # Exceptions
    LicenseRestrictionError,
)

help(SceneBuilder)
help(GeometricValidator)
```
