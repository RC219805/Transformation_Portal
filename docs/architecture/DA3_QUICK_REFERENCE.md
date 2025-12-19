# DA3 Integration Quick Reference

**For**: Developers and Architects  
**Date**: 2025-12-19  
**Module**: `lux_depth_v3/`

---

## Table of Contents

1. [Architecture at a Glance](#architecture-at-a-glance)
2. [Key Design Decisions](#key-design-decisions)
3. [API Quick Reference](#api-quick-reference)
4. [Common Use Cases](#common-use-cases)
5. [Integration Points](#integration-points)
6. [Security Checklist](#security-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│  Module:       lux_depth_v3/                                │
│  Purpose:      DA3 integration for advanced depth features   │
│  Status:       ✅ Implemented (Beta)                         │
│  Stability:    Production-ready with caveats                │
└─────────────────────────────────────────────────────────────┘

Layers:
  CLI / Python API  ←─────────────────────────┐
          ↓                                    │
  Configuration & License Validation          │
          ↓                                    │
  Input Manager → Preprocessing               │
          ↓                                    │
  DA3 Inference (Python API / CLI)            │
          ↓                                    │
  Postprocessing → Metric Conversion          │
          ↓                                    │
  Validation & Quality Gates                  │
          ↓                                    │
  Export Manager (NPZ, GLB, PLY, video)       │
          └────────────────────────────────────┘
```

**Coexistence with V2**:
- ✅ `lux_depth_v2/` - Production (DA2-based, stable)
- 🆕 `lux_depth_v3/` - Advanced features (DA3-based, beta)

---

## Key Design Decisions

### 1. Module Isolation

**Why separate module?**
- Zero regression in v2 pipeline
- Independent versioning
- Clear DA2/DA3 separation

**When to use v2 vs v3?**

| Use Case                        | Recommended Module |
|---------------------------------|--------------------|
| Production depth estimation     | lux_depth_v2       |
| Material segmentation           | lux_depth_v2       |
| Multi-view depth                | lux_depth_v3 ✨    |
| Metric depth (meters)           | lux_depth_v3 ✨    |
| Camera pose estimation          | lux_depth_v3 ✨    |
| Gaussian Splatting              | lux_depth_v3 ✨    |
| 3D reconstruction               | lux_depth_v3 ✨    |

### 2. Dual API Strategy

**Python API Mode** (Recommended):
```python
from lux_depth_v3 import estimate_depth

result = estimate_depth("image.jpg", "output/", model="large-1.1")
```

**CLI Wrapper Mode** (Fallback):
```bash
lux-depth-v3 process image.jpg -o output/ -m large-1.1
```

**When to use which?**
- Python API: Full features, best performance
- CLI: Simple scripts, shell integration

### 3. Model Caching

**Why pre-cache models?**
- Eliminate download latency (critical for production)
- Enable offline operation
- Consistent deployment

**Recommended sets**:
```bash
# Development
lux-depth-v3 cache-download --set essential  # ~6GB

# Production
lux-depth-v3 cache-download --set production  # ~12GB

# Benchmarking
lux-depth-v3 cache-download --set benchmark  # ~20GB
```

### 4. License Compliance

**Critical**: DA3 has mixed licenses!

| License      | Models                   | Commercial Use |
|--------------|--------------------------|----------------|
| Apache-2.0   | metric-large, base, small| ✅ Allowed      |
| CC-BY-NC-4.0 | nested-giant, giant, large| ❌ Not Allowed  |

**Production recommendation**: Use `metric-large` (Apache-2.0, ✅ commercial)

```python
# Strict mode (production)
from lux_depth_v3.license import LicenseValidator, LicenseMode

validator = LicenseValidator(mode=LicenseMode.STRICT)
validator.validate_model_for_use(model="metric-large", commercial=True)
# ✅ Passes
```

---

## API Quick Reference

### High-Level Convenience API

```python
from lux_depth_v3 import estimate_depth

# Simplest usage
result = estimate_depth("image.jpg", "output/")

# With model selection
result = estimate_depth("image.jpg", "output/", model="large-1.1")

# Check results
if result.success:
    depth = result.depth_array  # (H, W) numpy array
    print(f"Depth range: {depth.min():.2f} - {depth.max():.2f}")
```

### Class-Based API (Full Control)

```python
from lux_depth_v3 import DA3DepthEstimator, DA3Config, ModelVariant

# Create estimator
estimator = DA3DepthEstimator(
    model=ModelVariant.DA3_LARGE_V1_1,
    device="cuda",
    config=DA3Config(
        process_res=1024,
        export_format=["mini_npz", "glb", "depth_vis"],
    ),
)

# Monocular depth
result = estimator.process_image("image.jpg", "output/")

# Multi-view depth
result = estimator.process_images(
    input_paths=["view1.jpg", "view2.jpg", "view3.jpg"],
    output_dir="output/multiview/",
    estimate_poses=True,  # Automatic pose estimation
)

# Gaussian Splatting
result = estimator.process_images(
    input_paths=image_paths,
    output_dir="output/gs/",
    infer_gs=True,
    export_format=["gs_ply", "gs_video"],
)
```

### CLI Usage

```bash
# Basic depth estimation
lux-depth-v3 process image.jpg -o output/

# With model selection
lux-depth-v3 process image.jpg -o output/ -m large-1.1

# Multi-view with GLB export
lux-depth-v3 process images/ -o output/ -f "mini_npz-glb"

# Gaussian Splatting
lux-depth-v3 process images/ -o output/ -m giant-1.1 --infer-gs -f "gs_ply-gs_video"

# Full API access
lux-depth-v3 api-process images/ -o output/ \
  -m nested-giant-large-1.1 \
  -f "mini_npz-glb-gs_ply-depth_vis" \
  --use-ray-pose \
  --conf-thresh 0.5
```

---

## Common Use Cases

### 1. Monocular Depth Estimation

```python
from lux_depth_v3 import estimate_depth

result = estimate_depth("kitchen.jpg", "output/kitchen/", model="large-1.1")
depth = result.depth_array  # (H, W) relative depth
```

**Output**: `output/kitchen/mini.npz` containing depth array

### 2. Metric Depth (Absolute Scale)

```python
from lux_depth_v3 import DA3DepthEstimator
from lux_depth_v3.metric_depth import convert_to_metric_depth

# Use metric-large model (outputs metric natively)
estimator = DA3DepthEstimator(model="metric-large")
result = estimator.process_image("image.jpg", "output/")

# Already in meters
depth_meters = result.depth_array

# Or convert relative depth to metric
from lux_depth_v3.metric_depth import convert_to_metric_depth

metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    focal_length_px=1200.0,
    method="focal",
)
```

### 3. Multi-View Depth with Pose Estimation

```python
estimator = DA3DepthEstimator(model="nested-giant-large-v1.1")

result = estimator.process_images(
    input_paths=["view1.jpg", "view2.jpg", "view3.jpg"],
    output_dir="output/multiview/",
    estimate_poses=True,  # DA3 estimates camera poses
)

# Access results
depth = result.depth_array        # (3, H, W)
extrinsics = result.extrinsics    # (3, 4, 4) camera poses
intrinsics = result.intrinsics    # (3, 3, 3) camera intrinsics
```

### 4. Gaussian Splatting for Novel View Synthesis

```python
estimator = DA3DepthEstimator(model="giant-1.1")

result = estimator.process_images(
    input_paths=image_paths,
    output_dir="output/gs/",
    infer_gs=True,
    export_format=["gs_ply", "gs_video"],
)

# Outputs:
# - scene.ply: Gaussian Splatting point cloud
# - gs_video.mp4: Novel view synthesis video
```

### 5. Batch Processing

```python
estimator = DA3DepthEstimator(model="large-1.1")

# Process directory
results = estimator.process_directory(
    input_dir="renders/750_Picacho/",
    output_dir="output/750_Picacho_depth/",
    extensions=["jpg", "png"],
)

# Model loaded once, reused for all images (10-20x speedup)
```

### 6. Quality Validation

```python
from lux_depth_v3.validation import DepthQualityMetrics

result = estimator.process_image("image.jpg", "output/")

# Compute metrics (if ground truth available)
metrics = DepthQualityMetrics.compute(
    predicted_depth=result.depth_array,
    ground_truth_depth=gt_depth,
)

# Validate quality gates
if metrics.rmse < 0.5 and metrics.delta_1 > 0.85:
    print("✅ Quality gate passed")
    
# Export to validation framework
metrics.export_to_json("validation_v1_baseline_pack/metrics/da3.json")
```

---

## Integration Points

### 1. Validation Framework

```python
# lux_depth_v3 exports metrics to validation_v1_baseline_pack/

from lux_depth_v3.validation import DepthQualityMetrics

metrics = DepthQualityMetrics.compute(predicted, ground_truth)
metrics.export_to_json("validation_v1_baseline_pack/metrics/da3_results.json")
```

### 2. Lux Render Pipeline (Future)

```python
# Depth-aware AI enhancement (planned integration)

from lux_render_pipeline import LuxRenderPipeline
from lux_depth_v3 import estimate_depth

# Estimate depth
depth_result = estimate_depth("render.jpg", "output/")

# Use depth in render pipeline
pipeline = LuxRenderPipeline()
enhanced = pipeline.enhance(
    image="render.jpg",
    depth_map=depth_result.depth_array,
    preset="architectural",
)
```

### 3. Material Response (Future)

```python
# Material-aware depth processing (planned)

from material_response import MaterialResponse
from lux_depth_v3 import estimate_depth

depth_result = estimate_depth("interior.jpg", "output/")
material_response = MaterialResponse()

enhanced = material_response.enhance(
    image="interior.jpg",
    depth_map=depth_result.depth_array,
    surfaces=["wood", "metal", "glass"],
)
```

---

## Security Checklist

### Input Validation

- [x] File size limits (50MB default)
- [x] Image dimension limits (4096px default)
- [x] Extension whitelist (.jpg, .png, .tiff, .tif)
- [x] Path traversal prevention
- [x] MIME type validation

### Service Mode

- [x] Rate limiting (60 req/min default)
- [x] CORS configuration
- [x] Input sanitization
- [x] File upload limits

### Dependencies

- [x] No CVE-2024-27763 (basicsr excluded)
- [x] No realesrgan dependency
- [x] Version pinning in requirements.txt
- [x] Minimal dependency surface

### License Compliance

- [x] Automated license validation
- [x] Strict mode for production
- [x] Clear documentation of restrictions
- [x] Commercial alternatives suggested

---

## Troubleshooting

### Issue: Model download fails

**Symptom**: HuggingFace Hub connection error

**Solution**:
```bash
# Pre-cache models
lux-depth-v3 cache-download --set essential

# Verify cache
lux-depth-v3 cache-list
```

### Issue: License violation warning

**Symptom**: `LicenseViolationError` for commercial use

**Solution**:
```python
# Use Apache-licensed model
estimator = DA3DepthEstimator(model="metric-large")  # Apache-2.0 ✅

# Not: nested-giant-large-v1.1 (CC-BY-NC-4.0 ❌)
```

### Issue: Out of memory

**Symptom**: CUDA OOM or system memory exhausted

**Solution**:
```python
# Use smaller model
estimator = DA3DepthEstimator(model="base")  # 120M params vs 1.4B

# Reduce processing resolution
config = DA3Config(process_res=512)  # Default: 768
estimator = DA3DepthEstimator(model="large-1.1", config=config)

# Use CPU
estimator = DA3DepthEstimator(model="large-1.1", device="cpu")
```

### Issue: Slow performance

**Symptom**: >5s per image

**Solution**:
```bash
# Pre-cache models
lux-depth-v3 cache-download --set production

# Use GPU
estimator = DA3DepthEstimator(model="large-1.1", device="cuda")

# Batch processing (model loaded once)
results = estimator.process_directory("input/", "output/")
```

### Issue: V2 vs V3 confusion

**Question**: Which module should I use?

**Answer**:

| Scenario                     | Module         | Reason                      |
|------------------------------|----------------|-----------------------------|
| Production depth estimation  | lux_depth_v2   | Stable, battle-tested       |
| Multi-view depth             | lux_depth_v3   | DA3-specific feature        |
| Metric depth in meters       | lux_depth_v3   | DA3METRIC-LARGE model       |
| Material segmentation        | lux_depth_v2   | Already implemented         |
| Upscaling                    | lux_depth_v2   | Integrated upscaler         |
| Gaussian Splatting           | lux_depth_v3   | DA3-only feature            |

---

## Performance Reference

| Model              | Device | Resolution | Time/Image | Use Case              |
|--------------------|--------|------------|------------|-----------------------|
| base               | CPU    | 518        | 2.1s       | Fast preview          |
| large-v1.1         | CPU    | 518        | 5.8s       | High quality preview  |
| large-v1.1         | CUDA   | 518        | 0.3s       | **Production** ⭐     |
| metric-large       | CUDA   | 518        | 0.3s       | Commercial use ✅     |
| nested-giant-v1.1  | CUDA   | 518        | 0.8s       | Research/hero renders |

**Recommended for production**: `metric-large` on CUDA (Apache-2.0, fast, high quality)

---

## Related Documentation

- [Full Architecture](./DA3_INTEGRATION_ARCHITECTURE.md) - Comprehensive design
- [ADR-002](./adr/ADR-002-DA3-MODULE-ARCHITECTURE.md) - Architectural decision record
- [User Guide](../../lux_depth_v3/README.md) - End-user documentation
- [Integration Guide](../../lux_depth_v3/INTEGRATION_GUIDE.md) - Integration details
- [License Guide](../../lux_depth_v3/docs/LICENSE_GUIDE.md) - License compliance
- [Security Guidelines](../../lux_depth_v3/SECURITY.md) - Security best practices

---

**Version**: 1.0  
**Last Updated**: 2025-12-19  
**Maintainer**: Transformation Portal Architect
