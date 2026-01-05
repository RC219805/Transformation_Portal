# Depth Estimation Capabilities Analysis & Optimal Configuration Guide

**Author**: Transformation Portal Codebase Review
**Date**: 2025-01-05
**Status**: Complete Analysis
**Purpose**: Comprehensive review of depth estimation capabilities with optimal configuration recommendations for quality

---

## Executive Summary

The Transformation Portal repository contains **three primary depth estimation systems**:

| System | Model | Best For | Quality Tier | Status |
|--------|-------|----------|--------------|--------|
| **lux_depth_v2** | Depth Anything V2 | Production workflows | ⭐⭐⭐⭐⭐ | **Production-Ready** |
| **lux_depth_v3** | Depth Anything 3 | Multi-view, metric depth | ⭐⭐⭐⭐ | Beta |
| **high_fidelity_depth** | DA V2 + Tiling Fixes | Research/validation | ⭐⭐⭐⭐⭐ | Experimental |

**Recommended Configuration for Maximum Quality**: `lux_depth_v2` with:
- Preset: `interior_luxury_apex_quality` or `exterior_pool_apex_quality`
- Model: `depth-anything/Depth-Anything-V2-Large-hf`
- Tile Size: 1024px with 128-192px overlap (128px for standard scenes, 192px for texture-heavy/aerial)
- Scale Reconciliation: Enabled
- Edge Snapping: Enabled
- Production Refinement: CLAHE + Guided Filter + Edge Snap

---

## 1. Depth Estimation Systems Overview

### 1.1 lux_depth_v2 (Production Standard)

**Location**: `lux_depth_v2/`

**Key Features**:
- ✅ Production-validated (44 dedicated test suites covering inference, tiling, edge snapping, and I/O)
- ✅ Security-hardened ([CVE-2024-27763 mitigated](../../SECURITY.md#cve-2024-27763-basicsr-command-injection-vulnerability))
- ✅ 16-bit precision end-to-end
- ✅ Docker deployment ready
- ✅ FastAPI service mode with rate limiting
- ✅ Tiled inference with scale reconciliation
- ✅ Production refinement pipeline (CLAHE + guided filter + edge snap)

**Depth Inference Implementation** (`lux_depth_v2/depth_inference.py`):
- **Tiled inference** at native model resolution (1024-1536px tiles)
- **Per-tile scale reconciliation** using Theil-Sen regression
- **Global anchor fusion** for cross-tile consistency
- **Edge snapping** for RGB-aligned depth boundaries
- **Production refinement**: CLAHE + guided filter + edge snap

**Models Supported**:
```
depth-anything/Depth-Anything-V2-Large-hf  (671MB, CC-BY-NC-4.0)
depth-anything/Depth-Anything-V2-Base-hf   (195MB, CC-BY-NC-4.0)
depth-anything/Depth-Anything-V2-Small-hf  (49.8MB, Apache 2.0)
```

**Recommended Usage**:
```bash
# CLI (Golden Path)
lux-depth-v2 --input-dir renders/ --output-dir output/ --preset interior_luxury_apex_quality

# Python API
from lux_depth_v2.depth_inference import create_tiled_estimator
estimator = create_tiled_estimator(
    tile_size=1024,
    overlap=192,
    fusion_mode="weighted",
    use_global_anchor=True,
    use_edge_snapping=True,
    model_name="depth-anything/Depth-Anything-V2-Large-hf"
)
depth_map = estimator.estimate_depth(rgb_image)
```

### 1.2 lux_depth_v3 (Advanced DA3 Integration)

**Location**: `lux_depth_v3/`

**Key Features**:
- ✅ Depth Anything 3 (DA3) support
- ✅ Multi-view depth estimation with pose estimation
- ✅ Metric-scale depth output (relative scale; absolute with calibration)
- ✅ Gaussian Splatting (3DGS) support
- ✅ Camera pose estimation
- ✅ License validation (Apache 2.0 vs CC-BY-NC-4.0)

**DA3 Model Variants** (`lux_depth_v3/config.py`):
```
Non-Commercial Models (CC-BY-NC-4.0)
  DA3NESTED-GIANT-LARGE-1.1  (1.40B params) - RECOMMENDED for research/non-commercial use
  DA3-GIANT-1.1              (1.15B params) - Non-commercial use only
  DA3-LARGE-1.1              (0.35B params) - Non-commercial use only

Commercial-Friendly Models (Apache 2.0)
  DA3METRIC-LARGE            (0.35B params) - Commercial-friendly, metric-consistent output
  DA3-BASE                   (0.12B params) - Commercial-friendly
  DA3-SMALL                  (0.08B params) - Commercial-friendly, fast/testing
```

**Inference Modes**:
1. **Monocular** - Single image depth estimation (affine-invariant)
2. **Multi-view** - Multiple views with pose estimation
3. **Metric** - Metric-consistent depth using DA3METRIC-LARGE (requires scale calibration for absolute values)

**Recommended Usage**:
```python
from lux_depth_v3 import DA3InferenceEngine, DA3Config, ModelVariant

config = DA3Config(
    model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    inference_mode="monocular"
)
engine = DA3InferenceEngine(config)
result = engine.inference([image_input])
```

### 1.3 high_fidelity_depth (Research Module)

**Location**: `high_fidelity_depth/`

**Key Features**:
- ✅ Fixed tiling logic (no sliver tiles)
- ✅ Per-tile scale reconciliation with gradient-weighted sampling
- ✅ Spatial smoothing of calibrations
- ✅ Resolution policy (conditional inference based on image size)
- ✅ Seam validation metrics

**Key Implementation** (`high_fidelity_depth/depth_estimator.py`):
```python
from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig

config = DepthConfig(
    model_name="depth-anything/Depth-Anything-V2-Large-hf",
    tile_size=1024,
    overlap=192,  # CRITICAL: Increased for texture-heavy scenes
    reconcile_scales=True,
    reconcile_method="robust",  # Theil-Sen regression
    fusion_mode="weighted",
    validate_seams=True,
    seam_energy_threshold=1.2
)
estimator = HighFidelityDepthEstimator(config)
depth = estimator.estimate_depth(image, use_global_anchor=True)
```

---

## 2. Output Artifact Contract

All depth estimation systems in this repository produce **affine-invariant depth maps** unless explicitly calibrated. Understanding the output format is critical for downstream integration.

### 2.1 Depth Map Format Specification

| Property | lux_depth_v2 | lux_depth_v3 | high_fidelity_depth |
|----------|--------------|--------------|---------------------|
| **Output Type** | Numpy array (H×W) | Numpy array (H×W) | Numpy array (H×W) |
| **Data Type** | `float32` or `uint16` | `float32` | `float32` |
| **Value Range** | [0.0, 1.0] (normalized) or [0, 65535] (16-bit) | [0.0, ∞) (unnormalized) | [0.0, 1.0] (normalized) |
| **Depth Encoding** | Inverse depth (disparity-like) | Metric-consistent (relative) | Inverse depth (disparity-like) |
| **Scale** | Affine-invariant¹ | Affine-invariant² | Affine-invariant¹ |
| **Coordinate System** | Camera-relative, Z-forward | Camera-relative, Z-forward | Camera-relative, Z-forward |

**¹Affine-Invariant**: Output preserves depth *ordering* but not absolute scale. Values are monotonically related to scene depth but require calibration for metric accuracy.

**²Metric-Consistent**: DA3METRIC models produce depth in consistent units across scenes but still require scale calibration for absolute metric values (e.g., meters).

### 2.2 Export Formats

**lux_depth_v2** (`lux_depth_v2/depth_exporter.py`):
```python
# Available export formats
- PNG (16-bit grayscale): Lossless, normalized to [0, 65535]
- EXR (32-bit float): Full precision, preserves raw values
- NPY (Numpy binary): For Python pipelines
```

**lux_depth_v3** (`lux_depth_v3/exporter.py`):
```python
# Export configuration
export_formats = ["png", "npy", "pfm"]  # PFM for metric depth
color_map = "inferno"  # Visualization colormap
depth_range = "auto"   # Auto-scale or fixed [min, max]
```

### 2.3 Metadata Guarantees

**lux_depth_v2** outputs include:
- `depth_metadata.json`: Model name, tile size, overlap, fusion mode, timestamp
- `validation_report.json`: (if `--validate` flag) RMSE, δ₁, MAE against ground truth

**lux_depth_v3** outputs include:
- `depth_metadata.json`: Model variant, inference mode, camera pose (if multi-view)
- `camera_poses.json`: (if multi-view) 4×4 transformation matrices
- `point_cloud.ply`: (optional) 3D reconstruction

### 2.4 Calibration Requirements

⚠️ **Critical**: All models output relative depth. For absolute metric scale:

1. **Single Image**: Require known ground truth distance (e.g., "floor to ceiling = 3.2m")
2. **Multi-view**: Triangulation provides metric scale if camera baseline is known
3. **DA3METRIC**: Provides *consistent* scale across scenes but still requires one calibration point

**Calibration Example**:
```python
# User provides: "The wall is 4 meters wide"
wall_pixels = 800  # pixels in image
depth_at_wall = depth_map[wall_region].mean()
scale_factor = 4.0 / depth_at_wall  # meters per depth unit
metric_depth = depth_map * scale_factor
```

---

## 3. Optimal Configuration for Maximum Quality

### 3.1 Interior Scenes (APEX Quality)

**Preset**: `interior_luxury_apex_quality`

**Configuration**:
```python
# lux_depth_v2/config.py - Preset settings
Preset.INTERIOR_LUXURY_APEX_QUALITY:
    precision: "fp32"           # Maximum numerical precision
    tile: 1024                  # Upscaling tile size
    tile_pad: 32                # Extra padding for edge quality
    post_tile: 2048             # Post-processing tile size
    post_overlap: 128           # 100% more overlap for seamless blending

    # Segmentation (SegFormer-B5)
    segmentation.backend: "segformer"
    segmentation.segformer_model: "nvidia/segformer-b5-finetuned-ade-640-640"
    segmentation.input_long_side: 2048  # 60% higher than default
    segmentation.min_confidence: 0.15

    # Depth Zones (Interior-optimized)
    depth_zones.mode: "auto"
    depth_zones.scene_type: "interior"
    depth_zones.fg_percentile: 0.35
    depth_zones.bg_percentile: 0.65
```

**Depth Inference Settings**:
```python
from lux_depth_v2.depth_inference import TiledInferenceConfig, TiledDepthEstimator

config = TiledInferenceConfig(
    tile_size=1024,
    overlap=128,  # 128 for standard scenes, 192 for texture-heavy/aerial
    model_name="depth-anything/Depth-Anything-V2-Large-hf",
    bypass_image_processor=True,  # CRITICAL: Skip HF's 518px resize
    fusion_mode="weighted",
    blend_window="hann",
    reconcile_scales=True,
    reconcile_method="robust",
    validate_edges=True,
    edge_alignment_threshold=0.5,
    use_global_anchor=True,
    use_edge_snapping=True,
    use_production_refinement=True,
    refinement_use_clahe=True,
    refinement_use_edge_filter=True,
    refinement_use_edge_snap=True
)
```

### 3.2 Exterior/Pool Scenes (APEX Quality)

**Preset**: `exterior_pool_apex_quality`

**Key Differences from Interior**:
```python
# Depth Zones (Exterior-optimized)
depth_zones.mode: "auto"
depth_zones.scene_type: "exterior"
depth_zones.fg_percentile: 0.30      # More foreground for vegetation
depth_zones.bg_percentile: 0.70      # Extended background for sky
depth_zones.close_range_m: 1.5
depth_zones.mid_range_m: 8.0
depth_zones.far_range_m: 25.0
depth_zones.infinity_m: 5000.0       # Mountains/sky

# Material Thresholds (Exterior-specific)
materials_v2.confidence.material_thresholds = {
    "water": 0.30,       # Critical for pool
    "vegetation": 0.35,  # Critical for landscaping
    "sky": 0.25,         # Critical for twilight gradient
    "stone": 0.48,       # Pool deck/columns
}
```

### 3.3 Aerial/Texture-Heavy Scenes

**Key Adjustments** (`high_fidelity_depth/depth_estimator.py`):
```python
config = DepthConfig(
    tile_size=1024,
    overlap=192,  # INCREASED from 128 for texture-heavy scenes (Blocker B fix)
    reconcile_scales=True,
    reconcile_method="robust",
    fusion_mode="weighted"
)

# Use gradient-weighted sampling for scale reconciliation (Blocker C fix)
# Avoids flat regions (sky, blank walls) that cause calibration drift
```

---

## 4. Quality Comparison: DA2 vs DA3

### 4.1 When to Use Depth Anything V2 (lux_depth_v2)

✅ **Recommended for**:
- Production architectural rendering workflows
- Single-image depth estimation
- Security-critical deployments (CVE mitigated)
- Docker/containerized deployments
- Batch processing workflows with GPU acceleration

### 4.2 When to Use Depth Anything 3 (lux_depth_v3)

✅ **Recommended for**:
- Multi-view depth estimation with pose
- Metric depth requirements (absolute scale in meters)
- 3D reconstruction projects (Gaussian Splatting)
- Camera pose estimation needs
- Research/experimental workflows

### 4.3 Performance Benchmarks

| Model | Device | Resolution | Time/Image | Notes |
|-------|--------|------------|------------|-------|
| DA2-Small | MPS (M4 Max) | 518×518 | 24ms | Best for preview |
| DA2-Large | MPS (M4 Max) | 518×518 | 65ms | Production quality |
| DA2-Large | CUDA | 518×518 | ~30ms | GPU accelerated |
| DA3-Base | CPU | 518×518 | 2.1s | Testing only |
| DA3-Large-v1.1 | CUDA | 518×518 | 0.3s | Recommended DA3 |
| DA3-Nested-Giant | CUDA | 518×518 | 0.8s | Maximum quality |

---

## 5. Critical Configuration Parameters

### 5.1 Tile Size and Overlap (Most Important)

```python
# PRODUCTION RECOMMENDATION
tile_size: 1024      # Sweet spot for quality/memory
overlap: 128-192     # 128 for most scenes, 192 for texture-heavy

# DO NOT use smaller tiles unless memory-constrained
# Smaller tiles = more seam artifacts
```

### 5.2 Scale Reconciliation (Critical for Quality)

```python
# ALWAYS ENABLE for production
reconcile_scales: True
reconcile_method: "robust"  # Theil-Sen regression (outlier-resistant)

# How it works:
# 1. Compute global anchor at low-res (768px)
# 2. Per-tile affine calibration: depth_calibrated = a * depth_tile + b
# 3. Clamp a ∈ [0.7, 1.3], b ∈ [-0.3, 0.3]
# 4. Spatial smoothing with σ=1.5 Gaussian
```

### 5.3 Edge Snapping and Refinement

```python
# PRODUCTION REFINEMENT CHAIN
use_production_refinement: True
refinement_use_clahe: True      # Contrast-limited adaptive histogram equalization
refinement_use_edge_filter: True  # Guided filter for edge preservation
refinement_use_edge_snap: True    # RGB-aligned depth boundaries

# WARNING: Avoid double-application of edge snapping
# If use_production_refinement=True with refinement_use_edge_snap=True,
# standalone use_edge_snapping is automatically disabled
```

### 5.4 Model Selection

```python
# PRODUCTION (Quality Priority)
model_name: "depth-anything/Depth-Anything-V2-Large-hf"

# PRODUCTION (Commercial Use)
model_name: "depth-anything/Depth-Anything-V2-Small-hf"  # Apache 2.0

# DA3 (Maximum Quality, Non-Commercial)
model_variant: ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1

# DA3 (Commercial Use)
model_variant: ModelVariant.DA3_METRIC_LARGE  # Apache 2.0
```

---

## 6. Common Quality Issues and Fixes

### 6.1 Tile Seam Artifacts

**Symptoms**: Visible grid patterns, depth discontinuities at tile boundaries

**Root Cause**: Per-tile normalization without cross-tile calibration

**Fix**:
```python
# Enable scale reconciliation
reconcile_scales: True
reconcile_method: "robust"

# Increase overlap
overlap: 192  # Up from 128

# Enable spatial smoothing
# (Automatic in high_fidelity_depth, manual in lux_depth_v2)
```

### 6.2 Soft/Blurry Depth Edges

**Symptoms**: Depth boundaries don't align with RGB edges

**Root Cause**: Bicubic upsampling from low-res inference

**Fix**:
```python
# CRITICAL: Bypass HuggingFace's 518px resize
bypass_image_processor: True

# Enable edge snapping
use_edge_snapping: True
edge_snap_config: EdgeSnappingConfig()
```

### 6.3 Sliver Tiles (Small Edge Tiles)

**Symptoms**: Quality degradation at image edges, inconsistent tile sizes

**Root Cause**: Tiling logic without proper padding

**Fix**:
```python
# Use reflective padding to clean tiling geometry (Blocker A fix)
# Implemented in high_fidelity_depth._pad_to_tile_geometry()

# All tiles should be full tile_size × tile_size
# No tiles smaller than min(256, tile_size)
```

### 6.4 Calibration Drift in Low-Variance Regions

**Symptoms**: Scale mismatches in sky, blank walls, uniform surfaces

**Root Cause**: Robust regression fails when there's no texture to match

**Fix**:
```python
# Gradient-weighted sampling (Blocker C fix)
# Skip low-variance regions for calibration
# Prioritize structural regions (walls, ceilings, planes)

# Implemented in high_fidelity_depth._reconcile_tile_scale()
if tile_variance < 1e-4 or ref_variance < 1e-4:
    return tile_depth, 1.0, 0.0  # Identity transform
```

---

## 7. Quality Validation Metrics

### 7.1 Edge Alignment Score

```python
# Correlation between RGB edges and depth edges
# Target: > 0.5 (higher is better)

def compute_edge_alignment(rgb, depth):
    rgb_edges = cv2.Canny(rgb_gray, 50, 150)
    depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    correlation = np.corrcoef(rgb_edges.ravel(), depth_edges.ravel())[0, 1]
    return correlation
```

### 7.2 Seam Energy Ratio

```python
# Gradient energy at tile boundaries vs interior
# Target: < 1.2 (boundary should not have higher energy than interior)

def validate_seam_energy(depth, tile_boundaries):
    boundary_energy = grad_mag[boundary_mask].mean()
    interior_energy = grad_mag[~boundary_mask].mean()
    ratio = boundary_energy / interior_energy
    assert ratio < 1.2, "Seam artifacts detected"
```

### 7.3 Depth Quality Metrics (DA3)

```python
from lux_depth_v3.validation import DepthQualityMetrics

metrics = DepthQualityMetrics.compute(predicted_depth, ground_truth_depth)
# RMSE < 0.5
# MAE < 0.3
# δ1 (δ < 1.25) > 0.85
# δ2 (δ < 1.25²) > 0.95
# δ3 (δ < 1.25³) > 0.98
```

---

## 8. Recommended Configuration Summary

### 8.1 Production Workflow (Interior)

```bash
lux-depth-v2 \
    --input-dir renders/ \
    --output-dir output/ \
    --preset interior_luxury_apex_quality
```

### 8.2 Production Workflow (Exterior)

```bash
lux-depth-v2 \
    --input-dir renders/ \
    --output-dir output/ \
    --preset exterior_pool_apex_quality
```

### 8.3 Maximum Quality (Python API)

```python
from lux_depth_v2.depth_inference import TiledInferenceConfig, TiledDepthEstimator
from lux_depth_v2.global_anchor import GlobalAnchorConfig
from lux_depth_v2.edge_snapping import EdgeSnappingConfig

# Configure for maximum quality
config = TiledInferenceConfig(
    tile_size=1024,
    overlap=192,  # Increased for texture-heavy scenes
    model_name="depth-anything/Depth-Anything-V2-Large-hf",
    device="auto",
    bypass_image_processor=True,  # CRITICAL
    fusion_mode="weighted",
    blend_window="hann",
    reconcile_scales=True,
    reconcile_method="robust",
    validate_edges=True,
    edge_alignment_threshold=0.5,
    use_global_anchor=True,
    global_anchor_config=GlobalAnchorConfig(),
    use_edge_snapping=True,
    edge_snap_config=EdgeSnappingConfig(),
    use_production_refinement=True,
    refinement_use_clahe=True,
    refinement_use_edge_filter=True,
    refinement_use_edge_snap=True
)

estimator = TiledDepthEstimator(config)
depth = estimator.estimate_depth(rgb_image)

# Validate quality
alignment = estimator.compute_edge_alignment(rgb_image, depth)
print(f"Edge alignment: {alignment:.3f}")  # Should be > 0.5
```

### 8.4 Multi-View / Metric Depth (DA3)

```python
from lux_depth_v3 import DA3InferenceEngine, DA3Config, ModelVariant

config = DA3Config(
    model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    inference_mode="multi_view"
)
config.api.process_res = 1024
config.api.ref_view_strategy = "saddle_balanced"
config.postprocessing.refinement.enable_refinement = True

engine = DA3InferenceEngine(config)
result = engine.infer(
    images=image_paths,
    convert_to_metric=True,
    focal_length_px=1200.0
)

print(f"Metric depth range: {result.metric_depth.min():.2f}m - {result.metric_depth.max():.2f}m")
```

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `lux_depth_v2/depth_inference.py` | Tiled inference with scale reconciliation |
| `lux_depth_v2/config.py` | Preset definitions and configuration |
| `lux_depth_v2/depth_refinement.py` | Production refinement chain |
| `lux_depth_v2/global_anchor.py` | Global anchor fusion |
| `lux_depth_v2/edge_snapping.py` | RGB-aligned edge snapping |
| `lux_depth_v3/inference.py` | DA3 inference engine |
| `lux_depth_v3/config.py` | DA3 model variants and configuration |
| `lux_depth_v3/metric_depth.py` | Metric depth conversion |
| `high_fidelity_depth/depth_estimator.py` | Research-grade tiled inference |
| `high_fidelity_depth/quality_metrics.py` | Quality validation metrics |

---

## Conclusion

For **maximum depth quality** in architectural rendering:

1. **Use `lux_depth_v2`** with `interior_luxury_apex_quality` or `exterior_pool_apex_quality` presets
2. **Model**: `depth-anything/Depth-Anything-V2-Large-hf` (non-commercial) or DA2-Small (commercial)
3. **Tile size**: 1024px with 128-192px overlap
4. **Enable**: Scale reconciliation, global anchor fusion, edge snapping, production refinement
5. **Validate**: Edge alignment > 0.5, seam energy ratio < 1.2

For **multi-view or metric depth**, use `lux_depth_v3` with DA3-Nested-Giant-Large-1.1 (non-commercial) or DA3-Metric-Large (commercial).

---

**Document Version**: 1.0
**Last Updated**: 2025-01-05
**Maintained by**: Transformation Portal Team
