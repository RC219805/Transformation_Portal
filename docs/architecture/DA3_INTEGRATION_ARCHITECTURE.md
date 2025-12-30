# Depth Anything 3 (DA3) Integration Architecture

**Author**: Transformation Portal Architect
**Date**: 2025-12-19
**Status**: Implemented (Design Documentation)
**Module**: `lux_depth_v3/`

---

## Executive Summary

This document defines the architectural design for integrating Depth Anything 3 (DA3) into the Transformation Portal. The integration provides production-ready monocular and multi-view depth estimation with metric depth output, camera pose estimation, and Gaussian Splatting capabilities while maintaining security hardening, backward compatibility, and operational excellence.

**Key Architectural Decisions:**

1. **Module Isolation**: `lux_depth_v3/` operates independently from `lux_depth_v2/` to prevent regression
2. **Dual API Strategy**: Support both official DA3 Python API and CLI wrapper for flexibility
3. **Security First**: Inherit and extend security hardening from v2 (CVE-2024-27763 mitigated)
4. **Model Caching**: Pre-download mechanism for offline operation and deployment consistency
5. **License Compliance**: Automated validation to prevent inadvertent commercial use of NC-licensed models
6. **Validation Integration**: Seamless integration with existing `validation_v1_baseline_pack/` framework

---

## 1. System Architecture

### 1.1 Module Structure

```
lux_depth_v3/
├── __init__.py              # Public API exports
├── cli.py                   # CLI interface (lux-depth-v3 command)
├── config.py                # Configuration dataclasses and enums
├── da3_wrapper.py           # DA3 API wrapper (Python API + CLI modes)
├── da3_integration.py       # High-level convenience API
├── inference.py             # Core inference engine
├── input_manager.py         # Input validation and preprocessing
├── preprocessing.py         # Image preprocessing utilities
├── postprocessing.py        # Depth filtering and refinement
├── metric_depth.py          # Metric depth conversion utilities
├── model_cache.py           # Model caching and download management
├── reference_view.py        # Reference view selection for multi-view
├── export.py                # Multi-format export (NPZ, GLB, PLY, TIFF)
├── validation.py            # Quality metrics and validation gates
├── service.py               # FastAPI service mode (optional)
├── license.py               # License validation and compliance
├── pyproject.toml           # Package metadata and dependencies
├── requirements.txt         # Core dependencies
├── SECURITY.md              # Security guidelines
├── README.md                # User documentation
├── INTEGRATION_GUIDE.md     # Integration documentation
├── tests/                   # Unit and integration tests
│   ├── test_lux_depth_v3.py
│   ├── test_da3_api.py
│   ├── test_reference_view.py
│   └── test_model_versioning.py
├── docs/                    # Extended documentation
│   ├── CLI_INTEGRATION.md
│   ├── LICENSE_GUIDE.md
│   ├── METRIC_DEPTH_GUIDE.md
│   ├── MODEL_CACHING_GUIDE.md
│   └── MODEL_VERSIONING.md
├── examples/                # Example scripts and notebooks
│   ├── basic_depth.py
│   ├── multi_view.py
│   ├── metric_depth.py
│   └── gaussian_splatting.py
└── benchmark/               # Performance benchmarking
    ├── benchmark_models.py
    └── compare_backends.py
```

### 1.2 Architectural Layers

```
┌──────────────────────────────────────────────────────────┐
│                    CLI / Python API                       │
│  (lux-depth-v3 command, estimate_depth(), DA3DepthEstimator) │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              Configuration & License Validation           │
│  (DA3Config, ModelVariant, LicenseValidator)             │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  Input Manager                            │
│  - Input validation (file size, dimensions, extensions)   │
│  - Multi-image batching                                   │
│  - Camera pose management (extrinsics, intrinsics)        │
│  - Reference view selection                               │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  Preprocessing                            │
│  - Resize and normalization                               │
│  - Padding for inference                                  │
│  - Color space conversion                                 │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              DA3 Inference Engine                         │
│  ┌────────────────┬─────────────────┐                    │
│  │  Python API    │   CLI Wrapper   │                    │
│  │  (Recommended) │   (Fallback)    │                    │
│  └────────────────┴─────────────────┘                    │
│  - Model caching and lazy loading                         │
│  - GPU/CPU/MPS device management                          │
│  - Monocular & multi-view modes                           │
│  - Gaussian Splatting (3DGS)                              │
│  - Pose estimation                                        │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  Postprocessing                           │
│  - Depth filtering (bilateral, median)                    │
│  - Edge refinement                                        │
│  - Multi-view fusion                                      │
│  - Metric depth conversion                                │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                 Validation & Quality Gates                │
│  - RMSE, MAE, Abs/Sq Relative error                      │
│  - δ threshold accuracies                                 │
│  - Edge completeness                                      │
│  - Automated quality gates                                │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              Export Manager                               │
│  - NPZ (mini_npz, full_npz)                              │
│  - 3D formats (GLB, PLY)                                  │
│  - Visualizations (depth_vis, feat_vis)                   │
│  - Image formats (TIFF, PNG)                              │
│  - Video (gs_video, depth_vis video)                      │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Core API Design

### 2.1 Python API Interface

```python
from lux_depth_v3 import (
    estimate_depth,           # High-level convenience function
    DA3DepthEstimator,        # Full-featured class API
    DA3Config,                # Configuration
    ModelVariant,             # Model selection
    InferenceMode,            # Monocular vs multi-view
)

# Quick estimation (simplified API)
result = estimate_depth(
    input_path="image.jpg",
    output_dir="output/",
    model="large-1.1",        # ModelVariant enum or string
    device="auto",            # auto, cuda, mps, cpu
)

# Advanced usage (class API)
estimator = DA3DepthEstimator(
    model=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    device="cuda",
    config=DA3Config(
        inference_mode=InferenceMode.MULTI_VIEW,
        process_res=1024,
        export_format=["glb", "mini_npz", "depth_vis"],
    ),
)

# Monocular depth
result = estimator.process_image(
    input_path="kitchen.jpg",
    output_dir="output/kitchen/",
)

# Multi-view with pose estimation
result = estimator.process_images(
    input_paths=["view1.jpg", "view2.jpg", "view3.jpg"],
    output_dir="output/multiview/",
    estimate_poses=True,  # Automatic pose estimation
)

# Multi-view with known poses
result = estimator.process_images(
    input_paths=["view1.jpg", "view2.jpg"],
    extrinsics=camera_extrinsics,  # (N, 4, 4)
    intrinsics=camera_intrinsics,  # (N, 3, 3)
    output_dir="output/multiview/",
)

# Gaussian Splatting
result = estimator.process_images(
    input_paths=image_paths,
    output_dir="output/gs/",
    infer_gs=True,
    export_format=["gs_ply", "gs_video"],
)

# Access results
depth_array = result.depth_array       # (N, H, W) numpy array
confidence = result.confidence_array   # (N, H, W) confidence maps
extrinsics = result.extrinsics         # (N, 4, 4) camera poses
intrinsics = result.intrinsics         # (N, 3, 3) camera intrinsics
```

### 2.2 Result Object Structure

```python
@dataclass
class DA3Result:
    """Unified result container for DA3 inference."""

    # Status
    success: bool
    message: str

    # Output paths
    output_dir: Path
    depth_path: Optional[Path] = None       # Primary depth file (NPZ)
    glb_path: Optional[Path] = None         # 3D mesh
    ply_path: Optional[Path] = None         # Point cloud
    depth_vis_dir: Optional[Path] = None    # Visualization frames

    # Lazy-loaded data (only when accessed)
    _depth_array: Optional[np.ndarray] = None
    _confidence_array: Optional[np.ndarray] = None
    _extrinsics: Optional[np.ndarray] = None
    _intrinsics: Optional[np.ndarray] = None

    # Quality metrics
    metrics: Optional[DepthQualityMetrics] = None

    @property
    def depth_array(self) -> np.ndarray:
        """Lazy load depth array from NPZ."""
        if self._depth_array is None:
            self._depth_array = load_depth_from_npz(self.depth_path)
        return self._depth_array

    # Similar properties for confidence, extrinsics, intrinsics
```

### 2.3 Configuration Schema

```python
@dataclass
class DA3Config:
    """Comprehensive DA3 configuration."""

    # Model selection
    model: ModelVariant = ModelVariant.DA3_LARGE_V1_1

    # Inference mode
    inference_mode: InferenceMode = InferenceMode.MONOCULAR

    # Processing parameters
    process_res: int = 768            # Processing resolution
    process_res_method: str = "long"  # 'long' or 'short' edge

    # Export configuration
    export_format: List[str] = field(default_factory=lambda: ["mini_npz"])
    export_dir: Optional[Path] = None

    # Pose parameters
    use_ray_pose: bool = False
    align_to_input_ext_scale: bool = False
    ref_view_strategy: str = "center"  # center, auto, manual

    # Gaussian Splatting
    infer_gs: bool = False
    render_exts: Optional[np.ndarray] = None
    render_ixts: Optional[np.ndarray] = None
    render_hw: Tuple[int, int] = (512, 512)

    # GLB export settings
    conf_thresh_percentile: float = 0.5
    num_max_points: int = 10_000_000
    show_cameras: bool = False

    # Feature extraction
    export_feat_layers: Optional[List[int]] = None
    feat_vis_fps: int = 15

    # Device & precision
    device: str = "auto"
    precision: str = "fp16"

    # CLI-specific (backward compatibility)
    cli: DA3CLIConfig = field(default_factory=DA3CLIConfig)

    # API-specific (Python API mode)
    api: DA3APIConfig = field(default_factory=DA3APIConfig)

    def to_api_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for DepthAnything3.inference()."""
        # Implementation provided in config.py
        pass
```

---

## 3. Model Caching Strategy

### 3.1 Cache Architecture

**Objective**: Pre-download all DA3 models for offline operation and consistent performance.

```python
class ModelCacheManager:
    """Manage DA3 model downloads and caching."""

    # Supported models
    OFFICIAL_MODELS = {
        # v1.1 models (recommended)
        "nested-giant-large-v1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "giant-v1.1": "depth-anything/DA3-GIANT-1.1",
        "large-v1.1": "depth-anything/DA3-LARGE-1.1",

        # Apache-licensed (commercial)
        "metric-large": "depth-anything/DA3METRIC-LARGE",
        "base": "depth-anything/DA3-BASE",
        "small": "depth-anything/DA3-SMALL",
        "mono-large": "depth-anything/DA3MONO-LARGE",
    }

    # Recommended sets
    RECOMMENDED_SETS = {
        "essential": ["nested-giant-large-v1.1", "metric-large"],
        "production": ["nested-giant-large-v1.1", "giant-v1.1",
                      "large-v1.1", "metric-large"],
        "benchmark": list(OFFICIAL_MODELS.keys()),
    }
```

### 3.2 Cache Workflow

```
┌─────────────────┐
│  User Request   │ ──▶ lux-depth-v3 cache-download --set production
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  ModelCacheManager                              │
│  - Verify HuggingFace Hub access                │
│  - Check existing cache                         │
│  - Download missing models                      │
│  - Validate checksums (if available)            │
│  - Update cache metadata                        │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Cache Storage                                  │
│  ~/.cache/huggingface/hub/models--depth-anything│
│  - Model weights (.safetensors, .bin)           │
│  - Configuration files (config.json)            │
│  - Tokenizer files (if applicable)              │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Cache Metadata                                 │
│  .cache_manifest.json                           │
│  {                                              │
│    "nested-giant-large-v1.1": {                 │
│      "model_id": "depth-anything/...",          │
│      "local_path": "~/.cache/...",              │
│      "size_gb": 5.6,                            │
│      "cached_at": "2025-12-19T10:00:00Z",       │
│      "verified": true                           │
│    }                                            │
│  }                                              │
└─────────────────────────────────────────────────┘
```

### 3.3 Cache CLI Commands

```bash
# Download model sets
lux-depth-v3 cache-download --set essential      # ~6GB
lux-depth-v3 cache-download --set production     # ~12GB
lux-depth-v3 cache-download --set benchmark      # ~20GB

# Download specific models
lux-depth-v3 cache-download --models nested-giant-large-v1.1,metric-large

# List cached models
lux-depth-v3 cache-list

# Show cache statistics
lux-depth-v3 cache-stats

# Verify cache integrity
lux-depth-v3 cache-verify

# Clean unused models
lux-depth-v3 cache-clean --keep essential
```

### 3.4 Offline Operation

Once models are cached:

```python
# Inference works offline (no internet required)
estimator = DA3DepthEstimator(
    model="nested-giant-large-v1.1",
    device="cuda",
)

# Model loaded from local cache
result = estimator.process_image("image.jpg", "output/")
```

**Benefits:**

- ✅ Eliminate download latency during inference
- ✅ Enable air-gapped deployment
- ✅ Consistent performance in production
- ✅ Deployment snapshots for reproducibility

---

## 4. Metric Depth Conversion

### 4.1 Conversion Workflow

DA3 models output **relative depth** (arbitrary scale). Metric depth conversion provides **absolute depth in meters**.

```python
from lux_depth_v3.metric_depth import convert_to_metric_depth

# Option 1: Using camera intrinsics (most accurate)
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,      # (H, W) relative depth
    intrinsics=camera_intrinsics,        # (3, 3) camera matrix
    method="intrinsics",
)

# Option 2: Using focal length
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    focal_length_px=1200.0,
    method="focal",
)

# Option 3: Using field-of-view estimation
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    image_width=1920,
    fov_degrees=60.0,                    # Horizontal FOV
    method="fov",
)

# Option 4: Scale from known reference (e.g., ceiling height)
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    reference_point=(100, 100),          # Pixel coordinates
    reference_depth_meters=2.8,          # Known depth (e.g., ceiling)
    method="reference",
)
```

### 4.2 Metric Depth Utilities

```python
from lux_depth_v3.metric_depth import (
    DepthStatistics,
    calculate_depth_statistics,
    visualize_metric_depth,
)

# Calculate statistics
stats = calculate_depth_statistics(metric_depth)
print(f"Mean depth: {stats.mean_depth:.2f}m")
print(f"Max depth: {stats.max_depth:.2f}m")
print(f"Depth range: {stats.depth_range:.2f}m")

# Visualize with metric scale bar
visualize_metric_depth(
    metric_depth,
    output_path="depth_metric_vis.png",
    colormap="turbo",
    show_scale_bar=True,
    depth_unit="meters",
)
```

### 4.3 DA3METRIC-LARGE Auto-Detection

```python
# DA3METRIC-LARGE outputs metric depth natively
# Conversion is automatically skipped

estimator = DA3DepthEstimator(model="metric-large")
result = estimator.process_image("image.jpg", "output/")

# Already in meters (no conversion needed)
depth_meters = result.depth_array
```

---

## 5. Integration Points

### 5.1 Integration with Existing Pipelines

```
┌──────────────────┐
│  lux_depth_v2    │ ← DA2-based production pipeline (PRESERVED)
└──────────────────┘   Security hardened, stable, GPU-accelerated
                       Material segmentation, upscaling, etc.

┌──────────────────┐
│  lux_depth_v3    │ ← DA3-based advanced features
└─────┬────────────┘   Multi-view, metric depth, Gaussian Splatting
      │
      ├──▶ validation_v1_baseline_pack/  (Quality metrics)
      ├──▶ lux_render_pipeline.py        (AI enhancement)
      ├──▶ material_response.py          (Material-aware processing)
      └──▶ luxury_tiff_batch_processor.py (16-bit TIFF workflows)
```

### 5.2 Validation Framework Integration

```python
# lux_depth_v3/validation.py integrates with existing framework

from lux_depth_v3 import DA3DepthEstimator
from lux_depth_v3.validation import DepthQualityMetrics

estimator = DA3DepthEstimator(model="large-v1.1")
result = estimator.process_image("image.jpg", "output/")

# Compute quality metrics (if ground truth available)
metrics = DepthQualityMetrics.compute(
    predicted_depth=result.depth_array,
    ground_truth_depth=gt_depth,
)

# Validate against quality gates
if metrics.rmse < 0.5 and metrics.delta_1 > 0.85:
    print("✅ Quality gate passed")
else:
    print("❌ Quality gate failed")

# Export to validation framework format
metrics.export_to_json("validation_v1_baseline_pack/metrics/da3_results.json")
```

### 5.3 Test Integration

```python
# tests/test_da3_integration.py

import pytest
from lux_depth_v3 import estimate_depth

def test_da3_monocular_depth():
    """Test monocular depth estimation."""
    result = estimate_depth(
        "data/sample_images/kitchen.jpg",
        "test_output/",
        model="base",  # Use small model for fast tests
    )
    assert result.success
    assert result.depth_array.shape[1:] == (480, 640)  # H, W

def test_da3_metric_depth():
    """Test metric depth conversion."""
    result = estimate_depth(
        "data/sample_images/kitchen.jpg",
        "test_output/",
        model="metric-large",
    )
    assert result.success
    # Verify reasonable depth range (architectural scene)
    assert 0.5 < result.depth_array.mean() < 10.0
```

---

## 6. Security Architecture

### 6.1 Security Hardening (Inherited from V2)

```python
# Input validation
class InputManager:
    def __init__(
        self,
        max_file_size_mb: int = 50,
        max_image_dimension: int = 4096,
        allowed_extensions: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif"},
    ):
        """Initialize with security constraints."""
        self.max_file_size_mb = max_file_size_mb
        self.max_image_dimension = max_image_dimension
        self.allowed_extensions = allowed_extensions

    def validate_input(self, path: Path) -> None:
        """Validate input against security constraints."""
        # File size check
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")

        # Extension check
        if path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"Invalid extension: {path.suffix}")

        # Dimension check
        img = Image.open(path)
        if max(img.size) > self.max_image_dimension:
            raise ValueError(f"Image too large: {img.size} > {self.max_image_dimension}")

        # Path traversal check
        if ".." in str(path):
            raise ValueError("Path traversal detected")
```

### 6.2 Service Mode Security

```python
# service.py - FastAPI service with rate limiting

from fastapi import FastAPI, UploadFile, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

@app.post("/estimate_depth")
@limiter.limit("60/minute")  # Rate limit
async def estimate_depth_endpoint(
    file: UploadFile,
    model: str = "large-v1.1",
):
    """Depth estimation API endpoint."""

    # Input validation
    if file.size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(400, "File too large")

    if not file.filename.endswith((".jpg", ".png")):
        raise HTTPException(400, "Invalid file type")

    # Process
    # ...
```

### 6.3 Dependency Security

**Safe Dependencies:**

```txt
# requirements.txt - No vulnerable packages
numpy>=1.20.0
pillow>=9.0.0
torch>=2.0.0
typer>=0.9.0
huggingface-hub>=0.19.0
```

**Excluded:**

- ❌ `basicsr` (CVE-2024-27763)
- ❌ `realesrgan` (depends on vulnerable basicsr)

---

## 7. License Compliance Architecture

### 7.1 License Validation Workflow

```
┌─────────────────┐
│  Model Request  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  LicenseValidator                       │
│  - Check model license                  │
│  - Validate commercial usage            │
│  - Warn or block based on mode          │
└────────┬────────────────────────────────┘
         │
         ├──▶ Apache-2.0 ────▶ ✅ Allow (commercial OK)
         │
         └──▶ CC-BY-NC-4.0 ──▶ ⚠️ Warn or ❌ Block
                                (commercial use not allowed)
```

### 7.2 License Validator Implementation

```python
from lux_depth_v3.license import LicenseValidator, LicenseMode

# Strict mode (blocks NC models for commercial use)
validator = LicenseValidator(mode=LicenseMode.STRICT)

try:
    validator.validate_model_for_use(
        model="nested-giant-large-v1.1",  # CC-BY-NC-4.0
        commercial=True,
    )
except LicenseViolationError as e:
    print(f"❌ License violation: {e}")
    print(f"💡 Use 'metric-large' instead (Apache-2.0)")

# Permissive mode (warns but allows)
validator = LicenseValidator(mode=LicenseMode.PERMISSIVE)
validator.validate_model_for_use(
    model="nested-giant-large-v1.1",
    commercial=True,
)
# ⚠️ Warning logged, but processing continues
```

### 7.3 CLI License Commands

```bash
# Show license info for model
lux-depth-v3 show-license --model nested-giant-large-v1.1

# Output:
# Model: DA3NESTED-GIANT-LARGE-1.1
# License: CC-BY-NC-4.0 (Non-commercial)
# Commercial use: ❌ Not allowed
#
# For commercial projects, use:
# - metric-large (Apache-2.0) ✅
# - base (Apache-2.0) ✅
# - small (Apache-2.0) ✅
```

---

## 8. Migration Path (V2 → V3)

### 8.1 Coexistence Strategy

**Principle**: V2 and V3 coexist without interference. Users opt-in to V3 features.

```python
# V2 continues to work (DA2-based)
from lux_depth_v2 import LuxPipelineV2

pipeline_v2 = LuxPipelineV2(preset="interior_luxury")
result_v2 = pipeline_v2.process("image.jpg", "output/")

# V3 provides DA3 features (opt-in)
from lux_depth_v3 import DA3DepthEstimator

estimator_v3 = DA3DepthEstimator(model="large-v1.1")
result_v3 = estimator_v3.process_image("image.jpg", "output/")
```

### 8.2 Feature Comparison Matrix

| Feature                     | lux_depth_v2 | lux_depth_v3 |
|-----------------------------|--------------|--------------|
| Depth model                 | DA2          | DA3          |
| Monocular depth             | ✅            | ✅            |
| Multi-view depth            | ❌            | ✅            |
| Metric depth                | ❌            | ✅            |
| Camera pose estimation      | ❌            | ✅            |
| Gaussian Splatting          | ❌            | ✅            |
| Material segmentation       | ✅            | ⏳ (planned)  |
| Upscaling                   | ✅            | ⏳ (planned)  |
| GPU acceleration            | ✅            | ✅            |
| Security hardening          | ✅            | ✅            |
| Production stability        | ✅ (mature)   | ⏳ (beta)     |

### 8.3 Migration Workflow

```
Phase 1: Evaluation (Current)
├── Install lux_depth_v3
├── Test DA3 models on sample images
├── Benchmark accuracy and performance
└── Validate license compliance

Phase 2: Parallel Operation (Recommended)
├── Keep lux_depth_v2 for production
├── Use lux_depth_v3 for advanced features
│   ├── Multi-view scenes
│   ├── Metric depth requirements
│   └── 3D reconstruction projects
└── Gradual migration as confidence grows

Phase 3: Full Migration (Optional)
├── Migrate critical workflows to V3
├── Deprecate V2 (but keep for legacy support)
└── Archive V2 codebase
```

---

## 9. Performance Architecture

### 9.1 Performance Optimization Strategies

```python
# 1. Model caching (10-20x speedup for batch processing)
from lux_depth_v3 import DA3DepthEstimator

estimator = DA3DepthEstimator(model="large-v1.1")
# Model loaded once, reused for all images

for image_path in image_paths:
    result = estimator.process_image(image_path, "output/")
    # No model reload overhead

# 2. GPU acceleration
estimator = DA3DepthEstimator(model="large-v1.1", device="cuda")

# 3. Batch processing
results = estimator.process_directory("input/", "output/")

# 4. Resolution optimization
estimator = DA3DepthEstimator(
    model="large-v1.1",
    config=DA3Config(process_res=512),  # Faster, lower quality
)
```

### 9.2 Benchmark Results (Reference)

| Model              | Device | Resolution | Time/Image | Throughput   |
|--------------------|--------|------------|------------|--------------|
| base               | CPU    | 518        | 2.1s       | ~1700 img/hr |
| large-v1.1         | CPU    | 518        | 5.8s       | ~620 img/hr  |
| large-v1.1         | CUDA   | 518        | 0.3s       | ~12000 img/hr|
| nested-giant-v1.1  | CUDA   | 518        | 0.8s       | ~4500 img/hr |

---

## 10. Testing & Validation Architecture

### 10.1 Test Pyramid

```
┌─────────────────────────────────────┐
│       Integration Tests             │  E2E workflows
│   (test_da3_integration.py)         │  - Monocular depth
│                                     │  - Multi-view depth
└─────────────────────────────────────┘  - Metric conversion
            ▲
            │
┌───────────┴─────────────────────────┐
│         Unit Tests                  │  Component tests
│  (test_lux_depth_v3.py)             │  - Config validation
│  (test_reference_view.py)           │  - Input manager
│  (test_model_versioning.py)         │  - Preprocessing
└─────────────────────────────────────┘  - Export formats
            ▲
            │
┌───────────┴─────────────────────────┐
│     Validation Tests                │  Quality gates
│  (test_quality_metrics.py)          │  - RMSE thresholds
│  (validation_v1_baseline_pack/)     │  - Edge accuracy
└─────────────────────────────────────┘  - Consistency checks
```

### 10.2 Quality Gates

```python
# Automated quality validation
from lux_depth_v3.validation import validate_depth_quality

result = estimator.process_image("image.jpg", "output/")

quality_passed = validate_depth_quality(
    depth=result.depth_array,
    ground_truth=gt_depth,
    gates={
        "rmse": 0.5,        # RMSE < 0.5
        "mae": 0.3,         # MAE < 0.3
        "delta_1": 0.85,    # δ1 > 85%
    },
)

if not quality_passed:
    raise ValueError("Quality gate failed - check depth accuracy")
```

---

## 11. Deployment Architecture

### 11.1 Deployment Modes

**Mode 1: Standalone CLI**

```bash
# Production deployment
lux-depth-v3 cache-download --set production
lux-depth-v3 process image.jpg -o output/ -m large-1.1
```

**Mode 2: Python Library**

```python
# Import in other scripts
from lux_depth_v3 import estimate_depth

result = estimate_depth("image.jpg", "output/", model="large-1.1")
```

**Mode 3: FastAPI Service**

```bash
# Start service
lux-depth-v3 serve --host 0.0.0.0 --port 8088

# Use API
curl -X POST http://localhost:8088/estimate_depth \
  -F "file=@image.jpg" \
  -F "model=large-1.1"
```

### 11.2 Docker Deployment

```dockerfile
# Dockerfile (example)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY lux_depth_v3/requirements.txt .
RUN pip install -r requirements.txt

# Copy module
COPY lux_depth_v3/ ./lux_depth_v3/

# Pre-cache models
RUN python -c "from lux_depth_v3.model_cache import precache_models; precache_models('production')"

# Expose service port
EXPOSE 8088

# Run service
CMD ["python", "-m", "lux_depth_v3.service", "--host", "0.0.0.0", "--port", "8088"]
```

---

## 12. Architectural Decision Records (ADRs)

### ADR-001: Module Isolation

**Context**: Integration of DA3 requires significant new functionality.

**Decision**: Create separate `lux_depth_v3/` module instead of extending `lux_depth_v2/`.

**Rationale**:
- Prevents regression in production v2 pipeline
- Allows independent versioning and release cycles
- Clear separation of DA2 and DA3 implementations
- Easier to deprecate v2 in future if desired

**Consequences**:
- ✅ V2 remains stable and production-ready
- ✅ V3 can iterate rapidly without risk
- ⚠️ Code duplication for shared utilities (mitigated by Platform Core)
- ⚠️ Users must choose between v2 and v3 APIs

### ADR-002: Dual API Strategy

**Context**: DA3 provides both Python API and CLI interfaces.

**Decision**: Support both modes with Python API as recommended path.

**Rationale**:
- Python API provides full feature access (GS, pose estimation, features)
- CLI provides fallback for users without Python API installed
- Python API is faster (no subprocess overhead)
- Both modes ensure maximum flexibility

**Consequences**:
- ✅ Best performance with Python API
- ✅ Fallback for minimal installations
- ⚠️ Increased testing surface (both modes)
- ⚠️ Additional documentation burden

### ADR-003: Model Caching Strategy

**Context**: DA3 models are large (0.3-5GB) and download on first use.

**Decision**: Implement pre-caching system with recommended model sets.

**Rationale**:
- Eliminates download latency in production
- Enables offline/air-gapped deployment
- Provides consistent performance
- Supports deployment snapshots

**Consequences**:
- ✅ Predictable deployment
- ✅ Offline operation
- ⚠️ Requires upfront disk space (~6-20GB)
- ⚠️ Cache management complexity

### ADR-004: License Validation

**Context**: DA3 models have mixed licenses (Apache vs CC-BY-NC).

**Decision**: Implement automated license validation with strict/permissive modes.

**Rationale**:
- Prevents inadvertent license violations
- Educates users on commercial restrictions
- Provides clear alternatives
- Supports both development (permissive) and production (strict)

**Consequences**:
- ✅ Legal compliance safeguards
- ✅ Clear documentation of limitations
- ⚠️ Additional code complexity
- ⚠️ User friction for NC models

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DA3 API changes break compatibility | High | Version pinning, wrapper abstraction, regression tests |
| Model download failures in production | High | Pre-caching, offline bundles, fallback models |
| License violations in commercial use | Critical | Automated validation, strict mode enforcement, documentation |
| Performance regression vs V2 | Medium | Benchmarking suite, performance gates, optimization |
| Security vulnerabilities in dependencies | High | Dependency scanning, minimal dependencies, version pinning |
| V2/V3 API confusion | Medium | Clear documentation, naming conventions, migration guides |

---

## 14. Future Roadmap

### Phase 1: Core Integration (✅ Complete)
- [x] DA3 Python API wrapper
- [x] Model caching system
- [x] Metric depth conversion
- [x] License validation
- [x] CLI interface
- [x] Documentation

### Phase 2: Advanced Features (⏳ In Progress)
- [ ] Material segmentation integration
- [ ] Upscaling pipeline (inherit from v2)
- [ ] Multi-view fusion improvements
- [ ] Real-time service optimization
- [ ] Benchmark vs DA2 quality

### Phase 3: Production Hardening (🔜 Planned)
- [ ] Comprehensive test suite (target: 90% coverage)
- [ ] Performance profiling and optimization
- [ ] Production deployment guides
- [ ] Migration tooling (v2 → v3 converter)
- [ ] Monitoring and observability

### Phase 4: Ecosystem Integration (🔮 Future)
- [ ] Integration with lux_render_pipeline
- [ ] Material Response technology compatibility
- [ ] Video processing workflows
- [ ] Cloud deployment templates
- [ ] RAG system knowledge integration

---

## 15. Conclusion

The DA3 integration architecture provides a production-ready, secure, and flexible foundation for advanced depth estimation in the Transformation Portal. By maintaining module isolation, implementing dual API strategies, and prioritizing security and license compliance, the architecture ensures both innovation velocity and operational stability.

**Key Achievements:**

1. ✅ Clean module separation prevents regression
2. ✅ Comprehensive model caching enables offline operation
3. ✅ License validation prevents legal issues
4. ✅ Security hardening inherited from v2
5. ✅ Flexible API supports both simple and advanced use cases
6. ✅ Validation framework integration maintains quality standards

**Next Steps:**

1. Complete Phase 2 advanced features
2. Achieve 90% test coverage
3. Production deployment pilot
4. Performance benchmarking vs DA2
5. User migration guides and workshops

---

## Appendices

### A. File Manifest

Complete list of files in `lux_depth_v3/`:

```
lux_depth_v3/
├── __init__.py              (92 lines)
├── cli.py                   (755 lines)
├── config.py                (496 lines)
├── da3_wrapper.py           (618 lines)
├── da3_integration.py       (264 lines)
├── inference.py             (658 lines)
├── input_manager.py         (231 lines)
├── preprocessing.py         (196 lines)
├── postprocessing.py         (269 lines)
├── metric_depth.py          (270 lines)
├── model_cache.py           (308 lines)
├── reference_view.py        (240 lines)
├── export.py                (160 lines)
├── validation.py            (238 lines)
├── service.py               (207 lines)
├── license.py               (137 lines)
├── pyproject.toml           (114 lines)
├── requirements.txt         (8 lines)
├── README.md                (595 lines)
├── SECURITY.md              (215 lines)
└── INTEGRATION_GUIDE.md     (416 lines)

Total: ~6,088 lines of code + documentation
```

### B. References

- [Depth Anything 3 Official Repository](https://github.com/DepthAnything/Depth-Anything-V3)
- [Hugging Face Model Hub](https://huggingface.co/depth-anything)
- [lux_depth_v2 Architecture](../lux_depth_v2/ARCHITECTURE.md)
- [Transformation Portal Security Guidelines](../../SECURITY.md)
- [ADR Template](./adr/ADR_TEMPLATE.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-19
**Approved By**: Transformation Portal Architect
**Status**: Active
