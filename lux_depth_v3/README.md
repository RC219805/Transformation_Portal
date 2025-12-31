# Lux Depth V3 - Depth Anything 3 Integration

Production-ready integration of Depth Anything 3 (DA3) models for the Transformation Portal, providing unified any-view monocular and multi-view depth inference with metric depth output, camera pose estimation, and Gaussian Splatting support.

## ⚠️ License Notice

**DA3 models are released under two licenses:**

- **Apache-2.0** (Commercial-friendly): `DA3-BASE`, `DA3-SMALL`, `DA3METRIC-LARGE`, `DA3MONO-LARGE` ✅
- **CC-BY-NC-4.0** (Non-commercial only): `DA3NESTED-GIANT-LARGE`, `DA3-GIANT`, `DA3-LARGE` ⚠️

**For commercial use, use Apache-licensed models.** See [LICENSE_GUIDE.md](docs/LICENSE_GUIDE.md) for details.

## Features

✨ **Full DA3 Python API Integration**
- Complete access to official DA3 API features
- Monocular and multi-view depth estimation
- Gaussian Splatting (3DGS) for novel view synthesis
- Pose-conditioned depth with camera parameters
- Feature extraction from intermediate layers
- Multiple export formats (NPZ, GLB, PLY, videos)

🎯 **Model Variants (v1.1 Recommended)**

**v1.1 Models (Latest):**
- **nested-giant-large-v1.1** (1.40B, NC) - All features, best accuracy
- **giant-v1.1** (1.15B, NC) - Any-view with GS support
- **large-v1.1** (0.35B, NC) - General use

**Apache-Licensed (Commercial-Friendly):**
- **metric-large** (0.35B) - **Recommended for commercial** ✅
- **base** (0.12B) - Balanced performance
- **small** (0.08B) - Lightweight, mobile-friendly
- **mono-large** (0.35B) - Monocular only, high quality

**v1.0 Models (Deprecated):**
- ⚠️ Use v1.1 for new projects. See [MODEL_VERSIONING.md](docs/MODEL_VERSIONING.md)

🔒 **License Validation**
- Automatic warnings for commercial use of NC models
- Strict mode to enforce compliance
- CLI `--show-license` to view model license info
- Commercial alternative suggestions

🚀 **Performance**
- GPU/CPU/MPS acceleration (Apple Silicon optimized)
- Batch processing support
- **Model caching system** for offline operation ✨ NEW
- Pre-cache all 10 model variants (essential/production/benchmark sets)
- LRU caching for iterative workflows

🔒 **Security Hardening**
- Input validation and sanitization
- File size limits (50MB default)
- Rate limiting for service mode
- No vulnerable dependencies

📊 **Quality Metrics**
- RMSE, MAE, Abs/Sq Relative error
- δ threshold accuracies (δ < 1.25, 1.25², 1.25³)
- Edge completeness and accuracy
- Quality gates for automated validation

💾 **Export Formats**
- `mini_npz` / `full_npz`: NumPy compressed archives
- `glb`: GLTF binary 3D mesh with texture
- `gs_ply`: Gaussian Splatting point cloud
- `gs_video`: Novel view synthesis video
- `depth_vis`: Depth visualization video
- `feat_vis`: Feature visualization video

📏 **Metric Depth Conversion** ✨ NEW
- Convert DA3METRIC-LARGE output to metric depth in meters
- Automatic handling of already-metric models (DA3NESTED)
- Support for camera intrinsics, focal length, or FOV estimation
- Real-world measurements for architectural applications
- Depth statistics and spatial analysis utilities
- See [METRIC_DEPTH_GUIDE.md](docs/METRIC_DEPTH_GUIDE.md) for details

## Quick Start

### Installation

```bash
# Install DA3 official package
pip install depth-anything-3

# Install lux_depth_v3 dependencies
cd lux_depth_v3
pip install -r requirements.txt
```

#### Editable install (offline / no PyPI access)

If your environment cannot reach PyPI, editable installs may fail during PEP 517 build isolation (it tries to download build requirements). Use:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation --no-deps -e lux_depth_v3
```

### Model Caching (Recommended)

Pre-cache DA3 models for offline operation and faster startup:

```bash
# Cache essential models (recommended for development)
lux-depth-v3 cache-download --set essential

# Cache production models (recommended for deployment)
lux-depth-v3 cache-download --set production

# List cached models
lux-depth-v3 cache-list

# Show cache statistics
lux-depth-v3 cache-stats
```

**Python API:**

```python
from lux_depth_v3.model_cache import precache_models

# Cache models for offline use
precache_models("production")
```

**Benefits:**
- ✅ Eliminate download latency during inference
- ✅ Enable offline operation (no internet required)
- ✅ Consistent performance in production
- ✅ Deployment-ready model bundles

See [MODEL_CACHING_GUIDE.md](docs/MODEL_CACHING_GUIDE.md) for comprehensive documentation.

### License Compliance Examples

#### Check Model License

```python
from lux_depth_v3.config import ModelVariant
from lux_depth_v3.license import LicenseValidator

validator = LicenseValidator()

# Check license for a model
variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
info = validator.get_license_info(variant)

print(f"Model: {info['model']}")
print(f"License: {info['license']}")
print(f"Commercial: {'✅ Allowed' if info['commercial_allowed'] else '❌ Not Allowed'}")
if info['alternative']:
    print(f"Commercial alternative: {info['alternative']}")
```

#### Commercial Use with Validation

```python
from lux_depth_v3.config import DA3Config, ModelVariant
from lux_depth_v3.inference import DA3InferenceEngine

# Commercial use with Apache-licensed model (no warning)
config = DA3Config(model_variant=ModelVariant.DA3_METRIC_LARGE)
engine = DA3InferenceEngine(config, commercial_use=True)

# Commercial use with NC model (triggers warning)
config = DA3Config(model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1)
engine = DA3InferenceEngine(config, commercial_use=True)
# ⚠️  LICENSE WARNING: DA3NESTED-GIANT-LARGE-1.1
# License: CC-BY-NC-4.0 (Non-Commercial)
# Commercial use is NOT permitted.
# For commercial applications, use:
#   → DA3METRIC-LARGE (Apache-2.0)

# Strict mode (raises error instead of warning)
try:
    engine = DA3InferenceEngine(
        config,
        commercial_use=True,
        validate_license_strict=True
    )
except RuntimeError as e:
    print(f"License violation: {e}")
    # Switch to commercial alternative
    alternative = ModelVariant.get_commercial_alternative(
        ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
    )
    config = DA3Config(model_variant=alternative)
    engine = DA3InferenceEngine(config, commercial_use=True)
```

### Python API Usage

#### Basic Monocular Depth

```python
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper

# Initialize wrapper (v1.1 recommended)
wrapper = DepthAnything3Wrapper(model_name="da3-large-1.1", device="cuda")

# Run inference
prediction = wrapper.inference(
    image=["path/to/image.jpg"],
    export_dir="output",
    export_format="mini_npz-glb"
)

# Access results
depth = prediction.depth[0]  # (H, W) depth map
confidence = prediction.conf[0]  # (H, W) confidence map
print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")
```

#### Multi-View with Poses

```python
# Multiple views with automatic pose estimation
images = ["view1.jpg", "view2.jpg", "view3.jpg"]

prediction = wrapper.inference(
    image=images,
    ref_view_strategy="saddle_balanced",  # Automatic reference view selection
    export_format="full_npz-glb",
    export_dir="output/multiview"
)

# Access estimated poses
extrinsics = prediction.extrinsics  # (N, 4, 4) camera poses
intrinsics = prediction.intrinsics  # (N, 3, 3) camera matrices
```

#### Metric Depth Conversion ✨ NEW

```python
from lux_depth_v3.config import DA3Config, ModelVariant
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.metric_depth import get_depth_statistics
import numpy as np

# Initialize with DA3METRIC-LARGE (requires conversion)
config = DA3Config(model_variant=ModelVariant.DA3_METRIC_LARGE)
engine = DA3InferenceEngine(config)

# Option 1: With camera intrinsics (most accurate)
intrinsics = np.array([
    [2000.0, 0.0, 1920.0],  # fx, 0, cx
    [0.0, 2000.0, 1080.0],  # 0, fy, cy
    [0.0, 0.0, 1.0]
])

result = engine.infer(
    images=[Path("interior.jpg")],
    intrinsics=intrinsics[np.newaxis, :, :],
    export_dir=Path("output"),
    convert_to_metric=True
)

# Access metric depth
metric_depth = result.metric_depth  # Depth in meters
print(f"Focal length: {result.metric_depth_info.focal_length_px:.2f}px")
print(f"Scale factor: {result.metric_depth_info.scale_factor:.4f}")

# Get statistics
stats = get_depth_statistics(metric_depth[0])
print(f"Depth range: {stats['min_m']:.2f}m - {stats['max_m']:.2f}m")
print(f"Mean depth: {stats['mean_m']:.2f}m")

# Option 2: With explicit focal length
result = engine.infer(
    images=[Path("exterior.jpg")],
    convert_to_metric=True,
    focal_length_px=500.0,
    export_dir=Path("output")
)

# Option 3: With FOV estimation (less accurate)
result = engine.infer(
    images=[Path("image.jpg")],
    convert_to_metric=True,
    fov_degrees=60.0,  # Horizontal field of view
    export_dir=Path("output")
)

# Nested model (already metric, no conversion needed)
config = DA3Config(model_variant=ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1)
engine = DA3InferenceEngine(config)

result = engine.infer(
    images=[Path("image.jpg")],
    convert_to_metric=True,  # Safe to enable (no-op for metric models)
    export_dir=Path("output")
)

print(f"Already metric: {result.metric_depth_info.already_metric}")  # True
```

**See [Metric Depth Guide](docs/METRIC_DEPTH_GUIDE.md) for detailed examples.**

#### Reference View Selection

When processing multiple views (≥3), DA3 automatically selects the optimal reference view for depth estimation. Four strategies are available:

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `saddle_balanced` | **Default** - General purpose | Balances multiple feature metrics for robustness |
| `saddle_sim_range` | Wide baseline captures | Maximizes similarity range (information-rich anchor) |
| `middle` | Video/temporal sequences | Uses middle view for temporal coherence |
| `first` | Pre-sorted inputs | Uses first view (debugging/manual curation) |

```python
# Use different strategies
wrapper.inference(
    image=images,
    ref_view_strategy="saddle_balanced",  # Robust general-purpose (default)
    export_dir="output"
)

wrapper.inference(
    image=video_frames,
    ref_view_strategy="middle",  # Best for video sequences
    export_dir="output/video"
)

wrapper.inference(
    image=aerial_images,
    ref_view_strategy="saddle_sim_range",  # Best for wide baselines
    export_dir="output/aerial"
)
```

**Manual Selection:**
```python
from lux_depth_v3.reference_view import select_reference_view

# Get selection with metrics
result = select_reference_view(
    num_views=len(images),
    strategy="saddle_balanced",
    class_tokens=tokens  # Extracted from model
)

print(f"Selected view: {result.selected_index}")
print(f"Scores: {result.scores}")
print(f"Metrics: {result.metrics.keys()}")
```

See [Reference View Selection Guide](docs/REFERENCE_VIEW_SELECTION.md) for detailed documentation.

#### Gaussian Splatting

```python
# Requires da3-giant or da3nested-giant-large
wrapper = DepthAnything3Wrapper(model_name="da3-giant")

# Reconstruct scene
prediction = wrapper.inference(
    image=scene_images,
    infer_gs=True,
    export_format="gs_ply-gs_video",
    render_exts=camera_trajectory,  # (M, 4, 4) rendering poses
    render_hw=(1080, 1920),
    export_dir="output/gs"
)
```

### Command Line Interface

#### Python API Mode (Recommended)

```bash
# Show license information
lux-depth-v3 api-process image.jpg -o output --show-license

# Basic depth estimation (v1.1 model)
lux-depth-v3 api-process image.jpg -o output -m nested-giant-large-v1.1

# Commercial use with Apache-licensed model
lux-depth-v3 api-process renders/ -o output \
  -m metric-large --commercial

# Multi-view with GLB export
lux-depth-v3 api-process images/ -o output -f "mini_npz-glb"

# Gaussian Splatting (requires v1.1 giant or nested)
lux-depth-v3 api-process scene/ -o output \
  -m giant-v1.1 --infer-gs -f "gs_ply-gs_video"

# Metric depth with commercial use
lux-depth-v3 api-process outdoor.jpg -o output \
  -m metric-large --commercial -f "full_npz"

# Feature extraction
lux-depth-v3 api-process image.jpg -o output \
  --export-feat "0,3,6,9" -f "feat_vis"

# Strict license validation (raises error on violation)
lux-depth-v3 api-process image.jpg -o output \
  -m nested-giant-large-v1.1 --commercial --strict-license

# Metric depth conversion with focal length
lux-depth-v3 api-process interior.jpg -o output \
  -m metric-large --metric --focal-length 500.0 --depth-stats

# Metric depth with FOV estimation
lux-depth-v3 api-process exterior.jpg -o output \
  -m metric-large --metric --fov 60.0 --depth-stats
```

#### Legacy Mode

```bash
# Monocular metric depth
lux-depth-v3 process \
  --input-dir renders/ \
  --output-dir output/ \
  --model metric-large

# Multi-view with pose estimation
lux-depth-v3 process \
  --input-dir views/ \
  --output-dir 3d/ \
  --multi-view \
  --model nested-giant-large
```

### CLI Integration Mode

```bash
  --preset interior_luxury

# Export multiple formats
lux-depth-v3 process \
  --input-dir renders/ \
  --export-format png \
  --export-format npz \
  --export-format ply

# Enable validation
lux-depth-v3 process \
  --input-dir test/ \
  --ground-truth-dir ground_truth/ \
  --validate

# Benchmark model
lux-depth-v3 benchmark \
  --model metric-large \
  --device cuda \
  --iterations 100
```

#### CLI Mode (Official DA3 Integration) ✨ NEW

**Performance**: 10-20x faster for batch processing with backend service!

```bash
# Install DA3 CLI (one-time setup)
pip install git+https://github.com/DepthAnything/Depth-Anything-V3.git

# Use CLI mode
lux-depth-v3 process \
  --input-dir renders/ \
  --output-dir output/ \
  --use-cli

# Use backend service for batch processing (fastest)
# Terminal 1: Start backend
lux-depth-v3 backend-start \
  --model-dir ~/.cache/lux_depth_v3/models/depth-anything-3-metric-large \
  --device cuda

# Terminal 2: Process multiple batches (reuses model in GPU memory)
lux-depth-v3 process --use-cli --use-backend -i batch1/ -o out1/
lux-depth-v3 process --use-cli --use-backend -i batch2/ -o out2/
lux-depth-v3 process --use-cli --use-backend -i batch3/ -o out3/

# Backend management
lux-depth-v3 backend-status    # Check if running
lux-depth-v3 backend-stop      # Stop backend
```

**When to Use:**
- **Native Mode**: Single images, small batches (<10), development
- **CLI Mode**: Official DA3 features, reproducibility
- **CLI + Backend**: Large batches (100+), production, 10-20x speedup

See [CLI Integration Guide](docs/CLI_INTEGRATION.md) for detailed usage.

### Service Mode

```bash
# Start FastAPI service
python -m lux_depth_v3.service

# Or with Uvicorn
uvicorn lux_depth_v3.service:app --host 0.0.0.0 --port 8088
```

**API Usage:**

```bash
# Estimate depth from image
curl -X POST http://localhost:8088/depth/estimate \
  -F "file=@render.jpg" \
  -F "model_variant=metric-large" \
  -F "metric_scaling=true"

# Download result
curl http://localhost:8088/depth/download/depth_12345_depth.png \
  --output depth.png

# Health check
curl http://localhost:8088/health
```

## Architecture

```
Input Manager → Preprocessing → DA3 Inference → Postprocessing/Fusion
              → Validation/Quality Gates → Output/Export
```

### Components

1. **Input Manager** (`input_manager.py`)
   - Standardized input handling
   - Image loading and validation
   - Camera pose management
   - Security checks

2. **Preprocessing** (`preprocessing.py`)
   - Image resizing and normalization
   - Padding to multiples
   - ImageNet normalization
   - Maintain aspect ratio

3. **Inference** (`inference.py`)
   - DA3 model wrapper
   - Monocular and multi-view modes
   - GPU/CPU/MPS support
   - Batch processing

4. **Postprocessing** (`postprocessing.py`)
   - Metric scaling
   - Filtering (median, bilateral)
   - Edge preservation
   - Multi-view fusion

5. **Validation** (`validation.py`)
   - Quality metrics computation
   - Ground truth comparison
   - Quality gates
   - Validation reports

6. **Export** (`export.py`)
   - Multi-format output
   - Point cloud generation
   - Metadata preservation

## Configuration

### Presets

```python
from lux_depth_v3 import DA3Config, Preset

# Photo-realistic monocular depth
config = DA3Config.from_preset(Preset.PHOTO_REALISTIC)

# Interior luxury (metric depth)
config = DA3Config.from_preset(Preset.INTERIOR_LUXURY)

# Exterior showcase
config = DA3Config.from_preset(Preset.EXTERIOR_SHOWCASE)

# Architectural 3D reconstruction
config = DA3Config.from_preset(Preset.ARCHITECTURAL_3D)

# Metric scanning
config = DA3Config.from_preset(Preset.METRIC_SCAN)
```

### Custom Configuration

```python
from lux_depth_v3 import DA3Config, ModelVariant, InferenceMode
from lux_depth_v3.config import PostprocessingConfig, ExportConfig, ExportFormat

config = DA3Config(
    model_variant=ModelVariant.METRIC_LARGE,
    inference_mode=InferenceMode.METRIC,

    # Postprocessing
    postprocessing=PostprocessingConfig(
        apply_metric_scaling=True,
        apply_bilateral_filter=True,
        preserve_edges=True,
    ),

    # Export
    export=ExportConfig(
        formats=[ExportFormat.PNG, ExportFormat.NPZ],
        output_dir=Path("output"),
    ),
)
```

## Advanced Usage

### Multi-View Reconstruction

```python
from lux_depth_v3 import InputManager, CameraPose
import numpy as np

# Create input manager for multi-view
manager = InputManager(inference_mode=InferenceMode.MULTI_VIEW)

# Add images with camera poses
for i, img_path in enumerate(image_paths):
    # Define camera pose (rotation + translation)
    pose = CameraPose(
        rotation=rotation_matrices[i],  # 3x3
        translation=translations[i],     # 3x1
        focal_length=(fx, fy),
        principal_point=(cx, cy),
    )

    manager.add_image(path=img_path, pose=pose)

# Run multi-view inference
results = engine.inference(manager.get_images())

# Access fused point cloud
postprocessor = Postprocessor(config.postprocessing)
fused = postprocessor.fuse_multiview(results)
```

### Batch Processing with Validation

```python
from lux_depth_v3 import DepthValidator, ValidationReport

# Setup validator
validator = DepthValidator(ground_truth_dir=Path("ground_truth"))
report = ValidationReport()

# Process batch
for img_input in tqdm(manager.get_images()):
    result = engine.inference(img_input)
    result = postprocessor.process(result)

    # Validate
    metrics = validator.validate(result)
    report.add_result(metrics)

    # Check quality gate
    if not metrics.passes_quality_gate():
        print(f"Quality gate failed: {img_input.path}")

# Save report
report.save(Path("output/validation_report.json"))

# Print summary
summary = report.compute_summary()
print(f"Mean RMSE: {summary['mean_rmse']:.4f}")
print(f"Mean δ1: {summary['mean_delta_1']:.3f}")
```

### Point Cloud Export

```python
from lux_depth_v3 import Exporter

# Configure exporter
config.export.formats = [ExportFormat.PLY]
config.export.point_cloud_downsample = 2
config.export.point_cloud_max_points = 500_000

exporter = Exporter(config.export)

# Export
exported = exporter.export(result, "my_scene")
print(f"Point cloud: {exported['ply']}")
```

## Performance

### Benchmarks

| Model Variant | Device | Throughput | Latency |
|--------------|--------|------------|---------|
| METRIC-LARGE | M4 Max (MPS) | ~40 img/s | 25ms |
| METRIC-LARGE | RTX 4090 | ~80 img/s | 12ms |
| MONO-LARGE | M4 Max (MPS) | ~45 img/s | 22ms |
| NESTED-GIANT | RTX 4090 | ~30 img/s | 33ms |

*Benchmarks on 1024x1024 images with fp16 precision*

### Optimization Tips

1. **Use GPU/MPS**: 10-20x faster than CPU
2. **Enable torch.compile**: 10-15% speedup (PyTorch 2.0+)
3. **Batch processing**: Amortize model loading overhead
4. **Model caching**: Avoid re-downloading weights
5. **FP16 precision**: 2x speedup with minimal quality loss

## Testing

```bash
# Run all tests
pytest lux_depth_v3/tests/ -v

# Run with coverage
pytest lux_depth_v3/tests/ --cov=lux_depth_v3 --cov-report=html

# Run specific test
pytest lux_depth_v3/tests/test_lux_depth_v3.py::test_inference_monocular -v

# Run integration tests only
pytest lux_depth_v3/tests/ -k "integration" -v
```

## Security

### Input Validation

- File size limits (default: 50MB)
- Image dimension limits (default: 4096px)
- File type validation (image/* only)
- Path traversal protection

### Service Mode Security

- Rate limiting (60 requests/minute)
- CORS configuration
- Input sanitization
- Error handling

See `SECURITY.md` for detailed security guidelines.

## Migration from Depth Anything V2

### API Compatibility

```python
# V2 (old)
from lux_depth_v2.pipeline import DepthPipeline
pipeline = DepthPipeline()
depth = pipeline.process("image.jpg")

# V3 (new)
from lux_depth_v3 import DA3Config, InputManager, DA3InferenceEngine
config = DA3Config.from_preset(Preset.PHOTO_REALISTIC)
manager = InputManager()
manager.add_image(path="image.jpg")
engine = DA3InferenceEngine(config)
engine.load_model()
result = engine.inference(manager.get_images()[0])
depth = result.depth_map
```

### Feature Comparison

| Feature | V2 | V3 |
|---------|----|----|
| Monocular depth | ✅ | ✅ |
| Multi-view | ❌ | ✅ |
| Metric depth | ❌ | ✅ |
| Camera poses | ❌ | ✅ |
| Point clouds | ❌ | ✅ |
| Quality metrics | ⚠️ Basic | ✅ Comprehensive |
| Service mode | ✅ | ✅ |

## Troubleshooting

### Common Issues

**Model not loading:**
```
ImportError: No module named 'depth_anything_v3'
```
→ Install official DA3 package when available

**CUDA out of memory:**
```
RuntimeError: CUDA out of memory
```
→ Reduce batch size or use fp16 precision

**Slow inference:**
→ Check device selection (should use GPU/MPS, not CPU)
→ Enable model caching to avoid re-downloads

**Service rate limiting:**
```
429 Rate limit exceeded
```
→ Wait 60 seconds or configure higher rate limit

## Future Enhancements

We've analyzed the official DA3 repository and identified additional features for integration. See our planning documents:

- **[Feature Gap Analysis](docs/DA3_FEATURE_GAP_ANALYSIS.md)** - Comprehensive analysis of missing features
- **[Integration Tracker](docs/DA3_FEATURE_INTEGRATION_TRACKER.md)** - Implementation roadmap and progress
- **[Decision Matrix](docs/DA3_FEATURE_DECISION_MATRIX.md)** - Quick reference for stakeholders

### Planned Features (Sprint 1-2)

**Priority 1 (Critical):**
- ✅ **Model Versioning Support** - Access to `-1.1` models with bug fixes
- ✅ **Metric Depth Conversion Utilities** - Convert DA3METRIC output to meters

**Priority 2 (High-Value):**
- ✅ **License Validation** - Automatic warnings for CC-BY-NC commercial restrictions
- ✅ **XFormers Fallback** - Graceful degradation for older GPUs

**Priority 3 (Conditional):**
- ⏸️ **DA3-Streaming** - Ultra-long video support (if user demand emerges)
- 📝 **Gradio/Gallery UI** - Web-based visualization
- 📊 **Model Performance Docs** - Comprehensive benchmarks and comparison

See the planning documents for detailed technical analysis, implementation plans, and timeline estimates.

## Contributing

See main repository `CONTRIBUTING.md` for guidelines.

## License

See main repository LICENSE file.

## Citation

```bibtex
@inproceedings{depthanything3,
  title={Depth Anything 3: Unified Any-View Depth Estimation},
  author={...},
  booktitle={...},
  year={2025}
}
```

## Support

- **Issues**: GitHub Issues in main Transformation Portal repository
- **Documentation**: See `INTEGRATION_GUIDE.md` for detailed integration steps
- **Examples**: See `examples/` directory for usage examples
