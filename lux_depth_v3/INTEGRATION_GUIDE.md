# Lux Depth V3 Integration Guide

Comprehensive guide for integrating Depth Anything 3 into the Transformation Portal.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic Integration](#basic-integration)
5. [Advanced Integration](#advanced-integration)
6. [Service Deployment](#service-deployment)
7. [Performance Optimization](#performance-optimization)
8. [Testing & Validation](#testing--validation)
9. [Migration from V2](#migration-from-v2)
10. [Troubleshooting](#troubleshooting)

## Overview

Lux Depth V3 provides a production-ready integration of Depth Anything 3 models with:

- **Unified API** for monocular and multi-view depth estimation
- **Metric depth output** with absolute scale
- **Camera pose estimation** for 3D reconstruction
- **Quality validation** framework integration
- **Multiple export formats** (PNG, NPZ, PLY, TIFF)
- **Security hardening** for production deployment

### Architecture

```
┌─────────────────┐
│  Input Manager  │ ← Image loading, validation, pose management
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │ ← Resize, normalize, pad
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DA3 Inference   │ ← Model loading, GPU acceleration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Postprocessing  │ ← Filtering, metric scaling, fusion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation     │ ← Quality metrics, ground truth comparison
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Export       │ ← PNG, NPZ, PLY, TIFF output
└─────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd lux_depth_v3
pip install -r requirements.txt
```

### 2. Install DA3 Package

#### Option A: Native Mode (Placeholder)

For testing and development, the module includes a placeholder wrapper (`da3_wrapper.py`).

#### Option B: CLI Mode (Recommended for Production)

For production use with the official DA3 CLI:

```bash
# Install official DA3 repository
git clone https://github.com/DepthAnything/Depth-Anything-V3.git
cd Depth-Anything-V3
pip install -e .

# Verify installation
da3 --help
```

**Benefits of CLI Mode:**
- Official DA3 implementation
- Backend service for 10-20x speedup in batch processing
- Production-tested and maintained by DA3 team
- Access to latest features and bug fixes

See [CLI Integration Guide](docs/CLI_INTEGRATION.md) for detailed setup.

### 3. Verify Installation

```bash
python -c "import lux_depth_v3; print(lux_depth_v3.__version__)"
```

## Configuration

### Preset-Based Configuration

The simplest way to configure DA3 is using presets:

```python
from lux_depth_v3 import DA3Config, Preset

# Interior scenes with metric depth
config = DA3Config.from_preset(Preset.INTERIOR_LUXURY)

# Photo-realistic monocular
config = DA3Config.from_preset(Preset.PHOTO_REALISTIC)

# Multi-view 3D reconstruction
config = DA3Config.from_preset(Preset.ARCHITECTURAL_3D)
```

### Custom Configuration

For fine-grained control:

```python
from lux_depth_v3.config import (
    DA3Config,
    ModelVariant,
    InferenceMode,
    DeviceConfig,
    PreprocessingConfig,
    PostprocessingConfig,
    ExportConfig,
    ExportFormat,
)

config = DA3Config(
    # Model
    model_variant=ModelVariant.METRIC_LARGE,
    inference_mode=InferenceMode.METRIC,

    # Device
    device=DeviceConfig(
        device="cuda",  # or "mps", "cpu", "auto"
        precision="fp16",
        use_compile=True,  # PyTorch 2.0+ optimization
    ),

    # Preprocessing
    preprocessing=PreprocessingConfig(
        target_size=(1024, 1024),
        resize_mode="bicubic",
        normalize=True,
        pad_to_multiple=32,
    ),

    # Postprocessing
    postprocessing=PostprocessingConfig(
        apply_metric_scaling=True,
        scale_factor=1.0,
        apply_bilateral_filter=True,
        preserve_edges=True,
    ),

    # Export
    export=ExportConfig(
        formats=[ExportFormat.PNG, ExportFormat.NPZ],
        output_dir=Path("output"),
        depth_scale=1000.0,  # mm per unit
    ),

    # Cache
    cache_dir=Path.home() / ".cache" / "lux_depth_v3",
    enable_model_cache=True,

    # Batch
    batch_size=4,
    num_workers=4,
)
```

## Basic Integration

### Single Image Processing

```python
from pathlib import Path
from lux_depth_v3 import (
    DA3Config,
    Preset,
    InputManager,
    DA3InferenceEngine,
    Postprocessor,
    Exporter,
)

# 1. Configuration
config = DA3Config.from_preset(Preset.INTERIOR_LUXURY)
config.export.output_dir = Path("output")

# 2. Input
manager = InputManager()
manager.add_image(path="render.jpg")

# 3. Inference
engine = DA3InferenceEngine(config)
engine.load_model()
result = engine.inference(manager.get_images()[0])

# 4. Postprocessing
postprocessor = Postprocessor(config.postprocessing)
result = postprocessor.process(result)

# 5. Export
exporter = Exporter(config.export)
exported = exporter.export(result, "render")

print(f"Depth range: {result.get_depth_range()}")
print(f"Exported: {exported}")
```

### Batch Processing

```python
from tqdm import tqdm

# Load directory
manager = InputManager()
num_images = manager.add_directory(
    Path("renders"),
    pattern="*.jpg",
    recursive=True,
)

print(f"Processing {num_images} images...")

# Initialize pipeline
engine = DA3InferenceEngine(config)
engine.load_model()
postprocessor = Postprocessor(config.postprocessing)
exporter = Exporter(config.export)

# Process batch
for img_input in tqdm(manager.get_images()):
    result = engine.inference(img_input)
    result = postprocessor.process(result)

    filename_base = img_input.path.stem
    exporter.export(result, filename_base)
```

## Advanced Integration

### Multi-View Reconstruction

```python
from lux_depth_v3 import CameraPose, InferenceMode
import numpy as np

# Setup multi-view configuration
config = DA3Config.from_preset(Preset.ARCHITECTURAL_3D)
config.inference_mode = InferenceMode.MULTI_VIEW

# Create input manager
manager = InputManager(inference_mode=InferenceMode.MULTI_VIEW)

# Add images with camera poses
image_paths = [
    "view_001.jpg",
    "view_002.jpg",
    "view_003.jpg",
]

# Example camera poses (replace with actual calibration)
for i, img_path in enumerate(image_paths):
    # Camera extrinsics (rotation + translation)
    angle = i * np.pi / 4  # 45 degree increments
    rotation = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    translation = np.array([i * 0.5, 0, 0])

    # Camera intrinsics
    focal_length = (1000.0, 1000.0)  # fx, fy
    principal_point = (512.0, 512.0)  # cx, cy

    pose = CameraPose(
        rotation=rotation,
        translation=translation,
        focal_length=focal_length,
        principal_point=principal_point,
    )

    manager.add_image(path=img_path, pose=pose)

# Run multi-view inference
engine = DA3InferenceEngine(config)
engine.load_model()
results = engine.inference(manager.get_images())

# Fuse multi-view results
postprocessor = Postprocessor(config.postprocessing)
fused = postprocessor.fuse_multiview(results)

# Export point cloud
config.export.formats = [ExportFormat.PLY]
exporter = Exporter(config.export)
exporter.export(fused, "fused_pointcloud")
```

### Validation Integration

```python
from lux_depth_v3 import DepthValidator, ValidationReport

# Setup validator with ground truth
validator = DepthValidator(
    ground_truth_dir=Path("ground_truth")
)

# Create validation report
report = ValidationReport()

# Process with validation
for img_input in manager.get_images():
    result = engine.inference(img_input)
    result = postprocessor.process(result)

    # Validate against ground truth
    metrics = validator.validate(result)
    report.add_result(metrics)

    # Check quality gate
    if metrics.passes_quality_gate(min_delta_1=0.85, max_rmse=0.3):
        print(f"✓ {img_input.path}: Passed quality gate")
    else:
        print(f"✗ {img_input.path}: Failed quality gate")
        print(f"  RMSE: {metrics.rmse:.4f}, δ1: {metrics.delta_1:.3f}")

# Save validation report
report.save(Path("output/validation_report.json"))

# Print summary
summary = report.compute_summary()
print("\nValidation Summary:")
print(f"  Images: {summary['num_images']}")
print(f"  Mean RMSE: {summary['mean_rmse']:.4f}")
print(f"  Mean δ1: {summary['mean_delta_1']:.3f}")
print(f"  Mean Edge Completeness: {summary['mean_edge_completeness']:.3f}")
```

### Integration with Existing Pipelines

```python
# Integration with lux_render_pipeline.py
from lux_depth_v3 import DA3InferenceEngine, DA3Config

def enhance_with_depth(image_path: Path, output_path: Path):
    """Enhance render using DA3 depth information."""

    # Generate depth map
    config = DA3Config.from_preset(Preset.INTERIOR_LUXURY)
    engine = DA3InferenceEngine(config)
    engine.load_model()

    manager = InputManager()
    manager.add_image(path=image_path)

    result = engine.inference(manager.get_images()[0])
    depth = result.depth_map

    # Use depth for zone-based processing
    # (integrate with existing lux_depth_v2 patterns)
    from lux_depth_v2.pipeline import apply_depth_aware_enhancement

    enhanced = apply_depth_aware_enhancement(
        image=result.original_image,
        depth=depth,
        config=config,
    )

    # Save result
    Image.fromarray(enhanced).save(output_path)
```

## Service Deployment

### Local Development

```bash
# Start service
python -m lux_depth_v3.service

# Or with Uvicorn
uvicorn lux_depth_v3.service:app --reload --port 8088
```

### Production Deployment

```bash
# With Gunicorn + Uvicorn workers
gunicorn lux_depth_v3.service:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8088 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY lux_depth_v3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy module
COPY lux_depth_v3 /app/lux_depth_v3

# Expose port
EXPOSE 8088

# Run service
CMD ["uvicorn", "lux_depth_v3.service:app", "--host", "0.0.0.0", "--port", "8088"]
```

Build and run:

```bash
docker build -t lux-depth-v3:latest .
docker run -p 8088:8088 --gpus all lux-depth-v3:latest
```

### API Usage

```python
import requests

# Estimate depth
with open("render.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8088/depth/estimate",
        files={"file": f},
        data={
            "model_variant": "metric-large",
            "metric_scaling": "true",
        }
    )

result = response.json()
print(f"Processing time: {result['processing_time_ms']:.1f}ms")
print(f"Depth range: {result['depth_range']}")

# Download depth map
depth_path = result['export_paths']['png']
filename = depth_path.split('/')[-1]

depth_response = requests.get(
    f"http://localhost:8088/depth/download/{filename}"
)

with open("depth_output.png", "wb") as f:
    f.write(depth_response.content)
```

## Performance Optimization

### 1. Model Selection

Choose the right model for your use case:

```python
# Fastest: SMALL variant (lower quality)
config.model_variant = ModelVariant.SMALL

# Balanced: LARGE variant (recommended)
config.model_variant = ModelVariant.LARGE

# Highest quality: NESTED_GIANT_LARGE (slowest)
config.model_variant = ModelVariant.NESTED_GIANT_LARGE
```

### 2. Precision Optimization

```python
# FP16: 2x speedup, minimal quality loss
config.device.precision = "fp16"

# BF16: Better numerical stability (if supported)
config.device.precision = "bf16"

# FP32: Highest precision (slowest)
config.device.precision = "fp32"
```

### 3. Batch Processing

```python
# Process multiple images simultaneously
config.batch_size = 8  # Adjust based on VRAM

# With DataLoader for efficient batching
from torch.utils.data import DataLoader

dataset = [manager.get_images()]  # Wrap in dataset
loader = DataLoader(dataset, batch_size=8, num_workers=4)
```

### 4. Torch Compile (PyTorch 2.0+)

```python
# Enable compilation for 10-15% speedup
config.device.use_compile = True
```

### 5. Caching

```python
# Enable model caching to avoid re-downloads
config.enable_model_cache = True
config.cache_dir = Path.home() / ".cache" / "lux_depth_v3"
```

## Testing & Validation

### Running Tests

```bash
# All tests
pytest lux_depth_v3/tests/ -v

# Specific component
pytest lux_depth_v3/tests/test_lux_depth_v3.py::test_inference_monocular -v

# With coverage
pytest lux_depth_v3/tests/ --cov=lux_depth_v3 --cov-report=html
```

### Benchmark Performance

```bash
lux-depth-v3 benchmark \
  --model metric-large \
  --device cuda \
  --iterations 100
```

### Integration with CI/CD

Add to `.github/workflows/ci.yml`:

```yaml
- name: Test Lux Depth V3
  run: |
    pytest lux_depth_v3/tests/ -v --cov=lux_depth_v3

- name: Benchmark
  run: |
    lux-depth-v3 benchmark --model metric-large --iterations 10
```

## Migration from V2

### API Changes

| V2 API | V3 API | Notes |
|--------|--------|-------|
| `DepthPipeline` | `DA3InferenceEngine` | More explicit configuration |
| `pipeline.process()` | `engine.inference()` | Returns `DepthResult` object |
| Config via YAML | `DA3Config` dataclass | Type-safe configuration |
| N/A | Multi-view support | New feature in V3 |

### Migration Example

**Before (V2):**

```python
from lux_depth_v2.pipeline import DepthPipeline

pipeline = DepthPipeline()
depth = pipeline.process("image.jpg")
```

**After (V3):**

```python
from lux_depth_v3 import DA3Config, InputManager, DA3InferenceEngine

config = DA3Config.from_preset(Preset.PHOTO_REALISTIC)
manager = InputManager()
manager.add_image(path="image.jpg")

engine = DA3InferenceEngine(config)
engine.load_model()
result = engine.inference(manager.get_images()[0])
depth = result.depth_map
```

### Compatibility Layer

For gradual migration, create a compatibility wrapper:

```python
# compat.py
from lux_depth_v3 import DA3Config, InputManager, DA3InferenceEngine, Preset

class DepthPipelineV2Compat:
    """V2-compatible wrapper for V3 pipeline."""

    def __init__(self, preset=Preset.PHOTO_REALISTIC):
        self.config = DA3Config.from_preset(preset)
        self.engine = DA3InferenceEngine(self.config)
        self.engine.load_model()

    def process(self, image_path):
        """V2-compatible process method."""
        manager = InputManager()
        manager.add_image(path=image_path)
        result = self.engine.inference(manager.get_images()[0])
        return result.depth_map
```

## Troubleshooting

### Issue: Model fails to load

**Error:**
```
ImportError: No module named 'depth_anything_v3'
```

**Solution:**
Install the official DA3 package when available. For now, the placeholder wrapper in `da3_wrapper.py` is used.

### Issue: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `config.batch_size = 1`
2. Use FP16: `config.device.precision = "fp16"`
3. Reduce image size: `config.preprocessing.target_size = (512, 512)`
4. Clear cache: `torch.cuda.empty_cache()`

### Issue: Slow inference on Apple Silicon

**Problem:** Using CPU instead of MPS

**Solution:**
```python
config.device.device = "mps"  # Force MPS backend
```

Verify with:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Issue: Service rate limiting

**Error:**
```
429 Rate limit exceeded
```

**Solution:**
Wait 60 seconds or configure higher limit in `service.py`:
```python
RATE_LIMIT_REQUESTS_PER_MINUTE = 120  # Increase from 60
```

### Issue: Validation metrics are zero

**Problem:** No ground truth found

**Solution:**
Ensure ground truth files match expected naming:
- Input: `image.jpg`
- Ground truth: `image_depth.npy` or `image.npy`

Place in `--ground-truth-dir` directory.

## Next Steps

1. **Integrate with existing pipelines** - Add DA3 to `lux_render_pipeline.py`
2. **Benchmark against V2** - Compare performance and quality
3. **Deploy service** - Set up production API endpoint
4. **Create custom presets** - Tune for specific use cases
5. **Contribute improvements** - Submit PRs for enhancements

## Support

- **Documentation**: See `README.md` for quick reference
- **Examples**: Check `examples/` directory for code samples
- **Issues**: Report bugs in main Transformation Portal repository
- **Community**: Join discussions in GitHub Discussions

## Appendix

### Complete Example Script

See `examples/full_pipeline_example.py` for a complete end-to-end example.

### Configuration Reference

See `config.py` for all available configuration options and their defaults.

### API Reference

Full API documentation available in module docstrings.
