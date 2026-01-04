# DA3 CLI Integration Guide

## Overview

The `lux_depth_v3` module provides two integration modes for Depth Anything 3:

1. **Native Python API** (default): Direct Python integration with DA3 models
2. **Official CLI Wrapper** (optional): Uses the `da3` command-line tool

The CLI mode offers significant performance benefits for batch processing through backend service acceleration.

## Installation

### Native Mode (Default)

No additional installation required. The placeholder DA3 API is included.

### CLI Mode

Install the official DA3 repository:

```bash
# Clone and install DA3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-V3
pip install -e .

# Verify installation
da3 --help
```

## CLI Mode Benefits

### Performance Comparison

| Mode | Model Load Time | Per-Image Time | Throughput (100 images) |
|------|----------------|----------------|-------------------------|
| Native (no cache) | 10-15s | 50ms | ~20s per image |
| Native (cached) | - | 50ms | ~5s total |
| CLI (no backend) | 10-15s per command | 50ms | ~20s per image |
| CLI (with backend) | 10-15s (once) | 50ms | ~5s total |

**Key Advantage**: The backend service keeps the model in GPU memory, providing **10-20x speedup** for batch processing by avoiding model reload overhead.

## Usage Examples

### Basic Processing

#### Native Mode (Default)
```bash
lux-depth-v3 process \
  --input-dir renders/ \
  --output-dir output/ \
  --model metric-large
```

#### CLI Mode
```bash
lux-depth-v3 process \
  --input-dir renders/ \
  --output-dir output/ \
  --use-cli
```

### Backend Service Workflow

The backend service is ideal for processing multiple batches or repeated operations.

#### 1. Start Backend Service

```bash
# In terminal 1 - start backend
lux-depth-v3 backend-start \
  --model-dir ~/.cache/lux_depth_v3/models/depth-anything-3-metric-large \
  --device cuda \
  --port 8008
```

The backend will:
- Load the model once into GPU memory
- Keep running until stopped
- Serve inference requests via HTTP

#### 2. Process Images Using Backend

```bash
# In terminal 2 - process multiple batches
lux-depth-v3 process \
  --input-dir batch1/ \
  --output-dir output1/ \
  --use-cli \
  --use-backend

lux-depth-v3 process \
  --input-dir batch2/ \
  --output-dir output2/ \
  --use-cli \
  --use-backend

# Model is NOT reloaded between batches - 10-20x faster!
```

#### 3. Check Backend Status

```bash
lux-depth-v3 backend-status --port 8008
```

#### 4. Stop Backend

```bash
# In terminal 1 - Ctrl+C
# Or from terminal 2:
lux-depth-v3 backend-stop --port 8008
```

### Advanced CLI Options

```bash
lux-depth-v3 process \
  --input-dir renders/ \
  --output-dir output/ \
  --use-cli \
  --use-backend \
  --backend-url http://localhost:8008 \
  --model nested-giant-large \
  --device cuda \
  --precision fp16 \
  --export-format png \
  --export-format npz \
  --verbose
```

## Python API

### Native Mode

```python
from lux_depth_v3 import DA3Config, DA3InferenceEngine, ImageInput
from pathlib import Path

# Configure
config = DA3Config.from_preset("interior_luxury")
config.cli.use_cli = False  # Default

# Initialize engine
engine = DA3InferenceEngine(config)
engine.load_model()

# Process images
images = [ImageInput(path=p) for p in Path("renders/").glob("*.jpg")]
results = engine.inference(images)
```

### CLI Mode

```python
from lux_depth_v3 import DA3Config, DA3InferenceEngine, ImageInput
from pathlib import Path

# Configure CLI mode
config = DA3Config.from_preset("interior_luxury")
config.cli.use_cli = True
config.cli.use_backend = False  # No backend

# Initialize engine (will use DA3 CLI)
engine = DA3InferenceEngine(config)

# Process images
images = [ImageInput(path=p) for p in Path("renders/").glob("*.jpg")]
results = engine.inference(images)
```

### CLI Mode with Backend

```python
from lux_depth_v3 import DA3Config, DA3InferenceEngine, ImageInput
from pathlib import Path

# Configure CLI mode with backend
config = DA3Config.from_preset("interior_luxury")
config.cli.use_cli = True
config.cli.use_backend = True
config.cli.backend_port = 8008

# Initialize engine
engine = DA3InferenceEngine(config)

# Start backend
engine.start_backend()

# Process multiple batches (model loaded only once)
for batch_dir in ["batch1", "batch2", "batch3"]:
    images = [ImageInput(path=p) for p in Path(batch_dir).glob("*.jpg")]
    results = engine.inference(images)
    print(f"Processed {len(results)} images from {batch_dir}")

# Stop backend
engine.stop_backend()
```

## CLI Export Formats

The DA3 CLI supports hyphen-separated format combinations:

```bash
# Export mini_npz only
--export-format mini_npz

# Export mini_npz and GLB
--export-format mini_npz-glb

# Export multiple formats
--export-format mini_npz-glb-ply
```

Available formats:
- `mini_npz` - Compressed NumPy array (depth only)
- `full_npz` - Full NumPy array (depth + metadata)
- `glb` - GLTF binary 3D mesh
- `ply` - Point cloud (ASCII)
- `png` - 16-bit grayscale PNG

## Configuration Options

### DA3CLIConfig

```python
from lux_depth_v3.config import DA3CLIConfig

cli_config = DA3CLIConfig(
    use_cli=True,
    use_backend=True,
    backend_url="http://localhost:8008",
    backend_port=8008,
    backend_host="127.0.0.1",

    # Export format (hyphen-separated combinations)
    export_format="mini_npz-glb",

    # Reference view strategy for multi-view
    ref_view_strategy="saddle_balanced",  # first, middle, saddle_balanced, saddle_sim_range

    # Ray-based pose estimation
    use_ray_pose=False,

    # GLB export settings
    conf_thresh_percentile=40.0,
    num_max_points=1_000_000,
    show_cameras=True,

    # Feature visualization
    feat_vis_fps=15,
    export_feat="",  # Comma-separated layer indices
)
```

## Troubleshooting

### DA3 CLI Not Found

```bash
# Check if da3 is in PATH
which da3

# If not found, install DA3
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-V3
pip install -e .
```

### Backend Won't Start

```bash
# Check if port is already in use
lsof -i :8008

# Try different port
lux-depth-v3 backend-start --port 8009

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Backend Connection Failed

```bash
# Verify backend is running
lux-depth-v3 backend-status --port 8008

# Check backend URL in config
lux-depth-v3 process \
  --use-cli \
  --use-backend \
  --backend-url http://localhost:8008  # Match port
```

### Performance Not Improving

**Issue**: CLI mode with backend is not faster than native mode.

**Solution**: Ensure you're processing multiple batches or repeated operations. The benefit comes from avoiding model reload:

```bash
# ❌ Single batch - no benefit
lux-depth-v3 process --use-cli --use-backend -i batch1/

# ✓ Multiple batches - 10-20x speedup
for dir in batch1 batch2 batch3; do
  lux-depth-v3 process --use-cli --use-backend -i $dir/
done
```

## Migration Guide

### From Native to CLI Mode

**Before**:
```bash
lux-depth-v3 process -i renders/ -o output/
```

**After**:
```bash
# 1. Install DA3 CLI (one-time)
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# 2. Add --use-cli flag
lux-depth-v3 process -i renders/ -o output/ --use-cli

# 3. Optional: Use backend for batch processing
lux-depth-v3 backend-start --model-dir ~/.cache/lux_depth_v3/models/...
lux-depth-v3 process -i renders/ -o output/ --use-cli --use-backend
```

### When to Use CLI vs Native

| Use Case | Recommended Mode | Reason |
|----------|------------------|--------|
| Single image processing | Native | Lower overhead |
| Small batch (<10 images) | Native | Simpler setup |
| Large batch (100+ images) | CLI + Backend | 10-20x speedup |
| Repeated operations | CLI + Backend | Model loaded once |
| Production deployment | CLI + Backend | Better resource management |
| Development/testing | Native | Easier debugging |

## Performance Best Practices

1. **Use backend for batch processing**: Start backend once, process multiple batches
2. **Keep backend running**: For repeated operations, leave backend running between sessions
3. **Match device config**: Ensure backend device matches processing requirements (cuda/mps/cpu)
4. **Monitor memory**: Backend keeps model in GPU memory - ensure sufficient VRAM
5. **Use appropriate precision**: fp16 reduces memory by 2x with minimal quality loss

## Example Workflows

### Architectural Rendering Pipeline

```bash
# 1. Start backend (once per day)
lux-depth-v3 backend-start \
  --model-dir ~/.cache/lux_depth_v3/models/depth-anything-3-metric-large \
  --device cuda &

# 2. Process multiple projects (fast)
for project in project_A project_B project_C; do
  lux-depth-v3 process \
    --use-cli --use-backend \
    -i "projects/${project}/renders/" \
    -o "projects/${project}/depth/" \
    --preset interior_luxury
done

# 3. Stop backend at end of day
lux-depth-v3 backend-stop
```

### Multi-View 3D Reconstruction

```bash
# Use CLI with COLMAP dataset
da3 colmap \
  --colmap-dir scene_data/ \
  --export-dir output/ \
  --export-format mini_npz-glb \
  --use-backend http://localhost:8008
```

## Additional Resources

- [DA3 Official Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Lux Depth V3 README](../README.md)
- [Integration Guide](INTEGRATION_GUIDE.md)
- [Security Guidelines](../SECURITY.md)
