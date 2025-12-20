# High-Fidelity Depth Inference - Quick Start Guide

## TL;DR

```python
from lux_depth_v2.depth_inference import create_tiled_estimator

# Create estimator (all enhancements enabled by default)
estimator = create_tiled_estimator(
    tile_size=1024,
    overlap=128,
    device="auto"
)

# Process image
depth = estimator.estimate_depth(rgb_image)
```

## What This Fixes

**Problem**: HuggingFace pipeline resizes all images to 518×518 internally, making high-res depth impossible.

**Solution**: Bypass the pipeline, load model directly, preserve native resolution through tiling.

**Result**: True 1024×1024 tile processing (vs 518×518), enabling luxury-grade depth maps.

## Installation

```bash
pip install transformers torch opencv-python pillow numpy
```

## Basic Usage

### 1. Simple Inference
```python
from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig
import numpy as np
from PIL import Image

# Load image
rgb = np.array(Image.open("render.jpg"))

# Configure (with all enhancements)
config = TiledInferenceConfig(
    tile_size=1024,
    overlap=128,
    bypass_image_processor=True,  # REQUIRED for high-res
    use_global_anchor=True,       # Prevents tile artifacts
    use_edge_snapping=True        # Sharp edges for DOF/masking
)

# Process
estimator = TiledDepthEstimator(config)
depth = estimator.estimate_depth(rgb)

# Save
Image.fromarray((depth * 255).astype(np.uint8)).save("depth.png")
```

### 2. Disable Enhancements (Fast Mode)
```python
config = TiledInferenceConfig(
    tile_size=1024,
    overlap=128,
    bypass_image_processor=True,
    use_global_anchor=False,  # Faster, may have tile seams
    use_edge_snapping=False   # Faster, softer edges
)
```

### 3. Custom Enhancement Settings
```python
from lux_depth_v2.global_anchor import GlobalAnchorConfig, PRESETS as GLOBAL_PRESETS
from lux_depth_v2.edge_snapping import EdgeSnappingConfig, PRESETS as SNAP_PRESETS

config = TiledInferenceConfig(
    tile_size=1024,
    overlap=128,
    bypass_image_processor=True,
    
    # Global anchor: aggressive (more global coherence)
    use_global_anchor=True,
    global_anchor_config=GLOBAL_PRESETS["aggressive"],
    
    # Edge snapping: subtle (less aggressive)
    use_edge_snapping=True,
    edge_snap_config=SNAP_PRESETS["subtle"]
)
```

## Configuration Options

### TiledInferenceConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_size` | 1024 | Tile size in pixels (512-1536 recommended) |
| `overlap` | 128 | Overlap between tiles (128-256 recommended) |
| `bypass_image_processor` | True | **CRITICAL**: Bypass 518px resize |
| `fusion_mode` | "median" | Tile blending: median \| weighted |
| `use_global_anchor` | True | Enable global context preservation |
| `use_edge_snapping` | True | Enable edge sharpening |

### GlobalAnchorConfig Presets

```python
from lux_depth_v2.global_anchor import PRESETS

# Conservative: minimal global influence
config.global_anchor_config = PRESETS["conservative"]

# Balanced: recommended default
config.global_anchor_config = PRESETS["balanced"]

# Aggressive: maximum global coherence
config.global_anchor_config = PRESETS["aggressive"]
```

### EdgeSnappingConfig Presets

```python
from lux_depth_v2.edge_snapping import PRESETS

# Subtle: gentle snapping
config.edge_snap_config = PRESETS["subtle"]

# Balanced: recommended default
config.edge_snap_config = PRESETS["balanced"]

# Aggressive: maximum edge sharpness
config.edge_snap_config = PRESETS["aggressive"]

# Multiscale: robust multi-scale processing
config.edge_snap_config = PRESETS["multiscale"]
```

## Validation Tools

### 1. Validate Bypass Mode
```bash
python lux_depth_v2/tools/validate_bypass_mode.py
```

**Expected output**: ✓ PASS for 512×512 and 1024×1024

### 2. Run A/B Comparison
```bash
# With synthetic test pattern
python lux_depth_v2/tools/ab_comparison.py

# With your own image
python lux_depth_v2/tools/ab_comparison.py \
    --input path/to/render.jpg \
    --output-dir results/
```

**Output**:
- `comparison.png` - Side-by-side visualization
- `baseline_depth.png` - Baseline (518px pipeline)
- `enhanced_depth.png` - Enhanced (tiled)
- `report.txt` - Metrics and improvements

## Performance Expectations

| Image Size | Tile Size | Global Anchor | Edge Snap | Estimated Time* |
|------------|-----------|---------------|-----------|-----------------|
| 1024×1024 | 1024 | Yes | Yes | ~2-5s (GPU) |
| 2048×2048 | 1024 | Yes | Yes | ~8-15s (GPU) |
| 4096×4096 | 1024 | Yes | Yes | ~30-60s (GPU) |

*Times are estimates for GPU (CUDA/MPS). CPU will be 5-10x slower.

## Common Issues

### Issue: "do_resize parameter not recognized"
**Cause**: Older transformers version  
**Fix**: `pip install --upgrade transformers`

### Issue: Out of memory
**Cause**: Tile size too large or too many tiles in memory  
**Fix**: Reduce `tile_size` to 512 or 768

### Issue: Tile seams visible
**Cause**: Insufficient overlap or global anchor disabled  
**Fix**: Increase `overlap` to 256 or enable `use_global_anchor=True`

### Issue: Soft edges
**Cause**: Edge snapping disabled  
**Fix**: Enable `use_edge_snapping=True`

## Integration with Existing Pipeline

```python
from lux_depth_v2.pipeline import LuxDepthPipeline
from lux_depth_v2.depth_inference import TiledDepthEstimator, TiledInferenceConfig

# Create depth estimator
depth_config = TiledInferenceConfig(
    tile_size=1024,
    bypass_image_processor=True,
    use_global_anchor=True,
    use_edge_snapping=True
)
depth_estimator = TiledDepthEstimator(depth_config)

# Use in pipeline
pipeline = LuxDepthPipeline(depth_estimator=depth_estimator)
result = pipeline.process(rgb_image)
```

## Next Steps

1. **Run validation**: `python lux_depth_v2/tools/validate_bypass_mode.py`
2. **Test with your renders**: Use A/B comparison script
3. **Measure improvements**: Edge alignment, sharpness, quality
4. **Optimize settings**: Tune tile_size, overlap, presets for your use case
5. **Integrate into production**: Add to your rendering pipeline

## Questions?

- **Why bypass_image_processor?** Because HF pipeline resizes to 518px internally
- **Why global anchor?** Prevents tile seams and global drift
- **Why edge snapping?** Required for luxury-grade DOF/masking
- **Performance cost?** ~5-10x slower than baseline, but with true high-res output

---

**Status**: ✅ Implementation complete | ⏳ Empirical validation pending
