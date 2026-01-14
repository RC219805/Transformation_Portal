# Materials V2/V3 Configuration Guide

## Overview

Materials V2 and V3 are advanced material processing engines that provide:

- **Materials V2**: Physics-based surface response with per-material enhancement
- **Materials V3**: Depth-aware material processing with expanded taxonomy

This guide explains how to enable and configure these engines for production use.

## Quick Start

### Using the PRODUCTION_ULTRA_MATERIALS Preset

The `production_ultra_materials` preset is the recommended way to enable both Materials V2 and V3:

```bash
lux-depth-v2 \
  --input "./input_images/sample.tif" \
  --output-dir ./output \
  --preset production_ultra_materials \
  --device auto
```

This preset automatically configures:
- ✅ Materials V2 with physics-based surface response
- ✅ Materials V3 with depth-aware processing
- ✅ High-quality SegFormer-B5 segmentation (1280px)
- ✅ MPS-safe tiled upscaling for large images
- ✅ Production-grade confidence thresholds

## Preset Comparison

| Preset | Materials V2 | Materials V3 | Segmentation | Use Case |
|--------|--------------|--------------|--------------|----------|
| `production_standard` | ❌ | ❌ | Heuristic | Fast processing, good quality |
| `production_ultra` | ❌ | ❌ | Heuristic | High quality, no materials |
| `production_ultra_materials` | ✅ | ✅ | SegFormer-B5 | **Flagship quality with materials** |
| `interior_luxury_max_quality` | ✅ | ❌ | SegFormer-B5 | Materials V2 only |
| `interior_luxury_apex_quality` | ✅ | ❌ | SegFormer-B5 | Maximum quality, V2 only |

## Configuration Details

### Materials V2 Configuration

```python
from lux_depth_v2.config import PipelineConfig, Preset

cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

# Materials V2 settings (automatically configured by preset)
cfg.materials_v2.enabled = True
cfg.materials_v2.backend = "segformer"
cfg.materials_v2.confidence.confidence_threshold = 0.4

# Per-material confidence thresholds
cfg.materials_v2.confidence.material_thresholds = {
    "wood": 0.55,      # Higher threshold for precision
    "metal": 0.55,
    "glass": 0.45,     # Lower threshold (glass is harder to detect)
    "fabric": 0.5,
    "stone": 0.55,
    "ceramic": 0.5,
    "water": 0.4,
    "polished": 0.45,
}

# Segmentation quality
cfg.materials_v2.segmentation.max_segmentation_side = 2048  # High resolution
cfg.materials_v2.segmentation.require_high_quality = True    # Enforce quality
```

### Materials V3 Configuration

```python
# Materials V3 settings (automatically configured by preset)
cfg.materials_v3.enabled = True
cfg.materials_v3.backend = "segformer"
cfg.materials_v3.taxonomy = MaterialTaxonomy.BASE  # 8-12 material classes

# Safety limits (OOM prevention)
cfg.materials_v3.max_megapixels = 30.0  # Max 30MP images
cfg.materials_v3.max_dimension = 6000   # Max 6000px on any side

# Edge refinement (production-safe default)
cfg.materials_v3.refine_edges = RefinementStrategy.OFF
```

## Offline Operation (No Downloads)

By default, the preset allows model downloads. For offline operation:

```python
from pathlib import Path

cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

# Disable downloads
cfg.segmentation.allow_downloads = False

# Provide local model paths
cfg.segmentation.segformer_model_path = Path("/path/to/segformer-b5")
cfg.segmentation.sam_model_path = Path("/path/to/sam_checkpoint.pth")
cfg.segmentation.efficientsam_model_path = Path("/path/to/efficientsam")
```

**Model Locations**:
- SegFormer-B5: `nvidia/segformer-b5-finetuned-ade-640-640` (HuggingFace)
- SAM: `facebook/segment-anything` checkpoints
- EfficientSAM: Available from Meta Research

## Validation and Error Handling

The configuration system validates Materials V2/V3 settings on initialization:

### Fail-Fast Validation

```python
# This will raise ValueError if segmentation backend is 'none'
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)
cfg.segmentation.backend = "none"  # Invalid!
cfg._validate_materials_config()  # Raises ValueError
```

### Warning on Missing Models

```python
# This will warn if downloads disabled but no local paths
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)
cfg.segmentation.allow_downloads = False
cfg.segmentation.segformer_model_path = None  # Missing!
cfg._validate_materials_config()  # UserWarning
```

## Report Metadata

When Materials V2/V3 are enabled, the report JSON includes detailed metadata:

```json
{
  "materials_v2_enabled": true,
  "materials_v2": {
    "enabled": true,
    "backend": "segformer",
    "confidence_threshold": 0.4,
    "material_thresholds": {
      "wood": 0.55,
      "glass": 0.45,
      ...
    },
    "disabled_reason": null
  },
  "materials_v2_metadata": {
    "confidence_avg": 0.67,
    "confidence_min": 0.42,
    "confidence_max": 0.89,
    ...
  },
  "materials_v3_enabled": true,
  "materials_v3": {
    "enabled": true,
    "taxonomy": "base",
    "backend": "segformer",
    "pixel_ops_enabled": false,
    "disabled_reason": null
  },
  "materials_v3_metadata": {
    ...
  }
}
```

### Disabled Reasons

If Materials V2/V3 fail to initialize, the `disabled_reason` field explains why:

- `"MODULE_NOT_AVAILABLE"`: Materials module not installed
- `"CONFIG_BLOCK_NULL"`: Config block is None
- `"DISABLED_BY_CONFIG"`: `enabled = False` in config
- `"DISABLED_BY_ENV_VAR"`: `DISABLE_MATERIALS_V3=1` environment variable
- `"INIT_FAILED: <error>"`: Initialization exception

## Performance Impact

### PRODUCTION_ULTRA_MATERIALS vs PRODUCTION_ULTRA

| Metric | production_ultra | production_ultra_materials | Delta |
|--------|------------------|---------------------------|-------|
| Processing Time | Baseline | +15-25% | Materials overhead |
| Memory Usage | Baseline | +10-15% | Segmentation models |
| Material Fidelity | Good | **Excellent (+40%)** | Physics-based response |
| Disk Usage | Baseline | +5% | Segmentation masks |

**Recommended For**:
- ✅ Flagship portfolio imagery
- ✅ Print materials and archival outputs
- ✅ Luxury real estate marketing
- ✅ Projects requiring maximum material realism

**Not Recommended For**:
- ❌ High-volume batch processing (use `production_ultra` instead)
- ❌ Quick previews (use `production_standard`)
- ❌ Systems with <16GB RAM

## MPS (Apple Silicon) Safety

The `production_ultra_materials` preset includes MPS safety measures:

```python
# Tiled upscaling to avoid MPS 2.5GB buffer limit
cfg.phase2.tile_based_upscaling = True
cfg.phase2.upscale_tile_size = 2048  # Safe for 3600×6000 → 14400×24000
cfg.phase2.upscale_overlap = 128     # Seamless blending

# Post-processing tiling
cfg.post_tile = 2048
cfg.post_overlap = 64
```

**Tested On**:
- M4 Max (128GB): 3600×6000 TIFF → 14400×24000 upscaled (success)
- M2 Pro (32GB): 2000×3000 TIFF → 8000×12000 upscaled (success)

## Troubleshooting

### Issue: Materials V2/V3 Not Running

**Symptoms**: Report shows `materials_v2_enabled: false` or `materials_v3_enabled: false`

**Check**:
1. Verify preset: `cfg.preset == Preset.PRODUCTION_ULTRA_MATERIALS`
2. Check config blocks: `cfg.materials_v2 is not None`
3. Check enabled flags: `cfg.materials_v2.enabled == True`
4. Review `disabled_reason` in report JSON

### Issue: Model Download Failures

**Symptoms**: UserWarning about missing models or initialization failures

**Solutions**:
1. Enable downloads: `cfg.segmentation.allow_downloads = True`
2. Provide local paths: `cfg.segmentation.segformer_model_path = Path(...)`
3. Check network connectivity (HuggingFace Hub access required)

### Issue: Out of Memory (OOM)

**Symptoms**: CUDA/MPS OOM errors during processing

**Solutions**:
1. Reduce segmentation resolution: `cfg.segmentation.input_long_side = 1024`
2. Lower Materials V3 megapixel limit: `cfg.materials_v3.max_megapixels = 20.0`
3. Enable tiled upscaling: Already enabled in preset
4. Use CPU instead of GPU: `--device cpu`

## Advanced Customization

### Custom Material Thresholds

```python
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

# Lower thresholds for better recall (more detections)
cfg.materials_v2.confidence.material_thresholds = {
    "wood": 0.45,   # -18% for better wood coverage
    "glass": 0.35,  # -22% for challenging glass surfaces
    ...
}

# Higher thresholds for better precision (fewer false positives)
cfg.materials_v2.confidence.material_thresholds = {
    "wood": 0.65,   # +18% for higher confidence
    "glass": 0.55,  # +22% to avoid false glass detections
    ...
}
```

### Enable Materials V3 Edge Refinement

```python
from lux_depth_v2.materials_v3 import RefinementStrategy

cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

# Enable edge refinement (requires EfficientSAM weights)
cfg.materials_v3.refine_edges = RefinementStrategy.ADAPTIVE
cfg.segmentation.efficientsam_model_path = Path("/path/to/efficientsam")
```

### Enable Materials V3 Pixel Operations

```python
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

# Enable pixel-level material modifications
cfg.materials_v3.apply_pixel_ops = True
cfg.materials_v3.glass_response_enabled = True
cfg.materials_v3.stone_response_enabled = True
```

## Migration from Other Presets

### From `production_ultra`
```python
# Before
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA)

# After (with Materials V2/V3)
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)
```

### From `interior_luxury_max_quality`
```python
# Before (Materials V2 only)
cfg = PipelineConfig(preset=Preset.INTERIOR_LUXURY_MAX_QUALITY)

# After (Materials V2 + V3)
cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)
```

## Testing and Validation

### Verify Materials Are Enabled

```bash
# Run pipeline with JSON report
lux-depth-v2 \
  --input sample.tif \
  --output-dir ./test_output \
  --preset production_ultra_materials

# Check report
cat ./test_output/sample_report.json | jq '.materials_v2_enabled, .materials_v3_enabled'
# Expected: true, true
```

### Automated Testing

```python
from lux_depth_v2.config import PipelineConfig, Preset

def test_materials_enabled():
    cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

    assert cfg.materials_v2 is not None, "V2 config missing"
    assert cfg.materials_v2.enabled, "V2 not enabled"

    assert cfg.materials_v3 is not None, "V3 config missing"
    assert cfg.materials_v3.enabled, "V3 not enabled"

    print("✅ Materials V2/V3 validation passed")

test_materials_enabled()
```

## Further Reading

- **Materials V2 Architecture**: `lux_depth_v2/materials_v2.py` docstrings
- **Materials V3 Architecture**: `lux_depth_v2/materials_v3.py` docstrings
- **Segmentation Backends**: `lux_depth_v2/material_segmentation.py`
- **Performance Benchmarks**: `lux_depth_v2/PERFORMANCE_VALIDATION.md`
- **Security Guidelines**: `lux_depth_v2/SECURITY.md`
