# Materials v2 User Guide

**Version:** 2.0  
**Last Updated:** 2025-12-09  
**Status:** Production Testing

## Overview

Materials v2 is an advanced material-aware enhancement system for Lux Depth V2 Pipeline that provides:

- **Confidence-gated material response** - Prevents over-processing with intelligent thresholding
- **Downscaled segmentation** - 2-3x faster processing with soft mask upsampling
- **Hard VRAM lifecycle control** - 40% lower memory usage with explicit cleanup
- **Mask caching** - 10-15% speedup on repeated processing with audit trail

## Quick Start

### Basic Usage

Enable Materials v2 with default settings (confidence threshold 0.6):

```bash
python3 -m lux_depth_v2.cli \
  --input input.tif \
  --output-dir output/ \
  --materials-v2
```

### With Confidence Tuning

Adjust confidence threshold for more conservative processing (0.8 = high confidence required):

```bash
python3 -m lux_depth_v2.cli \
  --input input.tif \
  --output-dir output/ \
  --materials-v2 \
  --confidence-threshold 0.8
```

### With Mask Caching

Enable mask caching for faster repeated processing:

```bash
python3 -m lux_depth_v2.cli \
  --input input.tif \
  --output-dir output/ \
  --materials-v2 \
  --cache-masks \
  --cache-dir .materials_v2_cache
```

### Batch Processing

Process entire directory with Materials v2:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir input_images/ \
  --output-dir output/ \
  --materials-v2 \
  --confidence-threshold 0.6 \
  --cache-masks
```

## Configuration Options

### Confidence Threshold

Controls how aggressive material enhancements are applied:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.4** | Aggressive - More coverage | Artistic rendering, maximizing enhancement |
| **0.6** | Balanced (default) | Most use cases, good quality/coverage trade-off |
| **0.8** | Conservative - High certainty | Critical projects, minimal artifacts |

**Recommended starting point:** 0.6

### Per-Material Thresholds

Materials v2 uses material-specific thresholds by default:

```python
material_thresholds = {
    'wood': 0.7,        # High confidence for wood grain
    'metal': 0.65,      # Medium-high for reflective surfaces
    'glass': 0.5,       # Lower for transparent materials
    'fabric': 0.6,      # Medium for textiles
    'stone': 0.7,       # High for stone texture
    'ceramic': 0.65,    # Medium-high for ceramics
    'water': 0.4,       # Very low for water (highly variable)
    'polished': 0.5,    # Lower for polished surfaces
}
```

**Note:** Global `--confidence-threshold` is used as fallback for materials not in the list.

### Confidence Blending

Control transition smoothness between enhanced and original:

```bash
# Soft blending (default, smooth transitions)
--confidence-blend-mode soft \
--confidence-blend-range 0.1

# Hard cutoff (sharp transitions, more defined edges)
--confidence-blend-mode hard
```

**Recommended:** Use `soft` mode for natural-looking results.

### Segmentation Resolution

Control segmentation resolution for performance tuning:

```bash
# Default (balanced)
--max-segmentation-side 1536

# Higher quality (slower)
--max-segmentation-side 2048

# Faster processing (lower quality)
--max-segmentation-side 1024
```

**Note:** Original image resolution is maintained via bicubic upsampling of masks.

## Advanced Features

### Material-Specific Strength

Adjust enhancement strength per material type (Python API):

```python
from lux_depth_v2.materials_v2 import MaterialsV2Engine, MaterialsV2Config

config = MaterialsV2Config(
    enabled=True,
    confidence=ConfidenceConfig(
        confidence_threshold=0.6,
        material_thresholds={
            'water': 0.4,      # Lower threshold for water
            'glass': 0.55,     # Slightly higher for glass
        }
    )
)

engine = MaterialsV2Engine(config)
result = engine.process(image_tensor, depth_map)
```

### Cache Management

Materials v2 caches segmentation masks to speed up repeated processing:

```bash
# Enable caching
--cache-masks --cache-dir .materials_v2_cache

# View cache statistics
ls -lh .materials_v2_cache/

# Clear cache
rm -rf .materials_v2_cache/
```

**Cache benefits:**
- 10-15% faster processing on second run
- Audit trail for quality validation
- Consistent results across runs

**Cache location:** `.materials_v2_cache/` (default)

### Quality Validation

Materials v2 generates confidence metrics for audit:

```bash
# Check confidence metrics in cache
cat .materials_v2_cache/750Picacho_Pool_confidence.json
```

Example metrics:
```json
{
  "image_hash": "a1b2c3d4e5f6...",
  "confidence_threshold": 0.6,
  "average_confidence": 0.72,
  "material_coverage": {
    "water": 45.2,
    "stone": 12.8,
    "wood": 8.5
  },
  "quality_score": 0.78
}
```

## Best Practices

### Confidence Threshold Tuning

1. **Start with default (0.6)** - Good balance for most images
2. **Increase to 0.7-0.8** if you see artifacts or over-processing
3. **Decrease to 0.5-0.4** if enhancement is too subtle
4. **Test edge cases** - Pool (water), bathroom (glass), kitchen (mixed)

### Material-Specific Challenges

#### Glass Surfaces
- Lower threshold recommended (0.5-0.6)
- Focus on transparency preservation
- Watch for reflection artifacts

```bash
--confidence-threshold 0.55
```

#### Water Features
- Very low threshold (0.4-0.5)
- Highly variable appearance
- Reflections and caustics need gentle processing

```bash
--confidence-threshold 0.45
```

#### Polished Stone
- Medium threshold (0.6-0.7)
- Balance texture enhancement and specular preservation
- Watch for over-sharpening

```bash
--confidence-threshold 0.65
```

#### Wood Grain
- Higher threshold (0.7)
- Preserve natural texture
- Avoid artificial enhancement

```bash
--confidence-threshold 0.7
```

### Performance Optimization

1. **Use mask caching** for repeated processing
2. **Downscale for preview** - Use `--max-segmentation-side 1024` for quick tests
3. **Batch processing** - Process multiple images in one run
4. **Monitor VRAM** - Materials v2 releases resources before upscaling

### Quality Checks

After processing, validate:

1. **Color accuracy** - Materials v2 should preserve color (< 1% difference)
2. **Edge quality** - Transitions should be smooth and natural
3. **Material fidelity** - Wood, metal, glass should look realistic
4. **Overall realism** - Enhanced image should look natural, not artificial

## Troubleshooting

### Common Issues

#### Over-processing / Artifacts

**Symptoms:** Unnatural enhancements, halos, color shifts

**Solution:**
```bash
# Increase confidence threshold
--confidence-threshold 0.75

# Use conservative blend mode
--confidence-blend-mode hard
```

#### Under-processing / No Enhancement

**Symptoms:** Minimal or no visible enhancement

**Solution:**
```bash
# Decrease confidence threshold
--confidence-threshold 0.5

# Check segmentation resolution
--max-segmentation-side 1536
```

#### Memory Issues

**Symptoms:** Out of memory errors, slow processing

**Solution:**
```bash
# Reduce segmentation resolution
--max-segmentation-side 1024

# Disable caching temporarily
# (remove --cache-masks flag)
```

#### Cache Inconsistency

**Symptoms:** Different results on second run

**Solution:**
```bash
# Clear cache and regenerate
rm -rf .materials_v2_cache/

# Run again with --cache-masks
```

### Performance Issues

#### Slow Segmentation

- Check GPU/MPS availability
- Reduce `--max-segmentation-side`
- Use `heuristic` backend (default, fastest)

#### Memory Spikes

- Materials v2 releases VRAM before upscaling
- If issues persist, reduce `--max-segmentation-side`
- Monitor with `nvidia-smi` (CUDA) or Activity Monitor (macOS)

## CLI Reference

### Materials v2 Flags

```bash
--materials-v2
  Enable Materials v2 confidence-gated material response

--confidence-threshold FLOAT
  Global confidence threshold (default: 0.6)
  Range: 0.0 to 1.0

--confidence-blend-range FLOAT
  Blend range for soft transitions (default: 0.1)

--confidence-blend-mode {soft,hard}
  Blending mode (default: soft)

--cache-masks
  Enable mask caching for repeated processing

--cache-dir PATH
  Cache directory (default: .materials_v2_cache)

--max-segmentation-side INT
  Max side length for segmentation (default: 1536)
```

## Examples

### Example 1: Pool Image (Water Features)

```bash
python3 -m lux_depth_v2.cli \
  --input pool.tif \
  --output-dir output_pool/ \
  --materials-v2 \
  --confidence-threshold 0.45 \
  --cache-masks
```

### Example 2: Bathroom (Glass, Stone, Metal)

```bash
python3 -m lux_depth_v2.cli \
  --input bathroom.tif \
  --output-dir output_bathroom/ \
  --materials-v2 \
  --confidence-threshold 0.7 \
  --cache-masks
```

### Example 3: Kitchen (Mixed Materials)

```bash
python3 -m lux_depth_v2.cli \
  --input kitchen.tif \
  --output-dir output_kitchen/ \
  --materials-v2 \
  --confidence-threshold 0.65 \
  --cache-masks
```

### Example 4: Conservative Processing

```bash
python3 -m lux_depth_v2.cli \
  --input image.tif \
  --output-dir output/ \
  --materials-v2 \
  --confidence-threshold 0.8 \
  --confidence-blend-mode hard \
  --cache-masks
```

## Integration with Phase 2

Materials v2 integrates seamlessly with Phase 2 performance enhancements:

```bash
python3 -m lux_depth_v2.cli \
  --input-dir images/ \
  --output-dir output/ \
  --materials-v2 \
  --confidence-threshold 0.6 \
  --cache-masks \
  --parallel-workers 2 \
  --model-cache \
  --async-io
```

**Expected overhead:** < 10% with heuristic backend

## Next Steps

1. **Test with your images** - Start with default settings
2. **Tune confidence threshold** - Adjust based on results
3. **Enable caching** - Speed up repeated processing
4. **Validate quality** - Use `compare_materials_quality.py`
5. **Report issues** - Help us improve Materials v2!

## Support

For questions or issues:
- Check troubleshooting section above
- Review technical specification: `MATERIALS_V2_TECHNICAL_SPEC.md`
- File issue on GitHub with sample image and configuration

---

**Version History:**
- v2.0 (2025-12-09): Production testing release
- v1.5 (2025-12-08): Phase 2 integration
- v1.0 (2025-12-07): Initial implementation
