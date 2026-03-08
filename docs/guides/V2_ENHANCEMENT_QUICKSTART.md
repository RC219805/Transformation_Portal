# V2 Enhancement - Quick Start Guide

## Overview

V2 Enhancement is a **depth-aware perceptual finishing system** for luxury real estate photography. It applies professional-grade enhancements to rendered images including:

- **Depth-aware tone mapping** for spatial hierarchy
- **Clarity enhancement** with edge-preserving sharpening
- **Material-specific processing** (wood, metal, glass, textiles, leather)
- **Atmospheric effects** (ambient occlusion, depth haze, light wrap)
- **Color grading** for luxury aesthetic

**Performance:** <2 seconds per image (end-to-end pipeline with depth maps)
**Dependencies:** Image processing only (no ML models)
**License:** Commercial-safe (BSD/MIT)

---

## Installation

V2 Enhancement is included in the Transformation Portal package. No additional dependencies required beyond core:

```bash
pip install -e .
```

---

## Usage

### Command Line

#### Basic Enhancement
```bash
python scripts/enhance_image.py input.png --output-dir output/
```

#### With Preset
```bash
# Luxury estate marketing
python scripts/enhance_image.py input.png \
    --output-dir output/ \
    --preset luxury_estate

# Architectural visualization
python scripts/enhance_image.py input.png \
    --output-dir output/ \
    --preset architectural
```

#### With Depth Maps
```bash
python scripts/enhance_image.py input.png \
    --output-dir output/ \
    --depth-dir depth_maps/ \
    --preset default
```

#### Skip Enhancement (Passthrough)
```bash
python scripts/enhance_image.py input.png \
    --output-dir output/ \
    --preset none
```

### Python API

#### Simple Enhancement
```python
from pathlib import Path
from transformation_portal.lux_depth_v3.v2_enhance import enhance_image
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig

# With preset
report = enhance_image(
    input_path=Path("input.png"),
    output_path=Path("output/enhanced.png"),
    config=V2EnhancementConfig.from_preset("luxury_estate"),
)

print(f"Enhanced in {report['runtime_s']:.2f}s")
```

#### With Depth Maps
```python
report = enhance_image(
    input_path=Path("input.png"),
    output_path=Path("output/enhanced.png"),
    depth_map_path=Path("depth_maps/input_depth.png"),
    config=V2EnhancementConfig.from_preset("default"),
)
```

#### Custom Configuration
```python
config = V2EnhancementConfig(
    preset="custom",
    enhancement_strength=0.9,  # Global enhancement [0, 1]
    clarity_strength=0.8,       # Clarity/sharpness [0, 1]
    material_strength=0.7,      # Material-specific [0, 1]
    depth_aware_tone_mapping=True,
    atmospheric_effects=True,
)

report = enhance_image(
    input_path=Path("input.png"),
    output_path=Path("output/enhanced.png"),
    config=config,
)
```

---

## Presets

### `default` - Balanced Enhancement
- **Use Case:** General real estate photography
- **Enhancement:** 0.7 | **Clarity:** 0.5 | **Material:** 0.6
- **Atmosphere:** ✅ Enabled
- **Best For:** Everyday interior/exterior renders

### `luxury_estate` - Premium Marketing
- **Use Case:** High-end luxury real estate marketing
- **Enhancement:** 0.8 | **Clarity:** 0.6 | **Material:** 0.7
- **Atmosphere:** ✅ Enabled
- **Best For:** Luxury condos, estates, premium listings

### `architectural` - Technical Visualization
- **Use Case:** Architectural visualization and technical documentation
- **Enhancement:** 0.6 | **Clarity:** 0.7 | **Material:** 0.5
- **Atmosphere:** ❌ Disabled
- **Best For:** Design reviews, technical presentations

### `none` - Skip Enhancement (Passthrough)
- **Use Case:** Disable V2 enhancement entirely
- **Enhancement:** 0.0 | **Clarity:** 0.0 | **Material:** 0.0
- **Atmosphere:** ❌ Disabled
- **Best For:** PBR-only workflows, debugging

---

## Features

### Depth-Aware Tone Mapping
- **Foreground boost:** Enhances primary subjects with highlight/clarity boost
- **Background compression:** Subtle atmospheric handling for depth perception
- **Spatial hierarchy:** Preserves depth relationships from V3 stage

### Clarity Enhancement
- **Multi-scale unsharp masking:** Reveals detail across multiple frequency bands
- **Edge-preserving:** Prevents halo artifacts
- **Material-aware:** Modulates strength based on detected materials

### Material-Specific Processing
Uses Materials V3 taxonomy for physics-based enhancements:

- **Wood:** Warmth boost + grain enhancement
- **Metal:** Highlight enhancement + contrast lift
- **Glass:** Subtle highlight boost + transparency preservation
- **Textiles:** Micro-contrast for fabric texture
- **Leather:** Sheen enhancement

### Atmospheric Effects
When enabled (default, luxury_estate):

- **Ambient occlusion:** Grounding for furniture/floor contact
- **Depth haze:** Atmospheric perspective for exteriors
- **Light wrap:** Window reflections, fireplace glow simulation

---

## Performance

### Benchmarks (Apple M4 Pro)
- **Default preset:** 0.018s per image (~2,000 images/hour)
- **Luxury preset:** 0.019s per image (~1,900 images/hour)
- **With depth maps:** 0.020-0.050s per image (400-600 images/hour)
- **Passthrough (none):** 0.008s per image (~7,200 images/hour)

### Resource Usage
- **Memory:** ~200 MB per image (peak)
- **CPU:** Single-threaded (parallelization via orchestrator)
- **Dependencies:** numpy, scipy, PIL only (~500 MB install)

---

## Output

### Report JSON
Each enhancement generates a JSON report with metadata:

```json
{
    "status": "success",
    "preset": "luxury_estate",
    "runtime_s": 0.019,
    "config": {
        "enhancement_strength": 0.8,
        "clarity_strength": 0.6,
        "material_strength": 0.7,
        "depth_aware_tone_mapping": true,
        "atmospheric_effects": true
    },
    "stage_metadata": {
        "has_depth": true,
        "has_materials": false,
        "processing_ms": 15.2
    }
}
```

---

## Integration with Orchestrator

V2 Enhancement integrates seamlessly with the Lux Depth V3 orchestrator:

```bash
# Full pipeline: Depth (V3) → Enhancement (V2) → PBR
python -m transformation_portal.lux_depth_v3 \
    --input input_images/ \
    --output output/ \
    --enable-v2 on \
    --v2-preset luxury_estate \
    --generate-pbr
```

### CLI Options
- `--enable-v2 on|off` - Enable/disable V2 enhancement stage (default: on)
- `--v2-preset PRESET` - Enhancement preset (default: default)
- Use `--preset none` or `--enable-v2 off` for PBR-only workflows

---

## Troubleshooting

### Enhancement Too Subtle
- Try `--preset luxury_estate` for stronger enhancement
- Or create custom config with higher strength values (0.8-0.9)

### Enhancement Too Strong
- Try `--preset architectural` for more conservative enhancement
- Or reduce strength values (0.4-0.6)

### No Depth-Aware Effects
- Ensure depth maps are in `--depth-dir` with correct naming
- Check depth map naming: `{image_stem}_depth.png` or `{image_stem}_depth_u16.png`

### Disable Enhancement
- Use `--preset none` to skip V2 entirely (passthrough)
- Or use `--enable-v2 off` in orchestrator

---

## Dependencies

### Required (Core)
- `numpy` ≥ 1.20.0
- `scipy` ≥ 1.7.0
- `Pillow` ≥ 9.0.0

### Optional
- `scikit-image` ≥ 0.19.0 (for advanced transforms)

### ❌ NOT Required
V2 Enhancement is **image processing only** - no ML dependencies:
- ❌ `torch` - ML framework
- ❌ `diffusers` - diffusion models
- ❌ `realesrgan` - upscaling models

---

## Architecture

### Design Principles
1. **Reuse over reimplementation** - Leverages existing `EnhancementStage`
2. **Image processing only** - No ML model dependencies
3. **Commercial safety** - BSD/MIT licenses only
4. **Performance first** - <2s/image typical
5. **Minimal change** - Backward compatible with existing workflows

### Components
- `v2_enhance.py` - Core enhancement logic
- `v2_presets.py` - Preset configuration system
- `scripts/enhance_image.py` - CLI entry point
- `EnhancementStage` - Reused from stage_graph

### References
- `V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md` - Detailed architecture
- `docs/historical/V2_ENHANCEMENT_FINAL_REPORT.md` - Final implementation report
- `ADR-022-v2-enhancement-optional.md` - Decision record

---

## Examples

### Batch Processing
```bash
for img in input_images/*.png; do
    python scripts/enhance_image.py "$img" \
        --output-dir output/ \
        --preset luxury_estate
done
```

### With Orchestrator (Full Pipeline)
```bash
python -m transformation_portal.lux_depth_v3 \
    --input input_images/ \
    --output output/ \
    --enable-v2 on \
    --v2-preset luxury_estate \
    --generate-pbr \
    --device mps
```

### PBR-Only Workflow (Skip V2)
```bash
python -m transformation_portal.lux_depth_v3 \
    --input input_images/ \
    --output output/ \
    --enable-v2 off \
    --generate-pbr
```

---

## License

V2 Enhancement uses only commercially-safe dependencies:
- `numpy` - BSD License
- `scipy` - BSD License
- `Pillow` - HPND License (PIL)

No research-only or GPL dependencies.

---

## Support

For issues or questions:
1. Check the Final implementation report: `docs/historical/V2_ENHANCEMENT_FINAL_REPORT.md`
2. Review `docs/architecture/decisions/V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md`
3. See `tests/test_v2_*.py` for usage examples
