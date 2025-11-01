# VFX Extension for Transformation Portal

Depth-guided visual effects extension that integrates with the Transformation Portal's existing infrastructure.

## Overview

The VFX Extension (`realize_v8_unified_cli_extension.py`) provides advanced depth-aware visual effects for architectural rendering enhancement:

- **Depth-Aware Bloom** - Highlights glow based on depth information
- **Atmospheric Fog** - Exponential fog with depth falloff
- **Depth of Field** - Selective focus based on depth zones
- **Color Grading Zones** - Different color treatments for foreground/background
- **LUT Integration** - Apply LUTs with depth masking
- **Material Response** - Surface-aware enhancement

## Features

### VFX Presets

#### 1. **subtle_estate**
Minimal depth effects for signature estate look
- Bloom: 8% intensity, 12px radius
- Material boost: 15%
- Best for: Clean, professional real estate imagery

#### 2. **montecito_golden**
Warm coastal light with atmospheric depth
- Bloom: 20% intensity, 18px radius
- Fog: 25% density, warm golden color
- Material boost: 22%
- Default LUT: Montecito Golden Hour
- Best for: Coastal properties, warm lighting

#### 3. **cinematic_fog**
Atmospheric fog with gentle bloom
- Bloom: 25% intensity, 20px radius
- Fog: 40% density, cool atmospheric color
- Material boost: 18%
- Best for: Dramatic atmospheric shots

#### 4. **dramatic_dof**
Strong depth of field for hero shots
- Bloom: 30% intensity
- Depth of Field: Focus at 35%, 6px blur
- Material boost: 25%
- Color grading: Warm foreground, cool background
- Best for: Hero shots, product focus

## Installation

The VFX extension requires the core Transformation Portal dependencies:

```bash
# Install core dependencies
pip install -r requirements.txt

# Ensure depth pipeline is available
pip install -e ".[ml]"

# Verify installation
python -c "from realize_v8_unified_cli_extension import VFX_PRESETS; print('✓ VFX Extension ready')"
```

## Usage

### Command Line Interface

#### Single Image Enhancement

Process a single image with VFX:

```bash
python realize_v8_unified_cli_extension.py enhance-vfx \
    --input interior.jpg \
    --output enhanced.jpg \
    --base-preset signature_estate_agx \
    --vfx-preset cinematic_fog \
    --material-response \
    --save-depth \
    --out-bitdepth 16
```

**Arguments:**
- `--input`: Input image path
- `--output`: Output image path
- `--base-preset`: Base enhancement preset (signature_estate, signature_estate_agx, natural)
- `--vfx-preset`: VFX preset (subtle_estate, montecito_golden, cinematic_fog, dramatic_dof)
- `--material-response`: Enable Material Response enhancement
- `--lut`: Optional LUT path (overrides preset default)
- `--save-depth`: Save depth map alongside output
- `--out-bitdepth`: Output bit depth (8, 16, or 32)

#### Batch Processing

Process multiple images:

```bash
python realize_v8_unified_cli_extension.py batch-vfx \
    --input renders/ \
    --output finals/ \
    --base-preset signature_estate \
    --vfx-preset dramatic_dof \
    --material-response \
    --pattern "*.jpg" \
    --jobs 4 \
    --out-bitdepth 16
```

**Arguments:**
- `--input`: Input directory
- `--output`: Output directory
- `--pattern`: File pattern to match (default: *.jpg)
- `--jobs`: Number of parallel jobs (currently sequential)
- Other arguments same as single image

### Python API

```python
from realize_v8_unified_cli_extension import enhance_with_vfx
from realize_v8_unified import _open_any, _save_with_meta

# Load image
img, meta = _open_any("interior.jpg")

# Apply VFX
result = enhance_with_vfx(
    img,
    base_preset="signature_estate_agx",
    vfx_preset="montecito_golden",
    material_response=True,
    save_depth=True
)

# Save result
_save_with_meta(
    result["image"],
    result["array"],
    "enhanced.jpg",
    meta,
    out_bitdepth=16
)

# Access depth map
if result["depth"] is not None:
    depth_map = result["depth"]  # numpy array (H, W) in [0, 1]
    
# Check metrics
print(f"Total time: {result['metrics']['total_ms']}ms")
print(f"Depth estimation: {result['metrics']['depth_estimation_ms']}ms")
print(f"VFX processing: {result['metrics']['vfx_ms']}ms")
```

## Architecture

### Pipeline Flow

```
Input Image
    ↓
Base Enhancement (realize_v8_unified)
    ↓ (exposure, contrast, saturation, clarity)
Material Response (optional)
    ↓ (surface-aware enhancement)
Depth Estimation (ArchitecturalDepthPipeline)
    ↓ (24ms on M4 Max with CoreML)
VFX Operators
    ↓ (bloom, fog, DOF, color grading)
LUT Application (optional)
    ↓ (with depth masking)
Output Image + Depth Map
```

### Components

1. **realize_v8_unified.py** - Base enhancement pipeline
   - Core image I/O and metadata handling
   - Preset system for base adjustments
   - Color management foundation

2. **realize_v8_unified_cli_extension.py** - VFX extension
   - Depth estimation integration
   - VFX operators (bloom, fog, DOF)
   - Material Response integration
   - LUT processing with depth masking
   - CLI interface

3. **depth_pipeline/** - Depth estimation (existing)
   - Depth Anything V2 model
   - CoreML optimization for Apple Silicon
   - LRU caching for iterative workflows

4. **material_response.py** - Material Response (existing)
   - Physics-based surface enhancement
   - Material type detection

## Performance

Typical processing times on M4 Max (518px resolution):

- Base enhancement: 5-15ms
- Depth estimation: 24ms (CoreML) / 200ms (CPU)
- VFX operators: 10-30ms
- Material Response: 15-25ms
- **Total: 50-100ms per image**

Batch processing: 400-600 images/hour

## Integration with Existing Tools

### LUT Collection

The VFX extension integrates with the existing LUT collection:

```bash
# Use location-specific LUT
python realize_v8_unified_cli_extension.py enhance-vfx \
    --input render.jpg \
    --output enhanced.jpg \
    --vfx-preset subtle_estate \
    --lut 02_Location_Aesthetic/California/Montecito_Golden_Hour_HDR.cube
```

LUT directories:
- `01_Film_Emulation/` - Kodak and FilmConvert emulations
- `02_Location_Aesthetic/` - Location-specific color profiles
- `03_Material_Response/` - Physics-based surface enhancement

### Depth Pipeline Configuration

Use existing depth pipeline presets:

```python
result = enhance_with_vfx(
    img,
    base_preset="signature_estate",
    vfx_preset="cinematic_fog"
)
# Depth estimation uses config/interior_preset.yaml by default
```

## Examples

See `examples/vfx_extension_example.py` for detailed usage examples:

```bash
python examples/vfx_extension_example.py
```

## Testing

Run the test suite:

```bash
# Test VFX extension
pytest tests/test_realize_v8_vfx_extension.py -v

# Test specific functionality
pytest tests/test_realize_v8_vfx_extension.py::TestVFXExtension::test_apply_depth_bloom -v
```

## Troubleshooting

### Depth Pipeline Not Available

If depth estimation is not available, the extension will use a fallback gradient depth map:

```
[WARN] Depth pipeline not available - depth VFX will be disabled
```

**Solution:** Install ML dependencies:
```bash
pip install -e ".[ml]"
```

### Material Response Not Available

If Material Response is unavailable, a simplified fallback is used:

```
[WARN] Material Response not available - install with pip install -e .[ml]
```

This uses local contrast enhancement instead of full Material Response.

### LUT File Not Found

If a LUT file is missing, the original image is returned:

```
[WARN] LUT not found: path/to/lut.cube
```

**Solution:** Verify LUT path or omit `--lut` argument to use preset defaults.

## Customization

### Creating Custom VFX Presets

Edit `VFX_PRESETS` in `realize_v8_unified_cli_extension.py`:

```python
VFX_PRESETS = {
    "my_custom_preset": {
        "description": "Custom VFX preset",
        "bloom_intensity": 0.15,
        "bloom_radius": 16,
        "fog_density": 0.20,
        "fog_color": (0.85, 0.87, 0.90),
        "material_boost": 0.20,
        "lut_default": "path/to/custom.cube",
    },
}
```

### Creating Custom Base Presets

Edit `PRESETS` in `realize_v8_unified.py`:

```python
PRESETS["my_custom_base"] = Preset(
    name="My Custom Base",
    description="Custom base enhancement",
    exposure=0.15,
    contrast=1.10,
    saturation=1.08,
    clarity=0.20,
)
```

## Known Limitations

1. **Parallel Processing**: Batch processing is currently sequential (jobs parameter unused)
2. **Depth Fallback**: Without ML dependencies, uses gradient depth (reduced quality)
3. **LUT Size**: Only supports cube LUTs (not 1D or 3D LUT formats)
4. **Memory**: Large images (4K+) require 8-16GB RAM for full pipeline

## Roadmap

- [ ] Parallel batch processing with multiprocessing
- [ ] Additional VFX operators (lens flares, light leaks)
- [ ] Real-time preview mode
- [ ] GPU-accelerated VFX operators
- [ ] Additional depth model backends
- [ ] Support for more LUT formats

## License

Part of the Transformation Portal project. See main README for license information.

## Credits

Built on top of:
- **Depth Anything V2** - Monocular depth estimation
- **ArchitecturalDepthPipeline** - Optimized depth processing
- **Material Response** - Proprietary surface enhancement
- **realize_v8_unified** - Base enhancement system

## Support

For issues, questions, or contributions, see the main Transformation Portal repository.
