# Luxury Estate Master Pipeline - Documentation
## Elite HDR Processing for 750 Picacho

**Version:** 1.0.0  
**Date:** 2025-11-10  
**Author:** Transformation Portal

---

## Overview

The **Luxury Estate Master Pipeline** is a cutting-edge, production-ready HDR processing system designed specifically for luxury real estate architectural photography. It combines 7 advanced processing stages optimized for 32-bit TIFF HDR sources.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  32-bit HDR TIFF Sources                     │
│         (sRGB, alpha channel, full dynamic range)            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 1: HDR Precision Loader                               │
│  • 32-bit float preservation                                 │
│  • Alpha channel extraction                                  │
│  • Metadata capture (IPTC, XMP, GPS)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 2: Depth Anything V2                                  │
│  • Monocular depth estimation                                │
│  • CoreML/MPS acceleration (24-65ms per image)              │
│  • Zone-based depth segmentation                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 3: Material Response Technology                       │
│  • Physics-based surface enhancement                         │
│  • Material detection: wood, metal, glass, stone, textiles  │
│  • Micro-contrast optimization                              │
│  • Highlight preservation                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 4: Intelligent Tone Mapping                           │
│  • AgX OCIO / Filmic Hable / Reinhard                       │
│  • HDR-to-display conversion                                │
│  • Exposure & contrast control                              │
│  • Highlight detail retention                               │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 5: Location Color Grading                            │
│  • LUT stack processing                                     │
│  • California coastal aesthetic                             │
│  • Film emulation (Kodak 2393)                              │
│  • Saturation & vibrance control                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 6: AI Enhancement                                     │
│  • ControlNet edge-preserving refinement                    │
│  • Stable Diffusion XL photorealistic enhancement           │
│  • Canny edge detection                                     │
│  • Room-specific prompting                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  Stage 7: Real-ESRGAN 4x Upscaling                          │
│  • AI-powered 4x resolution enhancement                     │
│  • Tile-based processing (512px)                            │
│  • Final resize to delivery dimensions                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Output Files                              │
│  • 16-bit TIFF master (archival)                            │
│  • High-quality JPEG delivery (95%)                         │
│  • Intermediate tonemapped preview                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python --version  # Should be 3.10 or higher

# Install package in development mode
cd /Users/rc/Transformation_Portal
pip install -e .

# Install ML dependencies
pip install -e ".[ml]"  # PyTorch, Diffusers, ControlNet
pip install -e ".[tiff]"  # tifffile, imagecodecs

# Install Real-ESRGAN
pip install realesrgan
pip install basicsr

# Download Real-ESRGAN weights
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O weights/RealESRGAN_x4plus.pth
```

### Verify Installation

```bash
python luxury_estate_master_pipeline.py --dry-run --preset 750_picacho
```

---

## Quick Start

### Process Single Image

```bash
# Basic processing with default preset
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Great_Room_HDR_32-bit.tif

# Specify room type for AI prompting
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Aerial_HDR_32-bit.tif \
  --room-type aerial \
  --preset aerial
```

### Batch Process All Images

```bash
# Process entire directory
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/*.tif

# With custom output directory
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/*.tif \
  --output-dir output_750_picacho_$(date +%Y%m%d)
```

---

## Configuration

### Preset System

The pipeline uses **dataclass-based presets** for complete configuration control.

**Available Presets:**
- `750_picacho` - Default preset for luxury interiors
- `aerial` - Optimized for aerial photography with atmospheric effects

### Configuration Structure

```python
@dataclass
class PipelinePreset:
    name: str
    description: str
    depth: DepthConfig
    material_response: MaterialResponseConfig
    tone_mapping: ToneMappingConfig
    color_grading: ColorGradingConfig
    ai_enhancement: AIEnhancementConfig
    upscaling: UpscalingConfig
    output: OutputConfig
```

### Creating Custom Presets

```python
from luxury_estate_master_pipeline import PipelinePreset, DepthConfig, ToneMappingConfig

my_preset = PipelinePreset(
    name="Custom Estate",
    description="Custom configuration",
    depth=DepthConfig(
        enabled=True,
        model_variant="small",
        clarity_strength=0.6,
    ),
    tone_mapping=ToneMappingConfig(
        method="filmic",
        exposure=0.2,
        white_point=12.0,
    ),
    # ... other configs
)
```

### YAML Configuration

You can also save/load presets from YAML:

```bash
# Save preset to YAML
python luxury_estate_master_pipeline.py \
  --preset 750_picacho \
  --save-preset my_preset.yaml

# Edit my_preset.yaml
# Load in code:
```

```python
import yaml
from luxury_estate_master_pipeline import PipelinePreset

with open('my_preset.yaml') as f:
    config = yaml.safe_load(f)
    preset = PipelinePreset(**config)
```

---

## Stage Details

### Stage 1: HDR Precision Loader

**Purpose:** Load 32-bit TIFF with complete precision preservation

**Features:**
- Supports 32-bit float, 16-bit, and 8-bit TIFFs
- Alpha channel extraction and preservation
- Metadata capture (IPTC, XMP, GPS coordinates)
- Automatic linear RGB conversion

**Parameters:** None (automatic detection)

---

### Stage 2: Depth Anything V2

**Purpose:** Monocular depth estimation for depth-aware processing

**Key Parameters:**
```yaml
depth:
  model_variant: "small"     # small, base, large
  backend: "pytorch_mps"     # pytorch_mps, coreml, pytorch_cpu
  num_zones: 4               # Depth segmentation zones
  clarity_strength: 0.55     # Detail enhancement (0.0-1.0)
```

**Performance:**
- **Small model:** 24-65ms per image (M4 Max)
- **Base model:** 80-150ms per image
- **Large model:** 200-400ms per image

**Backend Selection:**
- `pytorch_mps` - Apple Silicon (M-series) - **RECOMMENDED**
- `coreml` - Apple Neural Engine - Fastest on M3/M4
- `pytorch_cpu` - CPU fallback

---

### Stage 3: Material Response Technology

**Purpose:** Physics-based surface enhancement for realistic rendering

**Key Parameters:**
```yaml
material_response:
  strength: 0.75              # Overall enhancement (0.0-1.0)
  preserve_highlights: true   # Protect specular highlights
  enhance_wood: true          # Wood grain detail
  enhance_metal: true         # Metallic reflections
  enhance_glass: true         # Glass transparency
  enhance_stone: true         # Stone texture
  enhance_textiles: true      # Fabric detail
```

**Material Detection:**
- Automatic surface analysis
- Micro-contrast optimization per material type
- Selective sharpening with highlight masking

---

### Stage 4: Intelligent Tone Mapping

**Purpose:** HDR-to-display conversion with cinematic quality

**Tone Mapping Operators:**

1. **Filmic (Hable)** - *Recommended*
   ```yaml
   tone_mapping:
     method: "filmic"
     white_point: 11.2  # 8.0-16.0
   ```
   - Film-like highlight roll-off
   - Excellent for architectural interiors
   - Preserves HDR detail

2. **AgX** - *Requires OCIO config*
   ```yaml
   tone_mapping:
     method: "agx"
     agx_config_path: "path/to/config.ocio"
   ```
   - Industry-standard ACES-like response
   - Perfect neutral color

3. **Reinhard** - *Simple fallback*
   ```yaml
   tone_mapping:
     method: "reinhard"
   ```

**Additional Controls:**
```yaml
tone_mapping:
  exposure: 0.0      # EV adjustment (-2.0 to +2.0)
  contrast: 1.05     # Contrast multiplier (0.5-1.5)
```

---

### Stage 5: Location Color Grading

**Purpose:** Apply location-specific aesthetic with LUT stacks

**LUT Stack Configuration:**
```yaml
color_grading:
  lut_stack:
    - ["assets/luts/location_aesthetic/California/Montecito_Golden_Hour_HDR.cube", 0.70]
    - ["assets/luts/film_emulation/Kodak/Kodak_2393_D55_HDR.cube", 0.50]
  saturation: 1.08
  vibrance: 0.15
```

**Available LUTs:**
- **Location Aesthetic:**
  - `California/Montecito_Golden_Hour_HDR.cube`
  - `Mediterranean/Spanish_Colonial_Warm_HDR.cube`
  
- **Film Emulation:**
  - `Kodak/Kodak_2393_D55_HDR.cube`
  - `FilmConvert/FilmConvert_Nitrate_HDR.cube`

**Saturation vs. Vibrance:**
- `saturation`: Uniform saturation boost (affects all colors equally)
- `vibrance`: Smart saturation (boosts muted colors, protects skin tones)

---

### Stage 6: AI Enhancement

**Purpose:** Edge-preserving photorealistic refinement with ControlNet + SDXL

**Key Parameters:**
```yaml
ai_enhancement:
  num_inference_steps: 30    # 20-50 (higher = better quality)
  guidance_scale: 7.5        # 5.0-15.0 (prompt adherence)
  strength: 0.30             # 0.0-1.0 (transformation strength)
  seed: 42                   # Reproducibility
```

**Room-Specific Prompts:**
- **Aerial:** "luxury coastal estate aerial photography, dramatic hillside architecture..."
- **Interior:** "luxury {room_type} architectural photography, montecito coastal estate..."
- **Pool:** "luxury infinity pool mediterranean architecture, golden hour reflection..."

**Performance:**
- Processing time: ~60-90 seconds per image (M4 Max)
- Memory: 8GB+ VRAM recommended
- Resolution: Processes at 768px, upscales to original

**Disabling AI Enhancement:**
```yaml
ai_enhancement:
  enabled: false
```

---

### Stage 7: Real-ESRGAN 4x Upscaling

**Purpose:** AI-powered super-resolution for maximum detail

**Key Parameters:**
```yaml
upscaling:
  method: "esrgan"       # esrgan, lanczos
  scale_factor: 4.0      # 2.0, 4.0
  tile_size: 512         # Memory management
  tile_padding: 10       # Avoid seam artifacts
```

**Resolution Example:**
- Input: 2048×1536 (32-bit TIFF)
- After AI: 768×576 (SD processing size)
- After ESRGAN: 3072×2304 (4x upscale)
- Final: 2048×1536 (resized to original dimensions)

**Fallback:**
If Real-ESRGAN is unavailable, automatically falls back to Lanczos interpolation.

---

## Output Files

### Master TIFF

**Path:** `output_750_picacho_elite/{filename}_master.tif`

**Specifications:**
- Bit depth: 16-bit (default) or 32-bit
- Compression: LZW lossless
- Color space: sRGB
- Metadata: Complete preservation (IPTC, XMP, GPS)

**Use:** Archival master for future reprocessing

### Delivery JPEG

**Path:** `output_750_picacho_elite/{filename}_delivery.jpg`

**Specifications:**
- Quality: 95% (configurable)
- Optimization: Enabled
- Color space: sRGB
- Metadata: Embedded

**Use:** Client delivery, web, print

### Intermediate Preview

**Path:** `output_750_picacho_elite/{filename}_tonemapped.jpg`

**Specifications:**
- Quality: 90%
- Shows tonemapped result before AI/upscaling

**Use:** Quick preview, QC checks

---

## Performance Optimization

### Hardware Acceleration

**Apple Silicon (M-series):**
```yaml
depth:
  backend: "pytorch_mps"  # Metal Performance Shaders

# ML models automatically use MPS when available
```

**CUDA (NVIDIA):**
Pipeline auto-detects CUDA and uses it for all ML operations.

**CPU Fallback:**
All stages gracefully fall back to CPU if GPU unavailable.

### Memory Management

**Large Images (4K+):**
```yaml
upscaling:
  tile_size: 512      # Reduce for 8GB VRAM
  tile_padding: 10

performance:
  use_fp16: true      # Half-precision (saves memory)
```

**Batch Processing:**
Images are processed sequentially by default to avoid memory exhaustion.

### Processing Times (M4 Max, 2048×1536 images)

| Stage | Time | Notes |
|-------|------|-------|
| Load HDR | 0.5s | TIFF decompression |
| Depth Estimation | 0.03s | CoreML accelerated |
| Material Response | 1.2s | CPU-intensive |
| Tone Mapping | 0.2s | Fast |
| Color Grading | 0.3s | LUT interpolation |
| AI Enhancement | 75s | SD inference |
| Upscaling | 8s | Real-ESRGAN |
| **Total** | **~85s** | Per image |

**Throughput:** ~40-50 images/hour (full pipeline)

---

## Room-Specific Processing

### Auto-Detection from Filename

The pipeline can detect room types from filenames:

```
750Picacho_Aerial_HDR_32-bit.tif → room_type: "aerial"
750Picacho_Kitchen_HDR_32-bit.tif → room_type: "kitchen"
```

### Manual Override

```bash
python luxury_estate_master_pipeline.py \
  image.tif \
  --room-type bedroom
```

### Room Type Mappings

```python
room_types = {
    '750Picacho_Aerial_HDR_32-bit': 'aerial',
    '750Picacho_Bathroom_HDR_32-bit': 'bathroom',
    '750Picacho_Bedroom_HDR_32-bit': 'bedroom',
    '750Picacho_Great_Room_HDR_32-bit': 'great_room',
    '750Picacho_Kitchen_HDR_32-bit': 'kitchen',
    '750Picacho_Pool_HDR_32-bit': 'pool',
}
```

---

## Batch Processing Report

After batch processing, a JSON report is generated:

**Path:** `output_750_picacho_elite/processing_report.json`

```json
{
  "preset": "750 Picacho Elite",
  "images_processed": 6,
  "total_time": 512.3,
  "average_time": 85.4,
  "results": [
    {
      "source_path": "750Picacho_Aerial_HDR_32-bit.tif",
      "room_type": "aerial",
      "stages": {
        "1_load": 0.5,
        "2_depth": 0.03,
        "3_material": 1.2,
        "4_tonemap": 0.2,
        "5_color": 0.3,
        "6_ai": 75.0,
        "7_upscale": 8.0
      },
      "output_paths": {
        "master_tiff": "output/.../master.tif",
        "delivery_jpeg": "output/.../delivery.jpg"
      },
      "total_time": 85.2
    }
  ]
}
```

---

## Troubleshooting

### Import Errors

**Issue:** `ImportError: No module named 'transformation_portal'`

**Solution:**
```bash
pip install -e .
# or add src/ to PYTHONPATH
export PYTHONPATH=/Users/rc/Transformation_Portal/src:$PYTHONPATH
```

### ESRGAN Not Available

**Issue:** `Real-ESRGAN not available - upscaling will use Lanczos`

**Solution:**
```bash
pip install realesrgan basicsr
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O weights/RealESRGAN_x4plus.pth
```

### AI Enhancement Failing

**Issue:** `AI enhancement not available`

**Solution:**
```bash
pip install diffusers controlnet-aux transformers
pip install torch torchvision  # Ensure PyTorch installed
```

### Out of Memory

**Issue:** CUDA/MPS out of memory during AI enhancement

**Solutions:**
1. Reduce inference steps:
   ```yaml
   ai_enhancement:
     num_inference_steps: 20  # Down from 30
   ```

2. Disable upscaling:
   ```yaml
   upscaling:
     enabled: false
   ```

3. Process smaller batches

### Depth Model Slow

**Issue:** Depth estimation taking >1 second per image

**Solution:**
```yaml
depth:
  backend: "coreml"  # Use Apple Neural Engine
  model_variant: "small"  # Use smaller model
```

---

## Advanced Usage

### Programmatic API

```python
from pathlib import Path
from luxury_estate_master_pipeline import (
    LuxuryEstateMasterPipeline,
    get_750_picacho_preset,
)

# Initialize pipeline
preset = get_750_picacho_preset()
pipeline = LuxuryEstateMasterPipeline(preset)

# Process single image
result = pipeline.process_image(
    Path('input.tif'),
    room_type='great_room'
)

# Batch process
image_paths = list(Path('input_images').glob('*.tif'))
results = pipeline.batch_process(image_paths)

# Access statistics
print(f"Processed {pipeline.stats['images_processed']} images")
print(f"Total time: {pipeline.stats['total_time']:.1f}s")
```

### Custom Processing Stages

You can selectively enable/disable stages:

```python
preset = get_750_picacho_preset()

# Disable AI enhancement for faster processing
preset.ai_enhancement.enabled = False

# Disable upscaling
preset.upscaling.enabled = False

# Tone mapping only
preset.depth.enabled = False
preset.material_response.enabled = False
preset.color_grading.enabled = False

pipeline = LuxuryEstateMasterPipeline(preset)
```

---

## Best Practices

### For Best Quality

1. **Use full pipeline** with all stages enabled
2. **AI Enhancement:** 30-40 inference steps
3. **Upscaling:** Real-ESRGAN 4x
4. **Tone Mapping:** Filmic with white_point 11.2
5. **Save 16-bit masters** for archival

### For Fast Turnaround

1. **Disable AI enhancement** (saves ~75s per image)
2. **Use Lanczos upscaling** (saves ~8s per image)
3. **Small depth model** with CoreML backend
4. **Reduce inference steps** to 20

### For Consistency

1. **Use fixed seed** for AI enhancement
2. **Apply same preset** to all images in set
3. **Enable metadata embedding** for traceability

---

## License & Credits

**Pipeline:** Transformation Portal  
**Depth Estimation:** Depth Anything V2 (MIT License)  
**Tone Mapping:** AgX (OCIO), Filmic (Hable)  
**AI Models:** Stable Diffusion (CreativeML), ControlNet (Apache 2.0)  
**Upscaling:** Real-ESRGAN (BSD 3-Clause)

---

## Support

For issues, questions, or feature requests:
- **Repository:** /Users/rc/Transformation_Portal
- **Documentation:** docs/LUXURY_ESTATE_PIPELINE.md
- **Configuration:** config/750_picacho_master_preset.yaml

---

**Last Updated:** 2025-11-10  
**Version:** 1.0.0
