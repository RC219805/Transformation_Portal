# Transformation Portal CLI Reference

Complete command-line interface reference for the Transformation Portal v2.0 production system.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Main Entry Points](#main-entry-points)
- [Rendering Commands](#rendering-commands)
  - [Lux Render (AI Enhancement)](#lux-render-ai-enhancement)
  - [Sky Render (Atmospheric Physics)](#sky-render-atmospheric-physics)
- [Processing Commands](#processing-commands)
  - [Material Response](#material-response)
  - [Video Master Grader](#video-master-grader)
  - [TIFF Batch Processor](#tiff-batch-processor)
- [Analysis Commands](#analysis-commands)
  - [Decision Decay Dashboard](#decision-decay-dashboard)
  - [System Info](#system-info)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Transformation Portal CLI provides unified access to professional image and video processing pipelines for luxury real estate rendering and architectural visualization.

**Key Capabilities**:
- ✅ Physics-based sky replacement with auto-correction
- ✅ AI-powered enhancement (Stable Diffusion XL, ControlNet)
- ✅ Depth-aware processing (Depth Anything V2)
- ✅ Material Response technology (surface-aware rendering)
- ✅ Professional color grading (16+ LUTs)
- ✅ Batch processing (400-600 images/hour)

**What Makes This Unique**:
The "New Paradigm" - Instead of naive AI hallucination, the system uses **physics-based atmospheric rendering** with **shadow analysis** and **auto-correction guardrails**. When you request a sky that conflicts with the existing scene lighting, the system analyzes the shadows, detects the inconsistency, and suggests the physically-correct parameters.

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.10 or higher
python --version  # Should be 3.10+

# Git for cloning
git --version
```

### Basic Installation

```bash
# Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Verify installation
python -m transformation_portal version
```

### Full Installation (with ML capabilities)

```bash
# Install with ML extras (Stable Diffusion, ControlNet, etc.)
pip install -e ".[ml]"

# For 16-bit TIFF support
pip install -e ".[tiff]"

# For everything
pip install -e ".[all]"
```

### Verify Core Systems

```bash
# Run verification script
python scripts/verification/verify_core.py

# Expected output:
# ✅ Input Tensor Created (1024x1024)
# ✅ Loaded Micro-Climate: Montecito
# ✅ Physics Engines Initialized
# ✅ Smart Render Completed
# 📊 Confidence: 0.00 (flat gray = no hallucination)
```

---

## Main Entry Points

### Unified Pipeline (Recommended)

```bash
# Recipe-driven batch processing
python -m transformation_portal process \
  --input "renders/*.jpg" \
  --recipe config/recipes/signature_estate.yaml \
  --output output/

# List available recipes
python -m transformation_portal list-recipes

# Validate recipe
python -m transformation_portal validate-recipe config/recipes/custom.yaml
```

### Individual Pipelines

```bash
# Direct script execution
python lux_render_pipeline.py --input image.jpg --output output/
python material_response.py --input image.jpg --output output/
python luxury_video_master_grader.py --input video.mp4 --preset golden_hour
```

---

## Rendering Commands

### Lux Render (AI Enhancement)

**Purpose**: AI-powered render refinement using Stable Diffusion XL and ControlNet.

**Basic Usage**:
```bash
python lux_render_pipeline.py \
  --input renders/bedroom_v01.jpg \
  --output output/ \
  --preset luxury_bedroom
```

**Advanced Usage**:
```bash
python lux_render_pipeline.py \
  --input renders/kitchen.exr \
  --output output/ \
  --prompt "Ultra luxury kitchen with marble countertops, professional lighting" \
  --negative-prompt "distortion, artifacts, unrealistic" \
  --controlnet-strength 0.7 \
  --upscale 4x \
  --material-response \
  --lut assets/luts/film_emulation/Kodak_2393.cube \
  --brand-overlay assets/brand/lantern_logo/logo.png
```

**Key Parameters**:
- `--preset`: Pre-configured style preset (e.g., `luxury_bedroom`, `modern_kitchen`)
- `--prompt`: Describe desired aesthetic and details
- `--negative-prompt`: Elements to avoid
- `--controlnet-strength`: How much to preserve original structure (0.0-1.0, default: 0.8)
- `--upscale`: Upscaling factor (2x, 4x via Real-ESRGAN)
- `--material-response`: Enable physics-based surface enhancement
- `--lut`: Apply color grading LUT
- `--brand-overlay`: Add logo to final output

**Output**:
- `{basename}_enhanced.tiff` - Main enhanced output (16-bit if tifffile available)
- `{basename}_depth.png` - Depth map visualization
- `{basename}_control.png` - ControlNet conditioning map

**Performance**: ~2-5 minutes per image on M4 Max with GPU acceleration

---

### Sky Render (Atmospheric Physics)

**Purpose**: Physics-based sky replacement with shadow analysis and auto-correction.

**The New Paradigm**:
Traditional sky replacement tools naively composite any sky onto any photo. Transformation Portal **analyzes the existing scene lighting** (shadow directions, highlights) and **prevents physically impossible transformations**. When conflicts are detected, it **auto-suggests corrected parameters**.

**Basic Usage**:
```bash
python -c "
from transformation_portal.atmosphere import SkyBlender, LocationPresets, SkyGANGenerator
import cv2

# Load image
image = cv2.imread('renders/estate_exterior.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get Montecito golden hour parameters
presets = LocationPresets()
sky_params = presets.get_sky_parameters(
    location='montecito',
    time_of_day=17.5,  # 5:30 PM
    condition='sundowner'
)
atmo_params = presets.get_atmospheric_parameters(
    location='montecito',
    condition='sundowner'
)

# Execute smart render with physics guardrails
blender = SkyBlender()
result, suggestion = blender.smart_render(
    source_image=image,
    sky_params=sky_params,
    atmo_params=atmo_params,
    auto_correct=True,     # Enable shadow analysis
    strict_physics=False   # Allow minor deviations
)

# Save
cv2.imwrite('output/sky_replaced.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

# Check suggestion
if suggestion.confidence < 0.8:
    print(f'⚠️  Physics Violation Detected!')
    print(f'   Requested: {suggestion.original_request_azimuth:.1f}°')
    print(f'   Measured: {suggestion.measured_source_azimuth:.1f}°')
    print(f'   Message: {suggestion.message}')
"
```

**Advanced: Manual Sky Parameters**:
```python
from transformation_portal.atmosphere import SkyGANGenerator, SkyParameters
import numpy as np

# Initialize generator
generator = SkyGANGenerator()

# Custom sky parameters
params = SkyParameters(
    sun_azimuth=220,      # Southwest (0=North, 90=East, 180=South, 270=West)
    sun_elevation=30,     # Low angle for golden hour (0-90 degrees)
    cloud_coverage=0.15,  # Light clouds (0.0-1.0)
    haze_density=0.10,    # Minimal haze (0.0-1.0)
    turbidity=1.3,        # Exceptional clarity (1.0=pristine, 10.0=hazy)
    latitude=34.4,        # Montecito
    longitude=-119.7
)

# Generate HDR sky
sky = generator.generate_sky(
    params,
    resolution=(2048, 1024),
    output_format="hdr"
)

# Save as EXR
generator.save_sky(sky, "sky_custom.exr", format="exr")
```

**Key Parameters**:
- **sun_azimuth**: Direction of sun (0-360°, where 0=North, clockwise)
- **sun_elevation**: Height of sun above horizon (0-90°)
- **cloud_coverage**: Amount of clouds (0.0=clear, 1.0=overcast)
- **haze_density**: Atmospheric haze (0.0=crystal clear, 1.0=very hazy)
- **turbidity**: Atmospheric clarity (1.0=pristine, 2-3=typical, 10.0=polluted)
- **auto_correct**: Enable shadow analysis and physics guardrails
- **strict_physics**: Reject physically impossible requests vs. suggest alternatives

**Physics Guardrails**:
1. **Shadow Analysis**: Detects dominant light direction from scene shadows
2. **Consistency Check**: Compares requested sun position vs. measured shadows
3. **Auto-Correction**: If conflict > tolerance (default 45°), suggests corrected parameters
4. **Confidence Score**: 0.0-1.0, where 0.0 = flat gray (no scene lighting detected)

**Output**:
- Enhanced image with physically-consistent sky
- Correction suggestion if physics violated
- Confidence score indicating quality of shadow analysis

**Performance**: ~500ms-2s per image (GPU-accelerated)

---

## Processing Commands

### Material Response

**Purpose**: Physics-based surface enhancement for wood, metal, glass, stone, and textiles.

**Basic Usage**:
```bash
python material_response.py \
  --input renders/interior.jpg \
  --output output/ \
  --surfaces wood metal glass \
  --strength 0.7
```

**Advanced Usage**:
```bash
python material_response.py \
  --input-dir renders/ \
  --output-dir output/ \
  --surfaces wood metal glass stone fabric \
  --strength 0.75 \
  --preserve-highlights \
  --enhance-micro-contrast \
  --batch-size 16 \
  --parallel
```

**Key Parameters**:
- `--surfaces`: Material types to enhance (wood, metal, glass, stone, fabric)
- `--strength`: Enhancement intensity (0.0-1.0, default: 0.7)
- `--preserve-highlights`: Protect specular highlights
- `--enhance-micro-contrast`: Boost surface texture detail
- `--batch-size`: Images per batch for parallel processing
- `--parallel`: Enable multi-core processing

**Material Detection**:
The system automatically detects material types and applies physics-based enhancements:
- **Wood**: Grain enhancement, warmth adjustment, micro-contrast
- **Metal**: Specular preservation, reflection sharpening
- **Glass**: Transparency, refraction, caustics
- **Stone**: Texture emphasis, porosity simulation
- **Fabric**: Thread detail, softness, drape

**Performance**: ~50-100ms per image, 400-600 images/hour batch

---

### Video Master Grader

**Purpose**: Professional color grading for video with HDR tone mapping and LUT application.

**Basic Usage**:
```bash
python luxury_video_master_grader.py \
  --input video.mp4 \
  --output output/graded.mp4 \
  --preset signature_estate
```

**Advanced Usage**:
```bash
python luxury_video_master_grader.py \
  --input video.mov \
  --output output/graded.mov \
  --preset golden_hour_courtyard \
  --lut assets/luts/film_emulation/Kodak_2393.cube \
  --lut-strength 0.75 \
  --exposure 0.2 \
  --contrast 1.08 \
  --saturation 1.05 \
  --codec prores_422_hq \
  --preserve-hdr \
  --dry-run
```

**Key Parameters**:
- `--preset`: Pre-configured grading preset
- `--lut`: Color grading LUT (.cube format)
- `--lut-strength`: LUT opacity (0.0-1.0, default: 0.6)
- `--exposure`: Exposure adjustment in stops (-2.0 to +2.0)
- `--contrast`: Contrast multiplier (0.5-2.0)
- `--saturation`: Saturation multiplier (0.0-2.0)
- `--codec`: Output codec (prores_422_hq, h264, h265)
- `--preserve-hdr`: Maintain HDR metadata (PQ, HLG)
- `--dry-run`: Preview FFmpeg command without executing

**HDR Handling**:
- Automatic HDR detection (PQ, HLG transfer functions)
- Configurable tone mapping operators (Hable, Reinhard, Mobius)
- Color metadata preservation (color_primaries, color_trc, colorspace)
- ACES ODT transforms for broadcast compliance

**Available Presets**:
- `signature_estate`: Warm, inviting, luxury aesthetic
- `golden_hour_courtyard`: Sunset/golden hour look
- `modern_minimal`: Clean, contemporary, high-contrast
- `coastal_breeze`: Cool, airy, California coastal

**Output Codecs**:
- `prores_422_hq`: Professional master (recommended)
- `h264`: Web delivery
- `h265`: 4K+ delivery with efficient compression

**Performance**: Real-time to 2x real-time on M4 Max

---

### TIFF Batch Processor

**Purpose**: High-fidelity 16-bit TIFF batch processing with metadata preservation.

**Basic Usage**:
```bash
python luxury_tiff_batch_processor.py \
  --input-dir renders/ \
  --output-dir output/ \
  --preset architectural_signature
```

**Advanced Usage**:
```bash
python luxury_tiff_batch_processor.py \
  --input-dir renders/ \
  --output-dir output/ \
  --preset custom \
  --exposure 0.15 \
  --contrast 1.10 \
  --saturation 1.05 \
  --clarity 0.18 \
  --glow 0.05 \
  --grain 0.012 \
  --lut assets/luts/film_emulation/Kodak_2393.cube \
  --preserve-metadata \
  --preserve-gps \
  --batch-size 32 \
  --parallel
```

**Key Parameters**:
- `--preset`: Pre-configured adjustment preset
- `--exposure`: Exposure adjustment in stops
- `--contrast`: Contrast multiplier
- `--saturation`: Saturation multiplier
- `--clarity`: Local contrast enhancement (0.0-1.0)
- `--glow`: Soft bloom effect (0.0-1.0)
- `--grain`: Film grain simulation (0.0-1.0)
- `--lut`: Color grading LUT
- `--preserve-metadata`: Maintain IPTC/XMP metadata
- `--preserve-gps`: Maintain GPS coordinates
- `--batch-size`: Images per batch
- `--parallel`: Multi-core processing

**Metadata Preservation**:
- IPTC keywords, title, description
- XMP technical metadata
- GPS coordinates (latitude, longitude, altitude)
- Color profile (ICC)
- Camera EXIF (if present)

**16-bit Precision**:
When `tifffile` is installed, maintains 16-bit precision throughout processing. Otherwise, converts to 8-bit with quality preservation.

**Performance**: 400-600 images/hour on M4 Max

---

## Analysis Commands

### Decision Decay Dashboard

**Purpose**: Monitor temporal contracts and decision annotations in codebase.

**Usage**:
```bash
python -m transformation_portal.analyzers.decision_decay_dashboard \
  --scan-path src/ \
  --output-dir reports/ \
  --format html
```

**Features**:
- Tracks `# Decision:` annotations
- Detects expired temporal contracts
- Generates HTML/JSON reports
- Shows decision distribution by category

**Output**: `reports/decision_decay_dashboard.html`

---

### System Info

**Purpose**: Display system capabilities, installed models, and configuration.

**Usage**:
```bash
python -m transformation_portal version

# Expected output:
# Transformation Portal v2.0
# Python: 3.11.5
# PyTorch: 2.1.0
# GPU: Apple M4 Max (MPS available)
# CoreML: Available
# Depth Model: Depth Anything V2 (Large)
# TIFF Support: tifffile 2023.9.26
```

**Detailed Info**:
```python
from transformation_portal.core.system_info import get_system_info

info = get_system_info()
print(f"GPU: {info['gpu']}")
print(f"Memory: {info['memory_gb']}GB")
print(f"CoreML: {info['coreml_available']}")
```

---

## Advanced Usage

### Recipe-Based Workflows

**Create Custom Recipe** (`config/recipes/my_estate.yaml`):
```yaml
name: "My Estate Recipe"
description: "Custom workflow for luxury estates"
version: "1.0"

stages:
  - depth_estimation
  - material_response
  - color_grading
  - ai_enhancement

depth:
  model: "depth_anything_v2_large"
  use_coreml: true

material_response:
  surfaces: [wood, metal, glass]
  strength: 0.75

color_grading:
  lut: "assets/luts/film_emulation/Kodak_2393.cube"
  lut_strength: 0.7
  exposure: 0.1
  contrast: 1.08

ai_enhancement:
  model: "sdxl"
  controlnet: "canny"
  strength: 0.8
  upscale: 4

output:
  format: "tiff"
  bit_depth: 16
  preserve_metadata: true
```

**Execute Recipe**:
```bash
python -m transformation_portal process \
  -i "renders/*.jpg" \
  -r config/recipes/my_estate.yaml \
  -o output/
```

---

### Batch Processing Patterns

**Parallel Processing**:
```bash
# Process 100 images in parallel
python material_response.py \
  --input-dir renders/ \
  --output-dir output/ \
  --parallel \
  --batch-size 32
```

**Dry-Run Preview**:
```bash
# Preview processing plan without execution
python -m transformation_portal process \
  -i "renders/*.jpg" \
  -r config/recipes/signature_estate.yaml \
  -o output/ \
  --dry-run
```

---

### Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Force CPU
export TRANSFORMERS_DEVICE=cpu

# Cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Logging level
export LOG_LEVEL=DEBUG
```

---

## Troubleshooting

### Import Errors

**Problem**: `ImportError: No module named 'transformation_portal'`

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Or install with extras
pip install -e ".[ml]"
```

---

### GPU/MPS Not Available

**Problem**: Processing is slow, GPU not being used.

**Check GPU**:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")
```

**Solution**:
- **Apple Silicon**: Ensure macOS 13+ for MPS support
- **NVIDIA**: Install CUDA toolkit matching PyTorch version
- **Fallback**: Set `export TRANSFORMERS_DEVICE=cpu` to acknowledge CPU usage

---

### Out of Memory

**Problem**: `RuntimeError: CUDA out of memory` or `OutOfMemoryError`

**Solution**:
```bash
# Reduce batch size
--batch-size 1

# Reduce image resolution before processing
# Downsample 4K to 2K

# For video, process shorter segments
ffmpeg -i input.mp4 -t 30 segment.mp4
```

---

### FFmpeg Not Found

**Problem**: `FileNotFoundError: ffmpeg not found`

**Solution**:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

---

### Depth Model Download

**Problem**: First run downloads large model (~1.5GB)

**Solution**: This is expected. The Depth Anything V2 model downloads automatically on first use. Subsequent runs use cached model.

**Manual Download**:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "depth-anything/Depth-Anything-V2-Large",
    trust_remote_code=True
)
```

---

### Physics Violation Warnings

**Problem**: `⚠️ Physics Violation Detected!` when using sky replacement

**Understanding**: This is a **feature, not a bug**. The system detected that your requested sky parameters conflict with the existing scene lighting (shadows).

**Solution**:
1. **Accept Auto-Correction**: Use the suggested parameters from `CorrectionSuggestion`
2. **Override**: Set `strict_physics=False` to allow the transformation anyway
3. **Re-render**: If possible, re-render the 3D scene with corrected sun position

**Example**:
```python
result, suggestion = blender.smart_render(
    source_image=image,
    sky_params=params,
    auto_correct=True,
    strict_physics=False  # Allow violation
)

if suggestion.confidence < 0.8:
    print(f"Suggestion: {suggestion.message}")
    # Use suggestion.suggested_params for corrected parameters
```

---

## Performance Tips

1. **Use CoreML on Apple Silicon**: 3-5x speedup for depth estimation
2. **Batch Processing**: Process multiple images together for I/O efficiency
3. **Parallel Processing**: Use `--parallel` for multi-core utilization
4. **LRU Caching**: Repeated processing of same images is 10-20x faster
5. **GPU Acceleration**: Ensure CUDA/MPS is available and configured

---

## Getting Help

**Documentation**:
- Main README: `README.md`
- Architecture Guide: `docs/architecture/ARCHITECTURE.md`
- SkyGAN Guide: `docs/SKYGAN_ATMOSPHERIC_RENDERING.md`
- Workflow Guide: `docs/ARCHITECTURAL_WORKFLOW.md`

**Custom Agent**:
Use the specialized Transformation Portal agent in GitHub Copilot:
```
@transformation-portal-specialist How do I optimize batch processing for 1000 images?
```

**Issues**:
Open an issue on GitHub: https://github.com/RC219805/Transformation_Portal/issues

---

## Version

**Transformation Portal v2.0** (January 2026)
- Physics-based sky replacement with auto-correction
- Enhanced material response
- Optimized batch processing
- Production-ready with comprehensive testing

---

**Last Updated**: January 28, 2026
