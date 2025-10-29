[![CI](https://github.com/RC219805/Transformation_Portal/actions/workflows/python-app.yml/badge.svg)](https://github.com/RC219805/Transformation_Portal/actions)
[![License](https://img.shields.io/badge/license-Attribution-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/build-passing-success.svg)](https://github.com/RC219805/Transformation_Portal)

# Transformation Portal

> Professional image and video processing toolkit for luxury real estate rendering, architectural visualization, and editorial post-production.

## Overview

**Transformation Portal** is a comprehensive suite of AI-powered tools and pipelines designed for high-end architectural rendering, real estate photography, and video post-production. It combines cutting-edge machine learning models, professional color grading techniques, and proprietary **Material Response** technology to transform raw renders and photographs into polished marketing visuals.

## Table of Contents

* [Features](#features)
* [Quick Start](#quick-start)
* [Installation](#installation)
* [Core Components](#core-components)
  * [Depth Pipeline](#depth-pipeline)
  * [Lux Render Pipeline](#lux-render-pipeline)
  * [Luxury TIFF Batch Processor](#luxury-tiff-batch-processor)
  * [Luxury Video Master Grader](#luxury-video-master-grader)
  * [Material Response System](#material-response-system)
  * [Board Material Aerial Enhancer](#board-material-aerial-enhancer)
* [LUT Collection](#lut-collection)
* [Developer Tools](#developer-tools)
* [Usage Examples](#usage-examples)
* [Configuration](#configuration)
* [Performance](#performance)
* [Testing](#testing)
* [License](#license)

---

## Features

### Core Capabilities

- ✅ **AI-Powered Enhancement** - Stable Diffusion XL, ControlNet, Real-ESRGAN upscaling
- ✅ **Depth-Aware Processing** - Depth Anything V2 with Apple Neural Engine optimization
- ✅ **Material Response Technology** - Physics-based surface enhancement for wood, metal, glass, textiles
- ✅ **Professional Color Grading** - 16+ LUTs with Film Emulation and Location Aesthetics
- ✅ **16-bit TIFF Support** - Metadata-preserving batch processing for high-end photography
- ✅ **HDR Production Pipeline** - ACES color space, adaptive debanding, halation effects
- ✅ **Batch Processing** - 400-600 images/hour throughput on M4 Max
- ✅ **Production-Ready** - Comprehensive test suite, CI/CD, performance profiling

### Technology Stack

| Technology | Purpose |
|------------|---------|
| **Depth Anything V2** | Monocular depth estimation (24ms @ 518px on M4 Max) |
| **Stable Diffusion XL** | AI-powered render refinement |
| **ControlNet** | Edge-preserving image-to-image translation |
| **Real-ESRGAN** | Intelligent 4x upscaling |
| **FFmpeg** | Video processing and LUT application |
| **PyTorch/CoreML** | GPU acceleration (CUDA, MPS, Apple Neural Engine) |
| **Colour Science** | Professional color space transformations |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install extras
pip install -e ".[tiff]"   # 16-bit TIFF processing
pip install -e ".[ml]"     # ML extras for AI pipelines
pip install -e ".[dev]"    # pytest, linting
pip install -e ".[all]"    # everything
```

### Verify Installation

```bash
# Test depth pipeline
python -c "from depth_pipeline import ArchitecturalDepthPipeline; print('✓ Depth Pipeline ready')"

# Test Material Response
python -c "from material_response import MaterialResponse; print('✓ Material Response ready')"

# Run test suite
make test-fast
```

### Process Your First Image

```bash
# Depth-aware enhancement
python depth_pipeline/pipeline.py --input render.jpg --output enhanced.jpg

# TIFF batch processing
python luxury_tiff_batch_processor.py input_folder/ output_folder/ --preset signature

# AI render refinement
python lux_render_pipeline.py --input bedroom.jpg --out ./enhanced --prompt "luxury bedroom interior" --material-response
```

---

## Core Components

### Depth Pipeline

State-of-the-art depth-aware image processing using **Depth Anything V2** for architectural rendering.

**Key Features:**
- Monocular depth estimation (24-65ms per image on M4 Max)
- Apple Neural Engine optimization via CoreML
- Depth-aware denoising with edge preservation
- Zone-based tone mapping (AgX, Reinhard, Filmic)
- Atmospheric effects (haze, aerial perspective)
- Depth-guided clarity enhancement
- LRU caching for 10-20x speedup in iterative workflows

**Usage:**
```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load preset configuration
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Process single image
result = pipeline.process_render('interior.jpg')
pipeline.save_result(result, 'output/')

# Batch process
image_paths = Path('input/').glob('*.jpg')
results = pipeline.batch_process(image_paths, output_dir='output/')
```

**Performance:** 855-950ms per 4K image | 400-600 images/hour batch throughput

📖 [Full Depth Pipeline Documentation](DEPTH_PIPELINE_README.md)

---

### Lux Render Pipeline

AI-powered render refinement combining ControlNet, Stable Diffusion, and intelligent upscaling.

**Capabilities:**
- Edge-preserving AI enhancement via ControlNet (Canny, Depth)
- SDXL refinement for photorealistic details
- Real-ESRGAN 4x upscaling
- Material Response finishing layer
- Brand overlay and text placement
- Batch processing with progress tracking

**Example:**
```bash
python lux_render_pipeline.py \
  --input bedroom_render.jpg \
  --out ./enhanced \
  --prompt "minimalist bedroom interior, natural daylight, oak wood floor" \
  --neg "low detail, cartoon, blurry" \
  --width 1024 --height 768 --steps 30 --strength 0.45 \
  --material-response --texture-boost 0.28 \
  --brand_text "The Veridian | Penthouse 21B" --logo ./brand/logo.png
```

**Material Response Mode:**
Activates detail boosts, contact shadowing, and volumetric tinting that reinforce wood grain, textile separation, and atmospheric haze. Requires `[ml]` extras and GPU.

---

### Luxury TIFF Batch Processor

High-end workflow for polishing large-format TIFF photography with metadata preservation.

**Features:**
- 16-bit TIFF support via `tifffile`
- Multiple processing presets (Signature, Vivid, Natural, Moody)
- Tonal and chroma refinements tuned for luxury real estate
- Non-destructive workflow
- Progress tracking and batch statistics

**Presets:**
- **Signature** - Warm, inviting aesthetic with enhanced mid-tones
- **Vivid** - High saturation and contrast for impactful marketing
- **Natural** - Subtle enhancement preserving authentic look
- **Moody** - Dramatic shadows and reduced saturation

**Usage:**
```bash
python luxury_tiff_batch_processor.py input_folder/ output_folder/ --preset signature
```

---

### Luxury Video Master Grader

FFmpeg-based video color grading with LUT application and batch processing.

**Features:**
- Apply professional LUTs to video content
- Support for multiple video formats (MP4, MOV, AVI)
- Batch processing capabilities
- Maintain source quality with configurable codecs
- Progress monitoring

**Usage:**
```bash
python luxury_video_master_grader.py input_video.mp4 output_video.mp4 --lut path/to/lut.cube
```

---

### Material Response System

Proprietary surface-aware rendering technology that analyzes and enhances how different materials interact with light.

**Concept:**
Traditional color grading applies global transforms uniformly. Material Response technology shifts to surface-aware rendering that respects highlights, midtones, and micro-contrast differently across materials (wood, metal, glass, fabric, stone).

**Implementation:**
```python
from material_response import MaterialResponse, SurfaceType

# Initialize Material Response
mr = MaterialResponse()

# Analyze and enhance image
result = mr.enhance(
    image,
    surfaces=[SurfaceType.WOOD, SurfaceType.METAL, SurfaceType.GLASS],
    strength=0.7
)
```

**Applications:**
- Reinforcing wood grain and texture
- Enhancing metal reflectivity and specularity
- Improving glass transparency and refraction
- Textile fiber separation and depth
- Contact shadowing at material boundaries

---

### Board Material Aerial Enhancer

Material-aware palette assignment for aerial photography using clustering and texture blending.

**Features:**
- K-means clustering for material segmentation
- MBAR-approved material palettes
- Texture-based enhancement
- Board aesthetic compliance
- Aerial perspective enhancement

**Usage:**
```bash
python board_material_aerial_enhancer.py aerial_image.jpg output_enhanced.jpg
```

📖 [Palette Assignment Guide](08_Documentation/Palette_Assignment_Guide.md)

---

## LUT Collection

Professional color grading LUTs for film emulation and location aesthetics.

### Film Emulation
- **Kodak 2393** - Classic print film look
- **FilmConvert Nitrate** - Modern cinematic aesthetic

### Location Aesthetic
- **Montecito Golden Hour** - Warm, coastal California light
- **Spanish Colonial Warm** - Mediterranean architectural warmth

### Material Response LUTs
Revolutionary physics-based surface enhancement LUTs that analyze material interaction with light.

**Usage:**
1. Import LUTs into DaVinci Resolve, Premiere Pro, or other color-grading software
2. Apply at **60-80% opacity** initially
3. Stack multiple LUTs for complex material interactions

**Locations:**
- `01_Film_Emulation/` - Film stock emulation LUTs
- `02_Location_Aesthetic/` - Location-specific color palettes
- `03_Material_Response/` - Material-aware enhancement LUTs

---

## Developer Tools

### Decision Decay Dashboard

Terminal dashboard for monitoring codebase philosophy, temporal contracts, and brand consistency.

```bash
python decision_decay_dashboard.py
```

**Features:**
- Codebase philosophy violation detection
- Temporal contract monitoring
- Brand color token drift analysis
- Real-time dashboard interface

### HDR Production Pipeline

Shell script orchestrating full HDR finishing with ACES tone mapping, debanding, and halation.

```bash
./hdr_production_pipeline.sh input.exr output.mp4
```

### Prophetic Orchestrator

Workflow automation and pipeline orchestration for complex multi-stage processing.

### Temporal Evolution

Tracks and analyzes how visual assets evolve through processing stages.

---

## Usage Examples

### Example 1: Interior Rendering Enhancement

```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load interior preset (4 depth zones, no atmospheric effects)
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Process interior render
result = pipeline.process_render('living_room.jpg')

# Save with depth visualization
pipeline.save_result(result, 'output/', save_depth=True, save_visualization=True)
```

**Output:**
- `living_room_enhanced.png` - Depth-aware enhanced image
- `living_room_depth.npy` - Raw depth map
- `living_room_depth_viz.png` - Colorized depth visualization

---

### Example 2: Exterior with Atmospheric Effects

```python
# Load exterior preset (atmospheric haze enabled)
pipeline = ArchitecturalDepthPipeline.from_config('config/exterior_preset.yaml')

# Process with atmospheric effects
result = pipeline.process_render('facade.jpg')
pipeline.save_result(result, 'output/')
```

**Effect:** Realistic haze, aerial desaturation, depth-based color shift

---

### Example 3: AI-Powered Render Refinement

```bash
python lux_render_pipeline.py \
  --input 'drafts/*.png' \
  --out ./final \
  --prompt "luxury penthouse interior, floor-to-ceiling windows, natural daylight, hardwood floors" \
  --neg "low detail, cartoon, blurry, oversaturated" \
  --width 1024 --height 768 \
  --steps 30 --strength 0.45 --gs 7.5 \
  --material-response \
  --brand_text "Marina Heights | Residence 42A"
```

---

### Example 4: Batch TIFF Processing

```bash
# Process folder of high-res TIFFs with Signature preset
python luxury_tiff_batch_processor.py \
  ./source_photography/ \
  ./output_enhanced/ \
  --preset signature
```

---

### Example 5: Custom Processing Pipeline

```python
from depth_pipeline import DepthAnythingV2Model, ModelVariant
from depth_pipeline.processors import DepthAwareDenoise, ZoneToneMapping, DepthGuidedFilters
from depth_pipeline.utils import load_image, save_image

# Initialize components
model = DepthAnythingV2Model(variant=ModelVariant.SMALL, backend="pytorch_mps")
denoiser = DepthAwareDenoise(sigma_spatial=2.5, edge_threshold=0.05)
tone_mapper = ZoneToneMapping(num_zones=4, method='agx')
filters = DepthGuidedFilters(clarity_strength=0.6)

# Load and process
image = load_image('render.jpg', normalize=True)
depth_result = model.estimate_depth(image)
depth = depth_result['depth']

# Apply custom processing chain
result = denoiser(image, depth)
result = tone_mapper(result, depth)
result = filters(result, depth)

save_image(result, 'custom_enhanced.png')
```

---

## Configuration

### Depth Pipeline Configuration

Configuration files are YAML-based and stored in `config/`:

```yaml
# config/interior_preset.yaml
depth_model:
  variant: "small"           # small | base | large
  backend: "pytorch_mps"     # pytorch_cpu | pytorch_mps | coreml
  cache_size: 100

processing:
  depth_aware_denoise:
    enabled: true
    sigma_spatial: 3.0
    edge_threshold: 0.05
    preserve_strength: 0.8

  zone_tone_mapping:
    enabled: true
    num_zones: 4
    method: "agx"            # agx | reinhard | filmic
    zone_params:
      - {contrast: 1.2, saturation: 1.1}  # Foreground
      - {contrast: 1.0, saturation: 1.0}  # Midground
      - {contrast: 0.9, saturation: 0.9}  # Background
      - {contrast: 0.8, saturation: 0.85} # Far

  atmospheric_effects:
    enabled: false           # Set true for exteriors

  depth_guided_filters:
    enabled: true
    clarity_strength: 0.5
```

### M4 Max Optimization

```yaml
# config/m4_max_optimized.yaml
depth_model:
  backend: "coreml"       # Use Apple Neural Engine
  precision: "fp16"

optimization:
  preview_resolution: 512     # 24ms depth estimation
  production_resolution: 1024 # 65ms depth estimation
  batch_size: 4
  memory_limit_gb: 27
```

---

## Performance

### Benchmarks (Apple M4 Max, 36GB RAM)

| Operation | Resolution | Time | Throughput |
|-----------|-----------|------|------------|
| Depth Estimation (ANE) | 518×518 | 24ms | - |
| Depth Estimation (ANE) | 1024×1024 | 65ms | - |
| Full Depth Pipeline | 4K | 855-950ms | 400-600 img/hr |
| AI Render Refinement | 1024×768 | 45-90s | 40-80 img/hr |
| TIFF Batch Processing | 16-bit 4K | 2-5s | 720-1800 img/hr |

### Performance Optimization Tips

1. **Use CoreML backend** on Apple Silicon (3-4x faster than PyTorch MPS)
2. **Enable disk cache** for iterative parameter tuning (10-20x speedup)
3. **Batch process** for maximum throughput
4. **Scale resolution** based on use case:
   - Preview: 512px → 24ms depth estimation
   - Production: 1024px → 65ms depth estimation
5. **Monitor memory** with `psutil` for large batches

---

## Testing

### Run Test Suite

```bash
# Fast tests only
make test-fast

# Full test suite
make test-full

# CI simulation (linting + tests)
make ci

# Specific test file
pytest tests/test_depth_pipeline.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

The project includes comprehensive tests for:
- Depth pipeline processing
- Material Response algorithms
- TIFF batch processing
- Video grading workflows
- LUT application
- Color science utilities
- Edge cases and error handling

---

## Developer Setup

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

Includes:
- `pytest` - Testing framework
- `flake8` - Linting
- `pylint` - Static analysis
- `pytest-cov` - Coverage reporting

### Console Scripts

After installation, the following command-line tools are available:

- `luxury_tiff_batch_processor.py` - Batch TIFF processing
- `luxury_video_master_grader.py` - Video color grading
- `lux_render_pipeline.py` - AI render refinement
- `decision_decay_dashboard.py` - Codebase auditing
- `board_material_aerial_enhancer.py` - Aerial enhancement

### Code Quality

```bash
# Linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Pylint
pylint *.py --max-line-length=127

# Format check
make lint
```

---

## Continuous Integration

GitHub Actions workflows automatically:
- Run tests on Python 3.10, 3.11, 3.12
- Perform linting (flake8, pylint)
- Generate Montecito Manifest
- Upload build artifacts

**Workflows:**
- `python-app.yml` - Main CI pipeline
- `pylint.yml` - Static analysis
- `summary.yml` - Issue summarization
- `codeql.yml` - Security scanning

---

## Project Structure

```
Transformation_Portal/
├── depth_pipeline/              # Depth Anything V2 processing
│   ├── models/                  # Depth estimation models
│   ├── processors/              # Image processors
│   └── utils/                   # Utilities and caching
├── luxury_tiff_batch_processor/ # TIFF processing module
├── presence_security_v1_2/      # Security and watermarking
├── tools/                       # Editorial and pipeline tools
├── tests/                       # Comprehensive test suite
├── config/                      # YAML configurations
├── 01_Film_Emulation/          # Film emulation LUTs
├── 02_Location_Aesthetic/      # Location LUTs
├── 03_Material_Response/       # Material Response LUTs
├── 08_Documentation/           # Guides and documentation
├── 09_Client_Deliverables/     # Client-specific projects
├── examples/                    # Usage examples
└── textures/                    # Material textures
```

---

## Requirements

### System Requirements
- **Python:** 3.10+
- **OS:** macOS (M1/M2/M3/M4), Linux, Windows
- **RAM:** 16GB minimum, 36GB+ recommended for batch processing
- **GPU:** CUDA-capable GPU or Apple Silicon with Neural Engine

### Python Dependencies

**Core:**
- `numpy>=1.24,<3`
- `Pillow>=10.0.0,<12`
- `scipy>=1.10,<2`
- `torch>=2.0,<3`
- `typer>=0.12,<1`

**ML Pipeline:**
- `diffusers>=0.20,<1`
- `transformers>=4.35.0`
- `controlnet-aux>=0.0.6`
- `realesrgan>=0.3.0`
- `torchvision>=0.15.0`
- `opencv-python>=4.8.0`

**TIFF Processing:**
- `tifffile>=2023.7.18`
- `imagecodecs>=2023.1.23`

See `requirements.txt` for complete dependency list.

---

## License

Professional use permitted with attribution.

**Component Licenses:**
- **Pipeline code:** Proprietary with attribution requirements
- **Depth Anything V2 Small:** Apache 2.0 License
- **Depth Anything V2 Base/Large:** CC-BY-NC-4.0 (non-commercial)
- **LUT Collection:** Attribution required

---

## Citation

If you use Depth Anything V2 in research, please cite:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

---

## Support and Contact

**Author:** Richard Cheetham
**Brand:** Carolwood Estates · RACLuxe Division
**Email:** [info@racluxe.com](mailto:info@racluxe.com)

**Resources:**
- GitHub Issues: [Report issues](https://github.com/RC219805/Transformation_Portal/issues)
- Documentation: See inline code documentation and `08_Documentation/`
- Examples: Check `examples/` directory

---

## Acknowledgments

- **Depth Anything V2** by LiheYoung and contributors
- **Stable Diffusion** by Stability AI
- **HuggingFace** for model hosting and diffusers library
- **Apple** for CoreML and Neural Engine optimization
- **PyTorch** for deep learning framework

---

**Status:** Production-Ready (v0.1.0)
**Last Updated:** 2025-10-29
