# Transformation Portal - Pipeline Operations Guide

**Complete step-by-step guide to operating the Transformation Portal pipelines**

## Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Depth Pipeline Operations](#depth-pipeline-operations)
3. [Lux Render Pipeline Operations](#lux-render-pipeline-operations)
4. [TIFF Batch Processor Operations](#tiff-batch-processor-operations)
5. [Video Master Grader Operations](#video-master-grader-operations)
6. [Material Response Operations](#material-response-operations)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)
9. [Quick Reference](#quick-reference)

---

## Prerequisites & Setup

### System Requirements

**Minimum:**
- Python 3.10 or higher
- 16GB RAM
- 10GB free disk space

**Recommended:**
- Python 3.10+
- 36GB+ RAM (for batch processing)
- Apple Silicon M1/M2/M3/M4 (for Neural Engine optimization)
- CUDA-capable GPU (for AI pipelines)

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install optional features (choose what you need)
pip install -e ".[tiff]"   # For 16-bit TIFF processing
pip install -e ".[ml]"     # For AI/ML pipelines (requires GPU/MPS)
pip install -e ".[dev]"    # For development and testing
pip install -e ".[all]"    # Install everything

# 5. Verify installation
python -c "from depth_pipeline import ArchitecturalDepthPipeline; print('✓ Ready')"
```

### FFmpeg Installation (Required for Video Processing)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Verify FFmpeg:
```bash
ffmpeg -version
```

---

## Depth Pipeline Operations

### What is the Depth Pipeline?

The Depth Pipeline uses **Depth Anything V2** to add depth understanding to architectural renders, enabling:
- Depth-aware denoising (sharp edges, smooth flat areas)
- Zone-based tone mapping (independent control for foreground/midground/background)
- Atmospheric effects (realistic haze and aerial perspective)
- Depth-guided clarity enhancement

**Performance:** 855-950ms per 4K image on M4 Max | 400-600 images/hour batch throughput

### Basic Operations

#### 1. Process Single Image (Default Settings)

```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load default configuration
pipeline = ArchitecturalDepthPipeline.from_config('config/default_config.yaml')

# Process image
result = pipeline.process_render('input/bedroom.jpg')

# Save enhanced image
pipeline.save_result(result, 'output/')
```

**Output files:**
- `bedroom_enhanced.png` - Enhanced image
- `bedroom_depth.npy` - Raw depth data
- `bedroom_depth_viz.png` - Depth visualization (colorized)

#### 2. Process with Interior Preset

```python
# Load interior-optimized preset (4 depth zones, no atmospheric haze)
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

result = pipeline.process_render('input/living_room.jpg')
pipeline.save_result(result, 'output/', save_depth=True, save_visualization=True)
```

**Interior preset includes:**
- 4 depth zones for layered control
- Edge-preserving denoising (σ=3.0)
- AgX tone mapping for natural contrast
- Depth-guided clarity (strength=0.5)
- No atmospheric effects (disabled for interiors)

#### 3. Process with Exterior Preset

```python
# Load exterior-optimized preset (atmospheric effects enabled)
pipeline = ArchitecturalDepthPipeline.from_config('config/exterior_preset.yaml')

result = pipeline.process_render('input/facade.jpg')
pipeline.save_result(result, 'output/')
```

**Exterior preset adds:**
- Atmospheric haze simulation
- Aerial desaturation (distance-based)
- Color shift for depth perception
- 6 depth zones for expansive scenes

#### 4. Batch Processing Multiple Images

```python
from pathlib import Path

# Load configuration
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Get all JPG images
image_paths = list(Path('input/').glob('*.jpg'))

# Process batch with progress tracking
results = pipeline.batch_process(
    image_paths,
    output_dir='output/',
    save_depth=True,
    save_visualization=True
)

# Print statistics
stats = pipeline.get_stats()
print(f"Processed: {stats['images_processed']} images")
print(f"Total time: {stats['total_time']:.2f}s")
print(f"Cache hits: {stats['cache_stats']['hit_rate']:.2%}")
```

#### 5. Command-Line Usage

```bash
# Using the provided example script
python examples/simple_process.py input/render.jpg output/

# Batch processing with preset selection
python examples/batch_process.py input_folder/ output_folder/ --preset interior

# Custom pattern matching
python examples/batch_process.py renders/ enhanced/ --preset exterior --pattern "*.png"

# Skip depth map saves for faster processing
python examples/batch_process.py input/ output/ --no-depth --no-viz
```

### Custom Configuration

Create your own YAML configuration file:

```yaml
# config/my_custom_preset.yaml
depth_model:
  variant: "small"           # small (49MB) | base (195MB) | large (671MB)
  backend: "pytorch_mps"     # pytorch_cpu | pytorch_mps | coreml
  cache_size: 100            # LRU cache size (speeds up iterations)

processing:
  depth_aware_denoise:
    enabled: true
    sigma_spatial: 2.5       # Lower = sharper, Higher = smoother
    edge_threshold: 0.05     # Edge detection sensitivity
    preserve_strength: 0.8   # Edge preservation (0-1)

  zone_tone_mapping:
    enabled: true
    num_zones: 4             # Number of depth zones (3-8)
    method: "agx"            # agx | reinhard | filmic
    zone_params:
      - {contrast: 1.2, saturation: 1.1}  # Foreground
      - {contrast: 1.0, saturation: 1.0}  # Midground
      - {contrast: 0.9, saturation: 0.9}  # Background
      - {contrast: 0.8, saturation: 0.85} # Far background

  atmospheric_effects:
    enabled: false           # Enable for exteriors
    haze_intensity: 0.3      # Atmospheric haze (0-1)
    haze_color: [0.8, 0.85, 0.9]  # RGB color tint
    distance_threshold: 0.5  # Where haze begins (0-1)

  depth_guided_filters:
    enabled: true
    clarity_strength: 0.5    # Micro-contrast boost (0-1)
    glow_strength: 0.0       # Soft glow effect (0-1)
```

Then use it:
```python
pipeline = ArchitecturalDepthPipeline.from_config('config/my_custom_preset.yaml')
```

### Performance Optimization

**For Apple Silicon (M1/M2/M3/M4):**
```yaml
depth_model:
  backend: "coreml"       # Uses Apple Neural Engine (3-5x faster)
  precision: "fp16"       # Half precision for speed
```

**For CUDA GPUs:**
```yaml
depth_model:
  backend: "pytorch_cuda"
  device: "cuda:0"
```

**For CPU-only systems:**
```yaml
depth_model:
  backend: "pytorch_cpu"
  variant: "small"        # Use smallest model for speed
```

---

## Lux Render Pipeline Operations

### What is Lux Render?

AI-powered render refinement that combines:
- **ControlNet** for edge-preserving enhancement
- **Stable Diffusion XL** for photorealistic detail
- **Real-ESRGAN** for intelligent 4x upscaling
- **Material Response** for surface realism

**Performance:** 45-90 seconds per 1024×768 image (GPU-dependent)

### Basic Operations

#### 1. Simple AI Enhancement

```bash
python lux_render_pipeline.py \
  --input render.jpg \
  --out ./enhanced \
  --prompt "luxury bedroom interior, natural daylight, hardwood floor"
```

**What happens:**
1. Loads render.jpg
2. Applies ControlNet-guided SDXL refinement
3. Saves enhanced image to `./enhanced/render_enhanced.png`

#### 2. Full Enhancement with Material Response

```bash
python lux_render_pipeline.py \
  --input bedroom.jpg \
  --out ./final \
  --prompt "minimalist bedroom, floor-to-ceiling windows, oak flooring" \
  --neg "low detail, cartoon, blurry, oversaturated" \
  --width 1024 --height 768 \
  --steps 30 \
  --strength 0.45 \
  --gs 7.5 \
  --material-response \
  --texture-boost 0.28
```

**Parameters explained:**
- `--prompt`: Describe desired aesthetic (be specific)
- `--neg`: What to avoid (quality issues, unwanted styles)
- `--width/height`: Output resolution (multiples of 64)
- `--steps`: Quality vs speed (20=fast, 50=high quality)
- `--strength`: How much to change (0.3=subtle, 0.6=significant)
- `--gs`: Guidance scale (7-9=balanced, 12+=very strict)
- `--material-response`: Enable surface-aware enhancement
- `--texture-boost`: Material detail intensity (0.2-0.4)

#### 3. Batch Processing with Branding

```bash
python lux_render_pipeline.py \
  --input 'drafts/*.png' \
  --out ./final \
  --prompt "luxury penthouse interior, modern minimalist" \
  --steps 30 --strength 0.45 \
  --material-response \
  --brand_text "Marina Heights | Residence 42A" \
  --logo ./brand/logo.png
```

**Brand overlay features:**
- Text placement (configurable position)
- Logo overlay with transparency
- Consistent branding across batch

#### 4. Upscaling with AI Refinement

```bash
python lux_render_pipeline.py \
  --input low_res.jpg \
  --out ./upscaled \
  --prompt "architectural exterior, detailed facade" \
  --width 2048 --height 1536 \
  --upscale \
  --steps 40 \
  --strength 0.35
```

**Upscaling workflow:**
1. Real-ESRGAN 4x upscaling
2. SDXL refinement for detail restoration
3. Material Response finishing

### Python API Usage

```python
from lux_render_pipeline import LuxRenderPipeline

# Initialize pipeline (loads models - takes 30-60 seconds)
pipeline = LuxRenderPipeline(
    device="cuda",  # or "mps" for Apple Silicon, "cpu" for CPU-only
    use_xformers=True  # Enable memory-efficient attention
)

# Process single image
result = pipeline.enhance(
    input_path="render.jpg",
    output_dir="./enhanced",
    prompt="luxury interior, natural light, hardwood floors",
    negative_prompt="low quality, blurry, oversaturated",
    width=1024,
    height=768,
    num_steps=30,
    strength=0.45,
    guidance_scale=7.5,
    enable_material_response=True,
    texture_boost=0.28
)

print(f"Enhanced image saved to: {result['output_path']}")
print(f"Processing time: {result['processing_time']:.2f}s")
```

### Common Prompt Patterns

**Interior spaces:**
```
"minimalist [room type], floor-to-ceiling windows, natural daylight, 
[material] flooring, clean lines, professional architectural photography"
```

**Exterior facades:**
```
"modern [building type] exterior, detailed facade, golden hour lighting, 
[material] cladding, professional architectural photography"
```

**Negative prompts (always include):**
```
"low detail, cartoon, blurry, oversaturated, low quality, artifacts, 
distorted, amateur"
```

---

## TIFF Batch Processor Operations

### What is the TIFF Batch Processor?

Professional-grade batch processing for high-end photography with:
- 16-bit TIFF preservation
- Metadata retention (IPTC, XMP, GPS)
- Multiple enhancement presets
- Non-destructive workflow

**Performance:** 2-5 seconds per 4K TIFF | 720-1800 images/hour

### Available Presets

| Preset | Use Case | Characteristics |
|--------|----------|-----------------|
| **Signature** | Luxury real estate | Warm tones, enhanced midtones, inviting |
| **Vivid** | Marketing materials | High saturation, bold contrast |
| **Natural** | Editorial photography | Subtle enhancement, authentic look |
| **Moody** | Dramatic interiors | Deep shadows, muted tones |

### Basic Operations

#### 1. Process with Default Preset

```bash
python luxury_tiff_batch_processor.py input_folder/ output_folder/
```

Uses "Signature" preset by default.

#### 2. Process with Specific Preset

```bash
# Vivid preset for marketing
python luxury_tiff_batch_processor.py \
  ./raw_photos/ \
  ./enhanced_photos/ \
  --preset vivid

# Natural preset for editorial
python luxury_tiff_batch_processor.py \
  ./editorial/ \
  ./output/ \
  --preset natural

# Moody preset for dramatic effect
python luxury_tiff_batch_processor.py \
  ./interiors/ \
  ./moody_output/ \
  --preset moody
```

#### 3. Process with Progress Tracking

```bash
python luxury_tiff_batch_processor.py \
  ./source/ \
  ./processed/ \
  --preset signature \
  --verbose
```

**Output shows:**
```
Processing: image_001.tif... ✓ (2.3s)
Processing: image_002.tif... ✓ (2.1s)
...
Batch complete: 150 images in 347s (25.6 images/minute)
```

### Python API Usage

```python
from luxury_tiff_batch_processor import TIFFBatchProcessor, PresetConfig

# Initialize processor
processor = TIFFBatchProcessor()

# Process with preset
processor.batch_process(
    input_dir="./raw_photos",
    output_dir="./enhanced",
    preset="signature"
)

# Custom preset
custom_preset = PresetConfig(
    name="My Custom Look",
    lut="01_Film_Emulation/Kodak_2393.cube",
    exposure=0.2,
    contrast=1.08,
    saturation=1.05,
    clarity=0.15,
    grain=0.012
)

processor.process_with_preset(
    input_dir="./raw",
    output_dir="./custom",
    preset=custom_preset
)
```

### Preset Characteristics

**Signature Preset:**
- Exposure: +0.15 EV
- Contrast: 1.08
- Saturation: 1.05
- Clarity: 0.12
- Adds subtle warmth and inviting feel

**Vivid Preset:**
- Exposure: +0.10 EV
- Contrast: 1.15
- Saturation: 1.20
- Clarity: 0.20
- Bold, impactful for marketing

**Natural Preset:**
- Exposure: 0.0 EV
- Contrast: 1.02
- Saturation: 1.00
- Clarity: 0.08
- Minimal intervention, authentic

**Moody Preset:**
- Exposure: -0.10 EV
- Contrast: 1.12
- Saturation: 0.90
- Clarity: 0.15
- Deep shadows, cinematic

---

## Video Master Grader Operations

### What is the Video Master Grader?

FFmpeg-based video color grading with:
- Professional LUT application
- HDR tone mapping
- Frame rate conformance
- Batch processing capabilities

### Basic Operations

#### 1. Apply LUT to Video

```bash
python luxury_video_master_grader.py \
  input_video.mp4 \
  output_video.mp4 \
  --lut 01_Film_Emulation/Kodak_2393.cube
```

**Default output:**
- Codec: ProRes 422 HQ (high quality master)
- Preserves source resolution and frame rate
- Maintains color metadata

#### 2. Apply Preset Look

```bash
# Signature Estate preset
python luxury_video_master_grader.py \
  property_tour.mp4 \
  graded_tour.mp4 \
  --preset signature_estate

# Golden Hour Courtyard preset
python luxury_video_master_grader.py \
  exterior.mp4 \
  golden_exterior.mp4 \
  --preset golden_hour_courtyard
```

#### 3. HDR to SDR Conversion

```bash
python luxury_video_master_grader.py \
  hdr_video.mp4 \
  sdr_output.mp4 \
  --tone-map hable \
  --lut 01_Film_Emulation/FilmConvert_Nitrate.cube
```

**Tone mapping operators:**
- `hable` - Filmic S-curve (recommended)
- `reinhard` - Classic tone mapping
- `mobius` - Smooth highlight rolloff

#### 4. Batch Process Directory

```bash
python luxury_video_master_grader.py \
  --batch \
  --input-dir ./raw_videos/ \
  --output-dir ./graded/ \
  --preset signature_estate
```

Processes all video files (.mp4, .mov, .avi) in directory.

### Advanced Options

#### Frame Rate Conformance

```bash
# Convert to 24fps cinema standard
python luxury_video_master_grader.py \
  source.mp4 \
  output.mp4 \
  --fps 24 \
  --lut 01_Film_Emulation/Kodak_2393.cube
```

#### Custom Output Codec

```bash
# H.264 for web delivery
python luxury_video_master_grader.py \
  master.mp4 \
  web_optimized.mp4 \
  --codec h264 \
  --preset signature_estate

# ProRes 422 HQ for archival
python luxury_video_master_grader.py \
  source.mp4 \
  archive.mov \
  --codec prores_hq \
  --lut 02_Location_Aesthetic/Montecito_Golden_Hour.cube
```

#### Dry Run (Preview FFmpeg Command)

```bash
python luxury_video_master_grader.py \
  input.mp4 \
  output.mp4 \
  --lut path/to/lut.cube \
  --dry-run
```

Prints FFmpeg command without executing.

---

## Material Response Operations

### What is Material Response?

Proprietary surface-aware rendering technology that analyzes material types and applies physics-based enhancements:
- Wood grain reinforcement
- Metal reflectivity enhancement
- Glass transparency optimization
- Textile fiber separation
- Contact shadowing at material boundaries

### Supported Materials

- **WOOD** - Grain enhancement, warmth
- **METAL** - Specularity, reflectivity
- **GLASS** - Transparency, refraction
- **FABRIC** - Fiber separation, depth
- **STONE** - Texture, mass
- **LEATHER** - Micro-texture, luxury feel

### Python API Usage

```python
from material_response import MaterialResponse, SurfaceType
from PIL import Image

# Initialize Material Response
mr = MaterialResponse()

# Load image
image = Image.open('interior.jpg')

# Automatic material detection and enhancement
result = mr.enhance(
    image,
    surfaces=[SurfaceType.WOOD, SurfaceType.METAL, SurfaceType.GLASS],
    strength=0.7,
    preserve_highlights=True
)

# Save enhanced image
result.save('enhanced.jpg', quality=95)
```

### Advanced Usage

```python
# Fine-tune per-material strengths
result = mr.enhance_with_controls(
    image,
    material_config={
        SurfaceType.WOOD: {'strength': 0.8, 'grain_boost': 0.3},
        SurfaceType.METAL: {'strength': 0.6, 'specularity': 0.5},
        SurfaceType.GLASS: {'strength': 0.5, 'clarity': 0.7}
    }
)

# Apply specific LUT stack for material interaction
result = mr.enhance_with_luts(
    image,
    lut_stack=[
        '03_Material_Response/Wood_Grain_Enhancement.cube',
        '03_Material_Response/Metal_Specularity.cube'
    ],
    blend_mode='multiply',
    opacity=0.75
)
```

### Material Response with Other Pipelines

**With Depth Pipeline:**
```python
from depth_pipeline import ArchitecturalDepthPipeline
from material_response import MaterialResponse

# Process with depth awareness
depth_pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')
depth_result = depth_pipeline.process_render('render.jpg')

# Apply material response to depth-enhanced image
mr = MaterialResponse()
final = mr.enhance(
    depth_result['enhanced_image'],
    surfaces=[SurfaceType.WOOD, SurfaceType.GLASS],
    strength=0.7
)
```

**With Lux Render:**
```bash
# Material Response is built into Lux Render
python lux_render_pipeline.py \
  --input render.jpg \
  --out ./enhanced \
  --prompt "luxury interior" \
  --material-response \
  --texture-boost 0.28
```

---

## Common Workflows

### Workflow 1: Interior Render Enhancement (Standard)

**Goal:** Polish architectural interior render for portfolio

```bash
# Step 1: Depth-aware enhancement
python examples/simple_process.py render.jpg ./step1/ --preset interior

# Step 2: Apply Material Response (optional)
# If Material Response Python API is available:
python -c "
from material_response import MaterialResponse, SurfaceType
from PIL import Image
mr = MaterialResponse()
img = Image.open('./step1/render_enhanced.png')
result = mr.enhance(img, surfaces=[SurfaceType.WOOD, SurfaceType.GLASS], strength=0.7)
result.save('./final/render_final.jpg', quality=95)
"
```

**Output:** Professional-grade render with depth awareness and material enhancement.

### Workflow 2: Luxury Real Estate Photography Batch

**Goal:** Process 100+ property photos with consistent look

```bash
# Process entire folder with Signature preset
python luxury_tiff_batch_processor.py \
  ./raw_photos/ \
  ./client_delivery/ \
  --preset signature \
  --verbose

# Expected time: ~5-10 minutes for 100 images
```

**Output:** Consistently graded photos ready for MLS and marketing.

### Workflow 3: AI-Enhanced Marketing Render

**Goal:** Create photorealistic marketing image from draft render

```bash
# Full AI enhancement pipeline
python lux_render_pipeline.py \
  --input draft_render.jpg \
  --out ./marketing \
  --prompt "luxury penthouse interior, floor-to-ceiling windows, 
           natural daylight, hardwood floors, modern furniture, 
           professional architectural photography" \
  --neg "low quality, blurry, cartoon, oversaturated, amateur" \
  --width 1920 --height 1080 \
  --steps 40 \
  --strength 0.45 \
  --gs 7.5 \
  --material-response \
  --texture-boost 0.30 \
  --brand_text "Skyline Tower | Penthouse 42" \
  --logo ./brand/logo.png

# Expected time: 60-120 seconds (GPU-dependent)
```

**Output:** Photorealistic render with branding, ready for marketing.

### Workflow 4: Video Property Tour Color Grading

**Goal:** Apply cinematic color grade to property walkthrough video

```bash
# Step 1: Grade with preset
python luxury_video_master_grader.py \
  property_tour_raw.mp4 \
  property_tour_graded.mp4 \
  --preset signature_estate

# Step 2 (optional): Apply custom LUT for brand consistency
python luxury_video_master_grader.py \
  property_tour_graded.mp4 \
  property_tour_final.mp4 \
  --lut 02_Location_Aesthetic/Montecito_Golden_Hour.cube \
  --codec prores_hq

# Expected time: Depends on video length (~1-2x real-time)
```

**Output:** Professionally graded video with consistent aesthetic.

### Workflow 5: Exterior with Atmospheric Effects

**Goal:** Add realistic atmospheric depth to exterior facade

```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load exterior preset (atmospheric effects enabled)
pipeline = ArchitecturalDepthPipeline.from_config('config/exterior_preset.yaml')

# Process facade
result = pipeline.process_render('facade.jpg')

# Save with depth visualization for review
pipeline.save_result(
    result, 
    'output/', 
    save_depth=True, 
    save_visualization=True
)
```

**Output:** Exterior with realistic haze, aerial perspective, and depth.

### Workflow 6: Production Pipeline (Batch + Quality Control)

**Goal:** Process 500+ renders overnight with automated quality control

```python
from pathlib import Path
from depth_pipeline import ArchitecturalDepthPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load pipeline
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')

# Get all renders
all_renders = list(Path('batch_input/').rglob('*.jpg'))
print(f"Found {len(all_renders)} renders to process")

# Process in batches
batch_size = 50
for i in range(0, len(all_renders), batch_size):
    batch = all_renders[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}/{len(all_renders)//batch_size + 1}")
    
    results = pipeline.batch_process(
        batch,
        output_dir='batch_output/',
        save_depth=False,  # Skip depth saves for speed
        save_visualization=False
    )
    
    # Quality check (ensure no errors)
    failed = [r for r in results if r.get('error')]
    if failed:
        print(f"WARNING: {len(failed)} images failed")
        for f in failed:
            print(f"  - {f['input_path']}: {f['error']}")

# Print final stats
stats = pipeline.get_stats()
print(f"\n=== Production Run Complete ===")
print(f"Total images: {stats['images_processed']}")
print(f"Total time: {stats['total_time']/3600:.1f} hours")
print(f"Throughput: {stats['images_processed']/(stats['total_time']/3600):.0f} images/hour")
```

**Output:** Batch-processed renders with quality control and performance metrics.

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'numpy'"

**Problem:** Dependencies not installed.

**Solution:**
```bash
pip install -r requirements.txt
# Or for specific features:
pip install -e ".[ml]"  # For AI features
pip install -e ".[tiff]"  # For TIFF processing
```

#### 2. "Out of memory" during AI processing

**Problem:** GPU/RAM insufficient for model size.

**Solutions:**
```bash
# Use smaller resolution
python lux_render_pipeline.py --input img.jpg --width 768 --height 512

# Reduce batch size (if using Python API)
# In code: batch_size=1

# Enable xformers (memory-efficient attention)
python lux_render_pipeline.py --input img.jpg --use-xformers
```

#### 3. Depth pipeline is slow

**Problem:** Not using optimal backend for hardware.

**Solutions:**

**For Apple Silicon:**
```yaml
# config/my_config.yaml
depth_model:
  backend: "coreml"  # Use Apple Neural Engine
```

**For CUDA GPU:**
```yaml
depth_model:
  backend: "pytorch_cuda"
  device: "cuda:0"
```

**For CPU (minimal):**
```yaml
depth_model:
  backend: "pytorch_cpu"
  variant: "small"  # Use smallest model
```

#### 4. FFmpeg not found

**Problem:** FFmpeg not installed or not in PATH.

**Solution:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

#### 5. LUT not applied to video

**Problem:** LUT file path incorrect or format unsupported.

**Solution:**
```bash
# Verify LUT exists
ls -la 01_Film_Emulation/Kodak_2393.cube

# Use absolute path
python luxury_video_master_grader.py \
  input.mp4 output.mp4 \
  --lut /absolute/path/to/lut.cube

# Verify LUT format (must be .cube format)
file 01_Film_Emulation/Kodak_2393.cube
```

#### 6. "CUDA out of memory" error

**Problem:** GPU VRAM insufficient.

**Solutions:**
```bash
# Use CPU instead (slower but works)
export CUDA_VISIBLE_DEVICES=""
python lux_render_pipeline.py --input img.jpg ...

# Or use MPS on Apple Silicon
python lux_render_pipeline.py --device mps ...

# Or reduce image dimensions
python lux_render_pipeline.py --width 512 --height 512 ...
```

#### 7. TIFF processing loses metadata

**Problem:** Need tifffile library for full metadata preservation.

**Solution:**
```bash
pip install -e ".[tiff]"  # Installs tifffile and imagecodecs
```

#### 8. Depth visualization looks strange

**Problem:** Depth estimation model not downloaded or corrupted.

**Solution:**
```python
# Force re-download
from depth_pipeline.models import DepthAnythingV2Model
model = DepthAnythingV2Model(variant='small', force_download=True)
```

### Performance Tips

1. **Use appropriate model size:**
   - Small (49MB): Fast, good quality for most uses
   - Base (195MB): Better quality, slower
   - Large (671MB): Best quality, requires significant GPU/MPS

2. **Enable caching for iterative work:**
   ```yaml
   depth_model:
     cache_size: 100  # Stores recent depth estimates
   ```

3. **Batch processing is faster:**
   - Processes multiple images ~30% faster than sequential
   - Better GPU utilization

4. **Skip unnecessary outputs:**
   ```python
   # Skip depth saves for production speed
   pipeline.batch_process(paths, output_dir, save_depth=False, save_visualization=False)
   ```

5. **Use SSD for I/O:**
   - Image loading/saving is significant bottleneck
   - SSD vs HDD can double throughput

---

## Quick Reference

### Command Cheat Sheet

```bash
# DEPTH PIPELINE
# Single image with default settings
python examples/simple_process.py input.jpg output/

# Batch with interior preset
python examples/batch_process.py input_folder/ output/ --preset interior

# Batch with exterior preset
python examples/batch_process.py input_folder/ output/ --preset exterior


# LUX RENDER PIPELINE
# Basic AI enhancement
python lux_render_pipeline.py --input img.jpg --out ./out --prompt "luxury interior"

# Full enhancement with Material Response
python lux_render_pipeline.py --input img.jpg --out ./out \
  --prompt "description" --material-response --texture-boost 0.28

# Batch with branding
python lux_render_pipeline.py --input 'folder/*.png' --out ./final \
  --prompt "description" --brand_text "Text" --logo logo.png


# TIFF BATCH PROCESSOR
# Default (Signature preset)
python luxury_tiff_batch_processor.py input/ output/

# Specific preset
python luxury_tiff_batch_processor.py input/ output/ --preset vivid


# VIDEO MASTER GRADER
# Apply LUT
python luxury_video_master_grader.py input.mp4 output.mp4 --lut path/to/lut.cube

# Apply preset
python luxury_video_master_grader.py input.mp4 output.mp4 --preset signature_estate

# HDR to SDR
python luxury_video_master_grader.py input.mp4 output.mp4 --tone-map hable
```

### Python API Quick Reference

```python
# DEPTH PIPELINE
from depth_pipeline import ArchitecturalDepthPipeline
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')
result = pipeline.process_render('image.jpg')
pipeline.save_result(result, 'output/')

# LUX RENDER (conceptual - check actual API)
from lux_render_pipeline import LuxRenderPipeline
pipeline = LuxRenderPipeline(device="cuda")
result = pipeline.enhance(
    input_path="render.jpg",
    prompt="luxury interior",
    enable_material_response=True
)

# MATERIAL RESPONSE
from material_response import MaterialResponse, SurfaceType
mr = MaterialResponse()
result = mr.enhance(image, surfaces=[SurfaceType.WOOD], strength=0.7)

# TIFF PROCESSOR
from luxury_tiff_batch_processor import TIFFBatchProcessor
processor = TIFFBatchProcessor()
processor.batch_process("input/", "output/", preset="signature")
```

### Configuration Files Quick Reference

**Depth Pipeline Presets:**
- `config/default_config.yaml` - Balanced settings
- `config/interior_preset.yaml` - 4 zones, no atmospheric effects
- `config/exterior_preset.yaml` - 6 zones, atmospheric effects enabled

**Video Grading Presets:**
- `signature_estate` - Warm, inviting for luxury properties
- `golden_hour_courtyard` - Warm, golden light

**TIFF Processing Presets:**
- `signature` - Balanced enhancement, warm tones
- `vivid` - High saturation, bold contrast
- `natural` - Subtle, authentic look
- `moody` - Dramatic shadows, muted tones

**LUT Collections:**
- `01_Film_Emulation/` - Kodak, FilmConvert LUTs
- `02_Location_Aesthetic/` - Montecito, Spanish Colonial
- `03_Material_Response/` - Surface-specific enhancement LUTs

---

## Additional Resources

- **Main README:** [README.md](../README.md) - Project overview and features
- **Depth Pipeline:** [DEPTH_PIPELINE_README.md](../DEPTH_PIPELINE_README.md) - Technical details
- **Architecture:** [docs/ARCHITECTURE.md](ARCHITECTURE.md) - System design
- **Performance:** [docs/PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) - Optimization guide
- **Examples:** [examples/](../examples/) - Code examples
- **Tests:** [tests/](../tests/) - Test suite for reference

---

## Support

**Issues:** [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)

**Contact:** info@racluxe.com

---

**Last Updated:** 2025-11-01
