# Transformation Portal - Quick Start Cheat Sheet

**Get started in 5 minutes**

## Setup (One-Time)

```bash
# 1. Clone and navigate
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install (choose one)
pip install -r requirements.txt        # Core features
pip install -e ".[all]"                # Everything
pip install -e ".[ml]"                 # AI features only
pip install -e ".[tiff]"               # TIFF processing only

# 4. Verify
python -c "from depth_pipeline import ArchitecturalDepthPipeline; print('✓ Ready')"
```

---

## Most Common Tasks

### 1. Enhance Interior Render

```bash
python examples/simple_process.py interior.jpg output/
```

**Output:** `output/interior_enhanced.png` with depth-aware enhancement

---

### 2. Batch Process Photos

```bash
python luxury_tiff_batch_processor.py raw_photos/ enhanced/ --preset signature
```

**Output:** Entire folder processed with consistent look

---

### 3. AI Render Refinement

```bash
python lux_render_pipeline.py \
  --input render.jpg \
  --out ./enhanced \
  --prompt "luxury bedroom interior, natural light, hardwood floor" \
  --material-response
```

**Output:** Photorealistic AI-enhanced render

---

### 4. Color Grade Video

```bash
python luxury_video_master_grader.py input.mp4 output.mp4 --preset signature_estate
```

**Output:** Professionally color-graded video

---

## Command Quick Reference

### Depth Pipeline

```bash
# Single image (default)
python examples/simple_process.py input.jpg output/

# Batch interior
python examples/batch_process.py input_folder/ output/ --preset interior

# Batch exterior (with atmospheric effects)
python examples/batch_process.py input_folder/ output/ --preset exterior

# Custom pattern
python examples/batch_process.py renders/ out/ --pattern "*.png"
```

### Lux Render Pipeline

```bash
# Basic
python lux_render_pipeline.py --input img.jpg --out ./out \
  --prompt "luxury interior"

# With Material Response
python lux_render_pipeline.py --input img.jpg --out ./out \
  --prompt "luxury interior" --material-response --texture-boost 0.28

# With branding
python lux_render_pipeline.py --input img.jpg --out ./out \
  --prompt "luxury interior" --brand_text "Property Name" --logo logo.png

# Batch
python lux_render_pipeline.py --input 'folder/*.png' --out ./final \
  --prompt "description" --material-response
```

### TIFF Batch Processor

```bash
# Default (Signature preset)
python luxury_tiff_batch_processor.py input/ output/

# Vivid (high saturation)
python luxury_tiff_batch_processor.py input/ output/ --preset vivid

# Natural (subtle)
python luxury_tiff_batch_processor.py input/ output/ --preset natural

# Moody (dramatic)
python luxury_tiff_batch_processor.py input/ output/ --preset moody
```

### Video Master Grader

```bash
# Apply LUT
python luxury_video_master_grader.py input.mp4 output.mp4 \
  --lut 01_Film_Emulation/Kodak_2393.cube

# Apply preset
python luxury_video_master_grader.py input.mp4 output.mp4 \
  --preset signature_estate

# HDR to SDR
python luxury_video_master_grader.py hdr.mp4 sdr.mp4 --tone-map hable

# Custom frame rate
python luxury_video_master_grader.py input.mp4 output.mp4 --fps 24

# Batch directory
python luxury_video_master_grader.py --batch \
  --input-dir ./raw/ --output-dir ./graded/ --preset signature_estate
```

---

## Python API Quick Reference

### Depth Pipeline

```python
from depth_pipeline import ArchitecturalDepthPipeline

# Load and process
pipeline = ArchitecturalDepthPipeline.from_config('config/interior_preset.yaml')
result = pipeline.process_render('image.jpg')
pipeline.save_result(result, 'output/')

# Batch process
from pathlib import Path
images = list(Path('input/').glob('*.jpg'))
results = pipeline.batch_process(images, 'output/')
```

### Material Response

```python
from material_response import MaterialResponse, SurfaceType
from PIL import Image

mr = MaterialResponse()
image = Image.open('render.jpg')
result = mr.enhance(
    image,
    surfaces=[SurfaceType.WOOD, SurfaceType.GLASS],
    strength=0.7
)
result.save('enhanced.jpg', quality=95)
```

### TIFF Processor

```python
from luxury_tiff_batch_processor import TIFFBatchProcessor

processor = TIFFBatchProcessor()
processor.batch_process('input/', 'output/', preset='signature')
```

---

## Configuration Files

**Depth Pipeline:**
- `config/interior_preset.yaml` - Interior spaces (4 zones, no haze)
- `config/exterior_preset.yaml` - Exterior views (6 zones, atmospheric)
- `config/default_config.yaml` - Balanced defaults

**LUT Collections:**
- `01_Film_Emulation/` - Film stock looks (Kodak, FilmConvert)
- `02_Location_Aesthetic/` - Location-specific (Montecito, etc.)
- `03_Material_Response/` - Material enhancement LUTs

---

## Presets at a Glance

### TIFF Processing Presets

| Preset | Best For | Look |
|--------|----------|------|
| `signature` | Luxury real estate | Warm, inviting |
| `vivid` | Marketing materials | Bold, saturated |
| `natural` | Editorial | Subtle, authentic |
| `moody` | Dramatic interiors | Deep shadows |

### Video Grading Presets

| Preset | Best For | Look |
|--------|----------|------|
| `signature_estate` | Property videos | Warm, professional |
| `golden_hour_courtyard` | Exteriors | Golden, cinematic |

---

## Common Parameters

### Lux Render Pipeline

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `--steps` | 20-50 | Quality vs speed (30 = good balance) |
| `--strength` | 0.3-0.6 | How much to change (0.45 = moderate) |
| `--gs` | 7-12 | Guidance scale (7.5 = balanced) |
| `--texture-boost` | 0.2-0.4 | Material detail (0.28 = good default) |

### Depth Pipeline (YAML)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `sigma_spatial` | 1.0-5.0 | Denoising strength (3.0 = balanced) |
| `num_zones` | 3-8 | Depth stratification (4-6 typical) |
| `clarity_strength` | 0.0-1.0 | Micro-contrast (0.5 = moderate) |

---

## Troubleshooting Quick Fixes

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"Out of memory"**
```bash
# Reduce resolution
--width 768 --height 512
```

**"FFmpeg not found"**
```bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg
```

**Slow depth processing**
```yaml
# config/my_config.yaml
depth_model:
  backend: "coreml"  # Apple Silicon
  # or
  backend: "pytorch_cuda"  # NVIDIA GPU
```

---

## File Outputs

**Depth Pipeline:**
- `image_enhanced.png` - Enhanced image
- `image_depth.npy` - Raw depth data
- `image_depth_viz.png` - Depth visualization

**Lux Render:**
- `image_enhanced.png` - AI-refined render

**TIFF Processor:**
- `image.tif` - Enhanced TIFF (preserves 16-bit + metadata)

**Video Grader:**
- `video.mp4` or `.mov` - Graded video (default: ProRes 422 HQ)

---

## Performance Expectations

| Operation | Resolution | Time | Throughput |
|-----------|------------|------|------------|
| Depth Pipeline | 4K | 0.9-1.0s | 400-600/hr |
| Lux Render | 1024×768 | 45-90s | 40-80/hr |
| TIFF Processing | 4K | 2-5s | 720-1800/hr |
| Video Grading | 1080p | ~1-2x realtime | - |

*Apple M4 Max / NVIDIA RTX 4090 benchmarks*

---

## Essential Workflows

### 1. Quick Interior Enhancement
```bash
python examples/simple_process.py render.jpg output/
```

### 2. Batch Property Photos
```bash
python luxury_tiff_batch_processor.py photos/ enhanced/ --preset signature
```

### 3. AI Marketing Render
```bash
python lux_render_pipeline.py --input draft.jpg --out ./final \
  --prompt "luxury penthouse interior" --material-response \
  --brand_text "Property Name"
```

### 4. Video Tour Grading
```bash
python luxury_video_master_grader.py tour.mp4 graded.mp4 --preset signature_estate
```

---

## Getting Help

**Full Guide:** [docs/PIPELINE_OPERATIONS_GUIDE.md](PIPELINE_OPERATIONS_GUIDE.md)

**Documentation:**
- [README.md](../README.md) - Project overview
- [DEPTH_PIPELINE_README.md](../DEPTH_PIPELINE_README.md) - Depth pipeline details

**Examples:** [examples/](../examples/) - Code samples

**Issues:** [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)

---

**Quick Tip:** Start with example scripts, then customize configurations as needed!
