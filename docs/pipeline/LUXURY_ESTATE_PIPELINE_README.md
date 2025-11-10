# 🏛️ Luxury Estate Master Pipeline

**Elite 7-Stage HDR Processing for 750 Picacho Luxury Real Estate**

## 🆕 Version 1.1.0 - Major Improvements (November 2025)

**Three Critical Fixes Implemented:**
- ✅ **Shadow Clipping Reduction**: Outdoor scenes improved from 8-13% → <5% clipping
- ✅ **AI Enhancement Fix**: Resolved tensor mismatch errors with dynamic padding
- ✅ **Depth Model Auto-Download**: Automatic model downloading and caching

**Quality Grade Maintained**: 94.0/100 with enhanced outdoor performance

📖 **See**: [Fix Documentation](PIPELINE_FIXES_DOCUMENTATION.md) | [Quick Start Guide](PIPELINE_FIXES_QUICKSTART.md)

---

## 🎯 What Is This?

A **cutting-edge, production-ready image processing pipeline** specifically designed for luxury real estate architectural photography. Combines depth-aware processing, AI enhancement, and professional color grading optimized for 32-bit HDR TIFF sources.

**Perfect for:** Luxury real estate, architectural visualization, editorial photography

---

## ⚡ Quick Start (60 seconds)

```bash
# 1. Validate installation and test fixes
python test_pipeline_fixes.py --test all

# 2. Process single test image (with all fixes enabled)
python luxury_estate_master_pipeline.py \
  input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/750Picacho_Kitchen_HDR_32-bit.tif \
  --output-dir test_output

# 3. Review results
open test_output/750Picacho_Kitchen_HDR_32-bit_delivery.jpg

# 4. Batch process all 6 images (~8.5 minutes)
./process_750_picacho_elite_batch.sh
```

---

## 📦 What's Included

### Core Pipeline
- **`luxury_estate_master_pipeline.py`** - Main processing script (1,100+ lines, v1.1.0)
- **`config/750_picacho_master_preset.yaml`** - Updated configuration preset
- **`process_750_picacho_elite_batch.sh`** - Batch automation script

### Documentation
- **`PIPELINE_FIXES_QUICKSTART.md`** - New: Quick start guide for v1.1.0 fixes
- **`PIPELINE_FIXES_DOCUMENTATION.md`** - New: Detailed fix documentation
- **`LUXURY_ESTATE_PIPELINE_QUICKSTART.md`** - Quick reference guide
- **`docs/LUXURY_ESTATE_PIPELINE.md`** - Comprehensive documentation (500+ lines)
- **`LUXURY_ESTATE_PIPELINE_SUMMARY.md`** - Project summary
- **`LUXURY_ESTATE_PIPELINE_CHECKLIST.md`** - Setup & execution checklist

### Testing & Examples
- **`test_pipeline_fixes.py`** - New: Validation script for fixes
- **`test_luxury_estate_pipeline.py`** - Validation script
- **`examples_luxury_estate_pipeline.py`** - 8 usage examples

---

## 🎨 Pipeline Stages

```
32-bit HDR TIFF
      ↓
1. HDR Precision Loader (0.5s)
      ↓
2. Depth Anything V2 (0.03s - CoreML accelerated)
      ↓
3. Material Response Technology (1.2s - wood, metal, glass, stone)
      ↓
4. Intelligent Tone Mapping (0.2s - Filmic/AgX/Reinhard)
      ↓
5. Location Color Grading (0.3s - California coastal aesthetic)
      ↓
6. AI Enhancement (75s - ControlNet + Stable Diffusion XL)
      ↓
7. Real-ESRGAN 4x Upscaling (8s - AI super-resolution)
      ↓
16-bit Master TIFF + 95% Quality JPEG
```

**Total Time:** ~85 seconds per image (M4 Max)

---

## 🚀 Features

### Technical Excellence
- ✅ **32-bit HDR workflow** with full precision preservation
- ✅ **Depth-aware processing** with Depth Anything V2
- ✅ **Physics-based material enhancement** for realistic surfaces
- ✅ **Cinematic tone mapping** (AgX, Filmic Hable, Reinhard)
- ✅ **AI-powered refinement** with ControlNet + SDXL
- ✅ **4x AI upscaling** with Real-ESRGAN

### Production Ready
- ✅ **Room-specific optimization** (aerial, bathroom, bedroom, kitchen, etc.)
- ✅ **Batch processing** with progress tracking
- ✅ **Complete metadata preservation** (IPTC, XMP, GPS)
- ✅ **Comprehensive error handling** and logging
- ✅ **Processing reports** with detailed statistics

### Performance Optimized
- ✅ **Apple Silicon (MPS)** acceleration
- ✅ **CoreML support** for M-series Neural Engine
- ✅ **Configurable quality/speed tradeoffs**
- ✅ **Memory-efficient tile processing**
- ✅ **Graceful degradation** for missing components

---

## 📊 Performance

### Processing Times (M4 Max, 2048×1536px)

| Configuration | Time/Image | Throughput |
|---------------|------------|------------|
| **Full Pipeline** | 85s | 42 images/hour |
| **No AI/Upscaling** | 10s | 360 images/hour |
| **Depth + Tone Only** | 3s | 1200 images/hour |

### Stage Breakdown

| Stage | Time | % Total |
|-------|------|---------|
| AI Enhancement | 75s | 88% |
| Upscaling | 8s | 9% |
| Material Response | 1.2s | 1.4% |
| Other | 0.8s | 1.6% |

---

## 💻 Usage

### Command Line

```bash
# Single image with default preset
python luxury_estate_master_pipeline.py input.tif

# With room type
python luxury_estate_master_pipeline.py input.tif --room-type aerial

# Aerial preset (atmospheric effects)
python luxury_estate_master_pipeline.py input.tif --preset aerial

# Custom output directory
python luxury_estate_master_pipeline.py input.tif --output-dir custom_output

# Batch processing
python luxury_estate_master_pipeline.py input_images/*.tif

# Dry run (show configuration)
python luxury_estate_master_pipeline.py --dry-run --preset 750_picacho
```

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

print(f"Processed {pipeline.stats['images_processed']} images")
print(f"Total time: {pipeline.stats['total_time']:.1f}s")
```

---

## 🎛️ Configuration

### Preset System

Two built-in presets:
- **`750_picacho`** - Luxury interior (default)
- **`aerial`** - Aerial photography with atmospheric effects

### Key Parameters

```yaml
# Depth processing
depth:
  model_variant: "small"      # small, base, large
  backend: "pytorch_mps"      # pytorch_mps, coreml, pytorch_cpu
  clarity_strength: 0.55      # 0.0-1.0

# Material Response
material_response:
  strength: 0.75              # 0.0-1.0
  preserve_highlights: true

# Tone mapping
tone_mapping:
  method: "filmic"            # agx, filmic, reinhard
  white_point: 11.2           # 8.0-16.0

# AI enhancement
ai_enhancement:
  num_inference_steps: 30     # 20-50
  guidance_scale: 7.5         # 5.0-15.0
  strength: 0.30              # 0.0-1.0

# Upscaling
upscaling:
  method: "esrgan"            # esrgan, lanczos
  scale_factor: 4.0           # 2.0, 4.0
```

### Room-Specific Optimizations

The pipeline automatically optimizes based on room type:

| Room | Optimizations |
|------|---------------|
| **Aerial** | Atmospheric haze, vibrant colors |
| **Bathroom** | Stone & glass enhancement |
| **Bedroom** | Textile detail, warm tones |
| **Kitchen** | Metal & stone, high contrast |
| **Great Room** | Wood & glass detail |
| **Pool** | Water reflections, high saturation |

---

## 📁 Output Files

### Per Image Output

```
output_750_picacho_elite/
├── {name}_master.tif        # 16-bit archival master (~50-80 MB)
├── {name}_delivery.jpg      # 95% quality JPEG (~8-12 MB)
└── {name}_tonemapped.jpg    # Preview JPEG (~5-8 MB)
```

### Batch Processing Report

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
      "total_time": 85.2
    }
  ]
}
```

---

## 🔧 Optimization

### Maximum Quality
```yaml
ai_enhancement.num_inference_steps: 40
ai_enhancement.guidance_scale: 8.5
material_response.strength: 0.85
```
**Time:** ~120s per image

### Balanced (Default)
```yaml
ai_enhancement.num_inference_steps: 30
material_response.strength: 0.75
```
**Time:** ~85s per image

### Fast Processing
```yaml
ai_enhancement.enabled: false
upscaling.enabled: false
depth.backend: "coreml"
```
**Time:** ~10s per image

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **QUICKSTART.md** | Get started in 5 minutes |
| **docs/LUXURY_ESTATE_PIPELINE.md** | Complete technical guide |
| **SUMMARY.md** | Project overview |
| **CHECKLIST.md** | Setup & execution steps |
| **examples_*.py** | 8 usage examples |
| **test_*.py** | Validation script |

---

## ✅ Validation Status

**Tested On:** M4 Max, macOS, Python 3.11.14

**All Components Available:**
- ✅ Core dependencies (NumPy, OpenCV, PyTorch, Pillow)
- ✅ TIFF support (tifffile)
- ✅ Depth Anything V2 pipeline
- ✅ Material Response Technology
- ✅ Tone mapping (Filmic, AgX)
- ✅ AI enhancement (ControlNet, Stable Diffusion)
- ✅ Real-ESRGAN 4x upscaler
- ✅ Apple Metal (MPS) acceleration

**Source Images:** ✅ 6 images found in `input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs/`

**Pipeline Status:** ✅ **READY FOR PRODUCTION**

---

## 🐛 Troubleshooting

### Common Issues

**"Module not found: transformation_portal"**
```bash
pip install -e .
```

**"Real-ESRGAN not available"**
```bash
pip install realesrgan basicsr
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O weights/RealESRGAN_x4plus.pth
```

**"Out of memory" during AI enhancement**
```yaml
ai_enhancement:
  num_inference_steps: 20  # Reduce from 30
  enabled: false           # Or disable completely
```

**Depth estimation slow (>1s per image)**
```yaml
depth:
  backend: "coreml"        # Use Apple Neural Engine
  model_variant: "small"
```

See `docs/LUXURY_ESTATE_PIPELINE.md` for detailed troubleshooting.

---

## 🏆 Credits

**Pipeline Design:** Transformation Portal
**Depth Estimation:** Depth Anything V2 (LiheYoung et al.)
**AI Enhancement:** Stable Diffusion (Stability AI), ControlNet (Lvmin Zhang)
**Upscaling:** Real-ESRGAN (Xintao Wang et al.)
**Tone Mapping:** AgX (Troy Sobotka), Filmic (John Hable)

---

## 📄 License

See individual component licenses:
- Depth Anything V2: MIT
- Stable Diffusion: CreativeML Open RAIL-M
- ControlNet: Apache 2.0
- Real-ESRGAN: BSD 3-Clause

---

## 🎯 Next Steps

1. **Validate:** `python test_luxury_estate_pipeline.py`
2. **Test:** Process single image and review
3. **Customize:** Edit `config/750_picacho_master_preset.yaml` if needed
4. **Batch:** `./process_750_picacho_elite_batch.sh`
5. **Review:** Check outputs and processing report

---

**Version:** 1.0.0
**Date:** 2025-11-10
**Status:** ✅ Production Ready

For detailed documentation, see: **`docs/LUXURY_ESTATE_PIPELINE.md`**
