# Image Processing Readiness Guide

## Quick Answer: Are We Ready to Process Images? ✅

**Yes!** The Transformation Portal can process images at multiple capability tiers depending on what you have installed.

## 🎯 Three Capability Tiers

### 📦 Tier 1: Minimal Setup (✓ Available Now)
**Requirements:** numpy, Pillow
**What you can do:**
- ✅ Image format conversion (JPG, PNG, TIFF)
- ✅ Resize and crop operations
- ✅ Basic adjustments (brightness, contrast, saturation)
- ✅ EXIF/IPTC metadata reading
- ✅ Batch processing of basic operations

**Quick Start:**
```bash
# Process an image with basic adjustments
python scripts/simple_image_processor.py input_images/my_image.jpg \
  --brightness 1.1 --contrast 1.05 --saturation 1.1

# Resize to specific dimensions
python scripts/simple_image_processor.py input_images/my_image.jpg \
  --width 1920 --height 1080 --output resized.jpg
```

### 📦 Tier 2: Standard Setup (Professional)
**Requirements:** + scipy, tifffile, imagecodecs, scikit-image
**What you can do:**
- ✅ All Minimal tier features
- ✅ LUT-based color grading
- ✅ 16-bit TIFF batch processing
- ✅ Professional metadata preservation
- ✅ Advanced color space operations
- ✅ Histogram equalization and advanced filters

**Installation:**
```bash
pip install scipy tifffile imagecodecs scikit-image opencv-python
```

**Quick Start:**
```bash
# Use luxury TIFF batch processor (when Standard tier ready)
python scripts/utilities/luxury_tiff_batch_processor.py \
  input_images/ output/ --preset signature_estate
```

### 📦 Tier 3: Full Setup (AI-Powered)
**Requirements:** + torch, diffusers, transformers, realesrgan
**What you can do:**
- ✅ All Standard tier features
- ✅ AI-powered depth estimation (Depth Anything V2)
- ✅ Stable Diffusion XL enhancement
- ✅ Real-ESRGAN 4x upscaling
- ✅ ControlNet refinement
- ✅ Material Response processing
- ✅ Context-aware rendering

**Installation:**
```bash
pip install -r requirements.txt
python scripts/setup/download_depth_models.py
```

**Quick Start:**
```bash
# AI-powered render enhancement
python scripts/pipelines/lux_render_pipeline.py input_images/render.tiff

# Context-aware processing
python scripts/context_aware_rendering.py input_images/interior.jpg

# Depth-aware processing
python scripts/depth_tools.py input_images/exterior.jpg --output depth_enhanced/
```

---

## 🚀 Getting Started Right Now

### Step 1: Check Your Current Status

```bash
# Run the readiness check
python scripts/check_image_processing_readiness.py

# Quick start guide only
python scripts/check_image_processing_readiness.py --quick-start
```

### Step 2: Get Sample Images

**Option A: Create a test image** (already done if you see `input_images/test_render.jpg`)

**Option B: Download official samples:**
```bash
python scripts/download_samples.py
```

**Option C: Use your own images:**
```bash
cp ~/Downloads/my_render.jpg input_images/
```

### Step 3: Start Processing!

**With Minimal Setup (numpy + Pillow):**
```bash
# Simple brightness/contrast adjustment
python scripts/simple_image_processor.py input_images/test_render.jpg \
  --brightness 1.1 --contrast 1.05 --verbose

# Resize for web
python scripts/simple_image_processor.py input_images/test_render.jpg \
  --width 1200 --height 675 --output web_preview.jpg
```

**With Standard Setup (+scipy, tifffile):**
```bash
# Professional TIFF processing with presets
python scripts/utilities/luxury_tiff_batch_processor.py \
  input_images/ output/ --preset golden_hour

# Batch process entire directory
python examples/batch_process.py input_images/ output/
```

**With Full Setup (+ML packages):**
```bash
# Full AI pipeline
python scripts/pipelines/lux_render_pipeline.py input_images/render.tiff \
  --prompt "luxury architectural interior, photorealistic" \
  --strength 0.3 --upscale

# Depth-aware enhancement
python scripts/depth_tools.py input_images/interior.jpg \
  --depth-model small --device mps
```

---

## 📊 What's Currently Installed?

Run this command to see your current setup:

```bash
python scripts/check_image_processing_readiness.py
```

You'll see:
- ✓ Installed packages (green checkmarks)
- ○ Optional packages (gray circles)
- ✗ Missing required packages (red X)
- Disk space available
- FFmpeg status
- Available operations
- Recommended next steps

---

## 💡 Common Scenarios

### Scenario 1: "I just want to batch process JPEGs"
**Tier needed:** Minimal
**Installation:**
```bash
pip install numpy Pillow typer tqdm
```
**Processing:**
```bash
python scripts/simple_image_processor.py input_images/*.jpg \
  --brightness 1.05 --contrast 1.08
```

### Scenario 2: "I need professional TIFF workflow"
**Tier needed:** Standard
**Installation:**
```bash
pip install numpy Pillow scipy tifffile imagecodecs scikit-image
```
**Processing:**
```bash
python scripts/utilities/luxury_tiff_batch_processor.py \
  input_images/ output/ --preset signature_estate --preserve-metadata
```

### Scenario 3: "I want AI-powered enhancement"
**Tier needed:** Full
**Disk space:** ~5GB for packages + ~2GB for models
**Installation:**
```bash
# Check disk space first
df -h /

# Install packages
pip install -r requirements.txt

# Download models
python scripts/setup/download_depth_models.py
```
**Processing:**
```bash
python scripts/pipelines/lux_render_pipeline.py input_images/render.tiff \
  --ai-enhance --upscale --material-response
```

### Scenario 4: "I'm on a tight disk budget"
**Solution:** Use Minimal tier with external processing

```bash
# Process locally with minimal setup
python scripts/simple_image_processor.py input_images/my_image.jpg \
  --brightness 1.1 --output processed.jpg

# For AI features, consider:
# - Cloud GPU (Google Colab, RunPod)
# - External machine with more space
# - Process images in batches (download models, process, clean up)
```

---

## 🔧 Troubleshooting

### "No space left on device" during pip install

```bash
# Clear pip cache
rm -rf ~/.cache/pip

# Install without cache
pip install --no-cache-dir numpy Pillow

# For ML packages, install individually
pip install --no-cache-dir torch
pip install --no-cache-dir diffusers
# etc.
```

### "No sample images found"

```bash
# Download official samples
python scripts/download_samples.py

# OR create test image
python -c "from PIL import Image; Image.new('RGB', (1920, 1080), (200,200,200)).save('input_images/test.jpg')"

# OR use your own
cp ~/Downloads/my_image.jpg input_images/
```

### "FFmpeg not found" (for video processing)

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Check installation
ffmpeg -version
```

### "Import error: No module named 'torch'"

This is expected if you haven't installed ML packages. You have two options:

1. **Install ML packages** (requires ~5GB):
   ```bash
   pip install torch diffusers transformers
   ```

2. **Use Minimal/Standard tier** instead:
   ```bash
   # Stick with simple_image_processor.py
   python scripts/simple_image_processor.py input_images/image.jpg
   ```

---

## 📖 Next Steps

### Learn More
- **README.md** - Full feature overview
- **docs/pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md** - Detailed pipeline documentation
- **docs/guides/CONTEXT_AWARE_RENDERING.md** - AI-powered context extraction

### Upgrade Your Setup
1. Start with Minimal tier (numpy, Pillow)
2. Upgrade to Standard when needed (scipy, tifffile)
3. Upgrade to Full when you have disk space and need AI features

### Get Help
- Check existing scripts in `scripts/pipelines/` for examples
- Run `python <script>.py --help` for any script
- Review test files in `tests/` for usage patterns

---

## ✅ Summary

**Yes, you can process images right now!**

- **Minimal setup** works with just numpy + Pillow
- **Standard setup** adds professional TIFF workflows
- **Full setup** enables all AI features

**Current recommended workflow:**
1. Run `python scripts/check_image_processing_readiness.py`
2. Start with Minimal tier operations using `simple_image_processor.py`
3. Upgrade to higher tiers as needed

The portal is designed to be useful at every tier - you don't need the full ML stack to get started!
