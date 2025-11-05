# Transformation Portal - Setup Guide

This guide walks you through setting up the Transformation Portal with all required dependencies and optional features.

## Table of Contents

- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Model Downloads](#model-downloads)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Verify installation
python scripts/verify_setup.py
```

---

## Detailed Installation

### Prerequisites

- **Python**: 3.10+ (tested on 3.10, 3.11, 3.12)
- **Git**: For cloning the repository
- **pip**: Python package manager (included with Python)
- **Optional**: CUDA-capable GPU or Apple Silicon for ML acceleration

### Step 1: Core Dependencies

Install the core packages required for basic image processing:

```bash
pip install -r requirements.txt
```

This installs:
- **numpy**: Numerical processing
- **Pillow**: Image I/O and manipulation
- **scipy**: Scientific computing
- **typer**: CLI framework
- **accelerate**: Fast model loading (reduces memory by 30-40%)

### Step 2: ML Dependencies (Optional)

For AI-powered features (Stable Diffusion, ControlNet, Real-ESRGAN):

```bash
# Full ML stack
pip install torch torchvision  # or pytorch with CUDA for GPU
pip install diffusers transformers controlnet-aux

# Real-ESRGAN for 4x upscaling
pip install realesrgan basicsr facexlib gfpgan
```

**Platform-specific notes:**

- **Apple Silicon (M1/M2/M3/M4)**: Install PyTorch with MPS support
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- **NVIDIA GPU**: Install PyTorch with CUDA
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- **CPU only**: Standard PyTorch (slower but works)
  ```bash
  pip install torch torchvision
  ```

### Step 3: Depth Processing (Optional)

For depth-aware processing with Depth Anything V2:

```bash
# Install transformers for model loading
pip install transformers huggingface-hub

# For Apple Silicon - CoreML support
pip install coremltools  # macOS only
```

---

## Model Downloads

### Automatic Download Script

Use the provided script to download models:

```bash
python scripts/download_depth_models.py --model all
```

Options:
- `--model depth`: Download depth models only
- `--model realesrgan`: Download Real-ESRGAN weights only
- `--model all`: Download everything (default)
- `--output-dir PATH`: Custom output directory (default: ./weights)

### Manual Download

#### Real-ESRGAN Weights

```bash
# Create weights directory
mkdir -p weights

# Download Real-ESRGAN 4x model (67MB)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth
mv RealESRGAN_x4plus.pth weights/
```

#### Depth Anything V2 (PyTorch)

The transformers library will auto-download the model on first use:

```python
from transformers import pipeline

# Model downloads automatically (~400MB)
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small")
```

#### Depth Anything V2 (CoreML - Apple Silicon)

For optimal performance on M-series chips, convert the model to CoreML:

```python
# Convert PyTorch to CoreML (requires coremltools)
import coremltools as ct
from transformers import AutoModel

model = AutoModel.from_pretrained("depth-anything/Depth-Anything-V2-Small")
# ... conversion code (see CoreML documentation)
```

**Note**: CoreML conversion requires macOS and additional steps. PyTorch models work cross-platform.

---

## Troubleshooting

### Issue #1: Dimension Errors (Tensor Mismatch)

**Error:**
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (88)
```

**Cause**: Stable Diffusion 1.5 requires dimensions that are multiples of 64.

**Solution**: The pipeline now auto-corrects dimensions with a warning:
```
⚠ Corrected dimensions from 1024×770 to 1024×768 (SD 1.5 compatible)
```

**Valid dimensions:**
- 512×512 (standard)
- 768×512 (landscape)
- 512×768 (portrait)
- 768×768 (square)
- 1024×768 (HD landscape)
- 1024×1024 (HD square, requires 8GB+ VRAM)

### Issue #2: Real-ESRGAN Not Available

**Error:**
```
⚠ Real-ESRGAN not available (upscaling disabled)
```

**Solution:**
1. Install package: `pip install realesrgan`
2. Download weights: `python scripts/download_depth_models.py --model realesrgan`
3. Ensure GPU support (CPU mode is very slow)

**Fallback**: The pipeline automatically uses Pillow's Lanczos resampling if Real-ESRGAN is unavailable.

### Issue #3: Slow Model Downloads

**Error:**
```
ZoeD_M12_N.pt: 0% | 703k/1.44G [00:30<10:48:36, 37.1kB/s]
```

**Solutions:**
- Use a faster internet connection
- Pre-download models using `scripts/download_depth_models.py`
- Use cached models: Set `TRANSFORMERS_CACHE` environment variable
  ```bash
  export TRANSFORMERS_CACHE=/path/to/cache
  ```

### Issue #4: Missing Accelerate Warnings

**Warning:**
```
Cannot initialize model with low cpu memory usage because `accelerate` was not found
```

**Solution**:
```bash
pip install accelerate
```

**Benefits**:
- 2-3x faster model loading
- 30-40% less memory during initialization
- Automatic device mapping for multi-GPU setups

### Issue #5: Depth Pipeline Module Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'depth_pipeline/pipeline.py'
```

**Explanation**: The depth pipeline documentation refers to a future module structure. Current implementation:

**Current usage:**
```python
# Use depth_predict_coreml.py for CoreML depth estimation
python depth_predict_coreml.py --in-dir input/ --out-dir output/

# Or use transformers directly
from transformers import pipeline
depth_estimator = pipeline("depth-estimation")
```

**Coming soon**: Unified `depth_pipeline/` module with CLI and Python API.

---

## Verification

### Run Verification Script

```bash
python scripts/verify_setup.py --verbose
```

**Output:**
```
======================================================================
TRANSFORMATION PORTAL - INSTALLATION VERIFICATION
======================================================================

Required Packages:
----------------------------------------------------------------------
✓ numpy               1.24.3
✓ Pillow              10.0.0
✓ scipy               1.10.1
✓ typer               0.12.0

Optional ML Packages:
----------------------------------------------------------------------
✓ torch               2.0.1
✓ diffusers           0.20.2
✓ transformers        4.35.0
✓ controlnet-aux      0.0.7
✓ realesrgan          0.3.0
✓ accelerate          0.24.1

PyTorch Backends:
----------------------------------------------------------------------
✓ MPS                 Available
   → Apple Silicon detected - CoreML models recommended

Model Files:
----------------------------------------------------------------------
○ Depth Anything V2 (CoreML)   Not found (run scripts/download_depth_models.py)
✓ Real-ESRGAN weights          Found

======================================================================
SUMMARY
======================================================================
✓ All required packages are installed
○ 6/6 optional ML packages installed
○ 1/2 model files found

✓ Installation verified - ready to use!
```

### Test Dimension Validation

```python
from transformation_portal.pipelines.lux_render_pipeline import validate_sd_dimensions

# Valid dimensions
width, height = validate_sd_dimensions(1024, 768)  # Returns (1024, 768)

# Auto-corrected dimensions
width, height = validate_sd_dimensions(1024, 770, auto_correct=True)
# Prints: ⚠ Corrected dimensions from 1024×770 to 1024×768 (SD 1.5 compatible)
# Returns: (1024, 768)
```

### Test CLI

```bash
# Test lux_render_pipeline CLI (dry run)
python lux_render_pipeline.py \
  --input-glob 'test.png' \
  --out 'output/' \
  --prompt "test" \
  --neg "blurry" \
  --width 768 \
  --height 512

# Should start processing (or show helpful errors if dependencies missing)
```

---

## Image Dimension Requirements

### Stable Diffusion 1.5

**Requirements:**
- Dimensions **must** be multiples of 64
- Recommended resolutions:
  - **512×512**: Standard, fast, low VRAM (2GB)
  - **768×512**: Landscape, moderate (4GB)
  - **512×768**: Portrait, moderate (4GB)
  - **768×768**: Large square (6GB)
  - **1024×768**: HD landscape (8GB)
  - **1024×1024**: HD square, slow (10GB+)

**Auto-correction**: Invalid dimensions are automatically corrected to the nearest valid size:
```
1024×770 → 1024×768 (✓ Valid)
800×600 → 768×576 (✓ Valid)
```

### Real-ESRGAN Upscaling

**Input**: Any dimension (after SD processing)
**Output**: 4x scale (e.g., 768×512 → 3072×2048)
**VRAM**: ~6-8GB for 1024×768 input

---

## Performance Tips

### Apple Silicon (M1/M2/M3/M4)

1. **Use MPS backend** (automatic with PyTorch 2.0+)
2. **Install CoreML tools**: `pip install coremltools`
3. **Use CoreML models** when available (3-5x faster)
4. **Enable Metal**: Ensure macOS 13+ for best performance

**Expected performance**:
- Depth estimation: 24-65ms per image
- Batch throughput: 400-600 images/hour
- SD inference: 3-8 seconds per image (768×512)

### NVIDIA GPU

1. **Install CUDA-enabled PyTorch**
2. **Use mixed precision**: `--fp16` or `torch.cuda.amp`
3. **Batch processing**: Process multiple images simultaneously
4. **Monitor VRAM**: Use `nvidia-smi` to track usage

### CPU Only

1. **Use smaller models**: Depth-Anything-V2-Small vs Large
2. **Reduce dimensions**: 512×512 instead of 1024×768
3. **Skip heavy features**: Disable Real-ESRGAN, use Lanczos
4. **Be patient**: CPU inference is 10-20x slower than GPU

---

## Next Steps

After setup:

1. **Read the main README**: `README.md` for usage examples
2. **Explore examples**: `examples/` directory
3. **Run tests**: `make test-fast` or `pytest tests/`
4. **Try the workflows**: `README_PIPELINE.md` for complete pipelines

---

## Getting Help

- **Issues**: https://github.com/RC219805/Transformation_Portal/issues
- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory

---

**Last Updated**: November 2025
**Version**: 1.0.0
