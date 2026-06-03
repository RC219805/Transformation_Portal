# Troubleshooting Guide

> Historical general troubleshooting guide.
>
> This guide captures November 2025 issues. Current Lux Depth V3 troubleshooting
> lives in [LUX_DEPTH_V3_TROUBLESHOOTING.md](LUX_DEPTH_V3_TROUBLESHOOTING.md);
> current navigation lives in [Documentation Map](../governance/DOCUMENTATION_MAP.md).

This guide addresses common issues encountered when using the Transformation Portal. Issues are organized by priority and include solutions implemented as of November 2025.

## Table of Contents

- [Critical Issues (Priority 1)](#critical-issues-priority-1)
- [High Priority Issues (Priority 2)](#high-priority-issues-priority-2)
- [Medium Priority Issues (Priority 3)](#medium-priority-issues-priority-3)
- [General Troubleshooting](#general-troubleshooting)

---

## Critical Issues (Priority 1)

### Issue #1: Lux Render Pipeline CLI Bug ✅ FIXED

**Status**: Fixed in November 2025 update

**Symptoms**:
```
TypeError: expected str, bytes or os.PathLike object, not OptionInfo
```

**Root Cause**: Earlier versions had incorrect Typer parameter handling.

**Solution**: Update to latest version. The CLI now properly handles all Typer parameters.

**Verification**:
```bash
python lux_render_pipeline.py --help
# Should display help without errors
```

---

### Issue #2: Dimension Validation Error ✅ FIXED

**Symptoms**:
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (88)
```

**Root Cause**: Stable Diffusion 1.5 requires image dimensions that are multiples of 64. User-provided dimensions like 1024×770 or 800×600 cause tensor dimension mismatches in the U-Net architecture.

**Solution**: The pipeline now includes automatic dimension validation and correction.

**How it works**:
```python
# Invalid dimensions are automatically corrected
--width 1024 --height 770  # Automatically corrected to 1024×768
```

**Output**:
```
⚠ Corrected dimensions from 1024×770 to 1024×768 (SD 1.5 compatible)
```

**Valid Dimensions** (multiples of 64):
| Resolution | Use Case | VRAM Required |
|------------|----------|---------------|
| 512×512 | Standard, fast | ~2GB |
| 768×512 | Landscape | ~4GB |
| 512×768 | Portrait | ~4GB |
| 768×768 | Square | ~6GB |
| 1024×768 | HD Landscape | ~8GB |
| 1024×1024 | HD Square | ~10GB+ |

**Disable auto-correction** (for testing):
```python
from transformation_portal.pipelines.lux_render_pipeline import validate_sd_dimensions

# Raises error if invalid
validate_sd_dimensions(1024, 770, auto_correct=False)
```

**Manual fix**:
Always use dimensions that are multiples of 64:
```bash
python lux_render_pipeline.py \
  --width 768 \    # ✓ 768 = 64 × 12
  --height 512 \   # ✓ 512 = 64 × 8
  --input 'images/*.png'
```

---

### Issue #3: Missing CoreML Depth Model

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory:
'DepthAnythingV2SmallF16.mlpackage'
```

**Root Cause**: CoreML model file is not included in repository (too large for git).

**Solutions**:

**Option 1: Use PyTorch Depth Model** (Recommended for cross-platform)
```bash
# Install transformers
pip install transformers torch

# Model downloads automatically on first use
from transformers import pipeline
depth_estimator = pipeline("depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small")
```

**Option 2: Convert to CoreML** (Apple Silicon only)
```bash
# Install CoreML tools (macOS only)
pip install coremltools

# Follow CoreML conversion guide
# See: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html
```

**Option 3: Skip Depth Processing**
```bash
# Use --no-depth flag to skip depth guidance
python lux_render_pipeline.py \
  --no-depth \
  --input 'images/*.png'
```

**Status**: PyTorch-based depth estimation works across all platforms. CoreML is optional for Apple Silicon optimization.

---

## High Priority Issues (Priority 2)

### Issue #4: Real-ESRGAN Not Available ✅ IMPROVED

**Symptoms**:
```
⚠ Real-ESRGAN not available (upscaling disabled)
```

**Root Cause**: Real-ESRGAN package or model weights not installed.

**Solution** (now with helpful error messages):

**Step 1: Install package**
```bash
pip install realesrgan basicsr facexlib gfpgan
```

**Step 2: Download model weights**
```bash
# Automatic download (recommended)
python scripts/setup/download_depth_models.py --model depth

# Or manual download
mkdir -p weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth
mv RealESRGAN_x4plus.pth weights/
```

**Step 3: Verify installation**
```bash
python scripts/verify_setup.py
```

**Fallback**: If Real-ESRGAN is unavailable, the pipeline automatically uses Pillow's Lanczos resampling (lower quality but functional).

**Performance notes**:
- Real-ESRGAN requires GPU for reasonable speed
- CPU mode is ~50x slower
- Lanczos is instant but produces softer results

---

### Issue #5: Slow Model Downloads

**Symptoms**:
```
ZoeD_M12_N.pt: 0% | 703k/1.44G [00:30<10:48:36, 37.1kB/s]
```

**Root Cause**: Large models (1-2GB) downloading over slow connection.

**Solutions**:

**Option 1: Pre-download models**
```bash
# Use download script with progress bar
python scripts/setup/download_depth_models.py
```

**Option 2: Use model cache**
```bash
# Set Hugging Face cache directory
export TRANSFORMERS_CACHE=/path/to/fast/storage
export HF_HOME=/path/to/fast/storage

# Models download to this location
python your_script.py
```

**Option 3: Use smaller models**
```python
# Use Small model instead of Base or Large
from transformers import pipeline
depth_estimator = pipeline("depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small")  # ~400MB
# vs. "Depth-Anything-V2-Large"  # ~1.4GB
```

**Option 4: Skip depth processing**
```bash
python lux_render_pipeline.py --no-depth --input 'images/*.png'
```

---

## Medium Priority Issues (Priority 3)

### Issue #6: Missing Accelerate Package ✅ FIXED

**Symptoms**:
```
Cannot initialize model with low cpu memory usage because `accelerate`
was not found in the environment.
```

**Solution**: Now included in requirements.txt (as of November 2025)

**Install manually** (if needed):
```bash
pip install accelerate
```

**Benefits**:
- 2-3x faster model loading
- 30-40% less memory during initialization
- Automatic multi-GPU support
- Better memory management

**Verification**:
```python
import accelerate
print(accelerate.__version__)  # Should print version number
```

---

### Issue #7: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'numpy'
ImportError: cannot import name 'pipeline' from 'transformers'
```

**Solutions**:

**Step 1: Verify Python version**
```bash
python --version  # Must be 3.10+
```

**Step 2: Install all dependencies**
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

**Step 3: Install package in editable mode**
```bash
pip install -e .
```

**Step 4: Verify installation**
```bash
python scripts/verify_setup.py
```

**Common fixes**:
- **Wrong Python version**: Use Python 3.10+
- **Wrong virtual environment**: Activate correct venv
- **Cached packages**: Clear pip cache: `pip cache purge`
- **Conflicting packages**: Create fresh venv

---

### Issue #8: GPU/CUDA Not Detected

**Symptoms**:
```
torch.cuda.is_available() returns False
Using CPU - processing will be slow
```

**Solutions**:

**For NVIDIA GPU**:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**For Apple Silicon (M1/M2/M3/M4)**:
```bash
# Install MPS-enabled PyTorch
pip install torch torchvision

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"
```

**For CPU-only systems**:
- Use smaller models
- Reduce dimensions (512×512 instead of 1024×768)
- Skip heavy features (--no-depth, no Real-ESRGAN)
- Increase timeout values

---

## General Troubleshooting

### Diagnostic Commands

**Check installation status**:
```bash
python scripts/verify_setup.py --verbose
```

**Test dimension validation**:
```python
from transformation_portal.pipelines.lux_render_pipeline import validate_sd_dimensions

# Test with your dimensions
validate_sd_dimensions(1024, 770, auto_correct=True)
```

**Check PyTorch backend**:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"CPU: {torch.device('cpu')}")
```

**Verify models**:
```bash
ls -lh weights/
ls -lh *.mlpackage 2>/dev/null || echo "No CoreML models found"
```

**Check memory usage**:
```bash
# During processing, monitor memory
watch -n 1 free -h  # Linux
top -l 1 | grep PhysMem  # macOS
```

---

### Performance Optimization

**Apple Silicon (M1/M2/M3/M4)**:
- Use MPS backend (automatic with PyTorch 2.0+)
- Install CoreML tools for depth processing
- Expected: 24-65ms per image for depth estimation

**NVIDIA GPU**:
- Install CUDA-enabled PyTorch
- Use mixed precision training
- Monitor VRAM with `nvidia-smi`

**CPU Only**:
- Use smaller models
- Reduce image dimensions
- Skip Real-ESRGAN upscaling
- Be patient (10-20x slower than GPU)

---

### Getting Help

1. **Check documentation**:
   - [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed installation
   - [README.md](../../README.md) - Feature overview
   - [PIPELINE_OPERATIONS_GUIDE.md](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md) - Usage examples

2. **Run diagnostics**:
   ```bash
   python scripts/verify_setup.py --verbose
   ```

3. **Check existing issues**:
   https://github.com/RC219805/Transformation_Portal/issues

4. **Create new issue** with:
   - Error message (full traceback)
   - Output of `verify_setup.py`
   - Python version (`python --version`)
   - OS and hardware specs
   - Steps to reproduce

---

## Quick Reference: Common Fixes

| Problem | Quick Fix |
|---------|-----------|
| Tensor dimension mismatch | Use dimensions that are multiples of 64 (auto-corrected) |
| Real-ESRGAN not found | `pip install realesrgan && python scripts/setup/install_models.py --dry-run` |
| Slow model downloads | `python scripts/setup/download_depth_models.py` |
| Missing accelerate | `pip install accelerate` (now in requirements.txt) |
| Import errors | `pip install -r requirements.txt && pip install -e .` |
| GPU not detected | Install CUDA or MPS-enabled PyTorch |
| Out of memory | Reduce dimensions or use --no-depth flag |
| CoreML model missing | Use PyTorch models (cross-platform) |

---

**Last Updated**: November 2025
**Version**: 1.0.0
**Related**: See [SETUP_GUIDE.md](SETUP_GUIDE.md) for installation details
