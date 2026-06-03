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

# 2. Create the repo virtual environment
make venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install pinned core dependencies and CLI
make install-core

# 4. Verify the environment
make check-environment
```

---

## Detailed Installation

### Prerequisites

- **Python**: 3.11+ (tested on 3.11, 3.12)
- **Git**: For cloning the repository
- **pip**: Python package manager (included with Python)
- **Optional**: CUDA-capable GPU or Apple Silicon for ML acceleration

### Step 1: Core Dependencies

Install the core packages required for basic image processing:

```bash
make venv
source .venv/bin/activate
make install-core
```

This installs:
- **numpy**: Numerical processing
- **Pillow**: Image I/O and manipulation
- **scipy**: Scientific computing
- **typer**: CLI framework and other pinned core utilities
- the local project in editable mode without re-resolving dependencies outside the checked-in lockfiles

### Step 2: ML Dependencies (Optional)

For AI-powered features such as DA3 depth inference, use a trusted target-specific ML profile rather than the disabled umbrella install path:

```bash
# Cross-platform baseline
make install-ml-core

# Or use the bootstrap profiles directly:
./scripts/bootstrap/install_ml_stack.sh --profile core-cpu
./scripts/bootstrap/install_ml_stack.sh --profile core-mps    # macOS Apple Silicon only
PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 ./scripts/bootstrap/install_ml_stack.sh --profile core-cuda  # Linux + NVIDIA only
```

Optional profiles for `raw`, `coreml`, `research`, and `full` are currently fail-closed until trusted target-correct lockfile contracts exist again.

These trusted ML profile flows are currently supported on macOS and Linux only. They are not Windows-native; on Windows, use WSL2 or another Unix-like environment. The `core-cuda` profile is Linux + NVIDIA only.

**Platform-specific notes:**

- **Apple Silicon macOS (M1/M2/M3/M4):** use `make install-ml-core` or `./scripts/bootstrap/install_ml_stack.sh --profile core-mps`
- **Linux with NVIDIA GPU:** use `PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 ./scripts/bootstrap/install_ml_stack.sh --profile core-cuda`
- **CPU only (macOS/Linux):** use `make install-ml-core` or `./scripts/bootstrap/install_ml_stack.sh --profile core-cpu`

### Step 3: Depth Processing (Optional)

For depth-aware processing with the Depth Anything backend:

```bash
# Install transformers for model loading
pip install transformers huggingface-hub

# For Apple Silicon - CoreML support
pip install coremltools  # macOS only
```

---

## Model Downloads

### Automatic Download Script

Use the provided script to print Depth Anything CoreML setup instructions and verify local artifact status:

```bash
python scripts/setup/download_depth_models.py --model depth
```

Options:
- `--model depth`: Depth model setup/verification workflow
- `--verify-only`: Verify local model artifact status without setup steps
- `--output-dir PATH`: Custom output directory (default: ./weights)

### Depth Anything (HuggingFace)

Transformation Portal uses Depth Anything for depth estimation. The transformers library will auto-download the model on first use:

```python
from transformers import pipeline

# Transformers-compatible IDs use the "-hf" suffix.
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
```

For the `lux-depth-v3` pipeline, V3 metric model IDs are attempted first and automatically
fallback to V2 metric `*-hf` IDs when the V3 variant is unavailable.

### Depth Anything (CoreML - Apple Silicon)

For optimal performance on M-series chips, convert the model to CoreML:

```python
# Convert PyTorch to CoreML (requires coremltools)
import coremltools as ct
from transformers import AutoModel

model = AutoModel.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
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

### Issue #2: Slow Model Downloads

**Error:**
```
ZoeD_M12_N.pt: 0% | 703k/1.44G [00:30<10:48:36, 37.1kB/s]
```

**Solutions:**
- Use a faster internet connection
- Use `python scripts/setup/download_depth_models.py --verify-only` to verify local artifacts
- HuggingFace models are still downloaded automatically on first model use
- Use cached models: Set `TRANSFORMERS_CACHE` environment variable
  ```bash
  export TRANSFORMERS_CACHE=/path/to/cache
  ```

### Issue #3: Missing Accelerate Warnings

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

### Issue #4: Depth Pipeline Module Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'depth_pipeline/pipeline.py'
```

**Explanation**: The depth pipeline is now part of the Lux Depth V3 module. Use the current implementation:

**Current usage:**
```bash
# Use the lux-depth-v3 CLI
lux-depth-v3 --input-dir ./input --output-dir ./output
```

```python
# Or use the Python API
from transformation_portal.lux_depth_v3 import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
```

See [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md) for detailed usage.

---

## Verification

### Run Verification Script

```bash
python scripts/verification/verify_core.py
```

Or for ML dependencies:

```bash
python scripts/verification/verify_ml_deps.py
```

**Example Output:**
```
======================================================================
TRANSFORMATION PORTAL - INSTALLATION VERIFICATION
======================================================================

Required Packages:
----------------------------------------------------------------------
✓ numpy
✓ Pillow
✓ scipy
✓ typer

Optional ML Packages:
----------------------------------------------------------------------
✓ torch
✓ diffusers
✓ transformers
✓ controlnet-aux
✓ accelerate

PyTorch Backends:
----------------------------------------------------------------------
✓ MPS                 Available
   → Apple Silicon detected - MPS acceleration available

======================================================================
SUMMARY
======================================================================
✓ All required packages are installed
✓ Optional ML packages installed

✓ Installation verified - ready to use!
```

### Test CLI

```bash
# Test lux-depth-v3 CLI
lux-depth-v3 --help

# List available presets
lux-depth-v3 --list-stable
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

---

## Performance Tips

### Apple Silicon (M1/M2/M3/M4)

1. **Use MPS backend** (automatic with PyTorch 2.0+)
2. **Install CoreML tools**: `pip install coremltools`
3. **Enable Metal**: Ensure macOS 13+ for best performance

**Expected performance**:
- Depth estimation: 24-65ms per image
- Batch throughput: 400-600 images/hour

### NVIDIA GPU

1. **Install CUDA-enabled PyTorch**
2. **Use mixed precision**: `--fp16` or `torch.cuda.amp`
3. **Batch processing**: Process multiple images simultaneously
4. **Monitor VRAM**: Use `nvidia-smi` to track usage

### CPU Only

1. **Use smaller models**: Depth-Anything-V2-Small-hf vs Large-hf
2. **Reduce dimensions**: 512×512 instead of 1024×768
3. **Be patient**: CPU inference is 10-20x slower than GPU

---

## Next Steps

After setup:

1. **Read the main README**: `README.md` for usage examples
2. **Explore examples**: `examples/` directory
3. **Run tests**: `make test-fast` or `pytest tests/`
4. **Try the CLI**: `lux-depth-v3 --help` for the main processing pipeline

---

## Getting Help

- **Issues**: https://github.com/RC219805/Transformation_Portal/issues
- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory

---

**Last Updated**: March 2026
**Version**: 2.0.0
