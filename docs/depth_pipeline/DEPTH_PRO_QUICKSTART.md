# Depth Pro Quick Start Guide

This guide helps you get started with Apple's Depth Pro for metric depth estimation in the Transformation Portal.

## Prerequisites

### 1. Create a Dedicated Depth Pro Environment

```bash
./scripts/setup/install_depth_pro_runtime.sh --skip-verify
```

Keep `depth-pro` out of the main Transformation Portal environment. Depth Pro
currently requires `numpy<2`, while the primary repository environment is pinned
around NumPy 2.x for OpenCV, imagecodecs, and related tooling. The repo-owned
setup script pins the governed Depth Pro runtime surface:
- `torch==2.13.0`
- `torchvision==0.28.0`
- `numpy==1.26.4`
- `depth_pro` from Apple `ml-depth-pro` git ref `9efe5c1def37a26c5367a71df664b18e1306c708`

### 2. Download Checkpoint (1.9 GB)

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download checkpoint
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o checkpoints/depth_pro.pt
```

**Download time:** 5-15 minutes depending on your internet connection.

### 3. Validate Checkpoint (Recommended)

Run the pinned runtime verification, then the slower end-to-end inference smoke:

```bash
./scripts/setup/install_depth_pro_runtime.sh
./.venv-depth-pro/bin/python scripts/validate_depth_pro_checkpoint.py
```

The setup script verifies the pinned runtime and requested device (`mps` on
Apple Silicon, `cpu` elsewhere). The checkpoint validator then confirms:
- ✓ Verify file exists and has correct size (~1.9 GB)
- ✓ Verify SHA-256 hash matches expected value
- ✓ Check depth-pro package is installed
- ✓ Run basic inference test

**Expected output:**
```
======================================================================
  Depth Pro Checkpoint Validation
======================================================================

[1] Checking checkpoint file existence
  ✓ Checkpoint found: checkpoints/depth_pro.pt

[2] Checking checkpoint file size
  File size: 1.85 GB (1885 MB)
  ✓ Size is within expected range (1.5-2.5 GB)

[3] Verifying SHA-256 hash
  This may take 1-2 minutes for a 1.9 GB file...
  Computed in 45.2s
  Expected: 3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce
  Actual:   3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce
  ✓ SHA-256 verified - checkpoint is authentic

[4] Checking depth-pro package
  ✓ depth-pro package installed

[5] Running basic inference test
  Creating test image (640x480)...
  Initializing DepthProStage...
  Running inference (this may take 10-30 seconds on CPU)...
  ✓ Inference successful in 12.34s

  Depth Statistics:
    Shape:  (480, 640)
    Range:  0.52 - 8.73 meters
    Median: 3.21 meters
    P95:    7.89 meters

======================================================================
  Validation Summary
======================================================================

✅ All validation checks passed!
```

---

## Usage

### Option 1: Using Presets (Recommended)

The easiest way to use Depth Pro is with one of the experimental presets:

```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir ./images \
  --output-dir ./output \
  --preset depth-pro-example \
  --depth-pro-python ./.venv-depth-pro/bin/python \
  --non-commercial-ok \
  --accept-apple-depth-pro-research-license
```

**Available Presets:**
- `depth-pro-example` - Default preset (auto-detects device)
- `depth-pro-metric-mps` - Apple Silicon optimized (M1/M2/M3/M4)
- `depth-pro-metric-cpu` - CPU fallback (slow, for compatibility)

### Option 2: CLI Flags

Use CLI flags for more control:

```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir ./images \
  --output-dir ./output \
  --depth-backend depth_pro \
  --depth-pro-python ./.venv-depth-pro/bin/python \
  --depth-device mps \
  --non-commercial-ok \
  --accept-apple-depth-pro-research-license
```

**Device Options:**
- `mps` - Apple Silicon (M1/M2/M3/M4) - **Recommended for Mac**
- `cuda` - NVIDIA GPU
- `cpu` - CPU fallback (slow)

### Option 3: Python API

For programmatic use:

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import EnhanceConfig, enhance_batch
from transformation_portal.depth.backends import DepthBackendRegistry

# Create config
config = EnhanceConfig(
    depth_backend="depth_pro",
    depth_device="mps",  # or "cuda", "cpu"
    non_commercial_ok=True,
    accept_apple_depth_pro_research_license=True,
    depth_pro_checkpoint_path="checkpoints/depth_pro.pt",  # optional
)

# Validate backend available
registry = DepthBackendRegistry()
backend = registry.get_backend("depth_pro", config)
backend.ensure_available()

# Run batch processing
enhance_batch(
    input_dir=Path("./images"),
    output_dir=Path("./output"),
    config=config,
)
```

---

## License Requirements

⚠️ **Depth Pro requires explicit license acceptance**

You **must** set both flags:
1. `--non-commercial-ok` (or `non_commercial_ok=True` in config)
2. `--accept-apple-depth-pro-research-license` (or `accept_apple_depth_pro_research_license=True` in config)

**License:** Apple Machine Learning Research License (AMLR)

**Permitted Uses:**
- ✅ Research and academic use
- ✅ Non-commercial experimentation
- ✅ Personal projects (non-commercial)

**Prohibited Uses:**
- ❌ Commercial products or services
- ❌ Revenue-generating applications
- ❌ Paid client work

**Full License:** https://github.com/apple/ml-depth-pro/blob/main/LICENSE

---

## Performance

### Inference Time (640x480 image)

| Device | Time | Speedup |
|--------|------|---------|
| **MPS (M1/M2/M3)** | ~0.4s | 25x |
| **CUDA (RTX 3090)** | ~0.3s | 33x |
| **CPU (12-core)** | ~10-15s | 1x |

### Memory Usage

- **Model:** ~400 MB VRAM/RAM
- **Checkpoint:** 1.9 GB disk space
- **Peak inference:** ~600 MB

---

## Outputs

Depth Pro produces metric depth maps (absolute scale in meters):

| Output | Description |
|--------|-------------|
| `depth_depthpro.npy` | Float32 depth map (source of truth) |
| `depth_depthpro_preview.png` | 16-bit PNG visualization |
| `depth_depthpro_provenance.json` | Full audit metadata |

**Provenance includes:**
- Checkpoint SHA-256 hash
- Inference time
- Depth statistics (min, median, p95)
- Environment (Python, platform, torch versions)

---

## Integration Tests

Run integration tests with the real checkpoint:

```bash
pytest tests/integration/test_depth_pro_integration.py -v -s
```

**Tests include:**
- Checkpoint validation (size, SHA-256)
- DepthProStage inference
- DepthProBackend inference
- Registry integration
- Cache key consistency
- Provenance completeness

---

## Troubleshooting

### Checkpoint not found

**Error:**
```
FileNotFoundError: Depth Pro checkpoint not found: checkpoints/depth_pro.pt
```

**Solution:**
```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o checkpoints/depth_pro.pt
```

### SHA-256 mismatch

**Error:**
```
CheckpointValidationError: Checkpoint SHA-256 validation failed!
```

**Solution:**
Re-download the checkpoint (file may be corrupted):
```bash
rm checkpoints/depth_pro.pt
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o checkpoints/depth_pro.pt
```

### depth-pro package not installed

**Error:**
```
ImportError: depth_pro package not installed.
```

**Solution:**
```bash
./scripts/setup/install_depth_pro_runtime.sh --skip-verify
```

Then point the main pipeline at that environment:
```bash
export TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON=./.venv-depth-pro/bin/python
```

### License flags missing

**Error:**
```
LicenseRestrictionError: Depth Pro requires non_commercial_ok=True
```

**Solution:**
Add both license flags:
```bash
--non-commercial-ok --accept-apple-depth-pro-research-license
```

### MPS not available on Mac

**Error:**
```
MPS not available
```

**Possible causes:**
- macOS < 12.3 (Monterey required)
- Intel Mac (Apple Silicon only)
- PyTorch < 1.12

**Solution:**
Use CPU fallback:
```bash
--depth-device cpu
```

---

## Next Steps

1. **Validate your checkpoint** with the validation script
2. **Run a test image** with the example preset
3. **Compare with DA3** to evaluate quality
4. **Share feedback** for potential tier promotion

For more details, see:
- [Full Integration Guide](DEPTH_PRO_INTEGRATION_COMPLETE.md)
- [ADR-018: Depth Pro Integration](architecture/ADR-018-depth-pro-integration.md)
- [Lux Depth V3 README](../src/transformation_portal/lux_depth_v3/README.md)

---

**Status:** Experimental (research use only)
**Tier:** Experimental
**Support:** Community support only
