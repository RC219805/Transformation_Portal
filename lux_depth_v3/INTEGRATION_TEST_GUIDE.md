# Lux Depth V3 - Integration Testing Guide

## Overview

This guide provides step-by-step instructions for setting up the testing environment and running comprehensive integration tests for lux_depth_v3 and the DA3 enhance orchestrator.

## Prerequisites

- Python 3.10+
- 5-10 architectural render test images (JPEG, PNG, or TIFF)
- ~10GB disk space for models
- GPU recommended (CPU fallback available)

---

## Quick Start

### Option 1: Automated Testing Script

```bash
cd lux_depth_v3
./scripts/run_integration_tests.sh
```

This script will:
1. Check Python version
2. Verify dependencies
3. Run unit tests
4. Validate model versioning
5. Test metric depth utilities
6. Validate license compliance
7. Check for test images

---

## Manual Setup

### Step 1: Install Dependencies

```bash
# Core dependencies
cd lux_depth_v3
pip install -r requirements.txt

# PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Official DA3 package
pip install depth-anything-3

# V2 module (for orchestrator)
cd ../lux_depth_v2
pip install -e .
cd ../lux_depth_v3
```

### Step 2: Verify Installation

```bash
# Check PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check DA3
python3 -c "import depth_anything_3; print('DA3 installed')"

# Check V2
python3 -c "import lux_depth_v2; print('V2 installed')"

# Check lux_depth_v3
python3 -c "from lux_depth_v3 import ModelVariant; print('V3 installed')"
```

### Step 3: Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Exclude integration tests (don't require DA3)
pytest tests/ -v -k "not integration"

# Specific test files
pytest tests/test_model_versioning.py -v
pytest tests/test_enhance.py -v
pytest tests/test_security.py -v
```

---

## Integration Tests

### Test 1: Model Versioning

**Objective:** Verify v1.1 model variants are accessible

```bash
python3 -c "
from lux_depth_v3.config import ModelVariant

# Check v1.1 variants
variants = [
    ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    ModelVariant.DA3_GIANT_V1_1,
    ModelVariant.DA3_LARGE_V1_1,
]

for v in variants:
    info = v.value
    print(f'{v.name}: {info.display_name} ({info.license.value})')
"
```

**Expected Output:**
```
DA3_NESTED_GIANT_LARGE_V1_1: DA3NESTED-GIANT-LARGE-1.1 (CC-BY-NC-4.0)
DA3_GIANT_V1_1: DA3-GIANT-1.1 (CC-BY-NC-4.0)
DA3_LARGE_V1_1: DA3-LARGE-1.1 (CC-BY-NC-4.0)
```

---

### Test 2: Metric Depth Conversion

**Objective:** Validate metric depth conversion utilities

```bash
python3 << 'EOF'
from lux_depth_v3.metric_depth import convert_to_metric_depth, get_depth_statistics
import numpy as np

# Simulate depth output
depth = np.random.rand(480, 640).astype(np.float32) * 10.0

# Test DA3METRIC-LARGE conversion
result = convert_to_metric_depth(
    depth,
    model_name='DA3METRIC-LARGE',
    focal_length_px=500.0
)

print(f'Converted depth shape: {result.depth_meters.shape}')
print(f'Focal length: {result.focal_length_px} px')
print(f'Scale factor: {result.scale_factor:.4f}')
print(f'Already metric: {result.already_metric}')

# Get statistics
stats = get_depth_statistics(result.depth_meters)
print(f'Depth range: {stats["min"]:.2f} - {stats["max"]:.2f} meters')
print(f'Mean depth: {stats["mean"]:.2f} meters')

print('\n✓ Metric depth conversion validated')
EOF
```

---

### Test 3: License Validation

**Objective:** Verify license metadata and commercial use flags

```bash
python3 << 'EOF'
from lux_depth_v3.config import ModelVariant, ModelLicense

# Apache-2.0 models (commercial-friendly)
apache_models = [
    ModelVariant.DA3_BASE,
    ModelVariant.DA3_SMALL,
    ModelVariant.DA3_METRIC_LARGE,
    ModelVariant.DA3_MONO_LARGE,
]

print("Apache-2.0 Models (Commercial Use OK):")
for variant in apache_models:
    info = variant.value
    assert info.is_commercial, f'{variant.name} should allow commercial use'
    print(f'  ✓ {info.display_name} ({info.params})')

# CC-BY-NC-4.0 models (non-commercial only)
nc_models = [
    ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    ModelVariant.DA3_GIANT_V1_1,
    ModelVariant.DA3_LARGE_V1_1,
]

print("\nCC-BY-NC-4.0 Models (Non-Commercial Only):")
for variant in nc_models:
    info = variant.value
    assert not info.is_commercial, f'{variant.name} should NOT allow commercial use'
    print(f'  ⚠️  {info.display_name} ({info.params})')

print('\n✓ License validation passed')
EOF
```

---

### Test 4: End-to-End Orchestrator (Requires Test Images)

**Objective:** Test V3 depth generation + V2 enhancement pipeline

#### 4.1 Prepare Test Images

```bash
# Create test_images directory
mkdir -p test_images

# Copy 5-10 architectural renders
# (JPEG, PNG, or TIFF format)
# Recommended: 1920x1080 or higher resolution
```

#### 4.2 Run Orchestrator

```bash
lux-depth-v3 enhance \
  --input-dir test_images/ \
  --output-dir test_output/ \
  --model metric-large \
  --v2-preset production_ultra \
  --non-commercial-ok \
  --verbose
```

**Note:** `--non-commercial-ok` is required even for Apache models to acknowledge license awareness.

#### 4.3 Validate Outputs

```bash
# Check depth outputs
ls -lh test_output/depth/
file test_output/depth/*.png  # Should show: PNG image data, 16-bit

# Check V2 outputs
ls -lh test_output/v2/
file test_output/v2/*_master16.tif  # Should show: TIFF image data

# Check manifests
cat test_output/manifests/*_combined.json | python3 -m json.tool

# Check logs
tail -n 50 test_output/logs/v3_enhance.log
```

#### 4.4 Visual Inspection

```bash
# View depth maps (requires ImageMagick or similar)
convert test_output/depth/image_depth.png test_output/depth/image_depth_preview.jpg

# View V2 outputs
open test_output/v2/image_marketing.png  # macOS
xdg-open test_output/v2/image_marketing.png  # Linux
```

---

## Test Coverage Summary

| Test | Status | Duration | Dependencies |
|------|--------|----------|--------------|
| **Unit Tests** | ✅ Automated | 10-30s | None (pytest) |
| **Model Versioning** | ✅ Automated | <1s | lux_depth_v3 |
| **Metric Depth** | ✅ Automated | <1s | lux_depth_v3, numpy |
| **License Validation** | ✅ Automated | <1s | lux_depth_v3 |
| **End-to-End Orchestrator** | ⏳ Manual | 2-5 min/image | PyTorch, DA3, V2, test images |

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch torchvision
```

---

**Problem:** `ModuleNotFoundError: No module named 'depth_anything_3'`

**Solution:**
```bash
pip install depth-anything-3
```

---

### Test Failures

**Problem:** `pytest` not found

**Solution:**
```bash
pip install pytest
```

---

**Problem:** Unit tests fail with "skip" marks

**Solution:** This is expected. Some tests are marked as `skip` because they require the full DA3 environment. Run with:
```bash
pytest tests/ -v -k "not integration"
```

---

### Orchestrator Errors

**Problem:** `FileNotFoundError: test_images/`

**Solution:** Create directory and add test images:
```bash
mkdir -p test_images
# Copy architectural renders to test_images/
```

---

**Problem:** `V2 not found` or `lux_depth_v2` import error

**Solution:** Install V2 module:
```bash
cd ../lux_depth_v2
pip install -e .
```

---

**Problem:** Out of memory during depth generation

**Solution:** Reduce batch size or use CPU:
```bash
lux-depth-v3 enhance \
  --input-dir test_images/ \
  --output-dir test_output/ \
  --depth-device cpu \
  --non-commercial-ok
```

---

## Performance Benchmarking

### Measure Throughput

```bash
# Time a batch run
time lux-depth-v3 enhance \
  --input-dir test_images/ \
  --output-dir benchmark_output/ \
  --model metric-large \
  --non-commercial-ok

# Calculate images/hour
# images_per_hour = (num_images / total_seconds) * 3600
```

### Expected Performance

| Hardware | Model | Throughput |
|----------|-------|------------|
| M4 Max (GPU) | METRIC-LARGE | 127-400 images/hour |
| RTX 4090 | METRIC-LARGE | 300-500 images/hour |
| CPU (12-core) | METRIC-LARGE | 20-40 images/hour |

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/test-lux-v3.yml`:

```yaml
name: Lux Depth V3 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          cd lux_depth_v3
          pip install -r requirements.txt
          pip install pytest

      - name: Run unit tests
        run: |
          cd lux_depth_v3
          pytest tests/ -v -k "not integration"
```

---

## Next Steps

After successful integration testing:

1. ✅ **Mark P1 features as complete** in tracker
2. ✅ **Update README** with current feature status
3. ⏳ **Production validation** with 50+ images
4. ⏳ **Performance optimization** if needed
5. ⏳ **Documentation updates** with real-world examples

---

## Support

For issues or questions:
- Check `lux_depth_v3/README.md` for feature documentation
- Review `lux_depth_v3/enhance/INTEGRATION_STATUS.md` for architecture
- See `lux_depth_v3/docs/` for detailed guides
- Run `lux-depth-v3 --help` for CLI reference

---

**Last Updated:** 2026-01-02
**Status:** Ready for integration testing
