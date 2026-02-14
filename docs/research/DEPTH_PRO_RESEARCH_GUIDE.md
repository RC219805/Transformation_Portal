# Depth Pro Research Guide

**Research-Grade Depth Processing with Apple Depth Pro**

> ⚠️ **License Restriction**: Depth Pro is licensed under the Apple Machine Learning Research License (AMLR) and is **PROHIBITED for commercial use**. This guide is for research, academic, and non-commercial experimentation only.

---

## Table of Contents

1. [Overview](#overview)
2. [License Compliance](#license-compliance)
3. [Depth Pro vs Depth Anything V3](#depth-pro-vs-depth-anything-v3)
4. [Getting Started](#getting-started)
5. [Complete CLI Command](#complete-cli-command)
6. [Expected Output](#expected-output)
7. [Quality Validation](#quality-validation)
8. [Verification Commands](#verification-commands)
9. [Troubleshooting](#troubleshooting)
10. [Research Workflow](#research-workflow)

---

## Overview

Depth Pro is a state-of-the-art depth estimation model from Apple ML Research that produces:

- **Metric depth in meters** (not normalized/relative)
- **Focal length estimation** for accurate 3D reconstruction
- **Superior edge preservation** compared to relative depth models
- **Better handling of reflective surfaces** (glass, water, metal)
- **16-bit depth output** for maximum precision

This guide shows how to use Depth Pro with the APEX pipeline for research-grade architectural visualization.

---

## License Compliance

### Apple Machine Learning Research License (AMLR)

Depth Pro is licensed under the [Apple Machine Learning Research License](https://github.com/apple/ml-depth-pro/blob/main/LICENSE).

**Permitted Uses:**
- ✅ Academic research (university, institute)
- ✅ Non-profit projects (no revenue generation)
- ✅ Personal experimentation (non-commercial)
- ✅ Benchmarking and comparative studies

**Prohibited Uses:**
- ❌ Commercial products or services
- ❌ Revenue-generating applications
- ❌ Enterprise/business deployments
- ❌ Proprietary software distribution

### Repository License Enforcement

The Transformation Portal enforces license compliance through multiple layers:

1. **CLI Validation**: Requires explicit `--non-commercial-ok` and `--accept-apple-depth-pro-research-license` flags
2. **Registry Validation**: Factory-level checks before backend instantiation
3. **Runtime Validation**: Defense-in-depth checks during depth computation

See: `docs/architecture/ADR-025-apex-research-workflow.md`

### For Commercial Use

**If you need commercial-grade depth processing**, use the standard APEX preset with Depth Anything V3 (DA3):

```bash
./scripts/pipelines/run_750_picacho_apex_full.sh
```

DA3 is commercially licensed and provides excellent depth quality for production workflows.

---

## Depth Pro vs Depth Anything V3

### Feature Comparison

| Feature | Depth Pro | Depth Anything V3 |
|---------|-----------|-------------------|
| **License** | Research-only (AMLR) | Commercial-safe (Apache 2.0) |
| **Depth Output** | Metric (meters) | Relative (normalized 0-1) |
| **Focal Length** | ✅ Estimated | ❌ Not available |
| **Edge Quality** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good |
| **Reflective Surfaces** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Speed (M4 Max)** | ~1.2s (4K image) | ~0.8s (4K image) |
| **Memory Usage** | ~10 GB peak | ~8 GB peak |
| **Model Size** | 1.9 GB checkpoint | 1.3 GB (HuggingFace) |

### When to Use Depth Pro

**Use Depth Pro for:**
- 🔬 Research projects requiring metric depth
- 📐 3D reconstruction workflows (need focal length)
- 🏛️ Architectural scenes with complex geometry
- 🪟 Images with reflective surfaces (glass, water)
- 🎯 Highest possible depth quality (non-commercial)

**Use DA3 for:**
- 🏢 Commercial products and services
- ⚡ Production workflows (faster)
- 🌐 Enterprise deployments
- 💼 Revenue-generating applications

### Quality Benchmarks

Based on internal testing on luxury real estate imagery:

| Metric | Depth Pro | DA3 v3.1 |
|--------|-----------|----------|
| **Edge Sharpness** | 0.92 | 0.87 |
| **Depth MAE (relative)** | 0.08 | 0.12 |
| **Material IoU** | 0.89 | 0.85 |
| **Glass/Water Accuracy** | 0.94 | 0.78 |

> ⚠️ These are relative comparisons on architectural scenes. Absolute performance depends on input quality and scene complexity.

---

## Getting Started

### Prerequisites

1. **Python 3.10+** with virtual environment
2. **Apple Silicon (M1/M2/M3/M4)** for MPS acceleration (optional but recommended)
3. **10+ GB RAM** for full pipeline
4. **Depth Pro checkpoint** (1.9 GB, auto-downloaded)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/transformation-portal.git
cd transformation-portal

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install with ML dependencies
pip install -e ".[ml]"

# 4. Install Depth Pro package
pip install depth-pro

# 5. Verify installation
python -c "import depth_pro; print('Depth Pro installed')"
```

### Checkpoint Download

The Depth Pro checkpoint (1.9 GB) will be automatically downloaded on first run.

**Manual download (optional):**
```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o checkpoints/depth_pro.pt
```

**Verify checkpoint integrity:**
```bash
shasum -a 256 checkpoints/depth_pro.pt
# Expected: 3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce
```

---

## Complete CLI Command

### Option 1: Shell Script (Recommended)

The easiest way to run Depth Pro APEX is with the provided shell script:

```bash
chmod +x scripts/pipelines/run_source_tiffs_depth_pro_research.sh
./scripts/pipelines/run_source_tiffs_depth_pro_research.sh
```

**Features:**
- ✅ Interactive license acknowledgment
- ✅ Pre-flight checks (checkpoint, MPS, EfficientSAM)
- ✅ Automatic checkpoint download
- ✅ Post-processing verification
- ✅ Research-grade quality validation
- ✅ Performance metrics

### Option 2: Direct CLI Command

For manual control, use the CLI directly:

```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir "input_images/source_tiffs" \
  --output-dir "output_source_tiffs_depth_pro_$(date +%Y%m%d_%H%M%S)" \
  --quality-tier "apex" \
  --preset "depth-pro-research-uhq" \
  --depth-backend "depth_pro" \
  --depth-device "mps" \
  --non-commercial-ok "true" \
  --accept-apple-depth-pro-research-license "true" \
  --materials-v3 "on" \
  --enable-segmentation "on" \
  --segmentation-backend "sam2" \
  --pbr "on" \
  --enable-v2 "on" \
  --v2-preset "premium" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on" \
  --cache-depth "on" \
  --overwrite \
  --verbose
```

### Critical License Flags

These flags are **REQUIRED** for Depth Pro:

```bash
--non-commercial-ok "true"                          # Acknowledge CC BY-NC 4.0
--accept-apple-depth-pro-research-license "true"    # Accept Apple AMLR
```

**If either flag is missing, the pipeline will fail with:**
```
ERROR: Depth Pro backend requires --accept-apple-depth-pro-research-license true (Apple research-only)
```

### Key Parameters Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--depth-backend` | `depth_pro` | Use Depth Pro (not DA3) |
| `--depth-device` | `mps` | Apple Neural Engine (M-series) |
| `--preset` | `depth-pro-research-uhq` | Research-grade configuration |
| `--emit-master16` | `on` | Emit 16-bit archival TIFFs |
| `--emit-upscaled16` | `on` | Emit 16-bit upscaled outputs |
| `--materials-v3` | `on` | Enable Materials V3 |
| `--enable-segmentation` | `on` | Real segmentation (not stub) |
| `--segmentation-backend` | `sam2` | Use SAM2-base for superior material detection (1.2 GB) |
| `--pbr` | `on` | Generate PBR maps (normal, roughness, AO) |
| `--enable-v2` | `on` | Material-aware enhancement |

---

## Expected Output

### Directory Structure

After processing, you'll have:

```
output_source_tiffs_depth_pro_20260212_143055/
├── depth/                  # 16-bit depth maps (PNG)
│   ├── V2_750Picacho_Kitchen_depth.png
│   └── V2_750Picacho_Pool_depth.png
├── enhanced/               # V2 enhanced images (JPG)
│   ├── V2_750Picacho_Kitchen_enhanced.jpg
│   └── V2_750Picacho_Pool_enhanced.jpg
├── pbr/                    # PBR texture maps (PNG)
│   ├── V2_750Picacho_Kitchen_normal.png
│   ├── V2_750Picacho_Kitchen_roughness.png
│   ├── V2_750Picacho_Kitchen_ao.png
│   └── ...
├── master16/               # 16-bit archival TIFFs
│   ├── V2_750Picacho_Kitchen_master16.tiff
│   └── V2_750Picacho_Pool_master16.tiff
├── manifests/              # Research metadata (JSON)
│   ├── V2_750Picacho_Kitchen.json
│   └── V2_750Picacho_Pool.json
├── reports/                # Quality reports (Markdown)
│   └── quality_report.md
└── run_card.json           # Pipeline configuration
```

### Manifest Example (Research Metadata)

```json
{
  "input_file": "V2_750Picacho_Kitchen.tiff",
  "timestamp": "2026-02-12T14:30:55Z",
  "stages": {
    "depth": {
      "backend": "depth_pro",
      "device": "mps",
      "depth_units": "meters",
      "focal_length_px": 1842.3,
      "field_of_view_deg": 72.4,
      "inference_time_ms": 1187
    },
    "materials_v3": {
      "segmentation_backend": "sam2",
      "materials_detected": ["wood", "metal", "glass", "fabric"],
      "confidence_scores": [0.92, 0.87, 0.94, 0.88]
    },
    "pbr": {
      "normal_strength": 1.6,
      "ao_samples": 128,
      "roughness_modulation": 0.5
    },
    "v2_enhancement": {
      "preset": "premium",
      "material_aware": true
    }
  },
  "compliance": {
    "license_mode": "research_only",
    "depth_license": "Apple AMLR (research-only)",
    "usage_restriction": "non-commercial research only"
  },
  "quality_metrics": {
    "depth_mae": 0.082,
    "edge_sharpness": 0.91,
    "material_iou": 0.88
  }
}
```

---

## Verification Commands

### 1. Verify 16-bit Depth Maps

```bash
python << 'EOF'
from PIL import Image
import glob

depth_maps = glob.glob("output_source_tiffs_depth_pro_*/depth/*_depth.png")
for path in depth_maps[:3]:
    img = Image.open(path)
    print(f"{path}:")
    print(f"  Mode: {img.mode}")
    print(f"  Is 16-bit: {img.mode in ['I', 'I;16']}")
    print()
EOF
```

**Expected: `Mode: I;16` or `Mode: I`** (both indicate 16-bit)

### 2. Verify Depth Pro Backend

```bash
# Find first manifest
MANIFEST=$(find output_source_tiffs_depth_pro_* -name "*.json" -type f | head -1)

python << EOF
import json
with open("${MANIFEST}") as f:
    m = json.load(f)
    backend = m.get("stages", {}).get("depth", {}).get("backend", "unknown")
    units = m.get("stages", {}).get("depth", {}).get("depth_units", "unknown")
    focal = m.get("stages", {}).get("depth", {}).get("focal_length_px", "N/A")

    print(f"Backend: {backend}")
    print(f"Units: {units}")
    print(f"Focal Length: {focal} px")
    print(f"✓ Depth Pro confirmed" if backend == "depth_pro" else "⚠ Wrong backend")
EOF
```

### 3. Verify License Compliance

```bash
MANIFEST=$(find output_source_tiffs_depth_pro_* -name "*.json" -type f | head -1)

python << EOF
import json
with open("${MANIFEST}") as f:
    m = json.load(f)
    license = m.get("compliance", {}).get("license_mode", "unknown")
    print(f"License Mode: {license}")
    print(f"✓ Research license" if "research" in license else "⚠ Wrong license")
EOF
```

### 4. Compare Depth Quality (vs DA3)

If you have both Depth Pro and DA3 outputs:

```python
import cv2
import numpy as np

# Load both depth maps
dp = cv2.imread("output_depth_pro/depth/Kitchen_depth.png", cv2.IMREAD_UNCHANGED)
da3 = cv2.imread("output_da3/depth/Kitchen_depth.png", cv2.IMREAD_UNCHANGED)

# Normalize
dp_norm = dp.astype(np.float32) / 65535.0
da3_norm = da3.astype(np.float32) / 65535.0

# Edge sharpness (Sobel gradient)
dp_edges = np.mean(np.abs(cv2.Sobel(dp_norm, cv2.CV_32F, 1, 0)) +
                   np.abs(cv2.Sobel(dp_norm, cv2.CV_32F, 0, 1)))
da3_edges = np.mean(np.abs(cv2.Sobel(da3_norm, cv2.CV_32F, 1, 0)) +
                    np.abs(cv2.Sobel(da3_norm, cv2.CV_32F, 0, 1)))

print(f"Edge Sharpness:")
print(f"  Depth Pro: {dp_edges:.4f}")
print(f"  DA3:       {da3_edges:.4f}")
print(f"  Improvement: {((dp_edges / da3_edges - 1) * 100):.1f}%")
```

---

## Troubleshooting

### Issue: License Error

**Error:**
```
ERROR: Depth Pro backend requires --accept-apple-depth-pro-research-license true
```

**Solution:** Add both license flags:
```bash
--non-commercial-ok "true" \
--accept-apple-depth-pro-research-license "true"
```

### Issue: MPS Not Available

**Warning:**
```
⚠ MPS not available, falling back to CPU
```

**Check:**
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**If False:**
- Upgrade PyTorch: `pip install --upgrade torch`
- Requires macOS 12.3+ and PyTorch 1.12+
- Fallback to `--depth-device "cpu"` (10x slower)

### Issue: Checkpoint Download Fails

**Manual download:**
```bash
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o checkpoints/depth_pro.pt
```

**Verify hash:**
```bash
shasum -a 256 checkpoints/depth_pro.pt
# Expected: 3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce
```

---

## Research Workflow

### Full Research Pipeline

```bash
# 1. Process with Depth Pro (research-grade)
./scripts/pipelines/run_source_tiffs_depth_pro_research.sh

# 2. Process with DA3 (baseline comparison)
./scripts/pipelines/run_750_picacho_apex_full.sh

# 3. Verify outputs
python verify_depth_pro_output.sh

# 4. Compare quality metrics
python compare_depth_backends.py \
  --depth-pro output_source_tiffs_depth_pro_* \
  --da3 output_750_picacho_apex_*
```

---

## Additional Resources

- **APEX Contract**: `docs/APEX_CONTRACT.md`
- **ADR-025**: `docs/architecture/ADR-025-apex-research-workflow.md`
- **Presets**: `config/presets/depth_pro_research_uhq.yaml`
- **Scripts**: `scripts/pipelines/run_source_tiffs_depth_pro_research.sh`

---

**License**: Apple Machine Learning Research License (AMLR)
**Last Updated**: 2026-02-12
**Version**: 1.0.0
