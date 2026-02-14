# APEX Pipeline Quick Reference

## 🚀 Quick Start (Copy-Paste)

```bash
# Production APEX command - ALL features enabled
python -m transformation_portal.lux_depth_v3 \
  --input-dir "input_images/750_picacho/source_jpegs" \
  --output-dir "output_750_picacho_apex_full_$(date +%Y%m%d_%H%M%S)" \
  --quality-tier "apex" \
  --depth-backend "da3" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --enable-segmentation "on" \
  --segmentation-backend "sam2" \
  --pbr "on" \
  --enable-v2 "on" \
  --v2-preset "default" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on" \
  --cache-depth "on" \
  --overwrite \
  --verbose
```

**OR use the shell script:**

```bash
./scripts/pipelines/run_750_picacho_apex_full.sh
```

---

## 📋 Pre-Flight Checklist

```bash
# 1. Check input directory
ls -lh input_images/750_picacho/source_jpegs/

# 2. Verify ML dependencies
python verify_ml_deps.py

# 3. Test MPS (Apple Silicon)
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 4. Test SAM2
python -c "from transformation_portal.lux_depth_v3.segmentation_backend import _get_backend_instance; print(_get_backend_instance('sam2').__class__.__name__)"
```

---

## ✅ Post-Run Verification

```bash
OUTPUT_DIR="output_750_picacho_apex_full_20260213_143022"  # Replace with actual

# Quick checks
find "${OUTPUT_DIR}/depth" -name "*.png" | wc -l         # Should be 6
find "${OUTPUT_DIR}/pbr" -name "*.png" | wc -l           # Should be 18
find "${OUTPUT_DIR}/enhanced" -name "*.jpg" | wc -l      # Should be 6

# Verify Materials V3 ran
cat "${OUTPUT_DIR}/manifests"/*.json | head -1 | \
  jq '.stages.materials_v3.segmentation_backend'
# Expected: "sam2"

# Check materials detected
cat "${OUTPUT_DIR}/manifests"/*.json | head -1 | \
  jq '.stages.materials_v3.materials_detected'
# Expected: ["wood", "glass", "fabric", "metal", ...]
```

---

## 🎚️ Alternative Modes

### Fast Preview (Standard Quality)
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir "input_images/750_picacho/source_jpegs" \
  --output-dir "output_750_picacho_preview_$(date +%Y%m%d_%H%M%S)" \
  --quality-tier "standard" \
  --depth-device "mps" \
  --enable-v2 "on" \
  --verbose
```

### PBR Only (No Enhancement)
```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir "input_images/750_picacho/source_jpegs" \
  --output-dir "output_750_picacho_pbr_$(date +%Y%m%d_%H%M%S)" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --pbr "on" \
  --enable-v2 "off" \
  --verbose
```

---

## 📊 Performance Targets

| Quality | Features | Time/Image | Throughput |
|---------|----------|------------|------------|
| **Standard** | Depth + V2 | 0.5s | ~700/hr |
| **Premium** | + Materials (stub) | 1.2s | ~300/hr |
| **APEX (SAM2)** | + SAM2 Seg + PBR + 16-bit | 5.5s | ~60/hr |
| **APEX (EfficientSAM)** | + EfficientSAM + PBR + 16-bit | 2.2s | ~160/hr |

**Expected for 6 images (APEX mode with SAM2)**: ~30-40 seconds total

> **SAM2 vs EfficientSAM**: SAM2 provides superior material boundary detection (~4s/image) vs EfficientSAM (~0.1s/image). Use SAM2 for production quality, EfficientSAM for fast iteration.

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| **SAM2 fallback to stub** | `pip install -e ".[ml]"` |
| **MPS not available** | `pip install --upgrade torch` or use `--depth-device "cpu"` |
| **SAM2 too slow** | Use `--segmentation-backend "efficientsam"` for faster processing |
| **Out of memory** | `--segmentation-backend "efficientsam"` or `--max-workers 1` |
| **V2 timeout** | `export V2_TIMEOUT=300` or `--enable-v2 "off"` |
| **No materials detected** | Lower thresholds via custom preset or check SAM2 installation |

---

## �� Full Documentation

- **Complete Guide**: `docs/production/750_PICACHO_APEX_COMMAND.md`
- **ADR-030**: Materials V3 Production Integration
- **Config**: `config/materials_v3_production.yaml`
- **CLI Help**: `python -m transformation_portal.lux_depth_v3 --help`

---

## 🎯 What to Expect

### Output Structure
```
output_750_picacho_apex_full_*/
├── depth/        # 6 depth maps (16-bit PNG)
├── pbr/          # 18 PBR maps (normal, roughness, AO)
├── enhanced/     # 6 enhanced JPEGs (V2 tone mapping)
├── master16/     # 6 archival 16-bit TIFFs
├── manifests/    # 6 JSON manifests (provenance)
├── reports/      # Batch reports (JSON + HTML)
└── run_card.json # Reproducibility card
```

### Features Enabled
- ✅ **Depth Anything V3** (commercial-safe)
- ✅ **Materials V3** (real SAM2 segmentation - superior quality)
- ✅ **V2 Enhancement** (material-aware tone mapping)
- ✅ **PBR Maps** (normal, roughness, ambient occlusion)
- ✅ **16-bit Output** (archival quality TIFFs)
- ✅ **Provenance** (complete processing metadata)

### Segmentation Backend Comparison
- **SAM2** (default): ~4s/image, ⭐⭐⭐⭐⭐ Excellent quality
- **EfficientSAM** (fast): ~0.1s/image, ⭐⭐⭐⭐ Very good quality
- Switch via `--segmentation-backend "efficientsam"` for faster processing

---

**Status**: Production-Ready ✅
**Last Updated**: 2026-02-13
