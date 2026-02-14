# Production APEX Pipeline Documentation

This directory contains production-ready documentation and scripts for running the APEX pipeline on luxury real estate photography.

## 📚 Documentation Index

### 1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Start here** for quick copy-paste commands and essential checklists.

- Production command (copy-paste ready)
- Pre-flight checklist
- Post-run verification
- Common issues & fixes
- Performance targets

**Use when**: You need to run the pipeline quickly with standard settings.

---

### 2. [750_PICACHO_APEX_COMMAND.md](750_PICACHO_APEX_COMMAND.md)
**Complete guide** (25+ pages) for production deployment.

Sections:
- Complete CLI command with parameter breakdown
- Configuration recommendations
- Expected output structure
- Performance estimates
- Verification commands
- Alternative configurations
- Custom preset examples
- Production deployment checklist
- Performance benchmarks
- Troubleshooting guide

**Use when**: You need deep understanding, custom configuration, or troubleshooting.

---

## 🚀 Quick Start

### Option 1: Shell Script (Recommended)

```bash
./scripts/pipelines/run_750_picacho_apex_full.sh
```

Features:
- Pre-flight checks
- Interactive confirmation
- Post-run verification
- Performance metrics
- Color-coded output

---

### Option 2: Direct CLI

```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir "input_images/750_picacho/source_jpegs" \
  --output-dir "output_750_picacho_apex_full_$(date +%Y%m%d_%H%M%S)" \
  --quality-tier "apex" \
  --depth-backend "da3" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --enable-segmentation "on" \
  --segmentation-backend "efficientsam" \
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

---

## ✅ Features Enabled

All advanced features are enabled in the production APEX command:

- ✅ **Depth Anything V3** (commercial-safe depth estimation)
- ✅ **Materials V3** (surface-aware finishing)
- ✅ **EfficientSAM** (real ML segmentation)
- ✅ **V2 Enhancement** (material-aware tone mapping)
- ✅ **PBR Maps** (normal, roughness, AO)
- ✅ **16-bit Output** (archival TIFF masters)
- ✅ **Provenance** (complete metadata)

---

## 📊 Expected Performance

| Quality | Time/Image | Throughput |
|---------|------------|------------|
| APEX (all features) | ~2.2s | ~160/hr |
| Premium | ~1.2s | ~300/hr |
| Standard | ~0.5s | ~700/hr |

**For 6 images (APEX)**: ~12-15 seconds total

---

## 📁 Output Structure

```
output_750_picacho_apex_full_*/
├── depth/        # 6 depth maps
├── pbr/          # 18 PBR maps (normal, roughness, AO)
├── enhanced/     # 6 enhanced JPEGs
├── master16/     # 6 archival 16-bit TIFFs
├── manifests/    # 6 JSON manifests
├── reports/      # Batch reports
└── run_card.json # Reproducibility card
```

---

## 🔍 Related Documentation

- **ADR-030**: Materials V3 Production Integration
  `docs/architecture/ADR-030-materials-v3-production-integration.md`

- **Config**: Production preset configuration
  `config/materials_v3_production.yaml`

- **CLI Help**: Full parameter reference
  `python -m transformation_portal.lux_depth_v3 --help`

---

## 🐛 Troubleshooting

Quick fixes for common issues:

```bash
# EfficientSAM fallback to stub
pip install -e ".[ml]"

# MPS not available
pip install --upgrade torch torchvision

# Out of memory
--quality-tier "standard" --max-workers 1

# V2 timeout
export V2_TIMEOUT=300
```

See [750_PICACHO_APEX_COMMAND.md](750_PICACHO_APEX_COMMAND.md) Section 7 for detailed troubleshooting.

---

## 📞 Support

For issues or questions:
1. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common issues
2. Review [750_PICACHO_APEX_COMMAND.md](750_PICACHO_APEX_COMMAND.md) for detailed guidance
3. Inspect manifests and logs in output directory
4. Review ADR-030 for architectural context

---

**Status**: Production-Ready ✅
**Last Updated**: 2026-02-13
**Version**: 1.0
