# Lux Depth V2 - Quick Start Guide

**For Transformation Portal Integration**  
**Date**: 2025-12-06

---

## Installation

### For Repository Integration (Recommended)

```bash
# Navigate to repository root
cd /Users/rc/Transformation_Portal

# Install with repository-aligned dependencies
pip install -r lux_depth_v2/requirements-repo.txt

# Install in editable mode with CLI entry points (after Phase 2)
pip install -e .
```

### For Standalone Use (External Users)

```bash
cd lux_depth_v2
pip install -r requirements.txt
python -m lux_depth_v2.cli --help
```

---

## Quick Usage

### CLI Mode (Batch Processing)

```bash
# Basic usage
lux-depth-v2 \
  --input-dir /path/to/images \
  --output-dir /path/to/output \
  --preset interior_luxury

# With depth maps
lux-depth-v2 \
  --input-dir /path/to/images \
  --depth-dir /path/to/depth_maps \
  --output-dir /path/to/output \
  --preset photo_realistic \
  --device cuda

# Advanced options
lux-depth-v2 \
  --input-dir images/ \
  --output-dir output/ \
  --preset interior_luxury \
  --seg-backend onnx \
  --seg-onnx-model /models/material_seg.onnx \
  --upscaler-backend torch \
  --device cuda \
  --validate-ai
```

**Output Files**:
- `*_master16.tif` - 16-bit graded, pre-upscale
- `*_upscaled16.tif` - 16-bit final output
- `*_marketing.png` - 8-bit for review
- `*_preview.jpg` - Small preview
- `*_report.json` - Processing metrics

### Service Mode (REST API)

```bash
# Start service
lux-depth-v2-service \
  --output-dir /data/output \
  --service \
  --host 0.0.0.0 \
  --port 8088

# Test endpoints
curl http://localhost:8088/health

# Process image
curl -X POST http://localhost:8088/v2/process \
  -F "image=@input.jpg" \
  -F "preset=interior_luxury"
```

⚠️ **Security Warning**: Service mode requires hardening before production use. See `lux_depth_v2/SECURITY.md`.

---

## Presets

| Preset | Use Case | Key Features |
|--------|----------|--------------|
| `interior_luxury` | High-end interiors | Balanced clarity, material enhancement |
| `photo_realistic` | Architectural renders | Photorealism, gentle processing |
| `signature_estate` | Luxury estates | Maximum quality, aggressive detail |
| `coastal_modern` | Coastal properties | Blue-hour color grading, atmospheric |

---

## Material Segmentation Backends

### ONNX (Production)
```bash
--seg-backend onnx \
--seg-onnx-model /models/material_seg.onnx
```
- Best performance
- Requires trained ONNX model
- Export tool: `lux_depth_v2/tools/export_material_model_to_onnx.py`

### SegFormer (Practical)
```bash
--seg-backend segformer \
--seg-segformer-model nvidia/segformer-b5-finetuned-ade-640-640 \
--seg-allow-downloads
```
- Good scene understanding
- Automatic download from HuggingFace
- Proxy for material detection via semantic labels

### Heuristic (Fallback)
```bash
--seg-backend heuristic
```
- No dependencies
- Fast but least accurate
- Good for testing

---

## Troubleshooting

### Issue: "Module not found: lux_depth_v2"

**Solution**:
```bash
# Ensure you're in repository root
cd /Users/rc/Transformation_Portal

# Install dependencies
pip install -r lux_depth_v2/requirements-repo.txt

# Add to PYTHONPATH (temporary)
export PYTHONPATH=/Users/rc/Transformation_Portal:$PYTHONPATH
```

### Issue: "CUDA out of memory"

**Solution**:
```bash
# Use CPU
--device cpu

# Or reduce batch size (in code)
# config.batch_size = 1
```

### Issue: "basicsr not found" or CVE warning

**Solution**:
```bash
# Uninstall vulnerable versions
pip uninstall basicsr realesrgan -y

# Use repository dependencies
pip install -r lux_depth_v2/requirements-repo.txt
```

---

## Documentation

- **Integration Plan**: `docs/LUX_DEPTH_V2_INTEGRATION_PLAN.md`
- **Security Guide**: `lux_depth_v2/SECURITY.md`
- **API Reference**: `lux_depth_v2/docs/` (Sphinx)
- **Examples**: `lux_depth_v2/examples/`
- **Checklist**: `LUX_DEPTH_V2_INTEGRATION_CHECKLIST.md`

---

## Integration Status

**Current Phase**: Phase 1 (Security Hardening) - Documentation Complete

**Next Steps**:
1. Implement security hardening (upscaling.py, service.py)
2. Run security scans (safety, bandit)
3. Execute Phase 2 (Integration)

**Production Ready**: ⏳ **NOT YET** - Complete Phase 1 first

---

## Support

**Questions**: Open GitHub issue with `[lux_depth_v2]` tag  
**Security**: See `/SECURITY.md` for secure reporting  
**Documentation**: `lux_depth_v2/README.md`

---

**Version**: 1.0  
**Last Updated**: 2025-12-06
