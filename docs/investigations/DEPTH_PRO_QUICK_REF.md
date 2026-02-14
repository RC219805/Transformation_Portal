# Depth Pro Research - Quick Reference

**Research-Grade APEX with Apple Depth Pro**

> ⚠️ **RESEARCH ONLY** - Non-commercial use. See [full guide](docs/research/DEPTH_PRO_RESEARCH_GUIDE.md)

---

## Quick Start (Copy & Paste)

### Option 1: Shell Script (Recommended)

```bash
chmod +x scripts/pipelines/run_source_tiffs_depth_pro_research.sh
./scripts/pipelines/run_source_tiffs_depth_pro_research.sh
```

### Option 2: Direct CLI

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
  --segmentation-backend "efficientsam" \
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

---

## Critical Flags (Required)

**These flags are MANDATORY for Depth Pro:**

```bash
--non-commercial-ok "true"                          # CC BY-NC 4.0 acknowledgment
--accept-apple-depth-pro-research-license "true"    # Apple AMLR acceptance
```

**Missing either flag will cause immediate failure.**

---

## Key Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--depth-backend` | `depth_pro` | Use Depth Pro (not DA3) |
| `--depth-device` | `mps` | Apple Neural Engine (M-series) |
| `--preset` | `depth-pro-research-uhq` | Research config |
| `--emit-master16` | `on` | 16-bit archival TIFFs |
| `--emit-upscaled16` | `on` | 16-bit upscaled |
| `--materials-v3` | `on` | Materials V3 |
| `--enable-segmentation` | `on` | Real segmentation |
| `--segmentation-backend` | `efficientsam` | EfficientSAM |
| `--pbr` | `on` | PBR maps |
| `--enable-v2` | `on` | V2 enhancement |

---

## Verification Commands

### 1. Verify 16-bit Depth

```bash
python -c "from PIL import Image; img = Image.open('output_*/depth/*_depth.png'); print('16-bit:', img.mode in ['I', 'I;16'])"
```

### 2. Verify Depth Pro Backend

```bash
MANIFEST=$(find output_source_tiffs_depth_pro_* -name "*.json" | head -1)
python -c "import json; m=json.load(open('${MANIFEST}')); print('Backend:', m['stages']['depth']['backend'])"
```

### 3. Check Focal Length (Depth Pro Feature)

```bash
MANIFEST=$(find output_source_tiffs_depth_pro_* -name "*.json" | head -1)
python -c "import json; m=json.load(open('${MANIFEST}')); print('Focal Length:', m['stages']['depth'].get('focal_length_px', 'N/A'), 'px')"
```

---

## Expected Performance (M4 Max)

- **Depth inference**: ~1.2s per 4K image
- **Total pipeline**: ~3s per 4K image (with Materials V3 + V2 + PBR)
- **Throughput**: ~1200 images/hour (batch processing)
- **Memory**: ~10 GB peak

---

## Output Structure

```
output_source_tiffs_depth_pro_YYYYMMDD_HHMMSS/
├── depth/          # 16-bit depth maps (PNG, metric depth)
├── enhanced/       # V2 enhanced images
├── pbr/            # PBR maps (normal, roughness, AO)
├── master16/       # 16-bit archival TIFFs
├── manifests/      # Research metadata (JSON)
└── reports/        # Quality reports
```

---

## Depth Pro Advantages

✅ **Metric depth in meters** (not normalized)
✅ **Focal length estimation** (unique feature)
✅ **Superior edge preservation** (~5% better than DA3)
✅ **Better reflective surfaces** (glass, water)
✅ **16-bit output** (full precision)

---

## License Compliance

**Permitted:**
- ✅ Academic research
- ✅ Non-profit projects
- ✅ Personal experimentation
- ✅ Benchmarking

**Prohibited:**
- ❌ Commercial products
- ❌ Revenue-generating apps
- ❌ Enterprise deployments

**For commercial use:** Use DA3 instead (`./scripts/pipelines/run_750_picacho_apex_full.sh`)

---

## Troubleshooting

### License Error
```
ERROR: Depth Pro backend requires --accept-apple-depth-pro-research-license true
```
**Fix:** Add both `--non-commercial-ok "true"` and `--accept-apple-depth-pro-research-license "true"`

### MPS Not Available
```
⚠ MPS not available, falling back to CPU
```
**Fix:** Check PyTorch MPS support: `python -c "import torch; print(torch.backends.mps.is_available())"`
**Fallback:** Use `--depth-device "cpu"` (10x slower)

### Checkpoint Missing
```
ERROR: Depth Pro checkpoint not found
```
**Fix:** Download manually:
```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
```

---

## Files

- **Preset**: `config/presets/depth_pro_research_uhq.yaml`
- **Script**: `scripts/pipelines/run_source_tiffs_depth_pro_research.sh`
- **Guide**: `docs/research/DEPTH_PRO_RESEARCH_GUIDE.md`

---

## Related Documentation

- [Full Research Guide](docs/research/DEPTH_PRO_RESEARCH_GUIDE.md) - Complete documentation
- [APEX Contract](docs/APEX_CONTRACT.md) - Quality standards
- [ADR-025](docs/architecture/ADR-025-apex-research-workflow.md) - Architecture decision

---

**License**: Apple Machine Learning Research License (AMLR)
**Updated**: 2026-02-12
