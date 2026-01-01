# V3 Orchestrator Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Install V3 + V2 dependencies
pip install -e lux_depth_v3/
pip install -e lux_depth_v2/

# Or from requirements
pip install -r lux_depth_v3/requirements.txt
pip install -r lux_depth_v2/requirements.txt
```

### 2. Verify Installation

```bash
# Check V3 CLI
lux-depth-v3 --help

# Check enhance command
lux-depth-v3 enhance --help
```

### 3. Process Your First Image

```bash
# Create test directory
mkdir -p test_renders

# Copy your images to test_renders/
# ...

# Run orchestrator (requires --non-commercial-ok)
lux-depth-v3 enhance \
  --input-dir test_renders/ \
  --output-dir test_output/ \
  --non-commercial-ok \
  --verbose
```

### 4. Verify Outputs

```bash
# Check output structure
ls -la test_output/
# Should see: depth/, v2/, manifests/, logs/

# View depth map
ls test_output/depth/
# Should have: <image_stem>_depth.png (uint16)

# Check V2 outputs
ls test_output/v2/
# Should have: <stem>_master16.tif, <stem>_report.json, etc.

# Inspect manifest
cat test_output/manifests/<stem>_combined.json
```

## Common Workflows

### High-Quality Production

```bash
lux-depth-v3 enhance \
  -i renders/ \
  -o production_output/ \
  --model metric-large \
  --v2-preset production_ultra \
  --v2-upscaler torch \
  --non-commercial-ok
```

### Fast Preview

```bash
lux-depth-v3 enhance \
  -i renders/ \
  -o preview_output/ \
  --model base \
  --v2-preset interior_luxury \
  --v2-upscaler none \
  --non-commercial-ok
```

### Batch with Error Tolerance

```bash
lux-depth-v3 enhance \
  -i large_batch/ \
  -o batch_output/ \
  --depth-fallback skip \
  --v2-timeout 900 \
  --non-commercial-ok
```

### Resume Previous Run

```bash
# Automatically skips existing outputs
lux-depth-v3 enhance \
  -i renders/ \
  -o production_output/ \
  --non-commercial-ok
```

## Output Files Explained

### Depth Directory (`depth/`)
- `<stem>_depth.png`: uint16 PNG, single-channel depth map
- Contract: Shape (H, W), dtype uint16, p1p99 quantization

### V2 Directory (`v2/`)
- `<stem>_master16.tif`: 16-bit master output (V2)
- `<stem>_upscaled16.tif`: Upscaled 16-bit (if enabled)
- `<stem>_marketing.png`: 8-bit marketing export
- `<stem>_preview.jpg`: Quick preview
- `<stem>_report.json`: V2 processing metadata

### Manifests Directory (`manifests/`)
- `<stem>_combined.json`: Links DA3 depth + V2 outputs
- Schema: `lux-depth-v3.enhance.v1`
- Contains: Input hash, depth metadata, V2 status, timings, git hashes

### Logs Directory (`logs/`)
- `v3_enhance.log`: Orchestrator log (if logging enabled)
- `v2_<stem>.log`: Per-image V2 stdout/stderr

## Troubleshooting

### "License Error"
**Problem**: Missing `--non-commercial-ok` flag

**Solution**:
```bash
# Add license acknowledgement
lux-depth-v3 enhance ... --non-commercial-ok
```

### "V2 Not Found"
**Problem**: `lux_depth_v2` not installed

**Solution**:
```bash
# Install V2
pip install -e lux_depth_v2/
```

### "Out of Memory"
**Problem**: GPU memory exhausted

**Solutions**:
```bash
# Use smaller model
lux-depth-v3 enhance ... --model base

# Or use CPU
lux-depth-v3 enhance ... --depth-device cpu --v2-device cpu
```

### "V2 Timeout"
**Problem**: V2 enhancement takes too long

**Solutions**:
```bash
# Increase timeout
lux-depth-v3 enhance ... --v2-timeout 1200

# Or use faster preset
lux-depth-v3 enhance ... --v2-preset standard
```

## Next Steps

1. **Review manifests** to understand processing stages
2. **Check V2 reports** for quality metrics
3. **Tune presets** for your specific use case
4. **Set up monitoring** for production batches
5. **Read full docs**: `lux_depth_v3/enhance/README.md`

## Advanced Topics

### Python API
See `lux_depth_v3/enhance/README.md` for Python API usage

### Custom V2 Presets
See `lux_depth_v2/config.py` for preset definitions

### Depth Quantization
See `depth_writer.py` for quantization methods

### Manifest Schema
See `manifest.py` for dataclass definitions
