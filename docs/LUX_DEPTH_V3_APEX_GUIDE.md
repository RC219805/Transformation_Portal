# Lux Depth V3 CLI - APEX Quality Guide

This guide demonstrates the **APEX quality tier** commands for the `lux-depth-v3` pipeline, which enables maximum-quality depth processing with all advanced features.

## Installation

First, install the package:

```bash
# Install with ML dependencies
pip install -e ".[ml]"

# Or install with all dependencies
pip install -e ".[all]"
```

The `lux-depth-v3` console script will be available in your PATH.

## Quick Start

### Info Command

View available quality tiers, presets, and features:

```bash
lux-depth-v3 info
```

### Basic Processing

Process images with default premium quality:

```bash
lux-depth-v3 process \
  --input-dir ./input_images \
  --output-dir ./output
```

## APEX Quality Commands

APEX is the highest quality tier, enabling all features and maximum quality settings.

### 1. Commercial-Safe APEX (Recommended)

This command uses **Depth Anything V3** (commercial-safe) with all APEX features:

```bash
lux-depth-v3 process \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-backend "depth_anything_v3" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on" \
  --overwrite
```

**What this enables:**
- **Quality Tier**: APEX (maximum quality settings)
- **Depth Model**: Depth Anything V3 Large (commercial-safe)
- **PBR Maps**: Normal, Roughness, Ambient Occlusion at 1.5x strength
- **Materials V3**: Surface-aware enhancement
- **Depth Cache**: Content-addressable caching for speed
- **Outputs**: All deliverable formats (master16, upscaled16, marketing, report, run-card)*

*Note: Deliverable outputs (master16, upscaled16, marketing, report, run-card) are accepted as CLI flags but implementation pending.

### 2. Python Module Invocation

If the console script is not in your PATH, use Python module execution:

```bash
python -m transformation_portal.lux_depth_v3 process \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-backend "depth_anything_v3" \
  --pbr "on" \
  --materials-v3 "on" \
  --cache-depth "on" \
  --overwrite
```

## Research-Only APEX+ Variants

⚠️ **WARNING**: The following commands use research-only models with **non-commercial licenses**. Only use if you can comply with license restrictions.

### 3. APEX+ with Depth Anything V3.1 (CC BY-NC 4.0)

**License**: Creative Commons Attribution-NonCommercial 4.0 (research/academic use only)

```bash
lux-depth-v3 process \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex_da31" \
  --preset "depth-anything-v3.1-research-m4" \
  --quality-tier "apex" \
  --non-commercial-ok "true" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on" \
  --overwrite
```

**Requirements:**
- Must set `--non-commercial-ok "true"` to acknowledge license
- Uses Depth Anything V3.1 (may have improved accuracy)
- Device `mps` recommended for Apple Silicon, use `cuda` for NVIDIA GPUs

### 4. APEX+ with Apple Depth Pro (AMLR Research License)

**License**: Apple Machine Learning Research License (research-only)

```bash
lux-depth-v3 process \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex_depthpro" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-backend "depth_pro" \
  --non-commercial-ok "true" \
  --accept-apple-depth-pro-research-license "true" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on" \
  --overwrite
```

**Requirements:**
- Must set `--non-commercial-ok "true"`
- Must set `--accept-apple-depth-pro-research-license "true"`
- Requires Apple Depth Pro model checkpoint
- Best performance on Apple Silicon (MPS device)

## Quality Tiers

| Tier | Speed | Model Size | PBR Strength | Use Case |
|------|-------|------------|--------------|----------|
| `draft` | 500-700 img/hr | Small | 0.8x | Fast previews, iteration |
| `standard` | 200-250 img/hr | Base | 1.0x | Production batches |
| `premium` | 100-150 img/hr | Large | 1.2x | Client deliverables |
| `apex` | 50-100 img/hr | Large | 1.5x | Maximum quality, hero shots |

## Available Presets

- `premium` - Alias for luxury_estate (recommended for APEX)
- `luxury_estate` - Premium quality for luxury real estate
- `architectural_interior` - Optimized for interior scenes
- `architectural_exterior` - Optimized for exterior scenes
- `default` - Standard balanced configuration

## Feature Flags

### PBR Maps (`--pbr`)
- `on`: Always generate PBR maps (normal, roughness, AO)
- `off`: Skip PBR generation
- `auto`: Enable for premium/apex tiers (default)

### Materials V3 (`--materials-v3`)
- `on`: Enable Materials V3 surface-aware enhancement
- `off`: Disable Materials V3
- `auto`: Default behavior (currently off, pending implementation)

### Depth Caching (`--cache-depth`)
- `on`: Enable content-addressable depth cache (saves time on re-runs)
- `off`: Disable caching
- `auto`: Enable for apex tier (default)

## Output Structure

```
output/
├── depth/              # Depth maps (16-bit PNG + optional float32 .npy)
├── v2/                 # V2 enhancement outputs (if enabled)
├── manifests/          # Processing manifests (JSON)
├── logs/               # Processing logs
└── zones/              # Zone-based processing data
```

Each processed image generates:
- `<name>_depth.png` - 16-bit depth visualization
- `<name>_depth_float.npy` - High-precision float depth (if `save_float_depth` enabled)
- `<name>_normal.png` - RGB normal map (if PBR enabled)
- `<name>_roughness.png` - Grayscale roughness map (if PBR enabled)
- `<name>_ao.png` - Grayscale ambient occlusion map (if PBR enabled)
- `<name>_manifest.json` - Processing metadata and provenance

## Common Options

### Processing Control

- `--limit N` - Process only first N images (for testing)
- `--overwrite` - Overwrite existing outputs
- `--fail-fast` - Stop on first error

### Logging

- `--verbose` / `-v` - Enable verbose logging
- `--quiet` / `-q` - Suppress all output except errors
- `--log-level LEVEL` - Set log level (DEBUG, INFO, WARNING, ERROR)
- `--json` - Output results as JSON (for scripting)

### Device Selection

- `--depth-device cpu` - Use CPU (slowest, most compatible)
- `--depth-device cuda` - Use NVIDIA GPU (fast)
- `--depth-device mps` - Use Apple Neural Engine (fastest on M-series)

## Examples

### Process with Limited Images (Testing)

```bash
lux-depth-v3 process \
  --input-dir ./test_images \
  --output-dir ./test_output \
  --quality-tier apex \
  --limit 5 \
  --verbose
```

### JSON Output for Scripting

```bash
lux-depth-v3 process \
  --input-dir ./input_images \
  --output-dir ./output \
  --quality-tier apex \
  --json > results.json
```

### GPU-Accelerated Processing

```bash
lux-depth-v3 process \
  --input-dir ./input_images \
  --output-dir ./output \
  --quality-tier apex \
  --depth-device cuda \
  --pbr on
```

## Troubleshooting

### Missing Dependencies

If you see warnings about missing packages:

```bash
# For ML features (Depth Anything V3)
pip install torch transformers diffusers

# For Apple Silicon optimization
pip install coremltools

# For all features
pip install -e ".[ml]"
```

### Performance Optimization

For maximum throughput:

1. Use GPU acceleration (`--depth-device cuda` or `mps`)
2. Enable depth caching (`--cache-depth on`)
3. Use lower quality tier for batches (`--quality-tier standard`)
4. Disable V2 enhancement if not needed (configuration option)

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce quality tier to `standard` or `draft`
2. Process fewer images at once using `--limit`
3. Use CPU instead of GPU (more memory available)
4. Close other applications

## License Compliance

### Commercial Use (Safe)

- Use `--depth-backend "depth_anything_v3"` (default)
- Do NOT set `--non-commercial-ok "true"`
- This uses Depth Anything V3, which is commercial-safe

### Research/Academic Use Only

If using research models:

1. **Depth Anything V3.1**: Requires `--non-commercial-ok "true"`
   - License: CC BY-NC 4.0
   - Cannot be used for commercial purposes

2. **Apple Depth Pro**: Requires both flags
   - `--non-commercial-ok "true"`
   - `--accept-apple-depth-pro-research-license "true"`
   - License: Apple Machine Learning Research License
   - Cannot be used for commercial purposes

## Support

For issues or questions:

- GitHub Issues: https://github.com/RC219805/Transformation_Portal/issues
- Documentation: `docs/architecture/ADR-001-PBR-Integration-Architecture.md`
- CLI Help: `lux-depth-v3 --help` or `lux-depth-v3 process --help`
