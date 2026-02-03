# Lux Depth V3 CLI - APEX Command Variants

This document provides usage examples for the `lux-depth-v3` CLI with APEX quality tier support, including both commercial-safe and research-only variants.

## Table of Contents
- [Installation](#installation)
- [Commercial-Safe APEX Mode](#commercial-safe-apex-mode)
- [Research-Only APEX+ Variants](#research-only-apex-variants)
- [Command Options Reference](#command-options-reference)
- [Quality Tiers](#quality-tiers)
- [Output Deliverables](#output-deliverables)

## Installation

First, ensure the package is installed with the CLI entry point:

```bash
pip install -e .
```

The `lux-depth-v3` command should now be available on your PATH. Alternatively, you can invoke it as a module:

```bash
python -m transformation_portal.lux_depth_v3 [options]
```

## Commercial-Safe APEX Mode

The commercial-safe APEX mode uses commercially-licensed depth backends (Depth Anything V3 with Apache 2.0 license) and provides the highest quality output for production use.

### Basic APEX Command

```bash
lux-depth-v3 \
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

### APEX with GPU Acceleration (CUDA)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/apex_cuda" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-device "cuda" \
  --materials-v3 "on" \
  --pbr "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on"
```

### APEX with Apple Silicon (MPS)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/apex_mps" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on"
```

## Research-Only APEX+ Variants

⚠️ **Important**: Research-only variants use non-commercial models that require explicit license acknowledgement. Only use these if you comply with the respective license restrictions.

### Variant A: Depth Anything V3.1 (CC BY-NC 4.0)

Depth Anything V3.1 provides state-of-the-art depth estimation but is restricted to non-commercial use under CC BY-NC 4.0.

```bash
lux-depth-v3 \
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

**License**: CC BY-NC 4.0 (Non-Commercial)
**Use Cases**: Research, academic projects, non-commercial portfolio work

### Variant B: Apple Depth Pro (AMLR Research License)

Apple Depth Pro provides high-quality depth estimation but requires both non-commercial acknowledgement and explicit Apple license acceptance.

```bash
lux-depth-v3 \
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

**License**: AMLR (Apple Machine Learning Research)
**Use Cases**: Research only, requires explicit license acceptance
**Requirements**: Both `--non-commercial-ok` and `--accept-apple-depth-pro-research-license` must be set to `true`

## Command Options Reference

### Required Options

- `--input-dir PATH`: Input directory containing images to process (required)
- `--output-dir PATH`: Output directory for all artifacts (required)

### Pipeline Configuration

- `--preset TEXT`: Pipeline preset (default: `premium`)
  - Options: `premium`, `depth-anything-v3.1-research-m4`, `default`, etc.
- `--quality-tier TEXT`: Quality tier (default: `standard`)
  - Options: `standard`, `premium`, `apex`

### Depth Backend Configuration

- `--depth-backend TEXT`: Depth estimation backend
  - Options: `depth_anything_v3` (default, commercial), `depth_pro` (research-only)
- `--depth-device TEXT`: Device for depth inference (default: `cpu`)
  - Options: `cpu`, `cuda`, `mps`

### Feature Toggles

- `--materials-v3 TEXT`: Enable Materials V3 surface-aware finishing (default: `off`)
  - Options: `on`, `off`, `true`, `false`, `yes`, `no`, `1`, `0`
- `--pbr TEXT`: Enable PBR map generation (normal, roughness, AO) (default: `off`)
  - Options: Same as above
- `--cache-depth TEXT`: Enable content-addressable depth cache (default: `off`)
  - Options: Same as above

### Output Deliverables

- `--emit-master16 TEXT`: Emit master 16-bit output (default: `off`)
- `--emit-upscaled16 TEXT`: Emit upscaled 16-bit output (default: `off`)
- `--emit-marketing TEXT`: Emit marketing-ready output (default: `off`)
- `--emit-report TEXT`: Emit processing report (default: `on`)
- `--emit-run-card TEXT`: Emit run card for reproducibility (default: `on`)

### License Acknowledgements

- `--non-commercial-ok TEXT`: Acknowledge non-commercial license restrictions (default: `false`)
  - Required for: Depth Anything V3.1, Depth Pro
- `--accept-apple-depth-pro-research-license TEXT`: Accept Apple Depth Pro research license (default: `false`)
  - Required for: Depth Pro backend

### Processing Flags

- `--overwrite`: Force reprocessing even if outputs exist
- `--force-depth`: Force depth recomputation (ignore cache)

### Logging

- `--verbose` / `-v`: Enable verbose logging
- `--quiet` / `-q`: Suppress all output except errors
- `--log-level TEXT`: Set log level (DEBUG, INFO, WARNING, ERROR)

## Quality Tiers

### Standard
- Default quality level
- Optimized for speed and moderate quality
- Suitable for drafts and previews

### Premium
- High-quality processing
- Balanced speed and quality
- Suitable for professional work

### APEX
- Maximum quality processing
- Full orchestrator path with all features enabled
- Includes PBR map generation, Materials V3, and all deliverables
- Suitable for final production and client deliverables

## Output Deliverables

When APEX mode is enabled with all emit flags, the following outputs are generated:

### Depth Assets
- `*_depth.png`: 16-bit PNG depth map (quantized for compatibility)
- `*_depth.npy`: Float32 depth array (high-precision, used for PBR)

### PBR Maps (when `--pbr on`)
- `*_normal.png`: Normal map for lighting calculations
- `*_roughness.png`: Roughness map for material appearance
- `*_ao.png`: Ambient occlusion map for contact shadows

### Enhanced Images
- `*_master16.tiff`: Master 16-bit output (when `--emit-master16 on`)
- `*_upscaled16.tiff`: Upscaled 16-bit output (when `--emit-upscaled16 on`)
- `*_marketing.jpg`: Marketing-ready 8-bit output (when `--emit-marketing on`)

### Metadata
- `*_combined.json`: Processing manifest with provenance (when `--emit-report on`)
- `*_run_card.json`: Run card for reproducibility tracking (when `--emit-run-card on`)

## Example Workflows

### Workflow 1: Draft Preview (Fast)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/draft" \
  --preset "default" \
  --quality-tier "standard" \
  --depth-device "cpu"
```

### Workflow 2: Client Deliverable (APEX Commercial)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/client_deliverable" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-device "cuda" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on" \
  --emit-report "on" \
  --emit-run-card "on"
```

### Workflow 3: Research Experiment (APEX+ Non-Commercial)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/research" \
  --preset "depth-anything-v3.1-research-m4" \
  --quality-tier "apex" \
  --non-commercial-ok "true" \
  --depth-device "cuda" \
  --materials-v3 "on" \
  --pbr "on" \
  --emit-report "on"
```

## Troubleshooting

### "Input directory does not exist"
Ensure the `--input-dir` path is correct and the directory exists.

### "Depth Pro backend requires --non-commercial-ok true"
When using `--depth-backend depth_pro`, you must set `--non-commercial-ok true` and `--accept-apple-depth-pro-research-license true`.

### "No images found in [directory]"
The input directory must contain at least one supported image format:
- `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.webp`

### Missing ML Dependencies
If you see warnings about missing torch, transformers, or coremltools:
```bash
pip install -e ".[ml]"
```

## Additional Resources

- [Depth Pipeline README](../depth_pipeline/DEPTH_PIPELINE_README.md)
- [Architecture Decision Record: Depth Backend Unification](../architecture/ADR-019-depth-backend-unification.md)
- [Architecture Decision Record: Depth Pro Integration](../architecture/ADR-018-depth-pro-integration.md)
- [PBR CLI Coverage Report](../PBR_CLI_COVERAGE_REPORT.md)
