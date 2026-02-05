# Lux Depth V3 - Orchestrated Depth + Enhancement Pipeline

Lux Depth V3 is a production-grade orchestrator for depth-aware image processing with optional AI-powered enhancement.

## Overview

The Lux Depth V3 pipeline provides:
- **Depth Estimation** using Depth Anything V3 (commercial) or Depth Pro (research)
- **PBR Map Generation** (normal, roughness, ambient occlusion)
- **Materials V3** surface-aware finishing
- **V2 Enhancement** (optional AI-powered refinement)
- **APEX Quality Tier** support for commercial production

## Quick Start

### Commercial Production (PBR-Only)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/commercial" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --pbr "on" \
  --enable-v2 "off" \
  --emit-master16 "on"
```

### Commercial Production (With Enhancement)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/enhanced" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --pbr "on" \
  --materials-v3 "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on"
```

### Research Experiment

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/research" \
  --preset "depth-anything-v3.1-research-m4" \
  --non-commercial-ok "true" \
  --quality-tier "apex" \
  --pbr "on"
```

## Key Concepts

### V2 Enhancement (Optional)

The V2 enhancement stage is **optional** and enabled by default for backward compatibility.

**To disable V2 enhancement:**
```bash
--enable-v2 "off"
```

**Why disable V2?**
- PBR-only workflows (depth + maps only)
- Enhancement script not available
- Custom post-processing pipelines
- Faster iteration during development

**V2 is independent:** All other pipeline features (depth, PBR, Materials V3) work without V2.

### Quality Tier vs Preset

**Use `--quality-tier` for most workflows:**
- `standard` - Fast/draft quality
- `premium` - Balanced quality
- `apex` - Maximum quality for production

**Use `--preset` for specialized scenarios:**
- Research model configurations
- Fine-tuned parameter combinations
- Non-commercial depth models

**Recommendation:** Start with `--quality-tier`, add `--preset` only when needed.

## Common Workflows

### PBR-Only (No Enhancement)

**Use Case:** Generate depth and PBR maps for 3D workflows, game engines, or technical visualization.

```bash
lux-depth-v3 \
  --input-dir "./input" \
  --output-dir "./output/pbr_only" \
  --quality-tier "apex" \
  --pbr "on" \
  --enable-v2 "off" \
  --depth-device "mps"
```

**Outputs:**
- `*_depth.png` - 16-bit depth map
- `*_depth.npy` - Float32 depth array
- `*_normal.png` - Normal map
- `*_roughness.png` - Roughness map
- `*_ao.png` - Ambient occlusion map

### Client Deliverable (APEX)

**Use Case:** Maximum quality output for client deliverables and final production.

```bash
lux-depth-v3 \
  --input-dir "./input" \
  --output-dir "./output/client" \
  --quality-tier "apex" \
  --depth-device "cuda" \
  --pbr "on" \
  --materials-v3 "on" \
  --enable-v2 "off" \
  --cache-depth "on" \
  --emit-master16 "on" \
  --emit-upscaled16 "on" \
  --emit-marketing "on"
```

**Outputs:**
- All PBR maps
- `*_master16.tiff` - Master 16-bit output
- `*_upscaled16.tiff` - Upscaled 16-bit output
- `*_marketing.jpg` - Marketing-ready 8-bit JPEG
- `*_combined.json` - Processing manifest
- `*_run_card.json` - Reproducibility card

### Fast Iteration (Development)

**Use Case:** Quick validation during development.

```bash
lux-depth-v3 \
  --input-dir "./input" \
  --output-dir "./output/dev" \
  --quality-tier "standard" \
  --depth-device "cpu" \
  --pbr "off" \
  --enable-v2 "off"
```

## Input Discovery

The pipeline automatically excludes derived artifacts from input discovery to prevent nonsensical reprocessing:

### Excluded Artifacts

- **Depth maps:** `*_depth.png`, `*_depthpro_depth16.png`
- **PBR maps:** `*_normal.png`, `*_roughness.png`, `*_ao.png`
- **Output directories:** `depth/`, `pbr/`, `v2/`, `manifests/`, `logs/`
- **Hidden files:** `.DS_Store`, `.cache/`
- **Intermediate directories:** `_non_source/`

### Default Behavior

The pipeline silently excludes artifacts and logs a summary:

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --quality-tier "standard"
```

**Output:**
```
INFO: Discovered 17 images, excluded 3 artifacts
```

### Validation Mode (Strict)

Use `--strict-inputs` to fail if artifacts are found (useful for CI/CD validation):

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --strict-inputs
```

**Error (if artifacts found):**
```
ERROR: Strict mode: 3 excluded artifacts found in ./input_images
ERROR:   - image_depthpro_depth16.png (matched: _depthpro_depth16)
ERROR:   - output/depth/result.png (matched: /depth/)
```

**Why exclude artifacts?** Processing depth maps as RGB inputs creates nonsensical results (depth of depth), feedback loops, and data corruption.

**Full documentation:** [docs/input_hygiene.md](../../../../docs/input_hygiene.md)

## Troubleshooting

### "Script not found" Error

**Error:**
```
ERROR: V2 enhancement script not found: scripts/enhance_image.py
```

**Fix:** Add `--enable-v2 "off"` to your command.

**Why:** V2 is enabled by default and validates the enhancement script at startup. This is **correct fail-fast design** to prevent wasted processing. For PBR-only workflows, disable V2.

### More Help

- **Full Troubleshooting Guide:** [docs/LUX_DEPTH_V3_TROUBLESHOOTING.md](../../../../docs/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- **CLI Reference:** [docs/LUX_DEPTH_V3_CLI_GUIDE.md](../../../../docs/LUX_DEPTH_V3_CLI_GUIDE.md)
- **CLI Help:** `lux-depth-v3 --help`

## Architecture

```
Input Images
    ↓
Depth Estimation (Depth Anything V3 or Depth Pro)
    ↓
PBR Generation (Normal, Roughness, AO) [Optional]
    ↓
Materials V3 (Surface-aware finishing) [Optional]
    ↓
V2 Enhancement (AI-powered refinement) [Optional]
    ↓
Output Deliverables
```

### Key Components

- **`__main__.py`** - CLI entry point with typer
- **`orchestrator.py`** - Main pipeline orchestration
- **`pbr_processor.py`** - PBR map generation
- **`materials_v3.py`** - Surface-aware finishing
- **`v2_runner.py`** - V2 enhancement execution
- **`config.py`** - Configuration and presets

## License Compliance

### Commercial-Safe (Default)

**Depth Anything V3** (Apache 2.0)
- ✅ Commercial use allowed
- ✅ No license flags required
- Recommended for production

### Research-Only (Explicit Opt-In)

**Depth Anything V3.1** (CC BY-NC 4.0)
```bash
--preset "depth-anything-v3.1-research-m4" \
--non-commercial-ok "true"
```

**Apple Depth Pro** (AMLR Research License)
```bash
--depth-backend "depth_pro" \
--non-commercial-ok "true" \
--accept-apple-depth-pro-research-license "true"
```

The CLI **enforces license compliance** at startup to prevent accidental violations.

## Performance Optimization

### GPU Acceleration

**NVIDIA CUDA:**
```bash
--depth-device "cuda"
```

**Apple Silicon (M1/M2/M3/M4):**
```bash
--depth-device "mps"
```

### Depth Caching

Enable content-addressable caching for faster iterations:
```bash
--cache-depth "on"
```

Cached depth maps are reused across runs, dramatically speeding up parameter exploration.

### Batch Processing

The pipeline automatically processes all images in `--input-dir` recursively. Supported formats:
- `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.webp`

## Output Structure

```
output_dir/
├── depth/                    # Depth maps (PNG and NPY)
├── pbr/                      # PBR maps (when --pbr on)
│   ├── normal/
│   ├── roughness/
│   └── ao/
├── master16/                 # Master 16-bit outputs (when --emit-master16 on)
├── upscaled16/               # Upscaled outputs (when --emit-upscaled16 on)
├── marketing/                # Marketing-ready JPEGs (when --emit-marketing on)
└── manifests/                # Processing metadata (when --emit-report on)
```

## Python API

### Basic Usage

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import EnhanceConfig, EnhanceOrchestrator

config = EnhanceConfig(
    quality_tier="apex",
    enable_v2=False,  # Disable V2 for PBR-only
    generate_pbr=True,
    depth_device="mps"
)

orchestrator = EnhanceOrchestrator(
    config=config,
    output_root=Path("./output")
)

results = orchestrator.enhance_batch(
    input_dir=Path("./input_images"),
    image_extensions=[".jpg", ".png"]
)
```

### PBR-Only Processing

```python
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

preset = get_preset("premium")
processor = PBRProcessor(config=preset)

# Process single image
pbr_maps = processor.process_image(
    image_path=Path("image.jpg"),
    output_dir=Path("./output/pbr")
)

# pbr_maps contains: normal_path, roughness_path, ao_path
```

## Development

### Running Tests

```bash
# Core tests (fast)
pytest tests/lux_depth_v3/ -v -m "not ml and not slow"

# ML tests (requires ML dependencies)
pytest tests/lux_depth_v3/ -v -m "ml"
```

### Linting

```bash
# From repository root
flake8 src/transformation_portal/lux_depth_v3/
pylint src/transformation_portal/lux_depth_v3/
```

## Additional Resources

- **Troubleshooting Guide:** [docs/LUX_DEPTH_V3_TROUBLESHOOTING.md](../../../../docs/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- **CLI Guide:** [docs/LUX_DEPTH_V3_CLI_GUIDE.md](../../../../docs/LUX_DEPTH_V3_CLI_GUIDE.md)
- **Architecture:** [docs/ARCHITECTURE.md](../../../../docs/ARCHITECTURE.md)
- **Main README:** [README.md](../../../../README.md)
