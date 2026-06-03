# Transformation Portal Quick Start Cheat Sheet

Use this page as a compact operator reference. For complete setup details, use
[SETUP_GUIDE.md](../guides/SETUP_GUIDE.md); for Lux Depth V3 commands, use
[LUX_DEPTH_V3_CLI_GUIDE.md](../cli/LUX_DEPTH_V3_CLI_GUIDE.md).

## Setup

```bash
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

make venv
make install-core
make check-environment

.venv/bin/python -c "import transformation_portal; print('Ready')"
```

Optional model runtimes are installed through governed Make targets and setup
scripts. The checked-in ML core lock is target-owned for native macOS Apple
Silicon; Linux and macOS Intel ML lanes are retired unsupported lanes that fail
closed until a governed lock is re-established.

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

Do not install ML, CUDA, RAW, or umbrella extras ad hoc into the repo `.venv`.
Use the current lock-backed targets and runtime scripts above.

## Common Tasks

### Check Image Processing Readiness

```bash
.venv/bin/python scripts/check_image_processing_readiness.py
.venv/bin/python scripts/check_image_processing_readiness.py --quick-start
```

### Minimal Image Adjustment

```bash
.venv/bin/python scripts/simple_image_processor.py input_images/render.jpg \
  --brightness 1.1 \
  --contrast 1.05 \
  --saturation 1.1 \
  --output output/render_basic.jpg
```

### Lux Depth V3 APEX Run

```bash
.venv/bin/lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/lux_depth_v3_apex \
  --preset premium \
  --quality-tier apex \
  --depth-backend da3 \
  --model-key da3-metric \
  --materials-v3 on \
  --pbr on \
  --cache-depth on \
  --emit-master16 on \
  --emit-upscaled16 on \
  --emit-marketing on \
  --emit-report on \
  --emit-run-card on \
  --run-card-version v2 \
  --overwrite
```

### Luxury TIFF Batch

```bash
.venv/bin/luxury-tiff-batch input_images/tiff output/tiff_lux \
  --preset signature \
  --profile balanced \
  --recursive
```

### AI Render Refinement

```bash
.venv/bin/lux_render \
  --input-glob "input_images/renders/*.png" \
  --out output/lux_render \
  --prompt "luxury bedroom interior, natural light, hardwood floor" \
  --material-response \
  --texture-boost 0.28
```

### Video Tour Grading

```bash
.venv/bin/luxury_video_grader input/tour.mp4 output/tour_graded.mp4 \
  --preset signature_estate
```

## CLI Quick Reference

| Surface | Current entrypoint | Notes |
|---------|--------------------|-------|
| Environment | `make venv`, `make install-core`, `make check-environment` | Repo-managed `.venv` contract |
| Image readiness | `.venv/bin/python scripts/check_image_processing_readiness.py` | Shows available tiers and missing optional runtimes |
| Minimal image edits | `.venv/bin/python scripts/simple_image_processor.py` | Core Pillow/numpy path |
| Lux Depth V3 | `.venv/bin/lux-depth-v3` | Main APEX/Lux Depth V3 CLI |
| TIFF batch | `.venv/bin/luxury-tiff-batch` | 16-bit TIFF batch processing |
| Lux render | `.venv/bin/lux_render` | Uses `--input-glob`, not the retired `--input` flag |
| Video grading | `.venv/bin/luxury_video_grader` | Preset-driven FFmpeg grader |

## Capability Tiers

| Tier | Install path | Typical use |
|------|--------------|-------------|
| Core | `make install-core` | Basic processing, CLI contracts, validation, docs |
| ML core | `make install-ml-core` | Native macOS Apple Silicon ML baseline |
| SAM2 | `make install-ml-sam2` | Native macOS Apple Silicon SAM2 support |
| DA3 | `./scripts/setup/install_da3_runtime.sh --profile baseline` | Lux V3 relative-depth runtime |
| Depth Pro | `./scripts/setup/install_depth_pro_runtime.sh` | Apple Depth Pro runtime |
| RAW | `./scripts/setup/install_raw_runtime.sh` | RAW image loader runtime |
| FastVLM | `./scripts/setup/install_fastvlm_runtime.sh` | Optional advisory captioning runtime |

## Output Expectations

| Workflow | Output shape |
|----------|--------------|
| Minimal image adjustment | Single adjusted image at the requested `--output` path |
| Lux Depth V3 | Depth artifacts, material/PBR outputs, marketing exports, report, and run card when enabled |
| TIFF batch | Processed TIFF files under the output directory, preserving high-fidelity metadata where supported |
| Lux render | AI-refined render outputs under `--out` |
| Video grading | Graded MP4/MOV master at the requested output path |

## Troubleshooting

### Module Import Fails

```bash
make install-core
make check-environment
```

### Optional Runtime Missing

```bash
.venv/bin/python scripts/check_image_processing_readiness.py
```

Install only the reported governed runtime. Do not substitute ad hoc package
installs for a failing Make target or setup script.

### Lux Render Flag Error

Use `--input-glob`:

```bash
.venv/bin/lux_render --input-glob "input_images/*.png" \
  --out output/lux_render \
  --prompt "luxury interior"
```

### FFmpeg Missing For Video Grading

```bash
brew install ffmpeg
```

Linux package names vary by distribution; install FFmpeg through the platform's
package manager before running `.venv/bin/luxury_video_grader`.

## Current References

- [Project README](../../README.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
- [Setup Guide](../guides/SETUP_GUIDE.md)
- [Image Processing Readiness](../guides/IMAGE_PROCESSING_READINESS.md)
- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md)
