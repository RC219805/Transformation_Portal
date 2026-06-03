# Image Processing Readiness Guide

This guide explains which image-processing workflows are available in a local
checkout and which governed setup commands enable each tier. For format policy,
see [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md). For end-to-end
processing commands, see
[../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md).

## Check Readiness

Start from the repo-managed environment:

```bash
make venv
make install-core
make check-environment
```

Then inspect the local capability tier:

```bash
.venv/bin/python scripts/check_image_processing_readiness.py
.venv/bin/python scripts/check_image_processing_readiness.py --quick-start
```

The readiness checker reports installed packages, disk capacity, FFmpeg
availability, sample-image availability, available operations, and the next
install command for missing optional tiers.

## Capability Tiers

| Tier | Setup command | Available workflows |
|------|---------------|---------------------|
| Minimal | `make install-core` | Format validation, resize/crop, brightness/contrast/saturation, basic metadata reads |
| Standard | `make install-core` | Minimal tier plus TIFF/LUT-oriented workflows when optional image packages are present |
| Full ML | `make install-ml-core` plus target runtime installers | Standard tier plus governed ML/depth/rendering workflows |

Optional runtime installers are explicit and isolated:

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
.venv/bin/python scripts/setup/download_depth_models.py
```

Use `make install-ml-core` as the supported Apple Silicon ML baseline. Advanced
Apple Silicon bootstrap profiles still route through
`./scripts/bootstrap/install_ml_stack.sh`, but current operator guidance should
prefer the Make target unless you are working directly on bootstrap coverage.

## Quick Starts

### Minimal Image Adjustment

```bash
.venv/bin/python scripts/simple_image_processor.py input_images/render.jpg \
  --brightness 1.1 \
  --contrast 1.05 \
  --saturation 1.1 \
  --output output/render_basic.jpg
```

### TIFF Batch Workflow

```bash
.venv/bin/luxury-tiff-batch input_images/tiff output/tiff_lux \
  --preset signature \
  --profile balanced \
  --recursive
```

### Lux Depth V3

```bash
.venv/bin/lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/lux_depth_v3 \
  --quality-tier apex \
  --model-key da3-metric \
  --emit-report on \
  --emit-run-card on \
  --overwrite
```

### Lux Render

```bash
.venv/bin/lux_render \
  --input-glob "input_images/renders/*.png" \
  --out output/lux_render \
  --prompt "luxury interior, natural light"
```

### Video Grading

```bash
.venv/bin/luxury_video_grader input/tour.mp4 output/tour_graded.mov \
  --preset signature_estate \
  --overwrite
```

## Samples

Use existing local images under `input_images/`, or download the repo samples:

```bash
.venv/bin/python scripts/download_samples.py
```

Client input images and generated outputs are local runtime data. Do not commit
them unless a test fixture or documentation example explicitly requires a small
tracked asset.

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Minimal tier is not ready | Run `make install-core` and `make check-environment`, then rerun the readiness checker |
| TIFF operations are unavailable | Run `make install-core`, then `.venv/bin/python scripts/check_image_processing_readiness.py` |
| RAW files do not load | Run `./scripts/setup/install_raw_runtime.sh` |
| DA3 depth runtime is missing | Run `./scripts/setup/install_da3_runtime.sh --profile baseline` |
| Depth Pro runtime is missing | Run `./scripts/setup/install_depth_pro_runtime.sh` |
| Legacy Lux Render flags fail | Use `.venv/bin/lux_render --input-glob ... --out ...` |
| Video grading fails before processing | Install FFmpeg with the host package manager and rerun the readiness checker |
| ML packages are absent | Run `make install-ml-core`; do not install unpinned ML packages ad hoc |

## Validation

```bash
.venv/bin/pytest tests/test_image_processing_readiness.py -q
.venv/bin/pytest tests/test_format_utils.py -q
python3 scripts/governance/check_docs_structure.py --all
./.auto-organize.sh --check --verbose
```

## Current Authority

- [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md) is the maintained
  format policy.
- [FILE_FORMAT_QUICK_REFERENCE.md](FILE_FORMAT_QUICK_REFERENCE.md) is the
  compact command reference.
- [../cli/LUX_DEPTH_V3_CLI_GUIDE.md](../cli/LUX_DEPTH_V3_CLI_GUIDE.md) is the
  detailed Lux Depth V3 contract.
- [../governance/DOCUMENTATION_MAP.md](../governance/DOCUMENTATION_MAP.md) is
  the current documentation navigation authority.
