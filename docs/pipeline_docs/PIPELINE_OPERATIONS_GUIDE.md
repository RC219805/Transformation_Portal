# Transformation Portal Pipeline Operations Guide

This guide covers the maintained local operator paths for image, depth, TIFF,
AI-render, and video workflows. It intentionally uses repo-managed setup,
console scripts, and governed runtime installers instead of retired root scripts
or ad hoc dependency installs.

For setup background, start with [SETUP_GUIDE.md](../guides/SETUP_GUIDE.md).
For Lux Depth V3 flag detail, use
[LUX_DEPTH_V3_CLI_GUIDE.md](../cli/LUX_DEPTH_V3_CLI_GUIDE.md).

## Operating Contract

- Run commands from the repository root.
- Use the repo-managed `.venv`; do not install optional runtimes ad hoc.
- Keep runtime output under `output/`, `.runtime/`, or the explicit output path
  requested by the command.
- Treat optional ML, RAW, Depth Pro, DA3, SAM2, and FastVLM failures as runtime
  readiness issues until evidence shows a product regression.
- Prefer console scripts installed by `make install-core` for maintained CLIs.

## Initial Setup

```bash
make venv
make install-core
make check-environment

.venv/bin/python -c "import transformation_portal; print('Ready')"
```

Optional runtime setup is governed by Make targets and setup scripts:

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

The checked-in ML core lock is target-owned for native macOS Apple Silicon.
Linux and macOS Intel ML lanes are retired unsupported lanes that fail closed
until a governed lock is re-established.

## Pipeline Overview

| Workflow | Maintained entrypoint | Primary output |
|----------|----------------------|----------------|
| Readiness check | `.venv/bin/python scripts/check_image_processing_readiness.py` | Capability tier report |
| Minimal image edits | `.venv/bin/python scripts/simple_image_processor.py` | Adjusted image |
| Lux Depth V3 | `.venv/bin/lux-depth-v3` | Depth, PBR/materials, reports, run cards |
| TIFF batch | `.venv/bin/luxury-tiff-batch` | Processed TIFF directory |
| AI render refinement | `.venv/bin/lux_render` | Refined render images |
| Video grading | `.venv/bin/luxury_video_grader` | Graded MP4/MOV master |

## Readiness Check

```bash
.venv/bin/python scripts/check_image_processing_readiness.py
.venv/bin/python scripts/check_image_processing_readiness.py --quick-start
```

Use this before processing when a host has changed or optional runtimes were
installed recently. The readiness script reports which capability tier is
available and which governed setup path to run next.

## Minimal Image Operations

Minimal operations require only the core environment:

```bash
.venv/bin/python scripts/simple_image_processor.py input_images/render.jpg \
  --brightness 1.1 \
  --contrast 1.05 \
  --saturation 1.1 \
  --output output/render_basic.jpg
```

Use this path for quick format conversion, resizing, and basic color adjustment
when optional ML runtimes are not required.

## Lux Depth V3 APEX Workflow

The maintained APEX depth workflow uses `lux-depth-v3` and the commercial-safe
`da3-metric` selector:

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
  --emit-run-card on \
  --run-card-version v2 \
  --overwrite
```

For research-only models, follow the license-acknowledgement flows in
[LUX_DEPTH_V3_CLI_GUIDE.md](../cli/LUX_DEPTH_V3_CLI_GUIDE.md). Do not replace
the governed selector with an unpinned model path in production guidance.

## TIFF Batch Workflow

Use `luxury-tiff-batch` for high-fidelity TIFF directory processing:

```bash
.venv/bin/luxury-tiff-batch input_images/tiff output/tiff_lux \
  --preset signature \
  --profile balanced \
  --recursive \
  --workers 4
```

Useful options:

| Option | Purpose |
|--------|---------|
| `--preset signature` | Baseline luxury marketing adjustment preset |
| `--profile balanced` | Balance quality, runtime, and output size |
| `--recursive` | Mirror nested input directories |
| `--dry-run` | Preview work without writing files |
| `--overwrite` | Replace existing outputs intentionally |

## AI Render Refinement Workflow

Use `lux_render` for diffusion-backed render refinement. The maintained input
flag is `--input-glob`.

```bash
.venv/bin/lux_render \
  --input-glob "input_images/renders/*.png" \
  --out output/lux_render \
  --prompt "luxury bedroom interior, natural light, hardwood floor" \
  --material-response \
  --texture-boost 0.28
```

If help or startup fails, first check runtime readiness:

```bash
.venv/bin/lux_render --help
.venv/bin/python scripts/check_image_processing_readiness.py
```

Missing optional ML dependencies should be resolved with the governed install
targets, not by direct package installation.

## Video Grading Workflow

Use `luxury_video_grader` for preset-driven FFmpeg grading:

```bash
.venv/bin/luxury_video_grader input/tour.mp4 output/tour_graded.mp4 \
  --preset signature_estate \
  --overwrite
```

Useful options:

| Option | Purpose |
|--------|---------|
| `--list-presets` | Print available grading presets |
| `--preset signature_estate` | Select the baseline grade |
| `--tone-map hable` | Force a tone-mapping operator |
| `--dry-run` | Print the FFmpeg command without executing |
| `--print-filter-graph` | Inspect the generated filter graph |

## Optional Advisory Captioning

FastVLM captioning is optional and advisory. It runs in an isolated runtime and
does not quality-gate visual output:

```bash
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

For live portal validation, use:

```bash
make validate-portal-fastvlm-captioning-live
```

## Validation Ladder

Run focused validation for the workflow you touched:

```bash
.venv/bin/pytest tests/test_image_processing_readiness.py tests/test_lux_render_pipeline_smoke.py -q
python3 scripts/governance/check_docs_structure.py --all
./.auto-organize.sh --check --verbose
make ci-quick
```

For Lux Depth V3 CLI contract changes:

```bash
.venv/bin/pytest tests/test_lux_depth_v3_cli.py -q
make validate-portal-lux-materials-live
```

For browser/frontdoor or orchestrator changes, use the contract gates listed in
[AGENTS.md](../../AGENTS.md) and the current documentation map.

## Troubleshooting

### Import Or Module Errors

```bash
make install-core
make check-environment
```

If optional runtime packages are missing, rerun the readiness script and install
only the governed runtime it identifies.

### Lux Render Argument Errors

Use `--input-glob`; the old `--input` form is not the maintained Lux Render
entrypoint.

```bash
.venv/bin/lux_render --input-glob "input_images/*.png" \
  --out output/lux_render \
  --prompt "luxury interior"
```

### FFmpeg Missing

Install FFmpeg through the host package manager, then rerun:

```bash
brew install ffmpeg
.venv/bin/luxury_video_grader --help
```

### Unsupported ML Lane

Linux, CUDA, and macOS Intel ML lock lanes are retired unsupported lanes in the
current repo state. If an ML setup target fails closed on those hosts, report it
as unsupported runtime posture rather than replacing the lock contract with ad
hoc packages.

## Current References

- [Quick Start Cheat Sheet](../reference/QUICKSTART_CHEATSHEET.md)
- [Image Processing Readiness](../guides/IMAGE_PROCESSING_READINESS.md)
- [Setup Guide](../guides/SETUP_GUIDE.md)
- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Lux Depth V3 Troubleshooting](../guides/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
