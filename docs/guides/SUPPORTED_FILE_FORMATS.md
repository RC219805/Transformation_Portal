# Supported File Formats

This maintained guide describes the file formats accepted by the current
Transformation Portal workflows and the supported entrypoints for processing
them. Commands assume the repository-managed `.venv` from `make install-core`.

For command sequencing across workflows, use
[Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md).

## Quick Reference

| Workflow | Maintained entrypoint | Primary input formats | Output shape |
|----------|----------------------|-----------------------|--------------|
| Minimal image operations | `.venv/bin/python scripts/simple_image_processor.py` | JPEG, PNG, TIFF, WebP, BMP | Single adjusted image |
| Lux Depth V3 | `.venv/bin/lux-depth-v3` | JPEG, PNG, TIFF, WebP, BMP, RAW via governed runtime | Depth/PBR/material outputs, report, run card |
| TIFF batch | `.venv/bin/luxury-tiff-batch` | TIFF | Processed TIFF directory |
| Lux render | `.venv/bin/lux_render` | PIL-readable images | Refined PNG outputs |
| Video grading | `.venv/bin/luxury_video_grader` | MP4, MOV, AVI, MKV, WebM | Graded MP4/MOV master |

## Setup

```bash
make venv
make install-core
make check-environment
```

Optional format/runtime support is installed through governed targets and setup
scripts:

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

Do not replace these with ad hoc optional dependency installs in current
operator guidance.

## Image Formats

### Core Image Formats

The core image utilities are extension-normalized and case-insensitive:

| Format | Extensions | Best use |
|--------|------------|----------|
| PNG | `.png` | Lossless render and web delivery |
| JPEG | `.jpg`, `.jpeg` | Web/gallery delivery |
| TIFF | `.tif`, `.tiff` | Professional and print workflows |
| WebP | `.webp` | Modern web previews |
| BMP | `.bmp` | Legacy uncompressed inputs |
| GIF | `.gif` | Limited support, first frame for image workflows |
| ICO | `.ico` | Icon inspection and conversion paths |
| Netpbm | `.ppm`, `.pgm`, `.pbm` | Utility conversion paths |
| TGA | `.tga` | Legacy render interchange |

Use `transformation_portal.utils.format_utils` for extension-level validation:

```python
from transformation_portal.utils.format_utils import (
    get_format_info,
    is_supported_image_format,
    validate_format,
)

validate_format("render.TIFF", "image")
assert is_supported_image_format("render.TIFF")
print(get_format_info("render.TIFF")["recommendations"])
```

For content-based detection, use
`transformation_portal.utils.format_utils_enhancements` when that optional
surface is available in the environment.

## Video Formats

Video grading uses FFmpeg through `.venv/bin/luxury_video_grader`.

| Format | Extensions | Notes |
|--------|------------|-------|
| MP4 | `.mp4`, `.m4v` | H.264/H.265 delivery |
| MOV | `.mov` | ProRes and QuickTime masters |
| AVI | `.avi` | Legacy interchange |
| MKV | `.mkv` | Matroska container |
| WebM | `.webm` | VP8/VP9 web delivery |
| FLV | `.flv` | Legacy input support |

HDR video support is handled by the video grader's FFmpeg tone-mapping options.

## Workflow Examples

### Minimal Image Adjustment

```bash
.venv/bin/python scripts/simple_image_processor.py input_images/render.jpg \
  --brightness 1.1 \
  --contrast 1.05 \
  --saturation 1.1 \
  --output output/render_basic.jpg
```

### Lux Depth V3 With TIFF Or PNG Inputs

```bash
.venv/bin/lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/lux_depth_v3_apex \
  --quality-tier apex \
  --depth-backend da3 \
  --model-key da3-metric \
  --materials-v3 on \
  --pbr on \
  --emit-master16 on \
  --emit-upscaled16 on \
  --emit-run-card on \
  --run-card-version v2 \
  --overwrite
```

### TIFF Batch Processing

```bash
.venv/bin/luxury-tiff-batch input_images/tiff output/tiff_lux \
  --preset signature \
  --profile balanced \
  --recursive
```

### Lux Render Refinement

```bash
.venv/bin/lux_render \
  --input-glob "input_images/renders/*.png" \
  --out output/lux_render \
  --prompt "luxury interior, natural light" \
  --material-response
```

### Video Grading

```bash
.venv/bin/luxury_video_grader input/tour.mp4 output/tour_graded.mov \
  --preset signature_estate \
  --overwrite
```

## Recommendations By Use Case

### Architectural Rendering

| Stage | Recommended format | Reason |
|-------|--------------------|--------|
| Raw renders | TIFF or PNG | Preserve render detail |
| Depth artifacts | PNG/TIFF plus run-card artifacts | Stable inspection and reproducibility |
| AI refinement | PNG inputs and PNG outputs | Good quality/size balance for diffusion workflows |
| Final delivery | PNG for web, TIFF for print | Match delivery channel |

### Real Estate Photography

| Stage | Recommended format | Reason |
|-------|--------------------|--------|
| Camera RAW ingest | RAW runtime or external 16-bit TIFF conversion | Preserve sensor data and metadata |
| Batch processing | TIFF | High-fidelity color pipeline |
| Web gallery | JPEG at high quality | Efficient delivery |
| Print/archive | 16-bit TIFF | Maximum quality |

### Video Production

| Stage | Recommended format | Reason |
|-------|--------------------|--------|
| Editing proxies | MP4 | Fast preview/editing |
| Master grading | MOV/ProRes | High-quality grading target |
| Web delivery | MP4/H.265 | Streaming efficiency |
| Archive | MOV/ProRes | Durable mezzanine format |

## RAW Files

RAW ingest is optional and isolated. Install and validate it through the
governed runtime:

```bash
./scripts/setup/install_raw_runtime.sh
.venv/bin/python scripts/check_image_processing_readiness.py
```

If RAW preview fallback is required for local investigation, keep it explicit in
the consuming command's documented flags and do not treat preview output as
quality-gate evidence.

## Technical Constraints

### Bit Depth

| Workflow | 8-bit | 16-bit input | 16-bit output |
|----------|-------|--------------|---------------|
| Minimal image operations | Yes | Limited by Pillow path | Output path dependent |
| TIFF batch | Yes | Yes | Yes where supported |
| Lux Depth V3 | Yes | Accepted for supported image types | Controlled by emit flags |
| Lux render | Yes | Accepted, converted for model path | PNG output by default |
| Video grading | Codec dependent | HDR video supported | Codec/flag dependent |

### Color And Metadata

- ICC profile preservation depends on the workflow and output format.
- TIFF batch workflows are the preferred path for metadata-sensitive stills.
- Video color metadata is controlled by `luxury_video_grader` options.
- Lux Depth V3 reproducibility metadata lives in emitted reports and run cards.

### File Size

There is no hard repository-level file-size limit for local inputs, but very
large images and videos are constrained by host memory, storage, and optional ML
runtime availability. Keep batch outputs under explicit `output/` subpaths.

## Error Handling

Typical failures and next steps:

| Symptom | Next step |
|---------|-----------|
| Unsupported image extension | Validate with `transformation_portal.utils.format_utils.validate_format()` |
| TIFF quality path missing | Run `make install-core` and the readiness check |
| RAW decode unavailable | Run `./scripts/setup/install_raw_runtime.sh` |
| Optional ML path unavailable | Run the readiness check and governed ML/runtime installer |
| FFmpeg missing | Install FFmpeg through the host package manager |

## Validation

```bash
.venv/bin/pytest tests/test_format_utils.py tests/test_image_processing_readiness.py -q
python3 scripts/governance/check_docs_structure.py --all
./.auto-organize.sh --check --verbose
```

## See Also

- [Quick Start Cheat Sheet](../reference/QUICKSTART_CHEATSHEET.md)
- [Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md)
- [Image Processing Readiness](IMAGE_PROCESSING_READINESS.md)
- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
