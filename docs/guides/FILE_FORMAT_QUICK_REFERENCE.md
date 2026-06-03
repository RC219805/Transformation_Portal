# File Format Quick Reference

Compact current reference for common image and video format workflows. For full
policy, see [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md).

## Supported Formats

| Category | Formats |
|----------|---------|
| Core images | PNG, JPEG, TIFF, WebP, BMP, GIF, ICO, Netpbm, TGA |
| Luxury stills | TIFF and PNG |
| Video | MP4, MOV, AVI, MKV, WebM, M4V, FLV |
| RAW | Optional isolated RAW runtime, then Lux Depth V3 or TIFF workflow |

Extensions are case-insensitive.

## Setup

```bash
make venv
make install-core
make check-environment
```

Optional format and ML runtimes:

```bash
./scripts/setup/install_raw_runtime.sh
make install-ml-core
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
```

## Validate Formats

```bash
.venv/bin/python examples/validate_file_formats.py image.jpg
.venv/bin/python examples/validate_file_formats.py --scan ./input_images
.venv/bin/python examples/validate_file_formats.py --formats
```

## Process Images

```bash
# Minimal adjustment
.venv/bin/python scripts/simple_image_processor.py input_images/render.jpg \
  --brightness 1.1 \
  --contrast 1.05 \
  --output output/render_basic.jpg

# Lux Depth V3
.venv/bin/lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/lux_depth_v3 \
  --quality-tier apex \
  --model-key da3-metric \
  --emit-report on \
  --emit-run-card on \
  --overwrite

# TIFF batch
.venv/bin/luxury-tiff-batch input_images/tiff output/tiff_lux \
  --preset signature \
  --profile balanced \
  --recursive

# Lux render
.venv/bin/lux_render \
  --input-glob "input_images/renders/*.png" \
  --out output/lux_render \
  --prompt "luxury interior, natural light"
```

## Process Video

```bash
.venv/bin/luxury_video_grader input/tour.mp4 output/tour_graded.mov \
  --preset signature_estate \
  --overwrite
```

## Recommended Workflow Formats

| Use case | Input | Processing | Delivery |
|----------|-------|------------|----------|
| Architectural render | PNG or TIFF | Lux Depth V3, Lux Render | PNG or TIFF |
| Real estate photography | RAW or TIFF | RAW runtime, TIFF batch | JPEG for web, TIFF for print |
| Video tour | MP4 or MOV | Video grader | MOV master, MP4 delivery |
| Quick preview | JPEG or PNG | Simple image processor | JPEG or PNG |

## Common Issues

| Symptom | Current fix |
|---------|-------------|
| Unsupported extension | Validate with `examples/validate_file_formats.py` and convert to PNG, JPEG, TIFF, WebP, BMP, or a supported video format |
| TIFF path unavailable | Run `make install-core` and `.venv/bin/python scripts/check_image_processing_readiness.py` |
| RAW decode unavailable | Run `./scripts/setup/install_raw_runtime.sh` |
| Lux Render flag error | Use `--input-glob` |
| FFmpeg missing | Install FFmpeg through the host package manager |

## Python API

```python
from transformation_portal.utils.format_utils import (
    get_format_info,
    is_supported_image_format,
    is_supported_video_format,
    validate_format,
)

validate_format("photo.jpg", "image")
assert is_supported_image_format("photo.jpg")
assert is_supported_video_format("tour.mp4")
print(get_format_info("render.tiff")["recommendations"])
```

## References

- [Supported File Formats](SUPPORTED_FILE_FORMATS.md)
- [Format Support Overview](FORMAT_SUPPORT_OVERVIEW.md)
- [Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md)
- [Documentation Map](../governance/DOCUMENTATION_MAP.md)
