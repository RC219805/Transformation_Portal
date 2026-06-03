# Image Processing Readiness Status

This status note is a quick pointer for local image-processing readiness. The
authoritative setup and tier details live in
[Image Processing Readiness Guide](../guides/IMAGE_PROCESSING_READINESS.md).

## Current Status

The repository supports a minimal local image-processing tier through the
repo-managed Python environment. Do not rely on a fixed historical machine
snapshot here; verify the current checkout before processing client images.

```bash
make venv
make install-core
.venv/bin/python scripts/check_image_processing_readiness.py
.venv/bin/python scripts/check_image_processing_readiness.py --quick-start
```

The readiness checker reports installed packages, disk capacity, FFmpeg
availability, sample-image availability, available operations, and the next
governed setup command for missing optional tiers.

## Immediate Minimal Processing

Use the minimal processor for basic local adjustments:

```bash
.venv/bin/python scripts/simple_image_processor.py input_images/test_render.jpg \
  --brightness 1.1 \
  --contrast 1.05 \
  --saturation 1.1 \
  --output output/test_render_processed.jpg \
  --verbose
```

Resize or convert formats with the same entrypoint:

```bash
.venv/bin/python scripts/simple_image_processor.py input_images/test_render.jpg \
  --width 1280 \
  --height 720 \
  --output output/web_preview.jpg

.venv/bin/python scripts/simple_image_processor.py input_images/image.png \
  --output output/converted.jpg \
  --quality 95
```

## Capability Tiers

| Tier | Setup command | Typical workflows |
|------|---------------|-------------------|
| Minimal | `make install-core` | Format conversion, resize/crop, brightness, contrast, saturation, basic metadata |
| Standard | `make install-core` | Minimal tier plus TIFF/LUT-oriented workflows when optional image packages are present |
| Full ML | `make install-ml-core` plus target runtime installers | Governed ML, depth, segmentation, and rendering workflows |

Optional runtimes are explicit and isolated:

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
.venv/bin/python scripts/setup/download_depth_models.py
```

## Maintained Processing Entrypoints

Use the maintained console scripts after the relevant setup tier is installed:

```bash
.venv/bin/luxury-tiff-batch input_images/tiff output/tiff_lux \
  --preset signature \
  --profile balanced \
  --recursive

.venv/bin/lux-depth-v3 \
  --input-dir ./input_images \
  --output-dir ./output/lux_depth_v3 \
  --model-key da3-metric \
  --quality-tier standard

.venv/bin/lux_render \
  --input-glob 'input_images/*.png' \
  --out output/lux_render \
  --prompt 'luxury interior, natural light'
```

## Related Documentation

- [Image Processing Readiness Guide](../guides/IMAGE_PROCESSING_READINESS.md)
- [Supported File Formats](../guides/SUPPORTED_FILE_FORMATS.md)
- [Pipeline Operations Guide](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md)
- [Repository README](../../README.md)
