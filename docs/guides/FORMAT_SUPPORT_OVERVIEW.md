# Format Support Overview

This page is a navigation layer for current file-format guidance. The maintained
format authority is [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md);
[FILE_FORMAT_QUICK_REFERENCE.md](FILE_FORMAT_QUICK_REFERENCE.md) is the compact
operator version.

## Current Format Docs

| Document | Purpose |
|----------|---------|
| [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md) | Maintained format policy, workflow examples, validation, and runtime setup |
| [FILE_FORMAT_QUICK_REFERENCE.md](FILE_FORMAT_QUICK_REFERENCE.md) | Short command reference for daily use |
| [IMAGE_PROCESSING_READINESS.md](IMAGE_PROCESSING_READINESS.md) | Capability tier detection and governed install guidance |
| [../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md](../pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md) | End-to-end processing commands |

## Setup Contract

Use the repo-managed environment:

```bash
make venv
make install-core
make check-environment
```

Optional format/runtime support is governed:

```bash
make install-ml-core
make install-ml-sam2
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_da3_runtime.sh --profile baseline
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
```

## Quick Format Checks

```bash
.venv/bin/python examples/validate_file_formats.py image.jpg
.venv/bin/python examples/validate_file_formats.py --scan ./input_images
.venv/bin/python examples/validate_file_formats.py --formats
```

Programmatic validation lives in `transformation_portal.utils.format_utils`:

```python
from transformation_portal.utils.format_utils import get_format_info, validate_format

validate_format("render.TIFF", "image")
print(get_format_info("render.TIFF")["recommendations"])
```

## Processing Entrypoints

| Workflow | Current entrypoint |
|----------|--------------------|
| Minimal image adjustments | `.venv/bin/python scripts/simple_image_processor.py` |
| Lux Depth V3 | `.venv/bin/lux-depth-v3` |
| TIFF batch | `.venv/bin/luxury-tiff-batch` |
| Lux render | `.venv/bin/lux_render` |
| Video grading | `.venv/bin/luxury_video_grader` |

Use `--input-glob` for Lux Render. Use `--model-key da3-metric` for the
commercial-safe Lux Depth V3 DA3 path.

## Validation

```bash
.venv/bin/pytest tests/test_format_utils.py -q
python3 scripts/governance/check_docs_structure.py --all
./.auto-organize.sh --check --verbose
```

## Current Authority

Current navigation is governed by
[DOCUMENTATION_MAP.md](../governance/DOCUMENTATION_MAP.md). Historical format
reports and old implementation notes are not operator guidance unless this map
promotes them.
