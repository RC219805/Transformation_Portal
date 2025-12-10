# ExportManager Quick Reference

**Phase 2 Slice 2 Component**

## Overview

ExportManager is a first-class abstraction layer for all pipeline export operations, providing clean separation between processing logic and I/O operations.

## Location

- **Module**: `src/transformation_portal/core/storage/export_manager.py`
- **Integration**: `lux_depth_v2/pipeline.py`
- **Tests**: `tests/core/storage/test_export_manager.py`

## Usage

### Basic Initialization

```python
from transformation_portal.core.storage import ExportManager, ExportConfig
from pathlib import Path
import lux_depth_v2.io_utils as io_utils

# Create configuration
config = ExportConfig(output_dir=Path("output"))

# Initialize ExportManager
export_manager = ExportManager(config, io_utils)
```

### Writing Files

```python
import numpy as np

# Create sample image (RGB float32, 0..1)
image = np.random.rand(100, 100, 3).astype(np.float32)

# Write master TIFF (16-bit)
master_path = export_manager.write_master("image001", image)
# Output: output/image001_master16.tif

# Write upscaled TIFF (16-bit)
upscaled_path = export_manager.write_upscaled("image001", image)
# Output: output/image001_upscaled16.tif

# Write preview JPG (8-bit)
preview_path = export_manager.write_preview("image001", image, quality=92)
# Output: output/image001_preview.jpg

# Write marketing PNG (8-bit)
marketing_path = export_manager.write_marketing_png("image001", image)
# Output: output/image001_marketing.png

# Write report JSON
report = {"status": "ok", "timing_s": 1.234}
report_path = export_manager.write_report("image001", report)
# Output: output/image001_report.json
```

### Path Resolution (Skip Existing)

```python
# Get expected paths before processing
master_path = export_manager.get_master_path("image001")
upscaled_path = export_manager.get_upscaled_path("image001")

# Check if files exist
if master_path.exists() and upscaled_path.exists():
    print("Already processed, skipping")
```

### Custom Configuration

```python
# Custom prefixes for organized output
config = ExportConfig(
    output_dir=Path("output"),
    master_prefix="gold_",
    upscaled_prefix="hires_",
    preview_prefix="thumb_"
)

export_manager = ExportManager(config, io_utils)

# Results:
# - gold_image001_master16.tif
# - hires_image001_upscaled16.tif
# - thumb_image001_preview.jpg
```

## API Reference

### ExportConfig

```python
@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path              # Required: Output directory
    master_prefix: str = ""       # Optional: Master file prefix
    upscaled_prefix: str = ""     # Optional: Upscaled file prefix
    preview_prefix: str = ""      # Optional: Preview file prefix
    report_suffix: str = "_report.json"
    master_suffix: str = "_master16"
    upscaled_suffix: str = "_upscaled16"
    marketing_suffix: str = "_marketing"
    preview_jpg_suffix: str = "_preview"
```

### ExportManager Methods

#### Write Operations

```python
def write_master(stem: str, master_arr: np.ndarray, compression: str = "deflate") -> Path
```
Write 16-bit master TIFF. Returns path to written file.

```python
def write_upscaled(stem: str, upscaled_arr: np.ndarray, compression: str = "deflate") -> Path
```
Write 16-bit upscaled TIFF. Returns path to written file.

```python
def write_preview(stem: str, preview_arr: np.ndarray, quality: int = 92) -> Path
```
Write preview JPG. Returns path to written file.

```python
def write_marketing_png(stem: str, png_arr: np.ndarray) -> Path
```
Write 8-bit marketing PNG. Returns path to written file.

```python
def write_report(stem: str, report_dict: dict) -> Path
```
Write processing report JSON. Returns path to written file.

#### Path Getters

```python
def get_master_path(stem: str) -> Path
def get_upscaled_path(stem: str) -> Path
def get_marketing_path(stem: str) -> Path
def get_preview_path(stem: str) -> Path
def get_report_path(stem: str) -> Path
```

## Pipeline Integration

### Automatic Usage

When using `LuxPipelineV2`, ExportManager is automatically initialized:

```python
from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2

config = PipelineConfig(
    output_dir="output",
    preset=Preset.PHOTO_REALISTIC
)

pipeline = LuxPipelineV2(config)
# ExportManager automatically initialized

# Check if available
if pipeline.export_manager:
    print("ExportManager active")
```

### Stage Timing

ExportManager operations are instrumented with timing stages:

```python
report = pipeline.process_one("image.jpg")

# Timing data in report
timing_stages = report["timing_stages_s"]
print(f"Master export: {timing_stages['export_master']:.3f}s")
print(f"Preview export: {timing_stages['export_preview']:.3f}s")
print(f"Upscaled export: {timing_stages['export_upscaled']:.3f}s")
print(f"Marketing export: {timing_stages['export_marketing']:.3f}s")
print(f"Report export: {timing_stages['export_report']:.3f}s")
```

## File Naming Conventions

### Default Naming

| File Type | Pattern | Example |
|-----------|---------|---------|
| Master TIFF | `{stem}_master16.tif` | `sunset_master16.tif` |
| Upscaled TIFF | `{stem}_upscaled16.tif` | `sunset_upscaled16.tif` |
| Marketing PNG | `{stem}_marketing.png` | `sunset_marketing.png` |
| Preview JPG | `{stem}_preview.jpg` | `sunset_preview.jpg` |
| Report JSON | `{stem}_report.json` | `sunset_report.json` |

### With Prefixes

```python
config = ExportConfig(
    output_dir=Path("output"),
    master_prefix="gold_",
    upscaled_prefix="hires_"
)
```

| File Type | Pattern | Example |
|-----------|---------|---------|
| Master TIFF | `{prefix}{stem}_master16.tif` | `gold_sunset_master16.tif` |
| Upscaled TIFF | `{prefix}{stem}_upscaled16.tif` | `hires_sunset_upscaled16.tif` |

## Testing

### Unit Tests

```bash
# Run ExportManager unit tests
pytest tests/core/storage/test_export_manager.py -v
```

### Integration Tests

```bash
# Run pipeline integration tests
pytest lux_depth_v2/tests/test_pipeline_export_manager_integration.py -v
```

### Mock I/O for Testing

```python
from unittest.mock import MagicMock

# Create mock I/O module
mock_io = MagicMock()
mock_io.atomic_write_rgb16_tiff = MagicMock()
mock_io.atomic_write_png8 = MagicMock()
mock_io.atomic_write_jpg8 = MagicMock()

# Use with ExportManager
config = ExportConfig(output_dir=Path("output"))
export_manager = ExportManager(config, mock_io)

# Verify delegation
export_manager.write_master("test", image)
mock_io.atomic_write_rgb16_tiff.assert_called_once()
```

## Error Handling

### Missing Dependencies

```python
try:
    export_manager.write_master("test", image)
except RuntimeError as e:
    print(f"Missing dependencies: {e}")
    # Handle gracefully
```

### I/O Errors

```python
try:
    export_manager.write_master("test", image)
except OSError as e:
    print(f"Write failed: {e}")
    # Disk full, permissions, etc.
```

### Graceful Fallback

```python
from lux_depth_v2.pipeline import EXPORT_MANAGER_AVAILABLE

if not EXPORT_MANAGER_AVAILABLE:
    print("ExportManager unavailable, using direct I/O")
    # Pipeline automatically falls back
```

## Performance

### Typical Timings (64x64 test image, CPU)

| Operation | Time | Notes |
|-----------|------|-------|
| `write_master` | ~0.8ms | 16-bit TIFF deflate |
| `write_preview` | ~0.2ms | JPG quality 92 |
| `write_upscaled` | ~3.0ms | 16-bit TIFF deflate |
| `write_marketing` | ~5.4ms | 8-bit PNG compression 7 |
| `write_report` | ~0.03ms | JSON indent 2 |

**Total Export Overhead**: ~9.5ms per image

### Optimization Tips

1. **Use compression=None for speed** (development only):
   ```python
   export_manager.write_master("test", image, compression="none")
   ```

2. **Lower JPEG quality for previews**:
   ```python
   export_manager.write_preview("test", image, quality=75)
   ```

3. **Disable unnecessary exports**:
   ```python
   config = PipelineConfig(
       save_preview_jpg=False,  # Skip preview
       save_marketing_png=False  # Skip marketing
   )
   ```

## Best Practices

1. **Use ExportManager in new code**: Prefer ExportManager over direct `io_utils` calls
2. **Test with dependency injection**: Mock I/O module for unit tests
3. **Monitor timing stages**: Use `timing_stages_s` for performance analysis
4. **Leverage path getters**: Use `get_*_path()` for skip_existing checks
5. **Keep config immutable**: Don't modify ExportConfig after initialization

## Future Extensions (Phase 2 Slice 3+)

ExportManager provides foundation for:

- **Scratch directory staging**: Write to fast local disk, move on completion
- **Async I/O**: Non-blocking exports with queue management
- **Chunked BigTIFF**: Stream large files to reduce memory
- **Cloud storage**: Direct S3/GCS uploads
- **Multi-destination**: Export to multiple locations simultaneously

## See Also

- **Implementation Details**: `docs/PHASE2_SLICE2_EXPORT_MANAGER_COMPLETE.md`
- **Phase 2 Summary**: `PHASE2_SLICE2_COMPLETE.md`
- **Pipeline Documentation**: `lux_depth_v2/README.md`

---

**Last Updated**: December 9, 2025  
**Phase**: 2.2 Complete  
**Status**: Production Ready
