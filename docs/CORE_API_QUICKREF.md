# Platform Core API - Quick Reference

**Version**: 1.0.0  
**Module**: `transformation_portal.core`

---

## Installation

```bash
# Core module is part of transformation_portal package
pip install -e .

# Verify installation
python -c "from transformation_portal.core import ConfigSchema, DeviceDetector"
```

---

## Config Module

### Basic Configuration

```python
from transformation_portal.core import ConfigSchema

# Create with defaults
config = ConfigSchema()

# Access sub-configs
print(config.device.device)          # "auto"
print(config.performance.batch_size) # 1
print(config.output.save_master)     # True

# Convert to dict
data = config.to_dict()
```

### Load Preset

```python
from transformation_portal.core.config import load_preset

# Load built-in preset
preset = load_preset("interior_luxury")
print(preset["performance"]["tile_size"])  # 2048

# Create config from preset
config = ConfigSchema(
    performance=preset["performance"],
    extras=preset["extras"]
)
```

### Custom Preset

```python
from transformation_portal.core.config import register_preset

# Register custom preset
register_preset("my_preset", {
    "performance": {
        "batch_size": 8,
        "tile_size": 4096
    },
    "extras": {
        "quality": "ultra"
    }
})

# Use custom preset
preset = load_preset("my_preset")
```

### Validate Config

```python
from transformation_portal.core.config import validate_config

config_dict = {
    "performance": {
        "tile_size": 512,
        "tile_overlap": 64
    }
}

errors = validate_config(config_dict)
if errors:
    print(f"Errors: {errors}")
```

---

## Device Module

### Detect Device

```python
from transformation_portal.core.device import DeviceDetector

# Auto-detect device
detector = DeviceDetector()
device_info = detector.detect()

# Use device
device = device_info.device  # torch.device
print(f"Using: {device}")

# Get capabilities
cap = device_info.capabilities
print(f"Memory: {cap.available_memory_gb:.1f} GB")
print(f"Batch size: {cap.recommended_batch_size}")
```

### Performance Profiling

```python
from transformation_portal.core.device import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile operation
with profiler.profile("load_model"):
    model = load_model()

with profiler.profile("inference"):
    result = model(input)

# Print summary
profiler.print_summary()

# Get specific result
result = profiler.get_result("inference")
print(f"Duration: {result.duration_ms:.1f}ms")
```

### Memory Management

```python
from transformation_portal.core.device import MemoryManager, calculate_safe_batch_size

# Track memory
manager = MemoryManager()
stats = manager.get_stats()
print(f"Available: {stats.available_mb:.1f} MB")

# Calculate batch size
batch_size = calculate_safe_batch_size(
    image_width=3840,
    image_height=2160,
    available_memory_gb=16.0
)
print(f"Safe batch size: {batch_size}")
```

---

## Security Module

### Validate Input File

```python
from transformation_portal.core.security import validate_input_file

# Validate with defaults
try:
    validate_input_file(input_path, strict=True)
    print("✅ File is valid")
except ValidationError as e:
    print(f"❌ Validation failed: {e}")
```

### Custom Validation

```python
from transformation_portal.core.security import InputValidator

validator = InputValidator(
    allowed_extensions=(".jpg", ".png", ".tif"),
    max_size_mb=100.0,
    enable_magic_bytes=True
)

# Validate
result = validator.validate_file(path, strict=False)
if result.valid:
    print(f"✅ Valid: {result.file_type}")
else:
    print(f"❌ Errors: {result.errors}")
```

### Path Security

```python
from transformation_portal.core.security import safe_resolve_path, PathValidator

# Validate path
validator = PathValidator(allowed_roots=[Path("/data")])
if validator.validate(user_path):
    safe_path = safe_resolve_path(user_path, root=Path("/data"))
    print(f"✅ Safe path: {safe_path}")
```

### Sanitize Filename

```python
from transformation_portal.core.security import sanitize_filename

# Sanitize dangerous filename
unsafe = "../../../etc/passwd"
safe = sanitize_filename(unsafe)
print(f"Safe: {safe}")  # "___etc_passwd"
```

---

## Artifacts Module

### Basic Caching

```python
from transformation_portal.core.artifacts import CacheManager

# Create cache
cache = CacheManager(Path(".cache"), max_size_gb=10.0)

# Compute key
key = cache.cache.compute_key("input.jpg", preset="interior")

# Get or compute
result = cache.get_or_compute(
    key,
    expensive_function,
    arg1, arg2
)
```

### Manual Cache Control

```python
from transformation_portal.core.artifacts import ContentAddressedCache

# Create cache
cache = ContentAddressedCache(Path(".cache"))

# Compute key
key = cache.compute_key("input.jpg", preset="interior")

# Check cache
cached = cache.get(key)
if cached:
    print(f"✅ Cache hit: {cached}")
else:
    # Compute and cache
    result = expensive_function()
    cache.put(key, result)

# Get stats
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
```

### Artifact Storage

```python
from transformation_portal.core.artifacts import ArtifactStorage, StorageBackend

# Setup storage
storage = ArtifactStorage(
    primary_path=Path("."),
    external_path=Path("/Volumes/T9"),
    auto_migrate_threshold_mb=2000.0
)

# Store artifact (auto-selects backend)
dest = storage.store(file_path, "renders/output.tif")

# Retrieve artifact
artifact = storage.retrieve("renders/output.tif")

# Migrate to external
storage.migrate("renders/output.tif", StorageBackend.EXTERNAL)
```

---

## Observability Module

### Setup Logging

```python
from transformation_portal.core.observability import setup_logging

# Basic logging
logger = setup_logging("my_pipeline", level=logging.INFO)
logger.info("Processing started")

# With log file
logger = setup_logging(
    "my_pipeline",
    level=logging.DEBUG,
    log_file=Path("pipeline.log")
)
```

---

## Common Patterns

### Complete Pipeline Setup

```python
from pathlib import Path
from transformation_portal.core import (
    ConfigSchema,
    DeviceDetector,
    validate_input_file,
    CacheManager,
)
from transformation_portal.core.device import PerformanceProfiler
from transformation_portal.core.observability import setup_logging

# Setup logging
logger = setup_logging("my_pipeline")

# Load configuration
config = ConfigSchema()

# Detect device
detector = DeviceDetector()
device_info = detector.detect()
logger.info(f"Using device: {device_info.device}")

# Setup profiling
profiler = PerformanceProfiler()

# Setup caching
cache = CacheManager(Path(".cache"))

# Validate input
with profiler.profile("validation"):
    validate_input_file(input_path, strict=True)

# Process with profiling
with profiler.profile("processing"):
    result = process_pipeline(input_path, config, device_info.device)

# Print performance summary
profiler.print_summary()
```

### Migration Pattern (Incremental)

```python
# Before (old code)
def _detect_device(self):
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# After (core module)
from transformation_portal.core.device import DeviceDetector

def _detect_device(self):
    detector = DeviceDetector()
    device_info = detector.detect()
    return device_info.device
```

---

## Built-in Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `photo_realistic` | Balanced quality | General photography |
| `interior_luxury` | High quality, enhanced materials | Interior real estate |
| `exterior_showcase` | Outdoor optimization | Exterior real estate |
| `architectural` | Precise geometry | Architectural renders |
| `archival_quality` | Maximum quality | Archive/print |
| `fast_preview` | Speed over quality | Quick previews |

---

## Error Handling

### Config Validation Errors

```python
from transformation_portal.core.config import validate_config, ConfigValidationError

try:
    errors = validate_config(config_dict)
    if errors:
        raise ConfigValidationError("Invalid config", errors=errors)
except ConfigValidationError as e:
    print(f"Config errors: {e.errors}")
```

### Input Validation Errors

```python
from transformation_portal.core.security import ValidationError

try:
    validate_input_file(path, strict=True)
except ValidationError as e:
    print(f"Path: {e.path}")
    print(f"Details: {e.details}")
```

---

## Performance Tips

1. **Cache device detection**: Call `detect()` once and reuse
2. **Use content-addressed caching**: Automatic invalidation on input changes
3. **Profile critical sections**: Use `PerformanceProfiler` for bottlenecks
4. **Validate early**: Check inputs before expensive operations
5. **Batch size**: Use `calculate_safe_batch_size()` for optimal memory usage

---

## Testing

```python
# Example test with core module
def test_pipeline_with_core():
    from transformation_portal.core import ConfigSchema
    from transformation_portal.core.device import DeviceDetector
    
    config = ConfigSchema()
    detector = DeviceDetector()
    device_info = detector.detect()
    
    assert device_info.device is not None
    assert config.device.device.value == "auto"
```

---

## Migration Checklist

- [ ] Import core modules
- [ ] Replace device detection
- [ ] Replace input validation
- [ ] Add performance profiling (optional)
- [ ] Add caching (optional)
- [ ] Update tests
- [ ] Validate performance
- [ ] Update documentation

---

## Support

- **Documentation**: `docs/PLATFORM_CORE_MIGRATION.md`
- **Tests**: `tests/core/`
- **Examples**: Module docstrings
- **Issues**: Create GitHub issue for problems

---

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: December 9, 2025
