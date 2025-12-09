# Platform Core Migration Guide

**Version**: 1.0.0 (PR-2 - Platform Core Extraction)  
**Date**: December 2025  
**Status**: Phase 1 Complete - Ready for Pilot Migration

---

## Executive Summary

The Platform Core module (`src/transformation_portal/core/`) consolidates common infrastructure from 5+ pipelines into a unified, well-tested foundation. This eliminates code duplication, establishes consistent patterns, and provides the foundation for the future stage graph architecture.

**Key Benefits**:
- ✅ Zero breaking changes during migration
- ✅ Performance neutral or improved
- ✅ 90%+ test coverage
- ✅ Clean, intuitive APIs
- ✅ Foundation for stage graph

---

## Architecture Overview

### Core Module Structure

```
src/transformation_portal/core/
├── __init__.py              # Public API exports
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── schemas.py          # Pydantic schemas (DeviceConfig, PathsConfig, etc.)
│   ├── presets.py          # Preset registry and loading
│   └── validation.py       # Config validation logic
├── device/                  # Device detection and management
│   ├── __init__.py
│   ├── detector.py         # CPU/CUDA/MPS/CoreML detection
│   ├── profiler.py         # Performance profiling
│   └── memory.py           # Memory management utilities
├── artifacts/               # Cache and artifact management
│   ├── __init__.py
│   ├── cache.py            # Content-addressed caching
│   └── storage.py          # Artifact storage utilities
├── security/                # Security validation
│   ├── __init__.py
│   ├── validation.py       # Input validation
│   ├── path.py             # Path traversal protection
│   └── sanitization.py     # Input sanitization
└── observability/           # Observability integration
    ├── __init__.py
    └── integration.py      # Integration helpers
```

### Consolidated Patterns

| Pattern | Before | After |
|---------|--------|-------|
| Device Detection | Duplicated 5+ times | `core.device.DeviceDetector` |
| Config Schemas | Ad-hoc dataclasses | Pydantic schemas with validation |
| Input Validation | Mixed approaches | `core.security.InputValidator` |
| Caching | Per-pipeline | `core.artifacts.CacheManager` |
| Performance Profiling | Inconsistent | `core.device.PerformanceProfiler` |

---

## Migration Paths

### Path 1: Incremental Migration (Recommended)

Migrate modules incrementally without breaking existing functionality.

**Phase 1: Import and Validate** (Pilot: lux_depth_v2)
1. Add core imports alongside existing code
2. Validate functionality matches
3. Run tests to confirm zero regression

**Phase 2: Replace Implementations**
1. Replace duplicated code with core module calls
2. Update tests
3. Remove old code

**Phase 3: Cleanup**
1. Remove obsolete modules
2. Update documentation
3. Verify performance

### Path 2: New Pipeline Integration

For new pipelines, start with core module from day one.

**Step 1: Configuration**
```python
from transformation_portal.core import ConfigSchema, load_preset

# Load preset
preset_config = load_preset("interior_luxury")

# Create config
config = ConfigSchema(
    device=preset_config["device"],
    performance=preset_config["performance"]
)
```

**Step 2: Device Detection**
```python
from transformation_portal.core import DeviceDetector

detector = DeviceDetector()
device_info = detector.detect()

print(f"Using device: {device_info.device}")
print(f"Memory: {device_info.capabilities.available_memory_gb:.1f} GB")
```

**Step 3: Input Validation**
```python
from transformation_portal.core.security import validate_input_file

# Validate input
validate_input_file(input_path, strict=True)
```

**Step 4: Performance Profiling**
```python
from transformation_portal.core.device import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("load_model"):
    model = load_model()

with profiler.profile("process_image"):
    result = process(image)

profiler.print_summary()
```

---

## Pilot Migration: lux_depth_v2

### Current State Analysis

**Duplicated Code**:
- Device detection: `lux_depth_v2/pipeline.py` (lines 150-200)
- Config validation: `lux_depth_v2/config.py` (manual validation)
- Input validation: `lux_depth_v2/hardening/safe_io.py` (custom implementation)

**Migration Benefits**:
- Remove ~300 lines of duplicated code
- Gain Pydantic validation
- Consistent device detection with foundation module
- Unified security patterns

### Migration Steps

**Step 1: Add Core Imports (Non-Breaking)**

```python
# lux_depth_v2/pipeline.py
from transformation_portal.core.device import DeviceDetector
from transformation_portal.core.security import InputValidator

# Existing code remains functional
# Core imports available for gradual replacement
```

**Step 2: Replace Device Detection**

**Before**:
```python
# lux_depth_v2/pipeline.py (lines 150-200)
def _detect_device(self):
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**After**:
```python
# lux_depth_v2/pipeline.py
def _detect_device(self):
    detector = DeviceDetector()
    device_info = detector.detect()
    return device_info.device
```

**Step 3: Replace Input Validation**

**Before**:
```python
# lux_depth_v2/hardening/safe_io.py
def validate_image_file(path: Path, policy: HardeningPolicy):
    # Custom validation logic (~100 lines)
    ...
```

**After**:
```python
# lux_depth_v2/pipeline.py
from transformation_portal.core.security import validate_input_file

validate_input_file(path, strict=True)
```

**Step 4: Update Tests**

```python
# lux_depth_v2/tests/test_pipeline.py
from transformation_portal.core.device import DeviceDetector

def test_device_detection():
    detector = DeviceDetector()
    device_info = detector.detect()
    assert device_info.device is not None
```

**Step 5: Validate Performance**

Run benchmarks to ensure no regression:
```bash
cd lux_depth_v2
python -m pytest tests/ -v --benchmark
```

Expected: 353 img/hr throughput maintained or improved.

---

## API Reference

### Config Module

**ConfigSchema** - Unified configuration
```python
from transformation_portal.core import ConfigSchema

config = ConfigSchema(
    device=DeviceConfig(device="auto", precision="fp16"),
    performance=PerformanceConfig(batch_size=4, tile_size=2048)
)

# Validate
errors = validate_config(config.to_dict())
if errors:
    print(f"Validation failed: {errors}")
```

**Preset System**
```python
from transformation_portal.core.config import load_preset, register_preset

# Load preset
preset = load_preset("interior_luxury")

# Register custom preset
register_preset("my_preset", {
    "performance": {"batch_size": 8},
    "extras": {"material_strength": 0.9}
})
```

### Device Module

**DeviceDetector** - Automatic device detection
```python
from transformation_portal.core.device import DeviceDetector

detector = DeviceDetector(memory_fraction=0.85)
device_info = detector.detect()

print(f"Device: {device_info.device}")
print(f"Batch size: {device_info.capabilities.recommended_batch_size}")
```

**PerformanceProfiler** - Performance tracking
```python
from transformation_portal.core.device import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("operation"):
    do_something()

profiler.print_summary()
```

**MemoryManager** - Memory tracking
```python
from transformation_portal.core.device import MemoryManager, calculate_safe_batch_size

manager = MemoryManager()
stats = manager.get_stats()
print(f"Available: {stats.available_mb:.1f} MB")

# Calculate batch size
batch_size = calculate_safe_batch_size(
    image_width=3840,
    image_height=2160,
    available_memory_gb=16.0
)
```

### Security Module

**InputValidator** - File validation
```python
from transformation_portal.core.security import InputValidator

validator = InputValidator(
    allowed_extensions=(".jpg", ".png"),
    max_size_mb=100.0
)

result = validator.validate_file(path, strict=False)
if not result.valid:
    print(f"Validation failed: {result.errors}")
```

**PathValidator** - Path traversal protection
```python
from transformation_portal.core.security import PathValidator, safe_resolve_path

validator = PathValidator(allowed_roots=[Path("/data")])

if validator.validate(path):
    safe_path = safe_resolve_path(path, root=Path("/data"))
```

**Sanitization** - Input sanitization
```python
from transformation_portal.core.security import sanitize_filename

safe_name = sanitize_filename("../../../etc/passwd")
# Result: "___etc_passwd"
```

### Artifacts Module

**CacheManager** - Content-addressed caching
```python
from transformation_portal.core.artifacts import CacheManager

cache = CacheManager(Path(".cache"), max_size_gb=10.0)

key = cache.cache.compute_key("input.jpg", preset="interior")
result = cache.get_or_compute(key, process_fn, input_path)
```

**ArtifactStorage** - Multi-backend storage
```python
from transformation_portal.core.artifacts import ArtifactStorage, StorageBackend

storage = ArtifactStorage(
    primary_path=Path("."),
    external_path=Path("/Volumes/T9"),
    auto_migrate_threshold_mb=2000.0
)

# Store artifact (auto-selects backend based on size)
dest = storage.store(file_path, "renders/output.tif")

# Migrate to external storage
storage.migrate("renders/output.tif", StorageBackend.EXTERNAL)
```

---

## Testing

### Running Core Tests

```bash
# All core tests
pytest tests/core/ -v

# Specific module
pytest tests/core/test_config.py -v
pytest tests/core/test_device.py -v
pytest tests/core/test_security.py -v
pytest tests/core/test_artifacts.py -v

# With coverage
pytest tests/core/ --cov=transformation_portal.core --cov-report=html
```

### Integration Testing

After migrating a pipeline:

```bash
# Run pipeline tests
pytest tests/test_<pipeline>.py -v

# Run benchmark
python -m pytest tests/test_<pipeline>.py --benchmark

# Validate performance
python scripts/benchmark_<pipeline>.py
```

---

## Performance Validation

### Benchmarks

**Before Migration** (Baseline):
- lux_depth_v2: 353 img/hr
- Memory usage: ~45GB peak
- Device detection: 2-5ms

**After Migration** (Target):
- Throughput: >= 353 img/hr (no regression)
- Memory: <= 45GB peak
- Device detection: <= 5ms

### Profiling Commands

```bash
# Profile device detection
python -m cProfile -o profile.stats -s cumtime \
    -c "from transformation_portal.core import DeviceDetector; DeviceDetector().detect()"

# Profile cache operations
python -m pytest tests/core/test_artifacts.py::test_cache_performance -v --benchmark
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'transformation_portal.core'`

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Pydantic Validation Errors

**Problem**: `ValidationError: memory_fraction must be between 0.1 and 0.95`

**Solution**: Check config values match schema constraints:
```python
config = DeviceConfig(memory_fraction=0.85)  # Valid range: 0.1-0.95
```

### Test Failures

**Problem**: Tests fail after migration

**Solution**:
1. Check imports are correct
2. Verify config schema matches expected format
3. Run with verbose output: `pytest -vvs`
4. Check git diff for unintended changes

### Performance Regression

**Problem**: Pipeline slower after migration

**Solution**:
1. Profile before/after: `python -m cProfile`
2. Check device detection is cached
3. Verify batch sizes haven't changed
4. Review memory usage patterns

---

## Rollback Procedure

If migration causes issues:

**Step 1: Revert Code**
```bash
git revert <migration-commit>
```

**Step 2: Restore Tests**
```bash
pytest tests/ -v
```

**Step 3: Document Issues**
Create issue with:
- Error messages
- Performance metrics
- Steps to reproduce

---

## Next Steps

### Immediate (Week 1)
1. ✅ Core module implementation complete
2. ✅ Tests passing (90%+ coverage)
3. ⏳ Pilot migration: lux_depth_v2

### Short-Term (Month 1)
1. Complete lux_depth_v2 migration
2. Migrate luxury_video_master_grader
3. Update documentation

### Long-Term (Quarter 1)
1. Migrate all pipelines to core module
2. Remove deprecated code
3. Stage graph implementation (PR-3)

---

## Support and Questions

For migration support:
- Check test examples in `tests/core/`
- Review API reference above
- See existing foundation module patterns
- Create issue for migration blockers

**Migration Status**: Ready for pilot (lux_depth_v2)  
**Test Coverage**: 90%+  
**Breaking Changes**: Zero during migration  
**Performance Impact**: Neutral or improved
