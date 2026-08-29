# Migration Guide

**Transformation Portal - Version Migration Guide**

Version: 2.0.0
Last Updated: 2026-03-21

---

## Overview

This guide helps you migrate your code between major versions of the Transformation Portal. All migrations maintain backwards compatibility during a transition period (minimum 6 months).

## Version Support Timeline

| Version | Release Date | Support Status | End of Support |
|---------|--------------|----------------|----------------|
| v0.1.x  | 2025-01-01   | 🔴 Deprecated  | 2026-06-30     |
| v1.0.x  | 2025-06-01   | ⚠️ Maintenance | 2026-12-31     |
| v2.0.x  | 2026-03-01   | ✅ Active      | -              |

## Current Version: v1.0.x → v2.0.x (Golden Path)

### Breaking Changes in v2.0.0

#### 1. Python Version Requirement

**v2.0.0 requires Python 3.11+**

```bash
# Check your Python version
python --version  # Must be 3.11 or higher
```

#### 2. Depth Backend Changes

The depth estimation backend system has been unified under ADR-019:

```python
# Old (v1.x) - deprecated
from depth_tools import estimate_depth
depth = estimate_depth(image, model="depth_anything_v2")

# New (v2.0.0) - recommended
from transformation_portal.depth import ArchitecturalDepthPipeline
pipeline = ArchitecturalDepthPipeline(backend="da3")  # Depth Anything V3
result = pipeline.run(image)
```

**Available backends:**
- `da3` (Depth Anything V3 family) - Default backend family; the default model is the Apache-2.0 `da3_metric` (repair 1.2, #2066). The bare `da3` model selector is deprecated: it still resolves the research model and requires `--non-commercial-ok`
- `da3-metric` - Apache-2.0 DA3 selector for the current Lux V3 relative-depth surface
- `depth_pro` (Apple Depth Pro) - Research only, requires explicit license acceptance

No-selector runs now record `requested_selector: "default"` plus a
`resolution_reason` naming the `da3_metric` default. The same reason is
published as `model_contract.resolution_reason` in run cards. This keeps the
default path distinct from the deprecated `da3` research alias and ensures
metric and research selections produce different cache/manifest fingerprints.

#### 3. Configuration Format

Configuration files now use structured YAML format:

```yaml
# Old format (still supported with deprecation warning)
depth_model: depth_anything_v2
output_dir: output/

# New format (recommended)
pipeline:
  depth:
    backend: da3
    model_variant: DA3-Large
  output:
    directory: output/
    format: png
```

### New Features in v2.0.0

#### 1. Plugin Architecture

**What's New**: Extensible plugin system for depth models, processors, and enhancers.

**Migration**: None required - this is a new optional feature.

**Example**:
```python
# New plugin system (optional)
from transformation_portal.plugins import get_global_registry

registry = get_global_registry()
depth_model = registry.get_plugin('depth_model', 'edge_depth_estimator')
```

#### 2. Real-Time Progress Tracking

**What's New**: Progress bars, checkpoints, and streaming results.

**Migration**: None required - this is a new optional feature.

**Example**:
```python
# New progress tracking (optional)
from transformation_portal.streaming import ProgressBar

with ProgressBar(total=100, description="Processing") as pbar:
    for i in range(100):
        process_item(i)
        pbar.update(1)
```

#### 3. Quality Firewall

**What's New**: Automatic quality enforcement with performance gates.

**Migration**: None required - enabled by default for production pipelines.

#### 4. Orchestrator Decomposition

The monolithic orchestrator has been decomposed into focused modules:
- `config_resolver.py` - Configuration handling
- `pipeline_coordinator.py` - Stage sequencing
- `artifact_manager.py` - Output management
- `execution_engine.py` - Image processing
- `validators/` - Validation logic

**Migration**: Backward-compatible aliases maintained in `orchestrator.py`.

### Deprecated Features

**v1.x APIs deprecated in v2.0.0** (removal planned for v3.0.0):

---

## Future Migrations

### Planned v1.0.x → v2.0.x (TBD)

The following changes are **under consideration** for v2.0.x (not finalized):

#### Potential Deprecations

1. **Root-level script imports** (tentative)
   ```python
   # May be deprecated in v2.0.x
   from depth_tools import estimate_depth

   # Recommended migration
   from transformation_portal.depth import DepthEstimator
   estimator = DepthEstimator()
   depth = estimator.estimate(image)
   ```

2. **Direct function calls** (tentative)
   ```python
   # May be deprecated in v2.0.x
   result = process_image(image, preset="golden_hour")

   # Recommended migration
   from transformation_portal.processors import ImageProcessor
   processor = ImageProcessor()
   result = processor.process(image, preset="golden_hour")
   ```

**Note**: These are tentative. Any actual deprecations will be announced with 6+ months warning.

---

## Automated Migration Tools

### Check for Deprecated Usage

```bash
# Scan your code for deprecated APIs
python -m transformation_portal.compat.analyze your_script.py

# Output:
# Found 3 deprecated usages:
#   Line 10: depth_tools.estimate_depth (deprecated in v1.5.0, removed in v2.0.0)
#   Line 25: luxury_tiff_batch_processor (deprecated in v1.3.0, removed in v2.0.0)
#   ...
```

### Auto-Migration Script (Future)

```bash
# Automatically update code (when v2.0.x is released)
python -m transformation_portal.compat.migrate --from 1.x --to 2.x your_script.py

# Creates: your_script.py.v2 with migrated code
# Creates: migration_report.txt with changes made
```

### Test with Warnings as Errors

Catch deprecation warnings during development:

```bash
# Run with warnings as errors
python -W error::DeprecationWarning your_script.py

# Or in pytest
pytest -W error::DeprecationWarning
```

---

## Step-by-Step Migration Process

### 1. **Audit Current Code**

Identify deprecated APIs:

```bash
# Enable deprecation warnings
export PYTHONWARNINGS=default

# Run your code
python your_script.py

# Check for deprecation warnings in output
```

### 2. **Update Dependencies**

```bash
# Update to latest version in current major series
pip install --upgrade transformation-portal

# Or pin to specific version
pip install transformation-portal==1.5.0
```

### 3. **Update Imports**

Replace deprecated imports:

```python
# Old (if deprecated)
from depth_tools import estimate_depth

# New
from transformation_portal.depth import DepthEstimator
```

### 4. **Update Function Calls**

Replace deprecated function calls:

```python
# Old (if deprecated)
result = estimate_depth(image)

# New
estimator = DepthEstimator()
result = estimator.estimate(image)
```

### 5. **Update Configuration Files**

Configuration files are automatically migrated, but you can manually update:

```yaml
# Old format (if deprecated)
depth_model: "depth_anything_v2"
output_dir: "output"

# New format (same, but explicit)
pipeline:
  depth_model: "depth_anything_v2"
  output:
    directory: "output"
```

### 6. **Test Thoroughly**

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=your_module

# Test specific migration
pytest tests/test_migration.py -v
```

### 7. **Update Documentation**

Update code comments and documentation to use new APIs:

```python
# Update docstring examples
def my_function(image):
    """Process image.

    Example:
        >>> from transformation_portal.depth import DepthEstimator  # Updated!
        >>> estimator = DepthEstimator()
        >>> depth = estimator.estimate(image)
    """
```

---

## Common Migration Scenarios

### Scenario 1: Batch Processing Script

**Current Code**:
```python
from luxury_tiff_batch_processor import batch_process

batch_process(
    input_dir="input/",
    output_dir="output/",
    preset="golden_hour"
)
```

**Migrated Code** (if needed in future):
```python
from transformation_portal.processors import TiffBatchProcessor

processor = TiffBatchProcessor()
processor.batch_process(
    input_dir="input/",
    output_dir="output/",
    preset="golden_hour"
)
```

### Scenario 2: Depth Estimation

**Current Code**:
```python
from depth_tools import estimate_depth

depth_map = estimate_depth("image.jpg")
```

**Migrated Code** (if needed in future):
```python
from transformation_portal.depth import DepthEstimator

estimator = DepthEstimator()
depth_map = estimator.estimate("image.jpg")
```

### Scenario 3: Video Processing

**Current Code**:
```python
from luxury_video_master_grader import process_video

process_video(
    input_path="video.mp4",
    output_path="graded.mp4",
    preset="signature_estate"
)
```

**Migrated Code** (if needed in future):
```python
from transformation_portal.processors import VideoGrader

grader = VideoGrader()
grader.process(
    input_path="video.mp4",
    output_path="graded.mp4",
    preset="signature_estate"
)
```

---

## Compatibility Guarantees

### What We Guarantee

✅ **API Stability**: No breaking changes within major version
✅ **Migration Period**: Minimum 6 months warning before removal
✅ **Compatibility Shims**: Old APIs delegate to new implementations
✅ **Configuration Migration**: Old configs automatically updated
✅ **Data Format Support**: Old TIFF/video formats always supported

### What We Don't Guarantee

⚠️ **Exact Output**: Algorithm improvements may change results slightly
⚠️ **Performance**: Optimizations may change timing characteristics
⚠️ **Internal APIs**: `_private` functions can change without notice
⚠️ **Experimental Features**: `@experimental` APIs may change in minor versions

---

## Getting Help

### Resources

- **Migration Issues**: [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)
- **API Documentation**: [docs/](../docs/)
- **Deprecation Policy**: [DEPRECATION_POLICY.md](DEPRECATION_POLICY.md)
- **Plugin Guide**: [docs/guides/PLUGIN_DEVELOPMENT.md](../guides/PLUGIN_DEVELOPMENT.md)

### Support Channels

1. **GitHub Issues**: Report migration problems
2. **Documentation**: Check docs/ for detailed guides
3. **Examples**: See examples/ for updated code samples

---

**Last Updated**: 2026-03-21
**Guide Version**: 2.0.0
