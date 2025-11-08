# Deprecation Policy

**Transformation Portal - API Stability and Deprecation Guidelines**

Version: 1.0.0  
Last Updated: 2025-11-08

---

## Overview

The Transformation Portal follows semantic versioning and maintains strict backwards-compatibility guarantees to ensure existing code continues to work across versions.

## Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (e.g., 1.0.0 → 2.0.0): Breaking changes to public APIs
- **MINOR** (e.g., 1.0.0 → 1.1.0): New features, backwards-compatible
- **PATCH** (e.g., 1.0.0 → 1.0.1): Bug fixes, backwards-compatible

## API Stability Levels

### 1. **Stable APIs** (Public API)
- **Guarantee**: No breaking changes within major version
- **Location**: `src/transformation_portal/` (all public modules)
- **Examples**: `DepthEstimator`, `MaterialResponse`, `LuxRenderPipeline`
- **Deprecation Timeline**: Minimum 6 months warning before removal

### 2. **Experimental APIs** (Beta)
- **Guarantee**: API may change in minor versions
- **Marking**: `@experimental` decorator or `_experimental` module prefix
- **Examples**: New depth models, experimental processors
- **Deprecation Timeline**: Minimum 3 months warning

### 3. **Internal APIs** (Private)
- **Guarantee**: No stability guarantee
- **Marking**: Single underscore prefix (`_internal_function`)
- **Examples**: `_load_model`, `_compute_depth_zones`
- **Note**: Not intended for external use

## Deprecation Process

### Phase 1: Deprecation Announcement (6+ months)

When an API is deprecated:

1. **Add `@deprecated` decorator**:
```python
from transformation_portal.compat import deprecated

@deprecated(
    replacement="new_function_name",
    removal_version="2.0.0"
)
def old_function():
    # Old implementation delegates to new one
    return new_function_name()
```

2. **Add deprecation warning**:
   - Function shows `DeprecationWarning` on first use
   - Warning includes replacement function name
   - Warning includes version when removal will occur

3. **Update documentation**:
   - Mark as deprecated in docstring
   - Add "See Also" section with replacement
   - Update examples to use new API

4. **Add to CHANGELOG.md**:
   ```markdown
   ### Deprecated
   - `old_function` deprecated in favor of `new_function_name`. Will be removed in v2.0.0.
   ```

### Phase 2: Migration Period (6+ months)

During migration period:

1. **Both APIs work**: Old and new APIs function identically
2. **Compatibility shims**: Provide automatic migration helpers
3. **Migration guide**: Document step-by-step upgrade process
4. **Test coverage**: Both APIs tested to ensure equivalence

### Phase 3: Removal (Next Major Version)

In next major version:

1. **Remove deprecated code**: Delete old API implementation
2. **Update tests**: Remove tests for deprecated APIs
3. **Update CHANGELOG.md**:
   ```markdown
   ### Removed
   - `old_function` (deprecated in v1.5.0, removed in v2.0.0). Use `new_function_name` instead.
   ```

## Using Deprecated APIs

### Suppress Deprecation Warnings (Not Recommended)

```python
import warnings

# Suppress all deprecation warnings (use sparingly!)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress specific warning
warnings.filterwarnings(
    "ignore",
    message=".*old_function.*",
    category=DeprecationWarning
)
```

### Check for Deprecated Usage

Run your code with warnings as errors to detect deprecated usage:

```bash
python -W error::DeprecationWarning your_script.py
```

## Migration Tools

### Automatic Migration Script

```bash
# Migrate from v1.x to v2.x
python -m transformation_portal.compat.migrate --from 1.x --to 2.x script.py
```

### Migration Guide

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions for each version.

## Examples

### Function Deprecation

```python
# Old way (deprecated in v1.5.0, removed in v2.0.0)
from transformation_portal.depth_tools import estimate_depth

# New way (use this!)
from transformation_portal.depth import DepthEstimator

estimator = DepthEstimator()
depth_map = estimator.estimate(image)
```

### Module Rename

```python
# Old import (deprecated in v1.5.0, removed in v2.0.0)
from transformation_portal.processors.luxury_video_master_grader import VideoGrader

# New import (use this!)
from transformation_portal.processors.video import VideoGrader
```

### Parameter Rename

```python
# Old parameters (deprecated in v1.5.0, removed in v2.0.0)
process_image(input_file="image.jpg", output_folder="output/")

# New parameters (use this!)
process_image(image_path="image.jpg", output_dir="output/")
```

## Backwards-Compatibility Guarantees

### What We Guarantee

1. **Public APIs**: No breaking changes within major version
2. **File Formats**: Old TIFF metadata formats always supported
3. **Configuration**: Old config files automatically migrated
4. **Command-line**: CLI options maintain backwards-compatibility

### What We Don't Guarantee

1. **Internal APIs**: `_private_function` can change anytime
2. **Experimental APIs**: `@experimental` APIs may change in minor versions
3. **Performance**: Optimizations may change performance characteristics
4. **Dependencies**: Dependency versions may change in minor versions
5. **Output Exactness**: Enhanced images may differ slightly due to algorithm improvements

## Version Support Policy

- **Current Major Version** (e.g., v2.x): Full support, active development
- **Previous Major Version** (e.g., v1.x): Security fixes for 12 months after v2.0 release
- **Older Versions**: No support (upgrade recommended)

## Questions?

- **Migration help**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Report deprecation issues**: [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)
- **API questions**: Check [docs/](../docs/) for detailed documentation

---

**Last Updated**: 2025-11-08  
**Policy Version**: 1.0.0
