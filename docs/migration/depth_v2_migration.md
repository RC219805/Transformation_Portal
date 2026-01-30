# Migration Guide: Depth Module v2.0

**Audience:** End Users, Pipeline Developers
**Timeline:** v1.8 (deprecation warnings) → v2.0 (removal, est. Q3 2026)
**Support Window:** 6 months minimum (may be extended)

---

## TL;DR

**What's changing:**
- Old modules (`depth/`, `lux_depth_v3/`, `depth_intelligence/`) are being consolidated
- New unified module: `depth_canonical/` with integrated PBR support
- **Your code will continue to work** via backward compatibility shims until v2.0

**What you need to do:**
1. Update import statements (5 minutes)
2. Test with deprecation warnings enabled (`python -Wd your_script.py`)
3. Update configuration presets if using custom YAML (optional)

**When you need to do it:**
- **Now:** Optional (but recommended to silence warnings)
- **Before v2.0 (Q3 2026):** Mandatory

---

## Breaking Changes

### None in v1.8-v1.9 ✅

All old imports continue to work via compatibility shims. You will see deprecation warnings but no functionality changes.

### v2.0.0 Removals ⚠️

The following modules will be **removed** in v2.0.0 (estimated Q3 2026):

- `transformation_portal.depth`
- `transformation_portal.lux_depth_v3`
- `transformation_portal.depth_intelligence`

**After v2.0, imports from these modules will raise `ModuleNotFoundError`.**

---

## Migration Steps

### Step 1: Update Import Statements

**Before (Deprecated):**
```python
# Old depth module
from transformation_portal.depth import (
    ArchitecturalDepthPipeline,
    DepthConfig,
)

# Old lux_depth_v3 module
from transformation_portal.lux_depth_v3 import (
    DA3Config,
    generate_pbr_maps,
    PBRConfig,
)

# Old depth_intelligence module
from transformation_portal.depth_intelligence import DepthEstimator
```

**After (Canonical):**
```python
# New unified module
from transformation_portal.depth_canonical import (
    DepthPipeline,  # Replaces ArchitecturalDepthPipeline
    DepthConfig,    # Replaces DA3Config
    generate_pbr_maps,
    PBRConfig,
)
```

### Step 2: Update Class Names

| Old Name | New Name | Notes |
|----------|----------|-------|
| `ArchitecturalDepthPipeline` | `DepthPipeline` | Functionality unchanged |
| `DA3Config` | `DepthConfig` | Unified config for DA2 and DA3 |
| `DepthEstimator` | `DepthPipeline` | Pipeline now handles estimation |
| `BatchOptions` | `BatchConfig` | Renamed for consistency |

**Example:**
```python
# Before
pipeline = ArchitecturalDepthPipeline(config)

# After
pipeline = DepthPipeline(config)
```

### Step 3: Test with Warnings Enabled

Run your code with Python's deprecation warnings enabled:

```bash
# Enable all warnings
python -Wd your_script.py

# Or explicitly enable FutureWarning
python -W default::FutureWarning your_script.py
```

You should see warnings like:
```
FutureWarning: transformation_portal.depth is deprecated and will be removed in v2.0.0 (est. Q3 2026).
Use transformation_portal.depth_canonical instead.
Migration guide: https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md
```

### Step 4: Update Configuration Presets (Optional)

If you use custom YAML presets, add a version field:

**Before:**
```yaml
# config/my_preset.yaml
depth_model:
  variant: "da3-metric-large"
  device: "cpu"
```

**After:**
```yaml
# config/my_preset.yaml
version: "1.0"  # NEW: Explicit version
preset_name: "my_preset"
last_updated: "2026-01-30"

depth_model:
  variant: "da3-metric-large"
  device: "cpu"
```

Presets without `version` will still work but will be assumed to be version 0.9. Explicit versioning helps with future schema migrations.

---

## Side-by-Side Examples

### Example 1: Basic Depth Processing

**Before (v1.7 and earlier):**
```python
from transformation_portal.depth import ArchitecturalDepthPipeline, DepthConfig

# Load config
config = DepthConfig.from_preset("architectural_interior")

# Create pipeline
pipeline = ArchitecturalDepthPipeline(config)

# Process image
depth_map = pipeline.process_image("input.jpg", "output/")
```

**After (v1.8+):**
```python
from transformation_portal.depth_canonical import DepthPipeline, DepthConfig

# Load config
config = DepthConfig.from_preset("architectural_interior")

# Create pipeline
pipeline = DepthPipeline(config)

# Process image (now returns dict with depth + optional PBR)
result = pipeline.process_image("input.jpg", "output/")
print(result["depth"])  # Path to depth map
```

### Example 2: PBR Map Generation

**Before (v1.7 and earlier):**
```python
from transformation_portal.lux_depth_v3 import generate_pbr_maps, PBRConfig
import numpy as np

# Generate PBR maps
depth = np.random.rand(2160, 3840).astype(np.float32)
config = PBRConfig(normal_strength=1.2)
normal, roughness, ao = generate_pbr_maps(depth, config)
```

**After (v1.8+):**
```python
from transformation_portal.depth_canonical import generate_pbr_maps, PBRConfig
import numpy as np

# Same API - no changes required
depth = np.random.rand(2160, 3840).astype(np.float32)
config = PBRConfig(normal_strength=1.2)
normal, roughness, ao = generate_pbr_maps(depth, config)
```

### Example 3: Integrated Pipeline with PBR

**New in v1.8+ (recommended):**
```python
from transformation_portal.depth_canonical import DepthPipeline, DepthConfig

# Load config and enable PBR
config = DepthConfig.from_preset("architectural_interior")
config.processing.pbr.enabled = True
config.processing.pbr.normal_strength = 1.5

# Create pipeline
pipeline = DepthPipeline(config)

# Process image (generates depth + PBR in one call)
result = pipeline.process_image("input.jpg", "output/")

print(result["depth"])      # output/input_depth.png
print(result["normal"])     # output/input_normal.png
print(result["roughness"])  # output/input_roughness.png
print(result["ao"])         # output/input_ao.png
```

### Example 4: Batch Processing

**Before (v1.7 and earlier):**
```python
from transformation_portal.depth import ArchitecturalDepthPipeline, DepthConfig
from pathlib import Path

config = DepthConfig.from_preset("default")
pipeline = ArchitecturalDepthPipeline(config)

# Manual batch loop
for image_path in Path("input/").glob("*.jpg"):
    pipeline.process_image(image_path, "output/")
```

**After (v1.8+):**
```python
from transformation_portal.depth_canonical import DepthPipeline, DepthConfig
from pathlib import Path

config = DepthConfig.from_preset("default")
pipeline = DepthPipeline(config)

# Built-in batch processing
image_paths = list(Path("input/").glob("*.jpg"))
results = pipeline.batch_process(image_paths, "output/")
```

---

## New Features in v2.0

### Integrated PBR Generation

No more separate calls to `generate_pbr_maps()`. Enable PBR in config:

```python
config = DepthConfig.from_preset("architectural_interior")
config.processing.pbr.enabled = True  # NEW: Built-in PBR

pipeline = DepthPipeline(config)
result = pipeline.process_image("input.jpg", "output/")

# Automatically includes PBR maps if enabled
assert "normal" in result
assert "roughness" in result
assert "ao" in result
```

### Unified Configuration

No more separate `DA3Config` vs. `DepthConfig`. One config for all:

```python
# Supports both DA2 and DA3 models
config = DepthConfig(
    model_variant="da3-metric-large",  # or "da2-small"
    device="cpu",  # or "cuda", "mps", "coreml"
    processing=ProcessingConfig(
        pbr=PBRConfig(enabled=True),
        apply_bilateral=True,
        enable_zone_mapping=True,
    )
)
```

### Better Error Messages

```python
# Before: Cryptic errors
# ImportError: cannot import name 'DA3Config' from 'transformation_portal.lux_depth_v3'

# After: Helpful errors with migration guidance
# FutureWarning: transformation_portal.lux_depth_v3.DA3Config is deprecated.
# Use transformation_portal.depth_canonical.DepthConfig instead.
# Migration guide: https://github.com/.../depth_v2_migration.md
```

### Performance Improvements

- 10-20x caching speedup for repeated processing
- LRU cache with SHA256-based keys
- Optional parallel PBR generation (ThreadPoolExecutor)

---

## FAQ

### Q: Do I need to change my code immediately?
**A:** No. Old APIs work until v2.0.0 (estimated Q3 2026, minimum 6 months from v1.8 release). However, updating now silences deprecation warnings and future-proofs your code.

### Q: Will my existing presets work?
**A:** Yes. Presets without `version` field will be auto-migrated to v1.0 schema. You'll see a warning recommending you add the version field.

### Q: How do I test the migration?
**A:** Enable deprecation warnings: `python -Wd your_script.py`. If you see no warnings, you're fully migrated.

### Q: What if I can't migrate before v2.0?
**A:** Contact the maintainers if you need an extended support window. Requests for extension will be evaluated at Month 5 (1 month before v2.0 release).

### Q: Will performance change?
**A:** No regression expected. Benchmarks show equivalent or better performance (10-20x caching speedup for iterative workflows).

### Q: Can I use both old and new APIs in the same project?
**A:** Yes, but not recommended. Old APIs are shims that redirect to new APIs, so you'll get deprecation warnings. Mixing makes code harder to maintain.

### Q: What if I'm using deprecated internal APIs?
**A:** Internal APIs (e.g., `InferenceEngine`, `BoundedCache`) are not exposed in the new module. If you depend on internals, open a GitHub issue to discuss your use case.

### Q: Will this break my CI/CD pipeline?
**A:** No, unless you treat warnings as errors (`-Werror`). If using `-Werror`, update imports before v1.8 or filter out `FutureWarning`.

### Q: How do I suppress warnings temporarily?
**A:**
```python
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformation_portal.depth")

# Or suppress all deprecation warnings (not recommended)
warnings.filterwarnings("ignore", category=FutureWarning)
```

### Q: Where can I get help?
**A:**
- **Documentation:** This guide and `docs/architecture/ADR-001-PBR-Integration-Architecture.md`
- **GitHub Issues:** https://github.com/RC219805/Transformation_Portal/issues
- **Community:** Open a discussion in GitHub Discussions

---

## Automated Migration Script

For projects with many files, use this script to automate import updates:

```python
#!/usr/bin/env python3
"""Automated migration script for depth module imports."""

import re
import sys
from pathlib import Path

REPLACEMENTS = [
    # Module imports
    (r'from transformation_portal\.depth import', 'from transformation_portal.depth_canonical import'),
    (r'from transformation_portal\.lux_depth_v3 import', 'from transformation_portal.depth_canonical import'),
    (r'from transformation_portal\.depth_intelligence import', 'from transformation_portal.depth_canonical import'),

    # Class renames
    (r'\bArchitecturalDepthPipeline\b', 'DepthPipeline'),
    (r'\bDA3Config\b', 'DepthConfig'),
    (r'\bBatchOptions\b', 'BatchConfig'),
]

def migrate_file(filepath: Path) -> bool:
    """Migrate a single Python file."""
    try:
        content = filepath.read_text()
        original = content

        for pattern, replacement in REPLACEMENTS:
            content = re.sub(pattern, replacement, content)

        if content != original:
            filepath.write_text(content)
            print(f"✅ Migrated: {filepath}")
            return True
        else:
            print(f"⏭️  Skipped: {filepath} (no changes)")
            return False
    except Exception as e:
        print(f"❌ Error: {filepath}: {e}")
        return False

def main():
    """Migrate all Python files in directory."""
    if len(sys.argv) < 2:
        print("Usage: python migrate_depth_imports.py <directory>")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    python_files = list(root_dir.rglob("*.py"))

    print(f"Found {len(python_files)} Python files")

    migrated = sum(migrate_file(f) for f in python_files)

    print(f"\n✅ Migrated {migrated}/{len(python_files)} files")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Backup your code first!
git commit -am "Backup before migration"

# Run migration
python migrate_depth_imports.py src/

# Test your code
python -Wd -m pytest tests/

# If all tests pass, commit
git commit -am "Migrate to depth_canonical"
```

---

## Timeline and Support

| Date (est.) | Milestone | Status | Actions |
|-------------|-----------|--------|---------|
| **Q1 2026** | v1.8 release | Deprecation warnings | Compatibility shims active |
| **Q2 2026** | v1.9 release | Final reminders | Migration tooling validated |
| **Q3 2026** | v2.0 release | Old modules removed | `depth_canonical` only |

**Support Commitment:**
- Minimum 6-month deprecation window
- May be extended if >20% of users still on deprecated APIs
- Maximum extension: +3 months (total 9 months)
- No extensions beyond v2.1

**Communication Channels:**
- GitHub Releases: Deprecation announcements
- README.md: Updated timeline
- This migration guide: Always current
- Email: Enterprise users with support contracts

---

## Troubleshooting

### Issue: Import Error After Update

**Symptom:**
```python
ImportError: cannot import name 'ArchitecturalDepthPipeline' from 'transformation_portal.depth_canonical'
```

**Solution:** Use new class name:
```python
# Change this:
from transformation_portal.depth_canonical import ArchitecturalDepthPipeline

# To this:
from transformation_portal.depth_canonical import DepthPipeline
```

### Issue: Config Validation Error

**Symptom:**
```
ValueError: Unknown config field: 'model_type'
```

**Solution:** Update config schema. Old field names may have changed:
- `model_type` → `model_variant`
- `cache_depth_maps` → `cache_enabled`

Check `docs/architecture/ADR-001-PBR-Integration-Architecture.md` for full schema.

### Issue: Missing PBR Maps

**Symptom:** `result` dict only contains `"depth"`, missing `"normal"`, `"roughness"`, `"ao"`.

**Solution:** Enable PBR in config:
```python
config.processing.pbr.enabled = True
```

### Issue: Performance Regression

**Symptom:** Processing slower after migration.

**Solution:** Enable caching:
```python
config.cache_enabled = True
config.cache_size = 128  # LRU cache size
```

### Issue: Warnings in Tests

**Symptom:** Test suite shows deprecation warnings.

**Solution:** Either:
1. Update test imports to new API (recommended)
2. Suppress warnings in test setup:
```python
import warnings
import pytest

@pytest.fixture(autouse=True)
def suppress_deprecation_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
```

---

## Feedback and Questions

If you encounter issues not covered in this guide:

1. **Check the docs:** `docs/architecture/ADR-001-PBR-Integration-Architecture.md`
2. **Search GitHub Issues:** Someone may have already reported the issue
3. **Open a new issue:** https://github.com/RC219805/Transformation_Portal/issues/new
4. **Community discussion:** GitHub Discussions for general questions

**Please include:**
- Python version
- Transformation Portal version
- Minimal code example reproducing the issue
- Full error message and traceback

---

**Last Updated:** 2026-01-30
**Version:** 1.0
**Status:** Active (v1.8 - v2.0)
