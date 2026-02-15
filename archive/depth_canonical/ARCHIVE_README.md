# Archive: depth_canonical Module

**Archived:** 2025-01-28
**Status:** Superseded
**Replacement:** `src/transformation_portal/depth/backends/` (ADR-019)

---

## Why Archived

The `depth_canonical` module was the original implementation of the unified depth estimation pipeline. It has been **superseded by ADR-019** which established a new backend architecture with the following improvements:

1. **Backend Abstraction:** ADR-019 introduced `src/transformation_portal/depth/backends/` with a cleaner separation between model implementations (Depth Anything V2, Depth Anything V3, Depth Pro).

2. **Improved Architecture:** The new architecture provides:
   - Standardized backend interface (`DepthBackend` ABC)
   - Better model registry and selection
   - Cleaner separation of concerns
   - More maintainable codebase

3. **Active Development:** The new backend system is actively maintained and receives ongoing improvements, while `depth_canonical` was frozen.

---

## What Was Replaced

### Module Structure
```
depth_canonical/
├── __init__.py           → depth/backends/__init__.py
├── config.py             → depth/backends/config.py
├── pipeline.py           → depth/backends/pipeline.py
├── models/
│   ├── da2_wrapper.py    → depth/backends/depth_anything_v2.py
│   ├── da3_wrapper.py    → depth/backends/depth_anything_v3.py
│   └── registry.py       → depth/backends/registry.py
├── processing/
│   └── pbr.py            → (functionality moved to separate modules)
├── security/
│   └── validation.py     → shared/validation.py (consolidated)
└── io/
    ├── io_atomic.py      → shared/io/atomic.py (consolidated)
    └── writers.py        → depth/io/writers.py
```

### Key Components Replaced
- **Pipeline:** `depth_canonical.pipeline` → `depth.backends.pipeline`
- **Models:** Individual wrappers → Backend implementations
- **Configuration:** `UnifiedDepthConfig` → Backend-specific configs
- **PBR Processing:** Moved to dedicated processing modules
- **Security Validation:** Consolidated into shared validation layer

---

## Migration Path

If you encounter code referencing `depth_canonical`, update as follows:

### Old Import Pattern
```python
from transformation_portal.depth_canonical import (
    UnifiedDepthConfig,
    run_depth_pipeline,
    ModelVariant
)
```

### New Import Pattern
```python
from transformation_portal.depth.backends import (
    DepthBackendConfig,
    create_backend,
    DepthBackendType
)
```

### Old Configuration
```python
config = UnifiedDepthConfig(
    model=ModelConfig(variant=ModelVariant.DA2_LARGE),
    processing=ProcessingConfig(apply_bilateral=True)
)
```

### New Configuration
```python
config = DepthBackendConfig(
    backend_type=DepthBackendType.DEPTH_ANYTHING_V2,
    model_size="large",
    processing_options={"bilateral_filter": True}
)
```

---

## Related References

- **ADR-019:** Architecture Decision Record for depth backend abstraction
- **PR #906:** Implementation of new backend architecture (merged 2026-01-21)
- **Last Commit:** `9af004ee` (2026-01-21) - Final updates before supersession
- **Replacement Location:** `src/transformation_portal/depth/backends/`

---

## Git History Preservation

This module was moved using `git mv` to preserve full commit history. To view the history:

```bash
# View history of archived module
git log --follow -- archive/depth_canonical/

# View specific file history
git log --follow -- archive/depth_canonical/pipeline.py

# Compare with replacement
git log -- src/transformation_portal/depth/backends/
```

---

## Test Coverage

Associated tests were also archived:
- `archive/depth_canonical_tests/` (formerly `tests/depth_canonical/`)
- `archive/test_depth_canonical_yaml.py` (formerly `tests/test_depth_canonical_yaml.py`)

The new backend architecture has comprehensive test coverage in `tests/depth/backends/`.

---

## Notes

- **Do not import from this archived module** - it is preserved for historical reference only
- The archived code is frozen and will not receive updates
- For new development, use `transformation_portal.depth.backends`
- If you need to reference this code, consider it read-only documentation of the previous architecture

---

**For Questions:** Consult ADR-019 or the current depth backend documentation in `docs/depth/backends/`
