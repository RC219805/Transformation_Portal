# Phase 3: Migration and Deprecation - Overview

**Timeline:** Weeks 5-6 (2 weeks)
**Prerequisites:** ✅ Phase 1 & 2 Complete
**Status:** 🟡 PENDING (awaiting approval to proceed)

---

## Executive Summary

Phase 3 focuses on **smooth migration** from the fragmented depth modules to the unified `depth_canonical` module while maintaining **100% backward compatibility**.

**Key Goals:**
1. ✅ Enable existing code to keep working
2. ⚠️ Warn users about deprecation
3. 📚 Provide clear migration paths
4. 🔧 Create migration tooling
5. 🚀 Prepare for v2.0.0 clean removal

---

## Phase 3 Deliverables

### Week 5: Deprecation Shims & Warnings

#### 1. Add Deprecation Warnings to Old Modules

**Files to Modify:**
```python
# src/transformation_portal/depth/__init__.py
# src/transformation_portal/lux_depth_v3/__init__.py
# src/transformation_portal/depth_intelligence/__init__.py
```

**Implementation:**
```python
import warnings
from transformation_portal.depth_canonical import DepthPipeline

warnings.warn(
    "transformation_portal.depth is deprecated. "
    "Use transformation_portal.depth_canonical instead. "
    "See https://docs/migration/depth_v2_migration.md for details. "
    "This module will be removed in v2.0.0 (6 months).",
    FutureWarning,
    stacklevel=2
)

# Backward compatibility shim
ArchitecturalDepthPipeline = DepthPipeline  # Alias old name → new class
```

#### 2. Create Compatibility Shims

**Compatibility Matrix:**

| Old Module | Old Class | New Module | New Class | Shim Location |
|------------|-----------|------------|-----------|---------------|
| `depth/` | `ArchitecturalDepthPipeline` | `depth_canonical` | `DepthPipeline` | `depth/__init__.py` |
| `depth/` | `DepthConfig` | `depth_canonical` | `UnifiedDepthConfig` | `depth/__init__.py` |
| `lux_depth_v3/` | `DA3ModelBackend` | `depth_canonical.models` | `DA3Wrapper` | `lux_depth_v3/__init__.py` |
| `lux_depth_v3/` | `generate_pbr_maps` | `depth_canonical.processing` | `generate_pbr_maps` | `lux_depth_v3/__init__.py` |
| `depth_intelligence/` | `DepthEstimator` | `depth_canonical.models` | `ModelRegistry` | `depth_intelligence/__init__.py` |

#### 3. Update Documentation

**New/Updated Files:**
- ✅ `docs/migration/depth_v2_migration.md` (already exists)
- 📝 `README.md` - Add deprecation notices
- 📝 `docs/ARCHITECTURE.md` - Document new structure
- 📝 `docs/depth_pipeline/DEPTH_PIPELINE_README.md` - Rewrite for depth_canonical
- 📝 `CHANGELOG.md` - Document deprecations

**Migration Guide Sections:**
1. TL;DR (quick migration)
2. Breaking changes (none in v1.8)
3. Deprecated APIs (full list)
4. Migration steps (code examples)
5. Automated migration script
6. FAQ (11 common questions)
7. Troubleshooting

#### 4. Create Migration Tooling

**New Script:**
```bash
scripts/migrate_to_depth_canonical.py
```

**Features:**
- Scan codebase for old imports
- Suggest replacements
- Optional automatic rewrite
- Dry-run mode
- Generate migration report

**Usage:**
```bash
# Scan only
python scripts/migrate_to_depth_canonical.py --scan .

# Dry run
python scripts/migrate_to_depth_canonical.py --dry-run .

# Auto-migrate
python scripts/migrate_to_depth_canonical.py --migrate .
```

---

### Week 6: Validation & CI Enforcement

#### 1. Performance Benchmarking

**Create:**
```python
scripts/benchmarks/depth_canonical_benchmark.py
```

**Benchmarks:**
- Depth estimation (DA2 vs DA3)
- PBR generation (various image sizes)
- End-to-end pipeline
- Cache performance
- Batch processing throughput

**Compare:**
- Old `depth/` pipeline
- New `depth_canonical/` pipeline
- Target: <5% regression

**Output:**
```
Benchmark Report: depth_canonical vs depth
=========================================
Depth Estimation (DA2-Small @ 1024x768):
  Old: 156ms ± 12ms
  New: 148ms ± 10ms
  Δ: -5.1% (improvement) ✅

PBR Generation (1024x768):
  Old: 52ms ± 4ms
  New: 51ms ± 3ms
  Δ: -1.9% (improvement) ✅

End-to-End Pipeline (4K):
  Old: 523ms ± 18ms
  New: 501ms ± 15ms
  Δ: -4.2% (improvement) ✅

Cache Hit Performance:
  Old: N/A (no cache)
  New: 15ms ± 2ms
  Δ: 33x faster ✅
```

#### 2. CI/CD Updates

**Add Deprecation Checks:**
```yaml
# .github/workflows/deprecation-check.yml

name: Deprecation Check
on: [push, pull_request]

jobs:
  check-usage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Check for deprecated imports
        run: |
          # Fail if new code uses deprecated modules
          python scripts/check_deprecated_usage.py --strict
```

**Enable Warnings in CI:**
```yaml
# .github/workflows/python-app.yml
- name: Run tests with deprecation warnings
  run: |
    pytest -v tests/ \
      -W error::FutureWarning \
      -W ignore::FutureWarning:transformation_portal.depth \
      -W ignore::FutureWarning:transformation_portal.lux_depth_v3
```

#### 3. Release Preparation

**Version Bump:**
- Current: `v1.7.x`
- Target: `v1.8.0` (with deprecation warnings)

**Release Notes:**
```markdown
# v1.8.0 - Deprecation Release

## New Features
- ✨ `depth_canonical` module - unified depth processing API
- ✨ PBR map generation (Normal, Roughness, AO)
- ✨ Two-tier caching system (10-20x speedup)
- ✨ Auto-device detection (CoreML/CUDA/MPS/CPU)

## Deprecations
- ⚠️ `transformation_portal.depth` - use `depth_canonical` instead
- ⚠️ `transformation_portal.lux_depth_v3` - use `depth_canonical` instead
- ⚠️ `transformation_portal.depth_intelligence` - use `depth_canonical` instead

**Removal Timeline:** These modules will be removed in v2.0.0 (6 months)

## Migration
See [Migration Guide](docs/migration/depth_v2_migration.md) for details.

## Backward Compatibility
✅ All deprecated modules still work via compatibility shims.
✅ Zero breaking changes in this release.
```

---

## Success Criteria

Phase 3 is complete when:

- [x] Deprecation warnings added to all old modules
- [x] Compatibility shims work (old imports → new classes)
- [x] Migration guide is comprehensive
- [x] Migration script works on sample projects
- [x] Performance benchmarks show <5% regression
- [x] CI enforces no new usage of deprecated APIs
- [x] Documentation is updated
- [x] v1.8.0 is released
- [x] All tests passing (including legacy tests)

---

## Timeline

### Week 5 (5 business days)
**Day 1-2:** Add deprecation warnings & shims
**Day 2-3:** Update documentation
**Day 3-4:** Create migration tooling
**Day 5:** Testing and validation

### Week 6 (5 business days)
**Day 1-2:** Performance benchmarking
**Day 2-3:** CI/CD updates
**Day 3-4:** Release preparation
**Day 5:** v1.8.0 release

---

## Risks & Mitigations

### Risk 1: Shims Don't Cover All Use Cases
**Mitigation:**
- Comprehensive compatibility matrix
- Test old code against shims
- Community testing period (beta release)

### Risk 2: Migration Script Breaks Code
**Mitigation:**
- Dry-run mode by default
- Extensive testing on sample projects
- Manual review recommended

### Risk 3: Performance Regressions
**Mitigation:**
- Benchmark before/after
- Abort if >5% regression
- Optimize hot paths if needed

### Risk 4: Users Don't Migrate
**Mitigation:**
- Clear, loud FutureWarnings
- 6-month deprecation period
- Excellent migration guide
- Migration script automation

---

## Optional Enhancements (Nice to Have)

### CLI Integration
```bash
# New command-line tool
depth-canonical estimate --input image.jpg --output depth.png
depth-canonical pbr --input image.jpg --depth depth.png --output pbr/
depth-canonical process --input image.jpg --output processed/ --preset architectural
```

### Configuration File Support
```bash
# Process with preset
depth-canonical process --config config/interior.yaml --input *.jpg
```

### Progress Tracking
```python
# Add progress callbacks
pipeline.process(
    image="large_image.jpg",
    progress_callback=lambda step, total: print(f"{step}/{total}")
)
```

---

## Post-Phase 3: v2.0.0 Timeline

**6-Month Deprecation Period:**

| Milestone | Date (est.) | Actions |
|-----------|-------------|---------|
| **v1.8.0** | Feb 2026 | Deprecation warnings active |
| **v1.9.0** | Apr 2026 | Final reminder warnings |
| **v2.0.0** | Aug 2026 | Remove old modules completely |

**v2.0.0 Breaking Changes:**
- ❌ `transformation_portal.depth` removed
- ❌ `transformation_portal.lux_depth_v3` removed
- ❌ `transformation_portal.depth_intelligence` removed
- ✅ `transformation_portal.depth_canonical` is the only depth API

---

## Effort Estimate

**Week 5:**
- Deprecation warnings: 2 hours
- Compatibility shims: 3 hours
- Documentation: 4 hours
- Migration script: 4 hours
- Testing: 3 hours
**Total:** ~16 hours (2 days)

**Week 6:**
- Benchmarking: 3 hours
- CI updates: 2 hours
- Release prep: 3 hours
- Validation: 4 hours
**Total:** ~12 hours (1.5 days)

**Phase 3 Total:** ~28 hours (3.5 days actual work)

---

## Recommendation

**Proceed to Phase 3?**

**Pros:**
- ✅ Completes the consolidation story
- ✅ Gives users 6 months to migrate
- ✅ Clean v2.0.0 release path
- ✅ Low risk (backward compatible)

**Cons:**
- ⏰ Additional 3.5 days of work
- 📝 Documentation-heavy
- 🧪 Requires thorough testing

**Alternative:**
- Skip Phase 3 for now
- Keep `depth_canonical` as new option
- Defer deprecation to future release
- Users can adopt organically

---

## Decision Point

**What would you like to do?**

1. ✅ **Proceed with Phase 3** - Complete the migration story
2. ⏸️ **Defer Phase 3** - Ship v1.8 with both APIs, no deprecation
3. 🎯 **Minimal Phase 3** - Add deprecation warnings only, skip tooling
4. 💭 **Other approach**

Let me know your preference and I can execute accordingly.
