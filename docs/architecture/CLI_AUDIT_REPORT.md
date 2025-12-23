# CLI Audit Report

**Date**: December 23, 2025  
**Auditor**: Transformation Portal Architect  
**Scope**: All command-line interfaces in repository

---

## Executive Summary

**Total CLIs identified**: 22  
**Production CLIs**: 2  
**Development/Testing CLIs**: 12  
**Project-Specific CLIs**: 8  

**Recommendation**: Relocate development/project-specific CLIs to `scripts/` or `tools/` to reduce root directory clutter and clarify production vs development boundaries.

---

## CLI Classification

### ✅ Tier 1: Production CLIs

**Definition**: User-facing, production-ready, documented, tested

| CLI | Location | Purpose | Status | Action |
|-----|----------|---------|--------|--------|
| `lux-depth-v2` | Entry point (pyproject.toml) | Batch image processing | ✅ Production | **KEEP** |
| `lux-depth-v2-service` | Entry point (pyproject.toml) | Service mode (API) | ✅ Production | **KEEP** |

**Characteristics**:
- Installed via `pip install -e .`
- Comprehensive documentation
- Security-hardened
- Feature-frozen
- Part of Golden Path

---

### 🔧 Tier 2: Development/Testing CLIs

**Definition**: Internal tooling for development, testing, validation

| CLI | Current Location | Purpose | Recommendation |
|-----|------------------|---------|----------------|
| `extract_validation_metrics.py` | Root | Extract metrics from validation runs | **MOVE** → `scripts/validation/` |
| `generate_validation_report.py` | Root | Generate validation reports | **MOVE** → `scripts/validation/` |
| `test_single_image.sh` | Root | Quick image processing test | **MOVE** → `scripts/testing/` |
| `apply-patches.sh` | Root | Apply code patches | **MOVE** → `scripts/maintenance/` |
| `tools/bench_perf.py` | tools/ | Performance benchmarking | ✅ Already in tools/ |
| `tools/comparison_tool.py` | tools/ | Image comparison | ✅ Already in tools/ |
| `tools/hdr_visualizer.py` | tools/ | HDR histogram visualization | ✅ Already in tools/ |
| `tools/qa_validator.py` | tools/ | QA validation | ✅ Already in tools/ |
| `tools/time_predictor.py` | tools/ | Processing time prediction | ✅ Already in tools/ |
| `tools/material_detector.py` | tools/ | Material detection testing | ✅ Already in tools/ |
| `tools/depth_aware_lut.py` | tools/ | Depth-aware LUT testing | ✅ Already in tools/ |
| `tools/validate_ad_pipeline_v3.py` | tools/ | Pipeline validation | ✅ Already in tools/ |

**Characteristics**:
- Used by developers/maintainers
- Not part of user-facing Golden Path
- May have incomplete documentation
- Tooling for quality assurance

---

### 📋 Tier 3: Project-Specific CLIs

**Definition**: One-off scripts for specific projects/phases

| CLI | Current Location | Purpose | Recommendation |
|-----|------------------|---------|----------------|
| `LAUNCH_PHASE1.sh` | Root | Launch Phase 1 processing | **MOVE** → `scripts/projects/phase1/` or **ARCHIVE** |
| `RUN_KITCHEN_PHASE1.sh` | Root | Run kitchen scene processing | **MOVE** → `scripts/projects/750_picacho/` or **ARCHIVE** |
| `RUN_STRUCTURE_EDGE_VALIDATION.sh` | Root | Edge validation testing | **MOVE** → `scripts/projects/` or **ARCHIVE** |
| `RUN_VALIDATION_NOW.sh` | Root | Immediate validation run | **MOVE** → `scripts/validation/` or **ARCHIVE** |
| `CLEANUP_NEXT_SESSION.sh` | Root | Session cleanup | **MOVE** → `scripts/maintenance/` |
| `MOVE_TO_EXTERNAL_SSD.sh` | Root | Data migration | **MOVE** → `scripts/maintenance/` |
| `tools/ad_editorial_post_pipeline.py` | tools/ | Editorial post-production | **KEEP** (advanced workflow) |
| `tools/montecito_manifest.py` | tools/ | Montecito project manifest | **MOVE** → `scripts/projects/montecito/` or **ARCHIVE** |

**Characteristics**:
- Specific to one project or phase
- Often one-time use
- May be outdated
- Clutters root directory

---

## Proposed Directory Restructure

### New Structure

```
Transformation_Portal/
├── lux_depth_v2/           # ✅ Production module (CLI entry points)
├── scripts/
│   ├── validation/         # Validation and metrics tools
│   │   ├── extract_validation_metrics.py
│   │   ├── generate_validation_report.py
│   │   └── run_validation.sh
│   ├── testing/            # Development testing
│   │   └── test_single_image.sh
│   ├── maintenance/        # Maintenance and cleanup
│   │   ├── cleanup_next_session.sh
│   │   ├── move_to_external_ssd.sh
│   │   └── apply_patches.sh
│   └── projects/           # Project-specific scripts
│       ├── phase1/
│       │   └── launch_phase1.sh
│       ├── 750_picacho/
│       │   └── run_kitchen_phase1.sh
│       └── montecito/
│           └── montecito_manifest.py
├── tools/                  # ✅ Already correct (development tools)
│   ├── bench_perf.py
│   ├── comparison_tool.py
│   ├── qa_validator.py
│   └── ...
└── ...
```

### Benefits

1. **Clarity**: Root directory only contains production entry (via pip)
2. **Organization**: Clear separation of production, development, projects
3. **Discoverability**: Related tools grouped together
4. **Maintenance**: Easier to identify outdated scripts
5. **Onboarding**: New contributors understand structure faster

---

## Migration Plan

### Phase 1: Immediate (This Session)

**Create directory structure**:
```bash
mkdir -p scripts/validation scripts/testing scripts/maintenance scripts/projects/phase1 scripts/projects/750_picacho scripts/projects/montecito
```

**Move root-level scripts**:
```bash
# Validation
mv extract_validation_metrics.py scripts/validation/
mv generate_validation_report.py scripts/validation/
mv RUN_VALIDATION_NOW.sh scripts/validation/run_validation.sh

# Testing
mv test_single_image.sh scripts/testing/

# Maintenance
mv CLEANUP_NEXT_SESSION.sh scripts/maintenance/cleanup_next_session.sh
mv MOVE_TO_EXTERNAL_SSD.sh scripts/maintenance/move_to_external_ssd.sh
mv apply-patches.sh scripts/maintenance/

# Project-specific
mv LAUNCH_PHASE1.sh scripts/projects/phase1/launch.sh
mv RUN_KITCHEN_PHASE1.sh scripts/projects/750_picacho/run_kitchen.sh
mv RUN_STRUCTURE_EDGE_VALIDATION.sh scripts/projects/edge_validation.sh

# From tools/
mv tools/montecito_manifest.py scripts/projects/montecito/
```

**Update documentation**:
- Update any references to old paths
- Add `scripts/README.md` explaining structure
- Update CONTRIBUTING.md with new structure

### Phase 2: Validation (Next Session)

**Verify no broken references**:
```bash
# Search for hardcoded paths
grep -r "RUN_KITCHEN_PHASE1" docs/
grep -r "LAUNCH_PHASE1" docs/
grep -r "extract_validation_metrics" docs/
```

**Update CI/CD**:
- Check if any workflows reference moved scripts
- Update paths in `.github/workflows/`

### Phase 3: Archive Evaluation (Future)

**Criteria for archiving**:
- Last used > 6 months ago
- Specific to completed project
- No longer maintained
- Superseded by newer tools

**Candidates**:
- `scripts/projects/phase1/*` (if Phase 1 complete)
- `scripts/projects/750_picacho/*` (if project complete)
- `scripts/projects/edge_validation.sh` (if validation complete)

**Archive process**:
1. Move to `archive/scripts/YYYY-MM-DD/`
2. Add README explaining why archived
3. Remove from main docs

---

## Entry Point Strategy

### Current State

**Production entry points** (via `pip install -e .`):
- `lux-depth-v2` → `lux_depth_v2.cli:main`
- `lux-depth-v2-service` → `lux_depth_v2.service:main`

**Benefits**:
- Installed in PATH automatically
- No need to find scripts manually
- Version-controlled via package
- Professional appearance

### Recommendation for Advanced Tools

**If a tool graduates to "commonly used"**:
1. Add entry point to `pyproject.toml`
2. Install via `pip install -e .[tools]`
3. Example:

```toml
[project.scripts]
lux-depth-v2 = "lux_depth_v2.cli:main"
lux-depth-v2-service = "lux_depth_v2.service:main"

[project.optional-dependencies]
tools = [
    "matplotlib",  # For visualization tools
]

[project.entry-points."console_scripts"]
lux-qa-validate = "tools.qa_validator:main"  # If commonly used
lux-compare = "tools.comparison_tool:main"    # If commonly used
```

**Criteria for entry point**:
- Used by external users (not just maintainers)
- Documented and tested
- Stable API
- Part of supported workflow

---

## Enforcement

### PR Template Update

Add checklist item:
- [ ] New CLI scripts added to appropriate directory (`scripts/` or `tools/`, not root)

### Pre-commit Hook (Optional)

Prevent new root-level scripts:
```bash
# .git/hooks/pre-commit
if git diff --cached --name-only | grep -E "^[^/]+\.(py|sh)$" | grep -v "setup.py"; then
  echo "❌ New root-level scripts not allowed"
  echo "   Add to scripts/ or tools/ instead"
  exit 1
fi
```

---

## Success Metrics

**After migration**:
- [ ] Root directory has 0 CLI scripts (only entry points via pip)
- [ ] All development tools in `tools/`
- [ ] All testing/validation tools in `scripts/validation/`
- [ ] All project-specific tools in `scripts/projects/`
- [ ] Documentation updated with new paths
- [ ] CI/CD workflows updated
- [ ] No broken references in docs

---

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [docs/architecture/STABILITY_POLICY.md](../docs/architecture/STABILITY_POLICY.md) - Stability tiers
- [MISSION_STATEMENT.md](../MISSION_STATEMENT.md) - Strategic priorities

---

**Status**: ⚠️ Recommendations pending implementation  
**Next Action**: Execute Phase 1 migration (create directories, move scripts)  
**Owner**: Repository maintainers

---

*A clean root directory signals project maturity and professionalism.*
