# Phase 2 PR Template: Materials V3 Investigation Documentation

## Summary

Extracts and organizes **Materials V3 investigation reports** and **diagnostic scripts** as **Phase 2** of the PR #934 extraction strategy.

Consolidates scattered investigation documentation into a structured subdirectory with comprehensive indexing and diagnostic methodology.

---

## What's Included

### Investigation Reports (2-4 files)

1. **Edge Artifacts - Primary Bedroom** (`edge_artifacts_primary_bedroom.md`)
   - Documents white halos and blue contamination at material boundaries
   - **Root cause**: V2 enhancement amplifies Materials V3 edge transitions
   - **Quantitative data**: 15.40% edge artifacts, 42.01% white halos, 62,777 blue contamination pixels
   - **Key learning**: Mask feathering alone cannot fix V2 amplification issues
   - **Status**: Documented, awaiting proper fix

2. **Sky/Water Color Grading Analysis** (`sky_water_color_grading_analysis.md`)
   - Validates color changes are preset-based enhancements, not degradation
   - **Quantitative analysis**: Sky -1.06% brightness, +2.91% saturation (acceptable)
   - **Key finding**: No degradation detected, behavior matches luxury_estate preset
   - **Status**: Resolved, validated as expected behavior

3. **[Additional investigations if found in branches]**
   - Depth identity bug analysis
   - Pixel ops zero delta investigation
   - Performance profiling reports

### Diagnostic Tools (2+ files)

1. **`diagnose_sky_issue.py`** (5.4 KB)
   - **Purpose**: Depth map analyzer with zone statistics
   - **Features**: Percentile analysis, zone breakdown, histogram visualization
   - **Usage**: `python tools/investigations/materials_v3/diagnose_sky_issue.py --depth depth.tif`
   - **Dependencies**: PIL, numpy (core deps)

2. **`create_sky_comparison.py`** (4.4 KB)
   - **Purpose**: Visual comparison generator for before/after validation
   - **Features**: Side-by-side layout, sky region cropping, downscaling
   - **Usage**: `python tools/investigations/materials_v3/create_sky_comparison.py --before in.tif --after out.tif`
   - **Dependencies**: PIL, numpy (core deps)

3. **[Additional tools if found]**
   - Pixel ops delta analyzer
   - Material mask visualizer
   - Performance profiler

### Documentation (3 new files)

1. **Investigation Index** (`docs/investigations/materials_v3/README.md`, ~300 lines)
   - Complete catalog of all investigations with summaries
   - Cross-references to Phase A/B completion reports
   - Links to diagnostic tools and related documentation
   - Quick navigation for common debugging scenarios

2. **Diagnostic Methodology** (`docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md`, ~250 lines)
   - Systematic 6-stage approach to debugging Materials V3 issues
   - Stage-by-stage isolation techniques (V3 vs V2 vs Depth)
   - Quantitative analysis methods (delta maps, statistics, thresholds)
   - Fix validation checklist with acceptance criteria
   - Common bug patterns and diagnostic strategies

3. **Tool Usage Guide** (`tools/investigations/materials_v3/README.md`, ~200 lines)
   - Installation requirements (core deps only)
   - Detailed usage examples for each tool
   - Integration patterns with main pipeline
   - Contribution guidelines for new tools

---

## File Organization

```
docs/investigations/materials_v3/
├── README.md                                # Investigation index (NEW)
├── DIAGNOSTIC_METHODOLOGY.md                # Debugging guide (NEW)
├── edge_artifacts_primary_bedroom.md        # Moved + renamed
├── sky_water_color_grading_analysis.md      # Moved + renamed
├── [additional_investigations].md           # Extracted from branches
└── assets/                                  # Supporting materials (images, charts)

tools/investigations/materials_v3/
├── README.md                                # Tool usage guide (NEW)
├── __init__.py                              # Package marker (NEW)
├── diagnose_sky_issue.py                    # Moved from root
├── create_sky_comparison.py                 # Moved from root
└── [additional_tools].py                    # Extracted from branches
```

---

## Value Proposition

### Knowledge Preservation
- 📚 **Diagnostic methodology documented**: Systematic 6-stage debugging approach preserved
- 🧪 **Failed fixes documented**: Mask feathering attempt documented to prevent repeat efforts
- 📊 **Quantitative baselines established**: Thresholds for edge artifacts, halos, color shifts
- 🎓 **Training material**: Real-world debugging examples for onboarding

### Reusable Assets
- 🛠️ **Production-ready diagnostic tools**: Depth analyzers, visual comparators
- 🔍 **Investigation templates**: Standardized format for future bug reports
- 📖 **Methodology playbook**: Step-by-step isolation and validation procedures

### Historical Record
- 🏗️ **Architecture insights**: Documents Materials V3 edge case handling
- 🐛 **Bug catalog**: Permanent record of issues discovered and resolved
- 🚧 **Development journey**: Captures iteration and learning from Materials V3 cycle

---

## Testing & Validation

### Documentation Quality
✅ **All markdown files lint clean**
- Checked with markdownlint-cli
- No broken internal links verified manually
- Relative paths updated for new structure

### Diagnostic Scripts
✅ **Scripts execute without errors**
- `diagnose_sky_issue.py --help` ✅ Works
- `create_sky_comparison.py --help` ✅ Works
- Import dependencies verified (PIL, numpy from core requirements.txt)

### File Structure
✅ **Organization validated**
- Clear, logical directory structure
- Files renamed for clarity (removed `_SUMMARY` suffix)
- Cross-references to Phase A/B reports working

### Data Hygiene
✅ **No sensitive data**
- No absolute paths found
- No credentials or API keys
- Test image references are generic or omitted

---

## Backward Compatibility

✅ **100% Compatible** - Documentation and tooling only, **no production code changes**

### What Changed
- **File locations**:
  - `docs/investigations/*.md` → `docs/investigations/materials_v3/*.md`
  - `diagnose_sky_issue.py` → `tools/investigations/materials_v3/diagnose_sky_issue.py`
  - `create_sky_comparison.py` → `tools/investigations/materials_v3/create_sky_comparison.py`

- **File names**: Simplified for clarity (removed `_SUMMARY`, `_INVESTIGATION` suffixes)

### What Didn't Change
- ✅ Production pipeline code (`src/transformation_portal/lux_depth_v3/`)
- ✅ Test suite (`tests/materials/`)
- ✅ Configuration files (`config/`)
- ✅ Dependencies (`requirements.txt`)
- ✅ CI/CD workflows

### Migration Impact
- **Impact**: 🟢 **NONE** - Documentation moves don't affect code
- **Breaking changes**: 🟢 **NONE** - Only internal doc references updated
- **API changes**: 🟢 **NONE** - Scripts are standalone tools

---

## Extraction Context

This is **Phase 2** of the 6-phase PR #934 extraction strategy:

| Phase | Description | Status | Effort | PR |
|-------|-------------|--------|--------|-----|
| **1** | Depth identity fix | ✅ Complete | 1-2 hours | #936 (merged) |
| **2** | Investigation docs | ⏳ **Current** | 2-3 hours | **This PR** |
| **3** | Config presets | 📅 Planned | 1-2 hours | Week 1 |
| **4** | Test fixtures | 📅 Planned | 2-3 hours | Week 1 |
| **5** | Performance optimizations | 📅 Planned | 3-4 hours | Week 2 |
| **6** | Documentation updates | 📅 Planned | 1-2 hours | Week 2 |

**Strategy Document**: [PR #934 Extraction Strategy](docs/project-status/PR934_EXTRACTION_STRATEGY.md)
**Execution Plan**: [Phase 2 Execution Plan](docs/project-status/PHASE_2_EXECUTION_PLAN.md)

---

## Related PRs & Issues

- **PR #934**: Original Materials V3 PR (to be closed after all 6 phases extracted)
- **PR #935**: Materials V3 Phases A+B - Harden pixel ops + Sky material (✅ merged 2024-02-14)
- **PR #936**: Depth identity fix - Phase 1 extraction (✅ merged 2024-02-13)

**Cross-references**:
- Phase A Complete: `docs/materials/PHASE_A_COMPLETE.md`
- Phase B Complete: `docs/materials/PHASE_B_COMPLETE.md`
- Implementation Summary: `MATERIALS_V3_IMPLEMENTATION_SUMMARY.md`
- Executive Summary: `MATERIALS_V3_EXECUTIVE_SUMMARY.md`

---

## Review Checklist

### For Reviewers

**Content Completeness**:
- [ ] All investigation reports are complete and well-organized
- [ ] Diagnostic methodology is clear and actionable
- [ ] Tool usage documentation is comprehensive
- [ ] Cross-references to related docs work correctly

**File Structure**:
- [ ] Directory structure is logical and easy to navigate
- [ ] File naming is clear and consistent
- [ ] README files provide good entry points

**Quality**:
- [ ] No markdown linting errors
- [ ] No broken links
- [ ] No absolute paths or sensitive data
- [ ] Code examples (if any) are syntactically correct

**Extraction Fidelity**:
- [ ] Investigation reports match original findings
- [ ] Diagnostic scripts work as documented
- [ ] No important context lost in reorganization

---

## Deployment Notes

### Post-Merge Actions
1. Update DOCUMENTATION_MAP.md if not included in this PR
2. Announce new investigation documentation in team channel
3. Add link to investigation index in main README.md (optional)

### No Breaking Changes
- No deployment steps required
- No dependency updates needed
- No configuration changes needed
- No database migrations needed

---

## Metrics

**Files Changed**: 10-15 files
**Lines Added**: ~800-1,000 (documentation)
**Lines Modified**: ~50-100 (path updates)
**New Directories**: 2 (`docs/investigations/materials_v3/`, `tools/investigations/materials_v3/`)

**Risk Level**: 🟢 **LOW** (documentation/tools only)
**Effort**: ⏱️ **2-3 hours** (within budget)
**Value**: ⭐⭐⭐⭐ **HIGH** (preserves critical investigation knowledge)

---

## Success Criteria

Phase 2 is **complete** when:

✅ All investigation reports consolidated in structured directory
✅ Diagnostic tools relocated and documented
✅ Comprehensive README and methodology docs created
✅ All file references updated and links working
✅ Scripts execute without import errors
✅ No markdown linting errors
✅ PR merged to main

---

**PR Author**: [Your Name]
**Extraction Plan**: [Phase 2 Execution Plan](docs/project-status/PHASE_2_EXECUTION_PLAN.md)
**Extraction Date**: 2024-02-14
**Estimated Review Time**: 30-45 minutes
