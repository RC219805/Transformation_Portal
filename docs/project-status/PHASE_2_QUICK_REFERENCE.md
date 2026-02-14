# Phase 2 Quick Reference

**Last Updated**: 2024-02-14
**Status**: Ready for Execution
**Estimated Time**: 2-3 hours

---

## One-Page Overview

### What Phase 2 Extracts

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Investigation Documentation & Diagnostic Tools    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📚 INVESTIGATION REPORTS (2 confirmed + 2-4 potential)     │
│  ├─ Primary Bedroom Edge Artifacts                          │
│  ├─ Sky/Water Color Grading Analysis                        │
│  ├─ [Depth Identity Bug - search branches]                  │
│  └─ [Pixel Ops Zero Delta - search branches]                │
│                                                              │
│  🛠️ DIAGNOSTIC TOOLS (2 confirmed + 1-2 potential)          │
│  ├─ diagnose_sky_issue.py                                   │
│  ├─ create_sky_comparison.py                                │
│  └─ [Additional analysis tools - search branches]           │
│                                                              │
│  📖 NEW DOCUMENTATION (3 files)                              │
│  ├─ Investigation Index (README.md)                         │
│  ├─ Diagnostic Methodology                                  │
│  └─ Tool Usage Guide                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Target File Structure

```
docs/investigations/materials_v3/
├── README.md                              ⭐ NEW - Investigation index
├── DIAGNOSTIC_METHODOLOGY.md              ⭐ NEW - Debugging guide
├── edge_artifacts_primary_bedroom.md      📦 MOVED + RENAMED
├── sky_water_color_grading_analysis.md    📦 MOVED + RENAMED
└── assets/                                📁 NEW - Supporting materials

tools/investigations/materials_v3/
├── README.md                              ⭐ NEW - Tool usage
├── __init__.py                            ⭐ NEW - Package marker
├── diagnose_sky_issue.py                  📦 MOVED from root
└── create_sky_comparison.py               📦 MOVED from root
```

---

## 8-Step Execution Path

### 1️⃣ Preparation (10 min)
```bash
git fetch --all
git checkout -b docs/materials-v3-investigations
mkdir -p docs/investigations/materials_v3/assets
mkdir -p tools/investigations/materials_v3
```

### 2️⃣ Search Branches (20 min)
```bash
# Search for additional investigation files
git checkout bugfix/materials-v3-critical-fixes
git checkout feature/materials-v3-water-sky-synthesis
# Document findings
```

### 3️⃣ Reorganize Existing (30 min)
```bash
# Move investigation reports
git mv docs/investigations/PRIMARY_BEDROOM_EDGE_ARTIFACTS_SUMMARY.md \
       docs/investigations/materials_v3/edge_artifacts_primary_bedroom.md

# Move diagnostic scripts
git mv diagnose_sky_issue.py \
       tools/investigations/materials_v3/
```

### 4️⃣ Extract Additional (40 min)
```bash
# Cherry-pick additional files from branches
git show branch:path/to/file.md > docs/investigations/materials_v3/new_file.md
```

### 5️⃣ Create Documentation (60 min)
- Write `docs/investigations/materials_v3/README.md` (~300 lines)
- Write `docs/investigations/materials_v3/DIAGNOSTIC_METHODOLOGY.md` (~250 lines)
- Write `tools/investigations/materials_v3/README.md` (~200 lines)

### 6️⃣ Update References (20 min)
```bash
# Fix relative paths in moved files
cd docs/investigations/materials_v3
sed -i '' 's|../materials/|../../materials/|g' *.md
```

### 7️⃣ Validate (20 min)
```bash
# Test scripts
python tools/investigations/materials_v3/diagnose_sky_issue.py --help
python tools/investigations/materials_v3/create_sky_comparison.py --help

# Lint docs
markdownlint docs/investigations/materials_v3/**/*.md
```

### 8️⃣ Commit & PR (15 min)
```bash
git add docs/investigations/materials_v3/ tools/investigations/materials_v3/
git commit -m "docs(materials-v3): extract investigation reports (Phase 2)"
git push -u origin docs/materials-v3-investigations
gh pr create --title "docs(materials-v3): extract and organize investigation reports (Phase 2)" \
             --base main \
             --body-file docs/project-status/PHASE_2_PR_TEMPLATE.md
```

---

## Critical Checklist

### Pre-Flight
- [ ] On main branch, clean working directory
- [ ] All remote branches fetched (`git fetch --all`)
- [ ] Phase 2 execution plan reviewed

### Execution
- [ ] Directories created (`docs/investigations/materials_v3/`, `tools/investigations/materials_v3/`)
- [ ] Development branches searched for additional files
- [ ] Existing files moved to new structure
- [ ] Additional files extracted (if found)
- [ ] Three README files created and comprehensive

### Quality Gates
- [ ] All diagnostic scripts have `--help` flags
- [ ] Scripts execute without import errors
- [ ] All markdown files lint clean
- [ ] All relative links resolve correctly
- [ ] No absolute paths in documentation
- [ ] No sensitive data or credentials

### Finalization
- [ ] File references updated to new paths
- [ ] DOCUMENTATION_MAP.md updated
- [ ] Changes committed in logical groups
- [ ] PR created with comprehensive description
- [ ] PR linked to #934, #935, #936

---

## Risk Mitigation Quick Guide

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Stale file references** | 🟡 Medium | Low | Update all relative paths, test links |
| **Import errors in scripts** | 🟡 Medium | Low | Test `--help`, verify core deps only |
| **Missing investigation files** | 🟡 Medium | Low | Search all dev branches thoroughly |
| **Merge conflicts** | 🟢 Low | Very Low | Separate directory, unlikely |

---

## Time Budget Tracker

| Step | Budgeted | Actual | Status |
|------|----------|--------|--------|
| Preparation | 10 min | _____ | ⬜ |
| Search branches | 20 min | _____ | ⬜ |
| Reorganize files | 30 min | _____ | ⬜ |
| Extract additional | 40 min | _____ | ⬜ |
| Create docs | 60 min | _____ | ⬜ |
| Update references | 20 min | _____ | ⬜ |
| Validate | 20 min | _____ | ⬜ |
| Commit & PR | 15 min | _____ | ⬜ |
| **TOTAL** | **~3.5 hours** | _____ | ⬜ |

---

## Success Metrics

### Deliverables
- ✅ **2+ investigation reports** organized in structured directory
- ✅ **2+ diagnostic tools** relocated and documented
- ✅ **3 comprehensive README files** created
- ✅ **All file references** updated and working
- ✅ **PR merged** to main

### Quality
- ✅ Zero markdown linting errors
- ✅ Zero broken links
- ✅ All scripts execute `--help` successfully
- ✅ No regressions in existing tests
- ✅ Documentation is clear and actionable

---

## Quick Links

- **📋 Full Execution Plan**: [PHASE_2_EXECUTION_PLAN.md](PHASE_2_EXECUTION_PLAN.md)
- **📝 PR Template**: [PHASE_2_PR_TEMPLATE.md](PHASE_2_PR_TEMPLATE.md)
- **🎯 Extraction Strategy**: [PR934_EXTRACTION_STRATEGY.md](PR934_EXTRACTION_STRATEGY.md)
- **📚 Related Documentation**:
  - Phase A Complete: `docs/materials/PHASE_A_COMPLETE.md`
  - Phase B Complete: `docs/materials/PHASE_B_COMPLETE.md`
  - Materials V3 Summary: `MATERIALS_V3_IMPLEMENTATION_SUMMARY.md`

---

## Emergency Contacts & Escalation

**If Issues Arise**:
1. **Import errors in scripts**: Check `requirements.txt`, verify core deps (PIL, numpy)
2. **Missing files from branches**: Document in "files not found" section, proceed with what exists
3. **Broken links after move**: Use search/replace for path patterns, test manually
4. **Time overrun**: Skip "nice to have" documentation sections, focus on core deliverables

**Escalation Criteria**:
- [ ] Cannot find any investigation files in branches (Phase 2 might be complete already)
- [ ] Scripts have complex dependencies not in requirements.txt
- [ ] Major conflicts with active development branches

**Escalate To**: Transformation Portal Architect (ADR/governance decisions)

---

## Post-Merge Checklist

After PR is merged:
- [ ] Announce new investigation documentation to team
- [ ] Update team onboarding materials with link to investigation index
- [ ] Close any related issues about missing documentation
- [ ] Archive original PR #934 (after all 6 phases complete)
- [ ] Move to Phase 3: Config presets extraction

---

**Document Version**: 1.0
**Created**: 2024-02-14
**Format**: Quick reference for Phase 2 execution
