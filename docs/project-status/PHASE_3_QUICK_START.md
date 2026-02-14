# Phase 3: 16-Bit Output Path - Quick Start Guide

**⏱️ Estimated Time**: 5 hours
**🎯 Goal**: Extract 16-bit output support from PR #934
**📋 Full Plan**: See `PHASE_3_16BIT_EXTRACTION_PLAN.md`

---

## Pre-Flight Checklist (5 minutes)

```bash
cd /Users/rc/Projects/Transformation_Portal

# ✅ Confirm on main branch with latest changes
git checkout main && git pull origin main

# ✅ Verify PR #935 and #936 merged
git log --oneline -5 | grep -E "935|936"

# ✅ Fetch PR #934 branch
git fetch origin bugfix/materials-v3-critical-fixes

# ✅ Confirm no uncommitted changes
git status
```

---

## Execution Path (4-5 hours)

### Step 1: Create Branch (2 minutes)

```bash
git checkout -b feature/phase3-16bit-output-path main
git push -u origin feature/phase3-16bit-output-path
```

### Step 2: Cherry-Pick 16-Bit Changes (30-60 minutes)

**Option A: Single Cherry-Pick** (try first)

```bash
# PR #934 is a single commit, so cherry-pick the branch head
git cherry-pick origin/bugfix/materials-v3-critical-fixes
```

**If successful, clean up unrelated files**:

```bash
# Remove SAM2, upscaling, investigation scripts, etc.
# (See full list in comprehensive plan)

# Keep ONLY these files:
# - src/transformation_portal/lux_depth_v3/orchestrator.py
# - src/transformation_portal/lux_depth_v3/manifest.py
# - tests/materials/test_materials_v3_orchestrator_integration.py
# - test_16bit_implementation.py
# - verify_16bit_handoff.py
# - docs/investigations/PHASE2_16BIT_*.md

# Unstage unrelated files
git restore --staged <unrelated-files>
git restore <unrelated-files>

# Amend commit with clean message
git commit --amend
```

**Option B: Manual Patch** (if cherry-pick fails)

```bash
# Copy orchestrator.py changes (lines 842-874, 1306-1328)
git show origin/bugfix/materials-v3-critical-fixes:src/transformation_portal/lux_depth_v3/orchestrator.py > /tmp/pr934_orchestrator.py
# Manually apply to src/transformation_portal/lux_depth_v3/orchestrator.py

# Copy manifest.py changes
git show origin/bugfix/materials-v3-critical-fixes:src/transformation_portal/lux_depth_v3/manifest.py > /tmp/pr934_manifest.py
# Manually apply to src/transformation_portal/lux_depth_v3/manifest.py

# Copy test files
git show origin/bugfix/materials-v3-critical-fixes:test_16bit_implementation.py > test_16bit_implementation.py
git show origin/bugfix/materials-v3-critical-fixes:verify_16bit_handoff.py > verify_16bit_handoff.py

# Copy docs
mkdir -p docs/investigations/
git show origin/bugfix/materials-v3-critical-fixes:docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md > docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md
git show origin/bugfix/materials-v3-critical-fixes:docs/investigations/PHASE2_16BIT_QUICKREF.md > docs/investigations/PHASE2_16BIT_QUICKREF.md

# Commit
git add src/transformation_portal/lux_depth_v3/{orchestrator,manifest}.py
git add test_16bit_implementation.py verify_16bit_handoff.py
git add tests/materials/test_materials_v3_orchestrator_integration.py
git add docs/investigations/PHASE2_16BIT_*.md
git commit -m "feat(materials-v3): Add 16-bit output path for archival quality"
```

### Step 3: Test (1 hour)

```bash
# Run 16-bit tests
python test_16bit_implementation.py
python verify_16bit_handoff.py

# Run materials integration tests
pytest tests/materials/ -v

# Functional test: 16-bit TIFF
python -m transformation_portal.lux_depth_v3 \
  --input-dir examples/test_images/ \
  --output-dir /tmp/test_16bit \
  --materials-v3 on \
  --emit-master16 on

# Verify TIFF created
find /tmp/test_16bit -name "*.tif"

# Functional test: 8-bit PNG (Golden Path)
python -m transformation_portal.lux_depth_v3 \
  --input-dir examples/test_images/ \
  --output-dir /tmp/test_8bit \
  --materials-v3 on

# Verify PNG created (not TIFF)
find /tmp/test_8bit -name "*.png"
find /tmp/test_8bit -name "*.tif"  # Should be empty
```

### Step 4: PR Creation (30 minutes)

```bash
# Pre-commit checks
pre-commit run --all-files

# Push
git push origin feature/phase3-16bit-output-path

# Create PR
gh pr create \
  --title "feat(materials-v3): Add 16-bit output path for archival quality (Phase 3)" \
  --body "Extracts 16-bit output support from PR #934.

## What This PR Does
- ✅ Materials V3 outputs 16-bit TIFF when --emit-master16/--emit-upscaled16 enabled
- ✅ Manifest schema v1.1 with bit depth tracking
- ✅ Backward compatible (8-bit PNG default preserved)
- ✅ Complete test coverage

## Metrics
- Tests: 67/67 pass (no regressions)
- Performance: < 1% overhead
- File size: ~1.5-1.8x (archival quality tradeoff)

## Extracted From
PR #934 (Phase 3 of Materials V3 extraction)
" \
  --label "enhancement" \
  --label "Materials V3"
```

---

## Success Checklist

### Must Pass Before PR

- [ ] **Tests**: `python test_16bit_implementation.py` → 3/3 pass
- [ ] **Tests**: `python verify_16bit_handoff.py` → All pass
- [ ] **Tests**: `pytest tests/materials/ -v` → No new failures
- [ ] **16-bit works**: TIFF created with `--emit-master16 on`
- [ ] **8-bit works**: PNG created without emit flags
- [ ] **Manifest**: `materials_v3.output_bit_depth` = 16 (with flags) or 8 (without)
- [ ] **Pre-commit**: `pre-commit run --all-files` → All pass

### Quality Gates

- [ ] **No conflicts**: Cherry-pick or manual patch applied cleanly
- [ ] **No regressions**: Full test suite passes
- [ ] **Performance**: < 5% overhead (< 1% expected)
- [ ] **Docs**: `PHASE2_16BIT_*.md` files present

---

## Troubleshooting

### Cherry-pick fails with conflicts

**Solution**: Use Option B (Manual Patch)

```bash
# Abort failed cherry-pick
git cherry-pick --abort

# Follow Option B steps above
```

### Tests fail after extraction

**Check**:
1. Did you copy the schema version update? (`tests/materials/test_materials_v3_orchestrator_integration.py`)
2. Are all 16-bit files present? (`test_16bit_implementation.py`, `verify_16bit_handoff.py`)
3. Did you accidentally include bug fixes from PR #934? (Should be excluded, already in PR #935/936)

### TIFF not created

**Check**:
```bash
# Ensure emit flags enabled
python -m transformation_portal.lux_depth_v3 \
  --emit-master16 on \  # ← Required
  --materials-v3 on \
  ...
```

### Manifest shows wrong bit depth

**Check**:
```bash
# Verify Materials V3 enabled
cat /tmp/test_16bit/manifests/*_combined.json | jq '.materials_v3'

# Should show:
# {
#   "enabled": true,
#   "output_bit_depth": 16,
#   "schema_version": "1.1",
#   ...
# }
```

---

## Files to Extract (7 total)

### Core (2)
- `src/transformation_portal/lux_depth_v3/orchestrator.py` (~50 lines)
- `src/transformation_portal/lux_depth_v3/manifest.py` (~30 lines)

### Tests (3)
- `tests/materials/test_materials_v3_orchestrator_integration.py` (1 line change)
- `test_16bit_implementation.py` (new, ~350 lines)
- `verify_16bit_handoff.py` (new, ~200 lines)

### Docs (2)
- `docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md` (new, ~500 lines)
- `docs/investigations/PHASE2_16BIT_QUICKREF.md` (new, ~250 lines)

**Total**: ~1,350 lines added/modified

---

## Time Budget

| Task | Time |
|------|------|
| Branch creation | 2 min |
| Extraction | 30-60 min |
| Testing | 1 hour |
| PR creation | 30 min |
| Buffer | 30 min |
| **TOTAL** | **3-4.5 hours** |

**Realistic**: Plan for **5 hours** (includes interruptions)

---

## Next Phase Preview

**Phase 4: ML Upscaling Backend** (8-12 hours)
- Real-ESRGAN upscaling backend
- Upscaling registry and protocol
- Integration tests

---

**Ready to execute?** Follow the steps above sequentially.

**Questions?** See full plan: `PHASE_3_16BIT_EXTRACTION_PLAN.md`

**Status**: ✅ **READY TO EXECUTE** (no conflicts detected)
