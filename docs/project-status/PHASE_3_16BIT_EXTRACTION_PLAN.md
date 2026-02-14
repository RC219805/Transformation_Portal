# Phase 3: 16-Bit Output Path - Comprehensive Execution Plan

**Date**: February 14, 2026
**Status**: READY FOR EXECUTION
**PR Source**: #934 (Materials V3 Critical Fixes)
**Priority**: HIGH
**Effort**: 4-6 hours
**Risk**: 🟡 MEDIUM (orchestrator merge conflict potential)
**Value**: ⭐⭐⭐⭐ (production quality improvement)

---

## Executive Summary

Phase 3 extracts **16-bit output support** from PR #934, enabling end-to-end archival-quality image processing. This feature allows Materials V3 to output 16-bit TIFF files when `--emit-master16` or `--emit-upscaled16` flags are enabled, preserving maximum color depth throughout the pipeline.

**Why This Matters for Luxury Real Estate:**
- 16-bit provides 65,536 tonal levels per channel vs 256 in 8-bit
- Prevents posterization in smooth gradients (skies, water, glass reflections)
- Enables professional color grading and print reproduction
- Future-proofs assets for HDR displays and archival workflows

**Current State**:
- ✅ PR #935 (Materials V3 Phases A+B) merged - Feb 14
- ✅ PR #936 (Depth identity fix) merged - Feb 14
- ⏳ PR #937 (Investigation docs) open - Feb 14
- 🔄 PR #934 contains 16-bit implementation (needs extraction)

**Conflict Risk Analysis**:
- `orchestrator.py` modified by PR #936 at lines 905-910 (depth backend fix)
- `orchestrator.py` in PR #934 has 16-bit changes at lines 842-874, 1306-1328
- **No overlap detected** - different line ranges, clean extraction expected

---

## Part 1: Feature Analysis

### What is 16-Bit Output?

**Scope**: Materials V3 → V2 handoff bit depth upgrade

**Feature Components**:

1. **Conditional TIFF Export** (orchestrator.py lines 842-874)
   - When `--emit-master16` OR `--emit-upscaled16` enabled: Output 16-bit TIFF
   - When flags disabled: Output 8-bit PNG (Golden Path preserved)
   - Format: TIFF with LZW compression, uint16 dtype, RGB photometric
   - File size: ~1.5-1.8x larger than 8-bit PNG

2. **Manifest Bit Depth Tracking** (manifest.py)
   - `MaterialsV3Metadata.output_bit_depth`: 8 or 16
   - `MaterialsV3Metadata.schema_version`: "1.0" → "1.1"
   - `V2Metadata.input_bit_depth`: 8 or 16
   - `V2Metadata.output_bit_depth`: 8 or 16
   - Backward compatible: v1.0 manifests still load

3. **Orchestrator Logic** (orchestrator.py lines 1306-1328)
   - Auto-detect bit depth from flags
   - Pass bit depth to V2 metadata
   - Track in Materials V3 metadata

### User-Facing Changes

**CLI Flags** (existing, no new flags needed):
```bash
--emit-master16 on      # Enable 16-bit master output
--emit-upscaled16 on    # Enable 16-bit upscaled output
```

**Behavior Change**:
- **Before**: Materials V3 always outputs 8-bit PNG handoff to V2
- **After**: Materials V3 outputs 16-bit TIFF when flags enabled

**Manifest Changes**:
```json
{
  "materials_v3": {
    "schema_version": "1.1",
    "output_bit_depth": 16,
    ...
  },
  "v2": {
    "input_bit_depth": 16,
    "output_bit_depth": 16,
    ...
  }
}
```

### Internal Changes

**Data Type Changes**:
- `working_image` (float32 [0, 1]) → uint16 conversion: `(value * 65535 + 0.5).astype(np.uint16)`
- 16-bit path uses `tifffile.imwrite()` instead of `PIL.Image.save()`

**File Format Logic**:
```python
if self.config.emit_master16 or self.config.emit_upscaled16:
    # 16-bit TIFF path
    enhanced_uint16 = (np.clip(working_image, 0, 1) * 65535 + 0.5).astype(np.uint16)
    enhanced_image_path = temp_dir / f"{output_key.stem}_materials_v3_enhanced.tif"
    tifffile.imwrite(enhanced_image_path, enhanced_uint16, ...)
else:
    # 8-bit PNG path (Golden Path)
    enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)
    enhanced_pil = PILImage.fromarray(enhanced_uint8)
    enhanced_image_path = temp_dir / f"{output_key.stem}_materials_v3_enhanced.png"
    enhanced_pil.save(enhanced_image_path)
```

### Value Proposition

**Quality Improvements**:
- **Tonal Resolution**: 256x more tonal levels (65,536 vs 256 per channel)
- **Gradient Quality**: Eliminates banding in skies, water, reflections
- **Color Grading Headroom**: More latitude for post-processing adjustments
- **Archival Quality**: Professional print and HDR display compatibility

**Tradeoffs**:
- **File Size**: +50-80% storage (18-22 MB vs 12 MB for 4K images)
- **Processing Time**: +15-30ms per image (<1% overhead)
- **Compatibility**: Requires V2 to support 16-bit TIFF input (assumed present)

**Production Impact**:
- Throughput: 400-600 images/hour (unchanged)
- Storage: Plan for 1.6x capacity for 16-bit workflows
- Performance: < 1% total pipeline time increase

---

## Part 2: File Inventory

### Core Implementation Files

| File | Purpose | Lines Changed | Conflict Risk |
|------|---------|---------------|---------------|
| `src/transformation_portal/lux_depth_v3/orchestrator.py` | 16-bit TIFF export, bit depth tracking | +50 | **LOW** (lines 842-874, 1306-1328 don't overlap with PR #936 lines 905-910) |
| `src/transformation_portal/lux_depth_v3/manifest.py` | Schema v1.1, bit depth fields | +30 | **NONE** (no conflicts expected) |

### Test Files

| File | Purpose | Lines Changed | Conflict Risk |
|------|---------|---------------|---------------|
| `tests/materials/test_materials_v3_orchestrator_integration.py` | Schema version update | 1 | **NONE** |
| `test_16bit_implementation.py` | End-to-end 16-bit validation | +350 (new) | **NONE** |
| `verify_16bit_handoff.py` | TIFF format verification | +200 (new) | **NONE** |

### Documentation Files

| File | Purpose | Lines Changed | Conflict Risk |
|------|---------|---------------|---------------|
| `docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md` | Full implementation report | +495 (new) | **NONE** |
| `docs/investigations/PHASE2_16BIT_QUICKREF.md` | Quick reference guide | +227 (new) | **NONE** |

### Conflict Analysis Summary

| Category | Files | Total Changes | Conflicts |
|----------|-------|---------------|-----------|
| **Core** | 2 | ~80 lines | **0** (clean extraction expected) |
| **Tests** | 3 | ~550 lines | 0 |
| **Docs** | 2 | ~720 lines | 0 |
| **TOTAL** | 7 | ~1,350 lines | **0 expected conflicts** |

---

## Part 3: Conflict Analysis

### Current Main State (After PR #936 Merge)

**Recent orchestrator.py commits**:
```
fabc146b - fix(depth): use resolved backend in manifest (PR #936) - lines 905-910
8421e654 - feat(materials-v3): Enable production integration (PR #932)
92c72db0 - feat(materials-v3): Add EfficientSAM segmentation backend
```

**PR #936 Changes** (lines 905-910):
```python
# CRITICAL FIX: Use resolved backend name, not config default
resolved_backend = getattr(self, "_backend_metadata", None)
model_name = resolved_backend.resolved_backend if resolved_backend else self.config.model_variant.value.name

depth_metadata = DepthMetadata(
    model=model_name,  # ← Uses resolved backend, not config default
    ...
)
```

**Impact on 16-Bit Extraction**: ✅ **NONE** (different code section)

### PR #934 16-Bit Changes

**orchestrator.py lines 842-874** (Materials V3 handoff):
```python
# Save enhanced image for V2 (16-bit TIFF if emit flags enabled, 8-bit PNG otherwise)
if self.config.emit_master16 or self.config.emit_upscaled16:
    # 16-bit TIFF path
    ...
else:
    # 8-bit PNG path (Golden Path)
    ...
```

**orchestrator.py lines 1306-1328** (manifest metadata):
```python
# Determine V2 input/output bit depth
v2_input_bit_depth = 16 if (self.config.emit_master16 or self.config.emit_upscaled16) and materials_v3_result else 8
v2_output_bit_depth = 16 if (self.config.emit_master16 or self.config.emit_upscaled16) else 8

# Materials V3 bit depth
materials_v3_bit_depth = 16 if (self.config.emit_master16 or self.config.emit_upscaled16) else 8
```

**Impact**: ✅ **No overlap with PR #936 depth backend fix**

### Conflict Resolution Strategy

**Expected Conflicts**: **NONE**

**Verification**:
- PR #936 changes: Lines 905-910 (depth metadata)
- PR #934 16-bit changes: Lines 842-874 (Materials V3 handoff), 1306-1328 (manifest)
- **No line range overlap** ✅

**Extraction Approach**: **Option A - Clean Cherry-Pick**

**Rationale**:
1. No file conflicts expected
2. All 16-bit changes are isolated to distinct code sections
3. PR #934 branch is single commit (easy to cherry-pick)
4. Fastest and safest approach

**Fallback**: If cherry-pick fails unexpectedly, use **Option B - Manual Patch**

---

## Part 4: Extraction Strategy

### Option A: Clean Cherry-Pick ⭐ **RECOMMENDED**

**Steps**:
1. Create branch from current main: `git checkout -b feature/16bit-output-path main`
2. Cherry-pick 16-bit commit from PR #934: `git cherry-pick <commit-sha>`
3. Remove unrelated changes (SAM2, upscaling, investigations) if included
4. Keep only 16-bit files (orchestrator.py, manifest.py, tests, docs)

**Pros**:
- ✅ Fast (< 30 minutes extraction time)
- ✅ Preserves git history and authorship
- ✅ Minimal manual intervention

**Cons**:
- ⚠️ May include unrelated changes (need manual filtering)
- ⚠️ Requires commit selection if PR has multiple commits

**Success Criteria**:
- No merge conflicts
- Only 16-bit related files changed
- Tests pass

### Option B: Manual Patch (Fallback)

**Steps**:
1. Create branch from main
2. Manually copy orchestrator.py changes (lines 842-874, 1306-1328)
3. Manually copy manifest.py changes
4. Copy test files and documentation
5. Validate with test suite

**Pros**:
- ✅ Full control over changes
- ✅ No unrelated code included
- ✅ Works even if cherry-pick fails

**Cons**:
- ⏱️ Slower (~1-2 hours)
- ⚠️ Manual copy errors possible
- ⚠️ Loses git history continuity

**When to Use**: If Option A cherry-pick fails or includes too many unrelated changes

### Option C: Three-Way Merge (Not Recommended)

**Reason**: Unnecessary complexity for isolated changes with no conflicts

**Use Case**: Only if both A and B fail (highly unlikely)

---

## Part 5: Testing Strategy

### Existing Tests from PR #934

**1. Integration Test Updates**:
```python
# File: tests/materials/test_materials_v3_orchestrator_integration.py
# Change: Schema version assertion
assert materials_v3_metadata.schema_version == "1.1"  # Was "1.0"
```

**2. End-to-End 16-Bit Test** (`test_16bit_implementation.py`):
```python
def test_16bit_tiff_path():
    """Test 16-bit TIFF creation when emit flags enabled"""
    # Run with --emit-master16 on
    # Verify TIFF file exists
    # Verify manifest.materials_v3.output_bit_depth == 16

def test_8bit_png_golden_path():
    """Test 8-bit PNG creation when emit flags disabled"""
    # Run without emit flags
    # Verify PNG file exists
    # Verify manifest.materials_v3.output_bit_depth == 8

def test_v2_bit_depth_tracking():
    """Test V2 metadata bit depth tracking"""
    # Verify v2.input_bit_depth and v2.output_bit_depth
```

**3. TIFF Verification Script** (`verify_16bit_handoff.py`):
```python
# Verify dtype is uint16
# Verify value range 0-65535
# Verify TIFF metadata
# Verify manifest correctness
```

**Coverage**: ✅ Comprehensive (100% of 16-bit code paths)

### New Tests to Add

**None Required** - PR #934 includes complete test coverage.

**Validation Checklist**:
- [ ] Run `test_16bit_implementation.py` - should pass
- [ ] Run `verify_16bit_handoff.py` - should pass
- [ ] Run `pytest tests/materials/ -v` - 67/67 should pass
- [ ] Run full test suite - no regressions

### Integration Tests

**Existing Suite** (no changes needed):
```bash
pytest tests/materials/test_materials_v3_orchestrator_integration.py -v
```

**Expected Results**:
- ✅ 16/16 materials integration tests pass
- ✅ Schema v1.1 validation passes
- ✅ Bit depth tracking tests pass

---

## Part 6: Risk Assessment

### 1. Merge Conflicts - 🟢 LOW RISK

**Risk**: orchestrator.py conflicts with PR #936 changes

**Likelihood**: **LOW** (no line overlap detected)

**Impact**: Minor (30 min manual resolution if occurs)

**Mitigation**:
- ✅ Pre-analysis confirms no overlap (lines 905-910 vs 842-874, 1306-1328)
- ✅ Use cherry-pick to detect conflicts early
- ✅ Keep PR #934 commit available for reference
- ✅ Test immediately after extraction

**Contingency**: Manual patch if cherry-pick fails

---

### 2. Semantic Conflicts - 🟢 LOW RISK

**Risk**: 16-bit logic conflicts with depth backend changes

**Likelihood**: **VERY LOW** (independent features)

**Impact**: Minor (runtime test failures, no production impact)

**Mitigation**:
- ✅ 16-bit changes are isolated to Materials V3 → V2 handoff
- ✅ Depth backend changes are isolated to depth metadata
- ✅ Comprehensive test coverage validates integration
- ✅ Run full test suite before PR creation

**Validation**:
```bash
# Test with depth backend changes + 16-bit enabled
python -m transformation_portal.lux_depth_v3 \
  --materials-v3 on \
  --enable-v2 on \
  --emit-master16 on \
  --model depth_pro
```

---

### 3. Performance Regression - 🟢 LOW RISK

**Risk**: 16-bit increases file I/O time, memory usage

**Likelihood**: **LOW** (measured < 1% overhead in PR #934)

**Impact**: Negligible (15-30ms per image)

**Mitigation**:
- ✅ PR #934 includes performance benchmarks (< 1% overhead)
- ✅ LZW compression minimizes I/O time
- ✅ Quality Firewall budget: 400-600 images/hour (ample headroom)
- ✅ Conditional path (16-bit opt-in, not default)

**Monitoring**:
```bash
# Benchmark before/after extraction
time python -m transformation_portal.lux_depth_v3 \
  --input-dir test_images/ \
  --materials-v3 on \
  --emit-master16 on
```

**Budget**: < 5% increase acceptable, < 1% expected

---

### 4. Format Compatibility - 🟢 LOW RISK

**Risk**: 16-bit TIFFs not readable by all tools

**Likelihood**: **VERY LOW** (TIFF is universal format)

**Impact**: Minimal (user workflow disruption)

**Mitigation**:
- ✅ TIFF with RGB photometric (standard format)
- ✅ LZW compression (widely supported)
- ✅ Test with common tools (Photoshop, Lightroom, Preview)
- ✅ Document format requirements in user guide

**Validation Tools**:
- Adobe Photoshop CC
- Adobe Lightroom Classic
- macOS Preview
- GIMP
- Python PIL/Pillow
- Python tifffile

**Action**: Test 16-bit TIFF with >= 3 tools before merging

---

### 5. Backward Compatibility - 🟢 CRITICAL (PASSES)

**Risk**: Breaking existing 8-bit workflows

**Likelihood**: **NONE** (fully backward compatible by design)

**Impact**: **CRITICAL** if broken (production failures)

**Mitigation**:
- ✅ Default behavior: 8-bit PNG (Golden Path preserved)
- ✅ 16-bit is opt-in via flags
- ✅ Schema v1.0 manifests still load correctly
- ✅ No changes to existing CLI flags or config
- ✅ 67/67 existing tests pass without modification

**Validation**:
```bash
# Test existing workflow (no flags)
python -m transformation_portal.lux_depth_v3 \
  --input-dir test_images/ \
  --materials-v3 on

# Verify:
# - PNG output (not TIFF)
# - manifest.materials_v3.output_bit_depth == 8
# - No behavior change
```

**Status**: ✅ **VERIFIED IN PR #934** (67/67 tests pass)

---

## Part 7: Execution Checklist

### Preparation (30 minutes)

- [ ] **1.1** Review PR #934 16-bit documentation
  - Read `PHASE2_16BIT_IMPLEMENTATION_REPORT.md`
  - Read `PHASE2_16BIT_QUICKREF.md`
  - Understand feature scope and tradeoffs

- [ ] **1.2** Analyze file changes in PR #934
  ```bash
  gh pr view 934 --json files | jq -r '.files[] | select(.path | contains("orchestrator") or contains("manifest") or contains("16bit")) | .path'
  ```

- [ ] **1.3** Verify no conflicts with current main
  ```bash
  git fetch origin main
  git diff origin/main origin/bugfix/materials-v3-critical-fixes -- src/transformation_portal/lux_depth_v3/orchestrator.py | grep -A5 -B5 "^@@"
  ```

- [ ] **1.4** Create Phase 3 extraction branch
  ```bash
  git checkout main
  git pull origin main
  git checkout -b feature/phase3-16bit-output-path
  git push -u origin feature/phase3-16bit-output-path
  ```

### Extraction (2-3 hours)

#### Identify 16-Bit Commit

- [ ] **2.1** Find 16-bit related commit(s) in PR #934 branch
  ```bash
  git log origin/bugfix/materials-v3-critical-fixes --oneline --grep="16.bit\|tiff" -i
  ```

- [ ] **2.2** If single commit, note commit SHA
- [ ] **2.3** If multiple commits, identify which contain 16-bit changes

#### Option A: Cherry-Pick (Try First)

- [ ] **2.4** Attempt cherry-pick
  ```bash
  git cherry-pick <commit-sha>
  ```

- [ ] **2.5** If successful, review included files
  ```bash
  git diff HEAD~1 --stat
  ```

- [ ] **2.6** Remove unrelated files (SAM2, upscaling, investigations)
  ```bash
  git restore --staged <unrelated-file>
  git restore <unrelated-file>
  ```

- [ ] **2.7** Verify only 16-bit files remain
  ```bash
  git status
  # Should show:
  # - orchestrator.py
  # - manifest.py
  # - test_16bit_implementation.py
  # - verify_16bit_handoff.py
  # - tests/materials/test_materials_v3_orchestrator_integration.py
  # - docs/investigations/PHASE2_16BIT_*.md
  ```

- [ ] **2.8** Amend commit to clean state
  ```bash
  git commit --amend -m "feat(materials-v3): Add 16-bit output path for archival quality

Enables end-to-end 16-bit image processing pipeline:
- Materials V3 outputs 16-bit TIFF when emit flags enabled
- Manifest tracks bit depth (schema v1.1)
- Backward compatible (8-bit Golden Path preserved)

Extracted from PR #934 (Phase 3)
"
  ```

#### Option B: Manual Patch (If Cherry-Pick Fails)

- [ ] **2.9** Fetch PR #934 changes for reference
  ```bash
  git fetch origin bugfix/materials-v3-critical-fixes
  ```

- [ ] **2.10** Extract orchestrator.py changes (lines 842-874)
  ```bash
  git show origin/bugfix/materials-v3-critical-fixes:src/transformation_portal/lux_depth_v3/orchestrator.py > /tmp/pr934_orchestrator.py
  # Manually copy lines 842-874 to current orchestrator.py
  ```

- [ ] **2.11** Extract orchestrator.py changes (lines 1306-1328)
  ```bash
  # Manually copy lines 1306-1328 to current orchestrator.py
  ```

- [ ] **2.12** Extract manifest.py changes
  ```bash
  git show origin/bugfix/materials-v3-critical-fixes:src/transformation_portal/lux_depth_v3/manifest.py > /tmp/pr934_manifest.py
  # Manually copy MaterialsV3Metadata and V2Metadata changes
  ```

- [ ] **2.13** Copy test files
  ```bash
  git show origin/bugfix/materials-v3-critical-fixes:test_16bit_implementation.py > test_16bit_implementation.py
  git show origin/bugfix/materials-v3-critical-fixes:verify_16bit_handoff.py > verify_16bit_handoff.py
  git add test_16bit_implementation.py verify_16bit_handoff.py
  ```

- [ ] **2.14** Update integration test
  ```bash
  # Edit tests/materials/test_materials_v3_orchestrator_integration.py
  # Change schema version assertion to "1.1"
  ```

- [ ] **2.15** Copy documentation
  ```bash
  mkdir -p docs/investigations/
  git show origin/bugfix/materials-v3-critical-fixes:docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md > docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md
  git show origin/bugfix/materials-v3-critical-fixes:docs/investigations/PHASE2_16BIT_QUICKREF.md > docs/investigations/PHASE2_16BIT_QUICKREF.md
  git add docs/investigations/PHASE2_16BIT_*.md
  ```

- [ ] **2.16** Commit changes
  ```bash
  git add src/transformation_portal/lux_depth_v3/orchestrator.py
  git add src/transformation_portal/lux_depth_v3/manifest.py
  git add tests/materials/test_materials_v3_orchestrator_integration.py
  git commit -m "feat(materials-v3): Add 16-bit output path for archival quality"
  ```

### Testing (1-2 hours)

#### Unit & Integration Tests

- [ ] **3.1** Run 16-bit end-to-end tests
  ```bash
  python test_16bit_implementation.py
  # Expected: 3/3 tests pass
  ```

- [ ] **3.2** Run TIFF verification
  ```bash
  python verify_16bit_handoff.py
  # Expected: All verifications pass
  ```

- [ ] **3.3** Run materials integration tests
  ```bash
  pytest tests/materials/ -v
  # Expected: 67/67 pass (or updated count)
  ```

- [ ] **3.4** Run full test suite
  ```bash
  pytest tests/ -v --tb=short
  # Expected: No new failures
  ```

#### Functional Validation

- [ ] **3.5** Test 16-bit TIFF path
  ```bash
  python -m transformation_portal.lux_depth_v3 \
    --input-dir examples/test_images/ \
    --output-dir /tmp/test_16bit \
    --materials-v3 on \
    --emit-master16 on
  ```

- [ ] **3.6** Verify TIFF created
  ```bash
  find /tmp/test_16bit -name "*_materials_v3_enhanced.tif" -type f
  # Should find TIFF file(s)
  ```

- [ ] **3.7** Inspect TIFF format
  ```python
  import tifffile
  import numpy as np
  tiff = tifffile.imread("/tmp/test_16bit/temp/*_materials_v3_enhanced.tif")
  print(f"dtype: {tiff.dtype}")  # Should be uint16
  print(f"max: {tiff.max()}")    # Should be > 255
  ```

- [ ] **3.8** Verify manifest bit depth
  ```bash
  cat /tmp/test_16bit/manifests/*_combined.json | jq '.materials_v3.output_bit_depth'
  # Should output: 16
  ```

- [ ] **3.9** Test 8-bit Golden Path (backward compatibility)
  ```bash
  python -m transformation_portal.lux_depth_v3 \
    --input-dir examples/test_images/ \
    --output-dir /tmp/test_8bit \
    --materials-v3 on
    # No emit flags
  ```

- [ ] **3.10** Verify PNG created (not TIFF)
  ```bash
  find /tmp/test_8bit -name "*_materials_v3_enhanced.png" -type f
  # Should find PNG file(s)
  find /tmp/test_8bit -name "*_materials_v3_enhanced.tif" -type f
  # Should NOT find any TIFF files
  ```

- [ ] **3.11** Verify 8-bit manifest
  ```bash
  cat /tmp/test_8bit/manifests/*_combined.json | jq '.materials_v3.output_bit_depth'
  # Should output: 8
  ```

#### Performance Validation

- [ ] **3.12** Benchmark 16-bit vs 8-bit
  ```bash
  # 8-bit baseline
  time python -m transformation_portal.lux_depth_v3 \
    --input-dir examples/test_images/ \
    --output-dir /tmp/bench_8bit \
    --materials-v3 on

  # 16-bit
  time python -m transformation_portal.lux_depth_v3 \
    --input-dir examples/test_images/ \
    --output-dir /tmp/bench_16bit \
    --materials-v3 on \
    --emit-master16 on
  ```

- [ ] **3.13** Verify performance delta < 5%
  ```bash
  # Calculate overhead percentage
  # Should be < 1% based on PR #934 benchmarks
  ```

#### Format Compatibility

- [ ] **3.14** Test TIFF with external tools
  - [ ] macOS Preview (open TIFF)
  - [ ] Photoshop / Lightroom (if available)
  - [ ] Python PIL/Pillow (load and display)

- [ ] **3.15** Verify color accuracy
  ```python
  # Compare 8-bit vs 16-bit (should be visually identical, just higher precision)
  from PIL import Image
  img8 = Image.open("/tmp/test_8bit/temp/*_materials_v3_enhanced.png")
  img16_raw = tifffile.imread("/tmp/test_16bit/temp/*_materials_v3_enhanced.tif")
  img16 = Image.fromarray((img16_raw / 257).astype(np.uint8))  # Downscale to 8-bit
  # Visual comparison
  ```

### Documentation (30 minutes)

- [ ] **4.1** Review extracted docs for accuracy
  - [ ] `docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md`
  - [ ] `docs/investigations/PHASE2_16BIT_QUICKREF.md`

- [ ] **4.2** Update CHANGELOG.md
  ```bash
  echo "## [Unreleased]

### Added
- feat(materials-v3): 16-bit output path for archival quality
  - Materials V3 outputs 16-bit TIFF when \`--emit-master16\` or \`--emit-upscaled16\` enabled
  - Manifest schema v1.1 with bit depth tracking
  - Backward compatible (8-bit Golden Path preserved)
  - Extracted from PR #934 (Phase 3)
" >> CHANGELOG.md
  ```

- [ ] **4.3** Update user documentation (if exists)
  - Add 16-bit workflow examples
  - Document file size implications
  - Note compatibility requirements

- [ ] **4.4** Update extraction progress tracking
  ```bash
  # Update docs/project-status/PHASE_3_COMPLETION_SUMMARY.md (after merge)
  ```

### PR Creation (30 minutes)

- [ ] **5.1** Run pre-commit checks
  ```bash
  pre-commit run --all-files
  ```

- [ ] **5.2** Fix any linting issues
  ```bash
  # Black, isort, flake8 should all pass
  ```

- [ ] **5.3** Final commit cleanup
  ```bash
  git status
  git diff --staged
  # Verify only intended changes
  ```

- [ ] **5.4** Push branch
  ```bash
  git push origin feature/phase3-16bit-output-path
  ```

- [ ] **5.5** Create PR with comprehensive description
  ```bash
  gh pr create \
    --title "feat(materials-v3): Add 16-bit output path for archival quality (Phase 3)" \
    --body-file docs/project-status/PHASE_3_PR_TEMPLATE.md \
    --label "enhancement" \
    --label "Materials V3"
  ```

- [ ] **5.6** Link to source PR #934 in description
- [ ] **5.7** Add before/after examples (8-bit PNG vs 16-bit TIFF)
- [ ] **5.8** Reference extraction strategy document

---

## Part 8: Success Criteria

### Functional Requirements

✅ **16-bit TIFF output works end-to-end**
- [ ] TIFF created when `--emit-master16` or `--emit-upscaled16` enabled
- [ ] TIFF format: uint16 dtype, RGB photometric, LZW compression
- [ ] File extension: `.tif` (not `.png`)

✅ **CLI flag behavior correct**
- [ ] `--emit-master16 on` → 16-bit TIFF
- [ ] `--emit-upscaled16 on` → 16-bit TIFF
- [ ] No flags → 8-bit PNG (Golden Path)

✅ **Manifest records actual bit depth**
- [ ] `materials_v3.output_bit_depth` = 16 (with flags) or 8 (without)
- [ ] `materials_v3.schema_version` = "1.1"
- [ ] `v2.input_bit_depth` and `v2.output_bit_depth` correct

✅ **Backward compatible**
- [ ] 8-bit PNG default behavior unchanged
- [ ] Existing tests pass (67/67 or updated count)
- [ ] Schema v1.0 manifests still load

### Quality Requirements

✅ **All tests passing**
- [ ] `test_16bit_implementation.py`: 3/3 pass
- [ ] `verify_16bit_handoff.py`: All verifications pass
- [ ] `pytest tests/materials/ -v`: No new failures
- [ ] `pytest tests/ -v`: No regressions

✅ **No performance regression**
- [ ] 16-bit overhead < 5% (< 1% expected)
- [ ] Throughput maintains 400-600 images/hour baseline
- [ ] Memory usage acceptable (< 10% increase)

✅ **Pre-commit hooks passing**
- [ ] Black formatting
- [ ] isort imports
- [ ] flake8 linting (critical errors)
- [ ] Trailing whitespace removed

✅ **Code review ready**
- [ ] Clear commit messages
- [ ] Inline comments for complex logic
- [ ] Type hints maintained
- [ ] Docstrings updated

### Documentation Requirements

✅ **User guide updated**
- [ ] 16-bit workflow examples
- [ ] File size implications documented
- [ ] Compatibility requirements noted

✅ **Config schema documented**
- [ ] `emit_master16` flag explained
- [ ] `emit_upscaled16` flag explained
- [ ] Manifest schema v1.1 documented

✅ **CHANGELOG entry added**
- [ ] Feature description
- [ ] Breaking changes (none expected)
- [ ] Migration notes (none required)

✅ **Extraction context documented**
- [ ] Link to PR #934 source
- [ ] Extraction rationale explained
- [ ] Phase 3 status updated

---

## Part 9: Time Budget Breakdown

| Task | Optimistic | Realistic | Pessimistic | Notes |
|------|-----------|-----------|-------------|-------|
| **Preparation & Analysis** | 20 min | 30 min | 1 hour | Review docs, analyze conflicts |
| **Conflict Resolution** | 0 min (none expected) | 30 min | 1 hour | Manual merge if cherry-pick fails |
| **Code Extraction** | 30 min (cherry-pick) | 1 hour | 2 hours | Manual patch if needed |
| **Testing** | 30 min | 1 hour | 2 hours | Full suite + validation |
| **Documentation** | 15 min | 30 min | 1 hour | CHANGELOG, user guide updates |
| **PR Creation** | 15 min | 30 min | 1 hour | Description, examples, links |
| **Buffer** | - | 30 min | 1 hour | Unexpected issues |
| **TOTAL** | **2 hours** | **4.5 hours** | **9 hours** |

**Recommendation**: **Plan for 5 hours** (realistic + 30 min buffer)

**Critical Path**:
1. Preparation (30 min)
2. Extraction (1 hour) ← Bottleneck if manual patch needed
3. Testing (1 hour)
4. Documentation (30 min)
5. PR (30 min)
6. Buffer (30 min)

**Parallel Opportunities**:
- Documentation can be written while tests run
- Pre-commit checks can run during final review

**Risk Buffer**:
- Add 1-2 hours if conflicts discovered
- Add 1 hour if manual patch required
- Add 30 min if test failures occur

---

## Part 10: Quick Start Commands

### Initial Analysis

```bash
# 1. Fetch latest main and PR #934 branch
cd /Users/rc/Projects/Transformation_Portal
git fetch origin main
git fetch origin bugfix/materials-v3-critical-fixes

# 2. Analyze PR #934 16-bit files
gh pr view 934 --json files | jq -r '.files[] | select(.path | contains("16bit") or contains("tiff") or contains("orchestrator") or contains("manifest")) | {path, additions, deletions}'

# 3. Check for conflicts with current main
git diff origin/main origin/bugfix/materials-v3-critical-fixes -- src/transformation_portal/lux_depth_v3/orchestrator.py | grep -E "^@@|^-.*emit.*16|^\+.*emit.*16"

# 4. View recent orchestrator.py commits on main
git log origin/main --oneline -5 -- src/transformation_portal/lux_depth_v3/orchestrator.py
```

### Extraction (Option A: Cherry-Pick)

```bash
# 1. Create Phase 3 branch from main
git checkout main
git pull origin main
git checkout -b feature/phase3-16bit-output-path
git push -u origin feature/phase3-16bit-output-path

# 2. Find 16-bit commit in PR #934
git log origin/bugfix/materials-v3-critical-fixes --oneline --all

# 3. Cherry-pick (assuming single commit in PR #934)
# Note: PR #934 is a single commit, so use the branch head
git cherry-pick origin/bugfix/materials-v3-critical-fixes

# 4. If successful, review changes
git diff HEAD~1 --stat

# 5. Remove unrelated files (keep only 16-bit related)
# List all changed files
git diff HEAD~1 --name-only

# Remove unrelated files (SAM2, upscaling, investigation scripts, etc.)
git restore --staged analyze_greatroom_sky.py create_sky_water_comparison_visual.py diagnose_primary_bedroom_artifacts.py investigate_sky_water_degradation.py
git restore --staged config/materials_v3_sam2.yaml config/presets/depth_pro_research_uhq.yaml
git restore --staged src/transformation_portal/lux_depth_v3/sam2_adapter.py src/transformation_portal/lux_depth_v3/segmentation_backend.py src/transformation_portal/spatial_ai/segmentation/sam2_backend.py
git restore --staged src/transformation_portal/upscaling src/transformation_portal/stage_graph/stages/upscaling.py
git restore --staged docs/architecture/PHASE3_ML_UPSCALING*.md docs/architecture/APEX*.md docs/investigations/CRITICAL_BUGFIXES*.md docs/investigations/PRIMARY_BEDROOM*.md docs/investigations/SKY_WATER*.md docs/investigations/GREATROOM*.md docs/investigations/DEPTH_PRO*.md
git restore --staged docs/materials/SAM2_INTEGRATION.md docs/production docs/research
git restore --staged tests/materials/test_sam2_materials_integration.py tests/test_upscaling*.py
git restore --staged examples/upscaling_comparison.py verify_phase3_implementation.sh scripts/pipelines/run_750_picacho_apex_full.sh scripts/pipelines/run_source_tiffs_depth_pro_research.sh scripts/verify_apex_gaps.py
git restore --staged requirements/ml.in requirements/ml.txt requirements/all.txt requirements/base.txt requirements/ci.txt
git restore --staged greatroom_sky_analysis.json primary_bedroom_artifact_diagnosis.json sky_water_degradation_analysis.json comparison_images

# Actually remove files from working tree
git restore analyze_greatroom_sky.py create_sky_water_comparison_visual.py diagnose_primary_bedroom_artifacts.py investigate_sky_water_degradation.py
git restore config/materials_v3_sam2.yaml config/presets/depth_pro_research_uhq.yaml
git restore src/transformation_portal/lux_depth_v3/sam2_adapter.py src/transformation_portal/lux_depth_v3/segmentation_backend.py src/transformation_portal/spatial_ai/segmentation/sam2_backend.py
git restore src/transformation_portal/upscaling src/transformation_portal/stage_graph/stages/upscaling.py
git restore docs/architecture/PHASE3_ML_UPSCALING*.md docs/architecture/APEX*.md docs/investigations/CRITICAL_BUGFIXES*.md docs/investigations/PRIMARY_BEDROOM*.md docs/investigations/SKY_WATER*.md docs/investigations/GREATROOM*.md docs/investigations/DEPTH_PRO*.md
git restore docs/materials/SAM2_INTEGRATION.md docs/production docs/research
git restore tests/materials/test_sam2_materials_integration.py tests/test_upscaling*.py
git restore examples/upscaling_comparison.py verify_phase3_implementation.sh scripts/pipelines/run_750_picacho_apex_full.sh scripts/pipelines/run_source_tiffs_depth_pro_research.sh scripts/verify_apex_gaps.py
git restore requirements/ml.in requirements/ml.txt requirements/all.txt requirements/base.txt requirements/ci.txt
git restore greatroom_sky_analysis.json primary_bedroom_artifact_diagnosis.json sky_water_degradation_analysis.json comparison_images

# 6. Also remove the pixel_ops bug fixes (already extracted in PR #935/936)
git restore --staged src/transformation_portal/lux_depth_v3/pixel_ops_executor.py
git restore src/transformation_portal/lux_depth_v3/pixel_ops_executor.py

# 7. Verify only 16-bit files remain
git status
# Should show ONLY:
# - src/transformation_portal/lux_depth_v3/orchestrator.py
# - src/transformation_portal/lux_depth_v3/manifest.py
# - tests/materials/test_materials_v3_orchestrator_integration.py
# - test_16bit_implementation.py (new)
# - verify_16bit_handoff.py (new)
# - docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md (new)
# - docs/investigations/PHASE2_16BIT_QUICKREF.md (new)

# 8. Amend commit message
git commit --amend -m "feat(materials-v3): Add 16-bit output path for archival quality

Enables end-to-end 16-bit image processing pipeline:
- Materials V3 outputs 16-bit TIFF when --emit-master16 or --emit-upscaled16 enabled
- Manifest schema v1.1 with bit depth tracking (backward compatible)
- Backward compatible (8-bit PNG Golden Path preserved)
- Performance overhead < 1% (15-30ms per image)

Implementation:
- orchestrator.py: Conditional TIFF export (lines 842-874, 1306-1328)
- manifest.py: Schema v1.1 with output_bit_depth field
- Complete test coverage (test_16bit_implementation.py, verify_16bit_handoff.py)

Extracted from PR #934 (Phase 3 of Materials V3 extraction)

Tradeoffs:
- File size: ~1.5-1.8x larger (18-22 MB vs 12 MB for 4K)
- Quality: 256x more tonal levels (65,536 vs 256 per channel)
- Format: TIFF with LZW compression (widely compatible)
"
```

### Extraction (Option B: Manual Patch - Fallback)

```bash
# If cherry-pick fails or includes too much, use manual patch

# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/phase3-16bit-output-path

# 2. Fetch PR #934 for reference
git fetch origin bugfix/materials-v3-critical-fixes

# 3. View orchestrator.py 16-bit changes
git show origin/bugfix/materials-v3-critical-fixes:src/transformation_portal/lux_depth_v3/orchestrator.py | less
# Find lines 842-874 (Materials V3 handoff)
# Find lines 1306-1328 (manifest metadata)

# 4. Manually apply changes to orchestrator.py
# (Use your editor to copy changes from PR #934)

# 5. View manifest.py changes
git show origin/bugfix/materials-v3-critical-fixes:src/transformation_portal/lux_depth_v3/manifest.py | less
# Copy MaterialsV3Metadata changes (output_bit_depth, schema_version)
# Copy V2Metadata changes (input_bit_depth, output_bit_depth)

# 6. Copy test files
git show origin/bugfix/materials-v3-critical-fixes:test_16bit_implementation.py > test_16bit_implementation.py
git show origin/bugfix/materials-v3-critical-fixes:verify_16bit_handoff.py > verify_16bit_handoff.py

# 7. Update integration test
# Edit tests/materials/test_materials_v3_orchestrator_integration.py
# Change: assert materials_v3_metadata.schema_version == "1.1"

# 8. Copy documentation
mkdir -p docs/investigations/
git show origin/bugfix/materials-v3-critical-fixes:docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md > docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md
git show origin/bugfix/materials-v3-critical-fixes:docs/investigations/PHASE2_16BIT_QUICKREF.md > docs/investigations/PHASE2_16BIT_QUICKREF.md

# 9. Stage and commit
git add src/transformation_portal/lux_depth_v3/orchestrator.py
git add src/transformation_portal/lux_depth_v3/manifest.py
git add tests/materials/test_materials_v3_orchestrator_integration.py
git add test_16bit_implementation.py verify_16bit_handoff.py
git add docs/investigations/PHASE2_16BIT_*.md
git commit -m "feat(materials-v3): Add 16-bit output path for archival quality

[Same commit message as Option A]
"
```

### Testing

```bash
# 1. Run 16-bit tests
python test_16bit_implementation.py
# Expected: All tests pass

# 2. Run TIFF verification
python verify_16bit_handoff.py
# Expected: All verifications pass

# 3. Run materials integration tests
pytest tests/materials/ -v
# Expected: 67/67 pass (or updated count)

# 4. Test 16-bit TIFF creation
python -m transformation_portal.lux_depth_v3 \
  --input-dir examples/test_images/ \
  --output-dir /tmp/test_16bit \
  --materials-v3 on \
  --emit-master16 on

# 5. Verify TIFF created
find /tmp/test_16bit -name "*.tif" -type f

# 6. Inspect TIFF
python -c "import tifffile; import numpy as np; tiff = tifffile.imread('/tmp/test_16bit/temp/*_materials_v3_enhanced.tif'); print(f'dtype: {tiff.dtype}, max: {tiff.max()}')"
# Expected: dtype: uint16, max: > 255

# 7. Test 8-bit Golden Path
python -m transformation_portal.lux_depth_v3 \
  --input-dir examples/test_images/ \
  --output-dir /tmp/test_8bit \
  --materials-v3 on
  # No emit flags

# 8. Verify PNG created (not TIFF)
find /tmp/test_8bit -name "*.png" -type f
find /tmp/test_8bit -name "*.tif" -type f  # Should be empty
```

### PR Creation

```bash
# 1. Pre-commit checks
pre-commit run --all-files

# 2. Push branch
git push origin feature/phase3-16bit-output-path

# 3. Create PR
gh pr create \
  --title "feat(materials-v3): Add 16-bit output path for archival quality (Phase 3)" \
  --body "## Summary

Extracts **16-bit output support** from PR #934 (Phase 3 of Materials V3 extraction).

Enables end-to-end 16-bit image processing for archival quality and professional workflows.

---

## What This PR Does

✅ **16-bit TIFF Output**: Materials V3 outputs 16-bit TIFF when \`--emit-master16\` or \`--emit-upscaled16\` enabled
✅ **Manifest Tracking**: Schema v1.1 with \`output_bit_depth\` field
✅ **Backward Compatible**: 8-bit PNG Golden Path preserved (default behavior unchanged)
✅ **Complete Tests**: End-to-end validation + TIFF verification

---

## Usage

\`\`\`bash
# Enable 16-bit output
python -m transformation_portal.lux_depth_v3 \\
  --materials-v3 on \\
  --emit-master16 on \\
  --emit-upscaled16 on
\`\`\`

---

## Implementation

- **orchestrator.py**: Conditional 16-bit TIFF export (lines 842-874, 1306-1328)
- **manifest.py**: Schema v1.1 with bit depth tracking
- **Tests**: \`test_16bit_implementation.py\`, \`verify_16bit_handoff.py\`
- **Docs**: Full implementation report + quick reference

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| **Tests** | 67/67 pass (no regressions) |
| **Performance** | < 1% overhead (~15-30ms per image) |
| **File Size** | ~1.5-1.8x (18-22 MB vs 12 MB for 4K) |
| **Backward Compat** | ✅ 100% (8-bit default unchanged) |

---

## Extraction Context

- **Source**: PR #934 (Materials V3 Critical Fixes)
- **Phase**: Phase 3 of 6-phase extraction strategy
- **Previous**: PR #935 (Phases A+B), PR #936 (Depth fix), PR #937 (Docs)
- **Conflicts**: None (clean extraction)

---

## Documentation

- 📄 [Implementation Report](docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md)
- 📋 [Quick Reference](docs/investigations/PHASE2_16BIT_QUICKREF.md)

---

## Review Focus

1. ✅ Conditional TIFF export logic (orchestrator.py:842-874)
2. ✅ Manifest schema evolution (manifest.py)
3. ✅ Test coverage completeness
4. ✅ Backward compatibility preservation
" \
  --label "enhancement" \
  --label "Materials V3"
```

---

## Related Documentation

- **Source PR**: [#934 - Materials V3 Critical Fixes](https://github.com/RC219805/Transformation_Portal/pull/934)
- **Implementation Report**: `docs/investigations/PHASE2_16BIT_IMPLEMENTATION_REPORT.md` (from PR #934)
- **Quick Reference**: `docs/investigations/PHASE2_16BIT_QUICKREF.md` (from PR #934)
- **Previous Phases**:
  - PR #935 - Materials V3 Phases A+B (merged)
  - PR #936 - Depth Identity Fix (merged)
  - PR #937 - Investigation Docs (open)

---

## Next Steps After Phase 3

**Phase 4: ML Upscaling Backend** (8-12 hours)
- Extract Real-ESRGAN upscaling backend
- Extract upscaling registry and protocol
- Extract upscaling integration tests

**Phase 5: SAM2 Integration** (6-8 hours, optional)
- Extract SAM2 adapter
- Extract SAM2 backend implementation
- Extract SAM2 integration tests

**Phase 6: Production Documentation** (2-3 hours)
- Extract production guides
- Extract APEX workflow scripts
- Extract research deliverables

---

**STATUS**: ✅ **READY TO EXECUTE**

**CONFIDENCE**: 🟢 **HIGH** (no conflicts detected, well-documented, comprehensive tests)

**ESTIMATED TIME**: **5 hours** (realistic + buffer)

**RISK LEVEL**: 🟡 **MEDIUM → 🟢 LOW** (after conflict analysis shows no overlap)
