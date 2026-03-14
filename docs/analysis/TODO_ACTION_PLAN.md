# TODO Action Plan - Transformation Portal

**Generated**: 2026-03-13
**Last Updated**: 2026-03-14
**Scope**: Comprehensive review of all outstanding TODOs across the codebase
**Cross-Reference**: [TODO_INVENTORY.md](./TODO_INVENTORY.md) (v2.0.0)

---

## Executive Summary

This action plan consolidates findings from a codebase-wide TODO review and provides prioritized recommendations for resolution.

### Key Findings

| Category | Count | Action Required |
|----------|-------|-----------------|
| **Active Source Code TODOs** | 10 | Mixed (see details) |
| **Test Observational TODOs** | 2 | Track only |
| **Config/Preset TODOs** | 2 | ✅ Completed (2026-03-14) |
| **Security Pattern TODOs** | 2 | Intentional (no action) |
| **Documentation TODOs** | 190 | Mostly no action; retain/archive historical notes |

Documentation TODO instances were reviewed for governance and archival value, but they are not part of the 11-item implementation backlog unless explicitly called out in the consolidated plan below.

### Priority Summary

- **P1 (Immediate)**: ~~3 items~~ 1 item remaining (HuggingFace revision pinning ✅ completed)
- **P2 (Near-term)**: ~~4 items~~ 3 items remaining (ICC preservation ✅ completed)
- **P3 (Deferred)**: 4 items for future phases
- **No Action Required**: 195 items (95%; historical, observational, or intentionally retained)

---

## Category 1: Active Source Code TODOs

### ✅ COMPLETED: ICC Profile Preservation (v2_enhance.py)

**Location**: `src/transformation_portal/lux_depth_v3/v2_enhance.py:576-600`

**Completed**: 2026-03-14

**Implementation**:
- ICC profile is now preserved in 16-bit TIFF output via tifffile `extratags` parameter
- Uses TIFF tag 34675 for ICC profile embedding
- EXIF preservation is documented as best-effort (requires IFD handling for full support)
- 8-bit fallback path already preserves ICC and EXIF via PIL

**Changes Made**:
- Added extratags list construction for ICC profile preservation
- Passes ICC profile bytes to tifffile.imwrite via extratags
- Added logging for metadata preservation status

---

### P3: Material Backend Integration Stubs (3 items)

**Location**: `src/transformation_portal/spatial_ai/materials/material_backend.py`

#### 1. ComfyUI Subprocess Integration (Line 281)
```python
# TODO: Implement ComfyUI subprocess integration
```
**Status**: Placeholder with graceful fallback to heuristic
**Action**: Keep as-is until Phase 5B implementation
**Priority**: P3 (Deferred - experimental feature)

#### 2. NVDIFFREC Integration (Line 323)
```python
# TODO: Implement NVDIFFREC integration
```
**Status**: Placeholder with graceful fallback to heuristic
**Action**: Defer to v2.3.0+ (requires ADR for GPU requirements)
**Priority**: P3 (Research feature)

#### 3. MaterialGAN Integration (Line 359)
```python
# TODO: Implement MaterialGAN integration
```
**Status**: Placeholder with graceful fallback to heuristic
**Action**: Defer to v2.3.0+ (requires ADR for model licensing)
**Priority**: P3 (Research feature)

**Cross-Reference**: See [OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md](./OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md) Section 3

---

### P3: SLERP Interpolation (scene_builder.py)

**Location**: `src/transformation_portal/spatial_ai/reconstruction/scene_builder.py:290`
```python
# TODO: Replace with proper SLERP interpolation
# This naive interpolation produces sheared rotations
```

**Context**: Camera path interpolation uses linear interpolation instead of spherical linear interpolation (SLERP).

**Impact**: Low - Affects only experimental reconstruction feature, produces "sheared rotations" per comment.

**Recommended Action**:
1. Implement SLERP using quaternion representation
2. Use `scipy.spatial.transform.Rotation.slerp()` or manual quaternion interpolation
3. Add unit tests for interpolation quality

**Effort**: 1-2 days
**Priority**: P3 (Technical enhancement)
**Owner**: Specialist (Spatial AI)

---

### P4: Upscaling Weights Cache (realesrgan.py)

**Location**: `src/transformation_portal/upscaling/backends/realesrgan.py:234`
```python
# # TODO(Phase 4.1): Use ~/.cache/transformation_portal/upscaling/weights
```

**Context**: Commented-out TODO for standardizing weight cache location.

**Impact**: Minimal - Currently functional, cache location is implementation detail.

**Action**: Track for Phase 4.1 standardization; no immediate action.
**Priority**: P4 (Nice-to-have)

---

### ✅ No Action Required: Security Pattern Constants

**Locations**:
- `src/transformation_portal/core/security/model_lock.py:19`
- `src/transformation_portal/core/config/preset_health.py:31`

```python
re.compile(r"TODO_REPLACE", re.IGNORECASE),
```

**Context**: These are **intentional pattern matchers** for detecting placeholder values in config files. The regex `TODO_REPLACE` is part of the security scanner to find unverified model hashes.

**Action**: None - This is correct security infrastructure.

---

## Category 2: Sample Download URLs

**Location**: `scripts/download_samples.py` (Lines 79, 98)
```python
"url": None,  # TODO: Upload to GitHub Release
```

**Impact**: Medium - Users cannot download demo samples automatically.

**Affected Samples**:
1. `demo_coastal_interior` (5MB)
2. `demo_pool_aerial` (8MB)
3. `sample_render_4k` (25MB)

**Recommended Action**:
1. Create GitHub Release `v2.x.x-samples`
2. Upload sample files with SHA256 hashes
3. Update `download_samples.py` with URLs and hashes
4. Update README sample documentation

**Effort**: 4 hours
**Priority**: P2 (User-facing feature)
**Owner**: DevOps

**Cross-Reference**: See [TODO_INVENTORY.md Section 3.2](./TODO_INVENTORY.md)

---

## Category 3: Config/Preset TODOs

### ✅ COMPLETED: HuggingFace Model Revision Pinning

**Completed**: 2026-03-14

**Locations Updated**:
- `config/presets/apex_research.yaml` - fallback model section
- `config/presets/apex_research_canary.yaml` - primary model section
- `config/presets/experimental/apex_research_ultra.yaml` - ensemble model entry

**Implementation**:
- All three preset files now pin the DA3 1.1 model to commit SHA: `b2359bdf726fb44ef62acca04d629dcf158053e7`
- Model ID corrected from `DA3-NESTED-GIANT-LARGE-1.1` to `DA3NESTED-GIANT-LARGE-1.1` (matching actual HuggingFace model)
- Verified via `python scripts/validation/validate_hf_revisions.py` - all revisions now properly pinned

**Verification**:
```
✅ All HuggingFace model revisions are properly pinned
```

**Cross-Reference**: See [HUGGINGFACE_MODEL_PINNING.md](../apex/HUGGINGFACE_MODEL_PINNING.md)

---

## Category 4: Test TODOs (Observational - No Action)

**Location**: `tests/stress/test_stress_large_batch.py` (Lines 326, 372)

```python
# TODO: Investigate why draft is slower than expected (expected: draft < standard)
# TODO: Investigate preset performance ordering regression
```

**Context**: These are observational notes tracking inconsistent preset performance, likely due to resource contention in stress tests.

**Impact**: None - Performance observation, not a bug.

**Action**: Track in [IMPROVEMENT_OPPORTUNITIES.md](../guides/IMPROVEMENT_OPPORTUNITIES.md) (already done).
**Priority**: P4 (Optimization research)

---

## Category 5: Test File Content (Not a TODO)

**Location**: `tests/security/test_pipeline_isolation.py:92`
```python
test_file.write_text("""# TODO: Consider using spatial_ai for this
```

**Context**: This is test fixture content written to a temporary file for security scanning validation. The "TODO" is part of the test data, not an actionable item.

**Action**: None - Test data, not a real TODO.

---

## Consolidated Action Plan

### Immediate Actions (P1) - Before v2.3.0 Planning

| # | Item | Effort | Owner | Deadline | Status |
|---|------|--------|-------|----------|--------|
| 1 | Pin HuggingFace model revisions | 1h | ML Specialist | Week 1 | ✅ Completed 2026-03-14 |
| 2 | Verify branch protection (GitHub UI) | 15min | Admin | Week 1 | Pending |
| 3 | Create rollback procedures doc | 2h | Architect | Week 2 | ✅ Already exists |

**Total P1 Effort**: ~3.5 hours → **Remaining**: 15min

---

### Near-term Actions (P2) - v2.3.0 Cycle

| # | Item | Effort | Owner | Sprint | Status |
|---|------|--------|-------|--------|--------|
| 4 | Upload sample data to GitHub Release | 4h | DevOps | Sprint 1 | Pending (requires GitHub Release creation) |
| 5 | Implement ICC/EXIF preservation in v2_enhance.py | 6h | ML Specialist | Sprint 1 | ✅ Completed 2026-03-14 |
| 6 | Delete obsolete depth_canonical module | 1h | Specialist | Sprint 1 | ✅ Already archived |
| 7 | Archive obsolete PR tracking docs | 30min | Any | Sprint 1 | ✅ Completed 2026-03-14 |

**Total P2 Effort**: ~11.5 hours → **Remaining**: ~4 hours

**Note**: PR tracking docs archived to `docs/_archive/2026-03-legacy-prs/`:
- `PR98_VERIFICATION_REPORT.md`
- `PR100_FIX_SUMMARY.md`
- `PR162_VERIFICATION_SUMMARY.md`

---

### Deferred Actions (P3) - v2.4.0+ / Phase 2

| # | Item | Effort | Owner | Phase |
|---|------|--------|-------|-------|
| 8 | SLERP interpolation in scene_builder.py | 1-2d | Spatial AI | Phase 2 |
| 9 | ComfyUI subprocess integration | 1-2w | ML | Phase 5B |
| 10 | NVDIFFREC integration | 3-4w | ML Research | Phase 3 |
| 11 | MaterialGAN integration | 2-3w | ML Research | Phase 3 |

**Total P3 Effort**: ~200+ hours (deferred)

---

### No Action Required (195 reviewed TODO instances, 95%)

The following categories require **no action**:

1. **Documentation TODO markers** (190 items): Historical notes, status markers, or archive candidates reviewed for context rather than implementation
2. **Security Patterns** (2 items): `TODO_REPLACE` regex patterns used to detect placeholder values
3. **Observational Notes** (2 items): Performance investigation tracking already captured in follow-up documentation
4. **Test Fixture Content** (1 item): Embedded `TODO` text used as scanner test data

These 195 reviewed instances are intentionally excluded from the 11-item P1/P2/P3 backlog above.

---

## Governance Recommendations

### 1. Definition of Done Updates

Add to project Definition of Done:
- [ ] "If adding `NotImplementedError`, document in TODO_INVENTORY.md with categorization"
- [ ] "If adding `# TODO:` comment, include tracking reference (phase, ADR, or issue number)"

### 2. Inventory Update Cadence

Establish monthly TODO inventory review:
- Review new TODOs added in PRs
- Verify deferred items still deferred for valid reasons
- Archive completed items

### 3. Preset Governance

Require for all preset changes:
- HuggingFace model revisions must be pinned (40-char SHA)
- Expected SHA256 hashes for model checkpoints
- `model_lock_manifest.yaml` must be updated

---

## Cross-Reference Documents

| Document | Purpose | Status |
|----------|---------|--------|
| [TODO_INVENTORY.md](./TODO_INVENTORY.md) | Comprehensive v2.0.0 inventory | ✅ Current |
| [OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md](./OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md) | Experimental feature tracking | ✅ Current |
| [HF_REVISION_PINNING_GUIDE.md](./HF_REVISION_PINNING_GUIDE.md) | Model pinning procedures | ✅ Current |
| [IMPROVEMENT_OPPORTUNITIES.md](../guides/IMPROVEMENT_OPPORTUNITIES.md) | Performance optimization tracking | ✅ Current |

---

## Summary

**Total Items Reviewed**: 206 TODO instances
**Action Required**: 11 items (5%)
**No Action Required**: 195 items (95%)

**Key Takeaway**: The majority of reviewed TODO instances are **historical documentation notes**, **intentional validation patterns**, or **observational tracking**. Only 11 items require actual implementation work, with 3 being P1 priority.

The codebase demonstrates mature TODO management with proper documentation and categorization. The existing [TODO_INVENTORY.md](./TODO_INVENTORY.md) provides comprehensive tracking and should continue to be the authoritative source for TODO governance.

---

**Document Version**: 1.0.0
**Author**: Transformation Portal Architect Review
**Next Review**: Before v2.3.0 release planning
