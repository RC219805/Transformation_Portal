# TODO Action Plan - Transformation Portal

**Generated**: 2026-03-13
**Last Updated**: 2026-05-11
**Scope**: Comprehensive review of all outstanding TODOs across the codebase
**Cross-Reference**: [TODO_INVENTORY.md](./TODO_INVENTORY.md) (v2.4.7)

---

## Executive Summary

This action plan consolidates findings from a codebase-wide TODO review and provides prioritized recommendations for resolution.

### Key Findings

| Category | Count | Action Required |
|----------|-------|-----------------|
| **Active Source Code TODOs** | 0 | ✅ All cleaned up |
| **Scanner-visible Test TODO comments** | 0 | ✅ No scanner-enforced test TODO comments |
| **Governed NotImplementedError items** | 24 | No governance action; all governed |
| **Config/Preset TODOs** | 0 | ✅ Completed (2026-03-14) |
| **Security Pattern TODOs** | 2 | Intentional (no action) |
| **Documentation TODOs** | Narrative only | Non-actionable historical markers (see Note below) |

All source code TODOs in `src/` have been cleaned up. The live scanner reports
24 governed `NotImplementedError` items, 0 ungoverned TODOs, and 1,570 files
scanned.

> **Note on Documentation TODO counts:** Markdown TODO prose is intentionally outside the CI scanner scope. Historical notes, status markers, and archive candidates remain narrative context unless promoted into the scanner-enforced code/test/script/tool/web surfaces. The experimental-feature-focused markers tracked in `OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md` remain documentation annotations, not implementation blockers.

### Recent Governance Completion (2026-05-11)

- `scripts/validation/scan_todo_inventory.py` now supports `--write-snapshot`,
  which writes the canonical JSON payload to
  `docs/analysis/todo_scanner_snapshot.json` without ad hoc shell redirection.
- Live scanner baseline: 24 `NotImplementedError` items, 0 ungoverned TODOs,
  1,570 files scanned.
- Baseline delta: 25 → 24 because `scripts/pipelines/tiff_enhancement_pipeline.py`
  is no longer present in the tracked tree. No new backlog item was opened.
- `docs/governance/todo_priority_schema.yaml` now records the current scan
  baseline, scan scope, generated-output exclusions, inventory-reference
  governance pattern, and snapshot refresh command.

### Recent Governance Completion (2026-05-01)

- TODO governance scanner is now enforced in CI. `scripts/validation/scan_todo_inventory.py --check-governance` runs in the `hf-revision-policy` job of `.github/workflows/enforcement.yml`. PRs introducing ungoverned action markers now fail. Enforced patterns: Python `# TODO:` / `# FIXME:` / `# HACK:` / `# XXX:` comments, `raise NotImplementedError(...)` AST nodes, and JS/TS `// TODO|FIXME|HACK|XXX` and block-comment markers. Enforced scopes: `src/`, `tests/`, `scripts/`, `tools/`, and `web/` (JS/TS). Required reference: `(Phase X)`, `(ADR-###)`, `(#1234)`, `(@owner)`, or `(TODO_INVENTORY.md ...)`. Bare `raise NotImplementedError` and abstract-method messages (e.g., "Subclass must implement") are auto-governed.
- NotImplementedError baseline corrected from 12 (src-only, with arithmetic errors) to the full scanner scope across `src/`, `tests/`, `scripts/`, `tools/`, and `web/`. The current 2026-05-11 baseline is 24, and all instances remain properly governed.
- Manual monthly inventory review is no longer the primary drift control; it now serves as a periodic narrative refresh on top of automated enforcement.

### Recent Governance Completion (2026-03-26)

- Exact-pinned web stack compatibility work is no longer pending.
- Dependabot PR `#1275` was closed and replaced by curated issue `#1277` and PR `#1278`.
- The merged validated set is FastAPI `0.135.1`, Starlette `1.0.0`, Uvicorn `0.42.0`.
- Invalid cross-platform ML lock regeneration was explicitly removed from that change set.
- Separate follow-up is now tracked in issue `#1421` ("Make target-owned ML lock generation fail closed by authoritative lane").

### Priority Summary

- **P1 (Immediate)**: ~~3 items~~ 1 item remaining (HuggingFace revision pinning ✅, Rollback docs ✅ verified existing; Branch protection verification: 15 min pending)
- **P2 (Near-term)**: ~~4 items~~ 3 items remaining (ICC preservation ✅ completed)
- **P3 (Deferred)**: ~~4 items~~ 3 items remaining (SLERP interpolation ✅ completed 2026-03-16)
- **No Scanner-Governance Action Required**: all 24 scanner-visible items are governed; historical documentation markers remain narrative context.

---

## Category 1: Active Source Code TODOs

Current scanner count: 0 active source-code TODO comments. Entries below are
completed work or research backlog items retained for planning context, not
scanner-governance violations.

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

### P3: Material Backend Research Tracks (3 items)

**Location**: `src/transformation_portal/spatial_ai/materials/material_backend.py`

These are not scanner-visible source TODO comments as of 2026-05-11. They remain
research-deferred feature tracks documented here and in
[OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md](./OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md).

#### 1. ComfyUI Subprocess Integration

**Status**: Optional integration path; current material backend routes unsupported modes through governed fallbacks and validation.
**Action**: Keep as-is until Phase 5B implementation
**Priority**: P3 (Deferred - experimental feature)

#### 2. NVDIFFREC Integration

**Status**: Research track; current material backend keeps licensing, GPU, and multi-view evidence requirements explicit.
**Action**: Defer to v2.3.0+ (requires ADR for GPU requirements)
**Priority**: P3 (Research feature)

#### 3. MaterialGAN Integration

**Status**: Research track; current material backend requires appropriate capture evidence before any real backend execution.
**Action**: Defer to v2.3.0+ (requires ADR for model licensing)
**Priority**: P3 (Research feature)

**Cross-Reference**: See [OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md](./OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md) Section 3

---

### ✅ COMPLETED: SLERP Interpolation (scene_builder.py)

**Location**: `src/transformation_portal/spatial_ai/reconstruction/scene_builder.py`

**Completed**: 2026-03-16

**Implementation**:
- ✅ SLERP interpolation using `scipy.spatial.transform.Slerp`
- ✅ Proper rotation matrix preservation (orthogonality maintained)
- ✅ LERP for translation vectors (separate from rotation)
- ✅ Comprehensive unit tests validating rotation orthogonality

**Changes Made**:
- `extract_camera_path()` now uses SLERP for rotations via scipy
- Translation uses linear interpolation (LERP)
- Tests verify orthogonality and proper determinant (+1) for all frames
- Boundary and midpoint validation included in test suite

**Tests Added**:
- `tests/spatial_ai/reconstruction/test_scene_utils.py::test_extract_camera_path_preserves_rotation_orthogonality`

**Verification**:
```bash
pytest tests/spatial_ai/reconstruction/test_scene_utils.py::TestSceneBuilder::test_extract_camera_path_preserves_rotation_orthogonality -v
```

**Original Context**: Camera path interpolation previously used naive linear interpolation of 4x4 extrinsic matrices, which produced sheared rotations. The SLERP implementation ensures rigid transformations are preserved.

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

## Category 4: Test TODOs (Resolved)

**Former Location**: `tests/stress/test_stress_large_batch.py` (historical lines 326, 372)

**Current Status**: The scanner-visible test TODO comments are gone. The stress
tests now assert bounded preset-performance behavior directly, so the current
scanner baseline reports 0 test TODO comments.

**Context**: The historical notes tracked inconsistent preset performance, likely
due to resource contention in stress tests.

**Impact**: None - Performance observation, not a bug.

**Action**: Continue tracking broader performance opportunities in
[IMPROVEMENT_OPPORTUNITIES.md](../guides/IMPROVEMENT_OPPORTUNITIES.md) if needed.
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

| # | Item | Effort | Owner | Phase | Status |
|---|------|--------|-------|-------|--------|
| 8 | SLERP interpolation in scene_builder.py | 1-2d | Spatial AI | Phase 2 | ✅ Completed 2026-03-16 |
| 9 | ComfyUI subprocess integration | 1-2w | ML | Phase 5B | Pending |
| 10 | NVDIFFREC integration | 3-4w | ML Research | Phase 3 | Pending |
| 11 | MaterialGAN integration | 2-3w | ML Research | Phase 3 | Pending |

**Total P3 Effort**: ~~200+~~ hours → **Remaining**: ~185+ hours (SLERP completed)

---

### No Action Required (scanner-governed backlog remains clean)

The following categories require **no action**:

1. **Documentation TODO markers:** Repository-wide historical notes, status markers, or archive candidates reviewed for context rather than implementation. Markdown TODO prose is outside scanner enforcement unless promoted into source/test/script/tool/web surfaces.
2. **Security Patterns** (2 items): `TODO_REPLACE` regex patterns used to detect placeholder values
3. **Observational Notes:** Performance investigation tracking already captured in follow-up documentation
4. **Test Fixture Content:** Embedded `TODO` text used as scanner test data

These reviewed instances are intentionally excluded from the 11-item P1/P2/P3 backlog above.

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
| [TODO_INVENTORY.md](./TODO_INVENTORY.md) | Comprehensive v2.4.7 inventory | ✅ Current |
| [OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md](./OUTSTANDING_TODOS_EXPERIMENTAL_FEATURES.md) | Experimental feature tracking | ✅ Current |
| [HF_REVISION_PINNING_GUIDE.md](./HF_REVISION_PINNING_GUIDE.md) | Model pinning procedures | ✅ Current |
| [IMPROVEMENT_OPPORTUNITIES.md](../guides/IMPROVEMENT_OPPORTUNITIES.md) | Performance optimization tracking | ✅ Current |

---

## Summary

**Scanner Baseline Reviewed**: 24 governed items across 1,570 files
**Scanner-Governance Action Required**: 0 items
  - 0 active source code TODOs (all cleaned up)
  - 0 ungoverned scanner-visible TODOs

**Planning Backlog Action Required**: 4 items total
  - 1 admin action (branch protection verification)
  - 3 future research/integration items (ComfyUI, NVDIFFREC, MaterialGAN)
**No Action Required**: all 24 scanner-visible items are governed

**Completion Progress (2026-05-11)**:
- ✅ 10 items completed (HF pinning, ICC preservation, rollback docs verified, depth_canonical archived, PR docs archived, SLERP interpolation, TODO_INVENTORY corrected, source code TODO cleanup)
- ⏳ 1 item pending external action (branch protection - requires admin UI access)
- ⏳ 3 items pending future phases (ComfyUI, NVDIFFREC, MaterialGAN)

**Key Takeaway**: The scanner-governed TODO surface has no open governance violations. The remaining action items are planning backlog entries, with 1 remaining P1 item (branch protection verification pending external admin access) and 3 deferred research/integration tracks. All source code TODOs in `src/` have been cleaned up.

The codebase demonstrates mature TODO management with proper documentation and categorization. The existing [TODO_INVENTORY.md](./TODO_INVENTORY.md) provides comprehensive tracking and should continue to be the authoritative source for TODO governance.

---

**Document Version**: 1.5.0
**Author**: Transformation Portal Architect Review
**Last Updated**: 2026-05-11
**Next Review**: Before v2.5.0 release planning (CI now blocks ungoverned TODOs at PR time)
