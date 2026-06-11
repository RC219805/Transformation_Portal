# TODO Inventory - Transformation Portal

**Document Version:** 2.4.7
**Date:** May 11, 2026
**Last Updated:** 2026-05-11 (scanner snapshot refreshed; 24 governed items)
**Previous Version:** 2.4.6 (2026-05-04)

---

## Purpose

This document provides a **complete, categorized inventory** of all TODOs, NotImplementedError stubs, pending integrations, and deferred work across the Transformation Portal codebase.

**Usage:**
- Reference for sprint planning
- Audit trail for architectural decisions
- Integration with issue tracking systems
- Binding inventory enforced by Architect governance

## Version 2.4.7 Changes (2026-05-11)

**Governance snapshot refresh:**
- ✅ **Scanner snapshot regenerated from live repo state.** `python scripts/validation/scan_todo_inventory.py --write-snapshot` now refreshes
  [`todo_scanner_snapshot.json`](todo_scanner_snapshot.json) directly from the scanner payload.
- 📊 **Current scanner baseline:** 24 `NotImplementedError` items, 0 ungoverned TODOs, 1,570 files scanned.
- 📊 **Baseline delta:** 25 → 24 governed items. The prior snapshot included
  `scripts/pipelines/tiff_enhancement_pipeline.py`, which is no longer present
  in the tracked tree. This is scanner-state reconciliation, not a newly closed
  action item.
- ✅ **No source-code `TODO|FIXME|HACK|XXX` backlog opened.** The only scanner
  hits remain governed `NotImplementedError` stubs: abstract/base-class
  contracts, phase gates, platform limits, or test protocol stubs.

## Version 2.4.6 Changes (2026-05-04)

**Minor cleanup:**
- ✅ **§2.2 ComfyUI `@abstractmethod` checkbox closed.** Verification of
  `src/transformation_portal/comfyui/custom_nodes.py` confirms `BaseNode`
  inherits from `ABC` and decorates `INPUT_TYPES`, `RETURN_TYPES`, and
  `execute` with `@abstractmethod`; `tests/test_comfyui.py::TestBaseNode::test_base_node_abstract_methods`
  asserts the canonical `__abstractmethods__` invariant. The previously open
  `[ ]` checkbox under §2.2 was a documentation residue from before that test
  landed. QW-1 in `docs/deliverables/QUICK_WINS.md` marked completed
  (2026-05-04) with AST-based verification command captured. No code changes.

## Version 2.4.5 Changes (2026-05-02)

**Minor cleanup:**
- ✅ **§4.1 Binary File Cleanup recommendation closed.** The TODO markers in
  the directory-structure diagram of `docs/fixes/BINARY_FILE_BEST_PRACTICES.md`
  had already been replaced with ✅ status rows in an earlier sweep; this
  release adds a status banner to the top of that document so the residual
  "⚠️ Currently tracked" warnings further down are no longer mistaken for
  current state, and marks QW-3 (`docs/deliverables/QUICK_WINS.md`) as
  completed (2026-05-02). Verified via
  `git ls-files input_images/ processed_images/` (no PNG/JPG/TIFF binaries
  tracked under either directory). No code changes.

## Version 2.4.4 Changes (2026-05-01)

**Major Updates — 2025 reorganization residue swept:**
- ✅ **Orphaned 2025 PR-review reports archived.** Four canonical files at `docs/pr_reports/` (`BUG_REPORT_CODE_REVIEW.md`, `PR_ACTIONABLE_FIXES.md`, `PR_CONSOLIDATION_ANALYSIS.md`, `PR_REVIEW_SUMMARY.md`, dated October 2025 / January 2025) moved to `docs/_archive/2025-pr-review-cycle/` with a provenance README.
- ✅ **`CODEBASE_OPTIMIZATION_2025.md` archived.** Nov-2025 historical record at `docs/operations/` was orphaned (no incoming references) and described an already-complete reorganization. Moved to `docs/_archive/2025-pr-review-cycle/` alongside the related PR-review reports.
- ✅ **`PR216_REVIEW_SUMMARY.md` archived.** Nov-2025 PR review summary at `docs/analysis/` (status "READY TO MERGE", zero incoming references) — same shape as `CODEBASE_OPTIMIZATION_2025.md`. Moved to `docs/_archive/2025-pr-review-cycle/` for consistency. Contains a dated `docs/` tree diagram that's now annotated as historical via the archive README.
- ✅ **All 12 redirect stubs from the 2025 reorg deleted.** Three at `docs/development/pr/` (pointed to the now-archived `pr_reports/` files), seven at `docs/development/` (pointed to canonical files in `docs/summaries/` / `docs/verification/`), and two at `docs/development/testing/` (pointed to `docs/summaries/`). Pre-deletion check: of the **stub files themselves**, only one had a live incoming reference (a `TODO_INVENTORY.md` cleanup-tracking line that flagged it for removal); all others were dangling. (Older docs — e.g., the QW-2 task spec in `QUICK_WINS.md` and a `docs/` tree diagram in `PR216_REVIEW_SUMMARY.md` — still mention the original canonical paths from the 2025 → 2026 reorg journey; those are historical/dated guidance and are now annotated with their resolution status. The dated `documentation-inventory-2026-04-29.csv` remains an immutable point-in-time snapshot.) Canonical copies in `docs/summaries/` and `docs/verification/` are untouched.
- ✅ **`DOCUMENTATION_POLICY.md` and `DOCUMENTATION_MAP.md` taxonomies updated.** Dropped `docs/pr_reports/` from both listings (the directory is now empty post-archive). Sibling `docs/pr_archive/` (19 files) and `docs/pr_summaries/` (1 file) remain in the taxonomy.
- ✅ **§6 "Old Verification Reports" cleanup closed.** The flagged `docs/development/VERIFICATION_COMPLETE.md` redirect stub is among the 12 deleted; canonical `docs/verification/VERIFICATION_COMPLETE.md` retained.

## Version 2.4.3 Changes (2026-05-01)

**Major Updates:**
- ✅ **§4.3 ADR-023 Manifest Implementation audit closed.** Audit confirmed all 8 implementation TODOs in the archived guide are shipped in production (Phase 2 in `tools/performance_ledger.py` v1.7; Phase 3 across `manifest.py`, `orchestrator.py`, and `tests/test_backend_selection.py`). The obsolete planning guide at `docs/_archive/2026-Q1-consolidation/ADR-023-implementation-guide.md` was deleted; the parent `ADR-023-post-pr841-hardening.md` (same archive directory) and active-docs successors remain the authoritative record. QW-4 in `docs/deliverables/QUICK_WINS.md` marked complete.

## Version 2.4.2 Changes (2026-05-01)

**Major Updates:**
- ✅ **TODO governance enforcement now runs in CI.** `scripts/validation/scan_todo_inventory.py --check-governance` is invoked from the `hf-revision-policy` job in `.github/workflows/enforcement.yml`. Any new ungoverned action marker now fails the PR build. Enforced patterns (per the scanner): Python `# TODO:` / `# FIXME:` / `# HACK:` / `# XXX:` comments, `raise NotImplementedError(...)` AST nodes, and JS/TS `// TODO|FIXME|HACK|XXX` and `* TODO|FIXME|HACK|XXX` comments. Enforced scopes: `src/`, `tests/`, `scripts/`, `tools/`, and `web/` (JS/TS). A "governance reference" is any of: `(Phase X)`, `(ADR-###)`, `(#1234)`, `(@owner)`, or `(TODO_INVENTORY.md ...)`. Bare `raise NotImplementedError` and recognized abstract-method messages (e.g., "Subclass must implement") are auto-governed. This replaces the manual monthly review cadence as the primary drift control.
- 📊 **NotImplementedError baseline corrected: 12 → 25.** The previous count was stale documentation; the live scanner reports 25, and all 25 are properly governed (abstract methods, phase gates, platform limits, categorized technical debt). No code changes required — this is a documentation reconciliation.
- 📊 **Source code TODO count: still 0** — verified by scanner.

## Version 2.4.1 Changes (2026-03-26)

**Major Updates:**
- ✅ Synchronized inventory references with the merged curated web-stack compatibility update
- ✅ Added documentation-level acknowledgement that Starlette 1.0 compatibility work is complete, not pending
- ✅ Clarified that the curated web-stack PR removed invalid ML lock regeneration churn, while the underlying ML lock-generation trust defect remains active and is tracked as follow-up issue #1279, so it is not duplicated here as a separate inventory item

## Version 2.4.0 Changes (2026-03-25)

**Major Updates:**
- ✅ **Quality Firewall documentation created** (`docs/guides/QUALITY_FIREWALL.md`) - landing page now exists
- 📊 **Source code TODO count: 0** - All `# TODO:` comments in `src/` have been cleaned up
- 📊 **Test TODO count: 3** - 2 observational (performance), 1 test fixture content
- 📊 **NotImplementedError count: 12** - All intentional (ABCs, phase gates, platform limits)
- 🔍 Verified depth_canonical module already archived (no action needed)
- 🔍 Verified PR tracking docs already archived (no action needed)
- 🔍 Updated CRITICAL_BIT_DEPTH_FIX_SUMMARY.md reference to `docs/guides/QUALITY_FIREWALL.md`

## Version 2.3.0 Changes (2026-03-24)

**Major Updates:**
- ✅ Rollback Procedures **VERIFIED AS COMPLETE** (docs/operations/ROLLBACK_PROCEDURES.md, v1.0.0)
- 📊 Reduced P1 items from 3 to 2 (rollback docs confirmed existing)
- 📊 Updated CI/CD Gaps: 5 Done, 1 Missing (was 4 Done, 2 Missing)
- 📊 Total completed items: 10 of 65 action-tracked items (15%)
- 🔍 Corrected documentation inconsistency: TODO_INVENTORY claimed rollback docs missing while they exist

## Version 2.2.0 Changes (2026-03-16)

**Major Updates:**
- ✅ SLERP interpolation **COMPLETED** (scene_builder.py using scipy.spatial.transform.Slerp)
- 📊 Reduced P3 items from 4 to 3 (SLERP completed)
- 📊 Total completed items: 16 of 65 action-tracked items (25%)

## Version 2.1.0 Changes (2026-03-15)

**Major Updates:**
- ✅ HuggingFace model revision pinning **COMPLETED** (config/model_lock_manifest.yaml)
- ✅ Rollback procedures **VERIFIED** (docs/operations/ROLLBACK_PROCEDURES.md exists)
- ✅ depth_canonical module **ARCHIVED** (removed from src/)
- ✅ context_aware_rendering.py **ARCHIVED** (archive/scripts/)
- ✅ lux_render_pipeline_plus_v3.py **ARCHIVED** (archive/scripts/pipelines/)
- ✅ ICC profile preservation in 16-bit TIFF **COMPLETED**
- 📊 Reduced P1 items from 3 to 1 (only branch protection pending)
- 📊 Updated statistics: 15 of 65 completed (23%), 5 pending action items (8%)

**Previous Version (2.0.0) Summary:**
- ✅ ADR-019 backend registry integration **COMPLETED** (PR #906, commit 9c7a281c)
- ✅ CI/CD coverage enforcement **IMPLEMENTED** (ci.yml dual-gate: 25% floor + 80% diff)
- ✅ Security scanning **IMPLEMENTED** (Bandit, pip-audit, gitleaks, CodeQL, Safety)
- ✅ Contract validation **IMPLEMENTED** (ingest_contract_validation.yml)
- ✅ Nightly regression suite **IMPLEMENTED** (nightly.yml with benchmarks)
- 🗑️ Identified 3 obsolete modules for cleanup → **NOW ARCHIVED**

---

## Inventory Categories

1. [Architectural Decisions (ADRs)](#1-architectural-decisions-adrs)
2. [Stub Implementations](#2-stub-implementations)
3. [Code TODOs](#3-code-todos)
4. [Documentation TODOs](#4-documentation-todos)
5. [CI/CD Gaps](#5-cicd-gaps)
6. [Test Infrastructure](#6-test-infrastructure)
7. [Obsolete Items](#7-obsolete-items)

---

## 1. Architectural Decisions (ADRs)

### 1.1 ADR-019 Backend Registry Integration

**Status:** ✅ **COMPLETED** (PR #906, commit 9c7a281c)
**Priority:** ~~P0~~ → DONE
**Effort:** 2-3 dev days (actual)
**Owner:** Specialist (Architect-supervised)
**Completion Date:** 2026-02-09 (APEX Research Ultra Phase 1)

**Context:**
- Original decision: Defer to v2.1.0 (Depth Pro not operational)
- New information: Depth Pro checkpoint verified at `checkpoints/depth_pro.pt`
- Revised decision: Immediate integration approved
- **IMPLEMENTATION COMPLETED** in PR #906 (APEX Research Ultra Phase 1)

**Scope (All Completed ✅):**
- ✅ Create `DA3Backend` adapter wrapping `DA3InferenceEngine`
- ✅ Integrate `DepthBackendRegistry` into orchestrator
- ✅ Implement backend selection: `config.depth_backend or "depth_anything_v3"`
- ✅ Add fallback policy: `depth_pro → da3`
- ✅ Update metadata capture for backend tracking
- ✅ Write integration tests (orchestrator + both backends)
- ✅ Update CLI documentation (`--depth-backend` flag)
- ✅ **BONUS:** Ensemble backend for depth fusion (depth_pro + da3)

**Files Implemented:**
- `src/transformation_portal/depth/backends/da3.py` ✅
- `src/transformation_portal/depth/backends/depth_pro.py` ✅
- `src/transformation_portal/depth/backends/ensemble.py` ✅
- `src/transformation_portal/depth/backends/registry.py` ✅
- `src/transformation_portal/depth/backends/protocol.py` ✅
- `src/transformation_portal/depth/backends/cache.py` ✅
- `tests/` (comprehensive integration tests) ✅
- `docs/` (CLI + architecture documentation) ✅

**References:**
- ADR-019-REVISED-DECISION.md (approved)
- ADR-019-INTEGRATION-APPROVAL.md (approved)
- ADR-019-IMPLEMENTATION-STATUS.md (superseded)
- CHANGELOG.md: "Backend Registry Integration (ADR-019)"
- PR #906: "feat(apex-ultra): APEX Research Ultra Phase 1"

**Architect Assessment:** **VERIFIED COMPLETE** - Implementation exceeds specification with ensemble backend.

---

### 1.2 Backend Strict Enforcement (ADR-024)

**Status:** 🔴 DEFERRED TO v2.1.0
**Priority:** P2 (Medium)
**Effort:** 1-2 days
**Blocker:** Requires ADR-019 completion + stable multi-backend system

**Scope:**
- Add `--strict-backend` CLI flag
- Fail-fast on backend unavailability (if strict mode)
- Default: permissive (warn + fallback)
- User-configurable fallback chains

**References:**
- ADR-024-backend-enforcement-strategy.md

---

### 1.3 YAML Configuration Loading (Phase 2)

**Status:** 🟡 DEFERRED TO v2.1.0
**Priority:** P3 (Low)
**Effort:** 1-2 days
**Impact:** Performance optimization (faster preset loading)

**Current State:**
- Config loader exists, uses `ConfigLoader` class
- YAML parsing functional
- Missing: Advanced preset composition, environment variable expansion validation

**Location:**
- `src/transformation_portal/config_loader.py`
- Mentioned in: `docs/guides/IMPROVEMENT_OPPORTUNITIES.md:315`

**Scope:**
- [ ] Implement preset inheritance/composition
- [ ] Add input validation for env var expansion
- [ ] Write integration tests for complex configs
- [ ] Document preset authoring guide

**Files Affected:**
- `src/transformation_portal/config_loader.py`
- `tests/test_config_loader.py`
- `docs/guides/preset_authoring.md` (new)

---

### 1.4 Depth Pro Preset Configuration

**Status:** 🔴 DEFERRED TO v2.1.0
**Priority:** P2 (Medium)
**Effort:** 4 hours
**Blocker:** Pending ADR-019 integration completion

**Current State:**
- Example presets exist:
  - `config/presets/depth_pro_example.yaml`
  - `config/presets/depth_pro_metric_mps.yaml`
  - `config/presets/depth_pro_metric_cpu.yaml`
- Not validated end-to-end with orchestrator

**Scope:**
- [ ] Validate existing presets with orchestrator
- [ ] Create presets for common scenarios (interior, exterior, aerial)
- [ ] Document license requirements (Apple AMLR)
- [ ] Add preset selection guide to README

**Files Affected:**
- `config/presets/depth_pro_*.yaml` (validate/extend)
- `docs/guides/depth_pro_presets.md` (new)
- `README.md` (update backend section)

---

### 1.5 CLI Backend Management Commands

**Status:** 🔴 DEFERRED TO v2.2.0
**Priority:** P4 (Nice-to-have)
**Effort:** 1 week

**Scope:**
- `lux-depth-v3 --list-backends` - Show available backends
- `lux-depth-v3 --download-models <backend>` - Automated checkpoint download
- `lux-depth-v3 --verify-backend <name>` - Health check

**References:**
- ADR-019-IMPLEMENTATION-STATUS.md (Future Work section)

---

### 1.6 Parallax Occlusion Mapping (POM)

**Status:** ✅ **ARCHIVED** (2026-02-10)
**Priority:** ~~P5~~ → COMPLETED
**Effort:** 1 hour (actual)
**Impact:** Advanced rendering feature, **no demonstrated user demand**

**Action Taken:**
- ✅ Audited `lux_render_pipeline_plus_v3.py` usage: **Zero production imports**
- ✅ Moved to `archive/scripts/pipelines/lux_render_pipeline_plus_v3.py`
- ✅ Documented archival rationale in `archive/scripts/README.md`
- ✅ Preserved git history via `git mv`

**Original State:**
- Stub existed in `scripts/pipelines/lux_render_pipeline_plus_v3.py:208`
- Raised `NotImplementedError` for Parallax Occlusion Mapping (POM)
- **Module abandoned** (last meaningful commit: 2025-09-15, 5+ months ago)
- **Superseded by:** Production PBR processor (`src/transformation_portal/lux_depth_v3/pbr_processor.py`)

**Location:**
```python
# archive/scripts/pipelines/lux_render_pipeline_plus_v3.py:208 (archived)
raise NotImplementedError(
    "Parallax Occlusion Mapping (POM) is not yet implemented in this pipeline. "
    "This is an advanced feature requiring additional research and validation."
)
```

**Architect Decision:**
- **ARCHIVED** - No user requests, no production usage, superseded by cleaner implementation
- If POM becomes a requirement, implement in production PBR processor (not archived code)
- Archived file preserved for historical reference only

**Removal from Active Inventory:** Moved to historical archive section in TODO_INVENTORY v3.0.0

---

## 2. Stub Implementations

**Updated Count:** 29 NotImplementedError instances (was 8 in v1.0.0)
**New Instances:** 21 added since 2026-02-05 (Spatial AI Phase 2, materials-v3)

### 2.0 NEW: Spatial AI Segmentation Stubs (SAM2)

**Status:** 🟢 INTENTIONAL FEATURE GATES
**Priority:** P2 (Medium - Phase 2 roadmap items)
**Action:** Track as Phase 2.1 implementation targets

**Location:** `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py`

#### 2.0.1 Auto Mask Generation (Line ~64)
```python
raise NotImplementedError("SAM2 auto mode not yet integrated")
```
**Context:** Automatic mask generation without prompts
**Status:** Phase 2.1 target feature
**Effort:** 3-4 days (model integration + testing)

#### 2.0.2 Prompted Segmentation (Line ~75)
```python
raise NotImplementedError("Prompted segmentation not yet implemented")
```
**Context:** Interactive segmentation with point/box prompts
**Status:** Phase 2.2 target feature
**Effort:** 5-6 days (prompt handling + UI consideration)

#### 2.0.3 Video Tracking (Line ~86)
```python
raise NotImplementedError("Video tracking not yet implemented")
```
**Context:** SAM2 video object tracking capabilities
**Status:** Phase 3 or later (video pipeline integration required)
**Effort:** 2 weeks (requires video pipeline design)

**Tests:** `tests/test_sam2_backend.py` validates NotImplementedError is raised correctly

**Architect Recommendation:**
- Keep stubs as feature gates
- Add to Phase 2.1/2.2 roadmap
- No action required for v2.2.0 release

---

### 2.0 NEW: Materials V3 Integration Stubs

**Status:** 🟢 INTENTIONAL RESEARCH FEATURES
**Priority:** P3 (Low - experimental ML features)
**Action:** Document as experimental, defer to v2.3.0 or v3.0.0

**Location:** `src/transformation_portal/spatial_ai/materials/material_backend.py`

#### 2.0.4 NVDIFFREC Integration (Line 323)
```python
# Phase 2.2 roadmap item: NVDIFFREC integration
```
**Context:** NVIDIA Differentiable Rendering for PBR material extraction
**Status:** Research phase (GPU-intensive, license considerations)
**Effort:** 3-4 weeks (model integration, dataset requirements, validation)

#### 2.0.5 MaterialGAN Integration (Line 359)
```python
# Phase 2.2 roadmap item: MaterialGAN integration
```
**Context:** GAN-based material synthesis
**Status:** Research phase (model availability, quality validation)
**Effort:** 2-3 weeks

**Architect Recommendation:**
- Mark as experimental research features
- Require ADR before implementation (license, GPU requirements, accuracy validation)
- Consider external plugin system instead of core integration

---

### 2.0 NEW: Spatial AI Reconstruction Stubs

**Status:** ✅ COMPLETED (SLERP implemented 2026-03-16)
**Priority:** ~~P4~~ → DONE
**Action:** None required

**Location:** `src/transformation_portal/spatial_ai/reconstruction/scene_builder.py`

#### 2.0.6 ✅ COMPLETED: SLERP Interpolation

**Previous State:**
```python
# TODO: Replace with proper SLERP interpolation
raise NotImplementedError("If interpolation != 'linear'")
```

**Current State (2026-03-16):**
- ✅ SLERP interpolation implemented using `scipy.spatial.transform.Slerp`
- ✅ LERP for translation vectors (separate from rotation)
- ✅ Comprehensive tests validating rotation orthogonality
- ✅ Test: `test_scene_utils.py::test_extract_camera_path_preserves_rotation_orthogonality`

**Implementation Details:**
- `extract_camera_path()` uses scipy.spatial.transform.Rotation for SLERP
- Boundary conditions (first/last frame) match original cameras exactly
- Midpoint validation confirms correct angular interpolation
- Determinant check ensures proper rotations (no reflections)

**Architect Assessment:** Implementation exceeds requirements with comprehensive validation.

---

### 2.0 NEW: Linear Decoder RAW Format

**Status:** 🟢 CORRECT PHASE GATE
**Priority:** P3 (Phase II Spatial AI roadmap)
**Action:** None until Phase II planning

**Location:** `src/transformation_portal/spatial_ai/ingest/linear_decoder.py`

#### 2.0.7 RAW Format Handling (Line ~34)
```python
raise NotImplementedError("RAW format handling is a Phase II feature")
```
**Context:** RAW camera format support for ingest pipeline
**Status:** Explicitly deferred to Phase II (ADR-027)
**Effort:** 1 week (requires rawpy/LibRaw integration)

**Architect Recommendation:** Keep as phase gate, validate against ADR-027

---

### 2.1 Lux Depth V3 Preprocessing (2 functions)

**Status:** ✅ **IMPLEMENTED** (2026-03-25 verification)
**Priority:** ~~P4~~ → DONE
**Action:** None - functions are fully implemented

**Location:** `src/transformation_portal/lux_depth_v3/preprocessing.py`

#### 2.1.1 `normalize_exif_orientation()`

**Line:** 542
**Purpose:** EXIF orientation normalization
**Status:** ✅ **FULLY IMPLEMENTED** - Function rotates images based on EXIF orientation tags, preserves metadata, and handles edge cases.

**Implementation Details:**
- Uses PIL `exif_transpose()` for rotation
- Preserves EXIF metadata with normalized orientation tag
- Avoids lossy re-encoding for JPEGs when no rotation needed
- Proper error handling with ValueError wrapping

#### 2.1.2 `validate_depth_image_alignment()`

**Line:** 631
**Purpose:** Validate depth map and image dimension matching
**Status:** ✅ **FULLY IMPLEMENTED** - Function validates RGB/depth dimension compatibility.

**Implementation Details:**
- Supports PNG, NPY, and TIFF depth formats
- Uses memory-mapped loading for NPY files (mmap_mode="r")
- Allows tolerance for DA3 padding (multiples of 14)
- Validates both exact and padded dimension matches

**Note:** The TODO_INVENTORY previously listed these as stubs, but code review and test verification confirm full implementations exist with test coverage in `tests/test_preprocessing.py`.

---

### 2.2 ComfyUI Custom Nodes (3 methods)

**Status:** 🟢 EXPERIMENTAL INTEGRATION
**Priority:** P4 (Community Feature)
**Action:** Mark as experimental, defer to plugin system

**Location:** `src/transformation_portal/comfyui/custom_nodes.py`

#### 2.2.1 `BaseNode.INPUT_TYPES()`

**Line:** 55
```python
@classmethod
def INPUT_TYPES(cls) -> Dict[str, Any]:
    raise NotImplementedError
```

**Purpose:** Abstract base class for ComfyUI node interface
**Status:** Correct usage (subclasses must override)

#### 2.2.2 `BaseNode.RETURN_TYPES()`

**Line:** 59
```python
@classmethod
def RETURN_TYPES(cls) -> Tuple[str, ...]:
    raise NotImplementedError
```

**Purpose:** Abstract base class method
**Status:** Correct usage

#### 2.2.3 `BaseNode.execute()`

**Line:** 62
```python
def execute(self, **kwargs) -> Tuple[Any, ...]:
    raise NotImplementedError
```

**Purpose:** Abstract base class method
**Status:** Correct usage

**Decision:**
- [x] Add `@abstractmethod` decorators for clarity — completed (verified 2026-05-04, see QW-1 in `docs/deliverables/QUICK_WINS.md`); `BaseNode(ABC)` plus `@abstractmethod` on `INPUT_TYPES`, `RETURN_TYPES`, and `execute` are locked in by `tests/test_comfyui.py::TestBaseNode::test_base_node_abstract_methods`
- [ ] Document ComfyUI integration as experimental
- [ ] Create example subclass in `examples/comfyui/`
- [ ] Defer full integration to plugin system (v2.2.0)

**Recommendation:** Add documentation, keep stubs (correct pattern for abstract base class)

---

### 2.3 Depth Canonical Model Registry

**Status:** 🔴 **OBSOLETE - RECOMMEND DELETION**
**Priority:** ~~P3~~ → CLEANUP REQUIRED
**Action:** Remove entire `depth_canonical` module

**Location:** `src/transformation_portal/depth_canonical/models/registry.py:31`

```python
raise NotImplementedError
```

**Context:**
- Older registry pattern from depth_canonical module
- **SUPERSEDED** by `src/transformation_portal/depth/backends/registry.py` (ADR-019)
- Last commit: 9af004ee (2026-01-21, "Gate 0: black+isort baseline")
- **No active development** for 23+ days

**Audit Results:**
- Only 14 Python files in module
- Only 2 imports found: `__init__.py` and `README.md`
- No usage in active pipelines (lux_depth_v3, spatial_ai, stage_graph)

**Architect Decision:** **DELETE MODULE**

**Deletion Plan:**
- [ ] Verify no imports in src/ (grep confirms only self-references)
- [ ] Remove `src/transformation_portal/depth_canonical/` entirely
- [ ] Update imports if any (likely none)
- [ ] Remove from pyproject.toml if listed
- [ ] Add MIGRATION.md note if external users exist (unlikely)
- [ ] Update CHANGELOG.md: "Removed obsolete depth_canonical module (superseded by ADR-019 backends)"

**Effort:** 1 hour (deletion + verification)
**Risk:** LOW (no active usage detected)

**Recommendation:** Schedule for immediate cleanup in next PR

---

### 2.4 Security Utils - Windows Platform Restriction

**Status:** ✅ INTENTIONAL LIMITATION
**Priority:** N/A
**Action:** None (documented constraint)

**Location:** `src/transformation_portal/utils/security.py:418`

```python
raise NotImplementedError(
    "Security hardening features are Unix-specific. "
    "Windows support requires separate implementation."
)
```

**Context:**
- Security features use Unix-specific system calls (chroot, seccomp, etc.)
- Windows requires completely different approach (ACLs, AppContainers)
- Current user base: macOS/Linux only

**Decision:** Keep as documented platform limitation (no action required)

---

### 2.5 Depth Anything V2 Backend (Legacy)

**Status:** 🟢 CORRECT STUB
**Priority:** N/A
**Action:** None (correct error handling)

**Location:** `src/transformation_portal/depth/models/depth_anything_v2.py:211`

```python
raise NotImplementedError(f"Backend {self.backend} not implemented")
```

**Context:**
- Handles unknown backend gracefully
- Part of model initialization error path
- Not actually a stub - defensive programming

**Decision:** No action (correct error handling pattern)

---

## 3. Code TODOs

### 3.1 Context-Aware Rendering Integration

**Status:** 🔴 **OBSOLETE - RECOMMEND ARCHIVAL**
**Priority:** ~~P3~~ → CLEANUP/ARCHIVE
**Effort:** 0h (deletion) or 2h (move to examples/)
**Owner:** Architect (decision)

**Status:** ✅ **ARCHIVED** (2026-02-10)
**Priority:** ~~P2~~ → COMPLETED
**Effort:** 1 hour (actual)
**Owner:** Architect
**Completion Date:** 2026-02-10

**Action Taken:**
- ✅ Moved `scripts/context_aware_rendering.py` → `archive/scripts/`
- ✅ Moved `scripts/premium_context_pipeline.py` → `archive/scripts/` (depends on context_aware_rendering)
- ✅ Moved `scripts/pipelines/lux_render_pipeline_plus_v3.py` → `archive/scripts/pipelines/`
- ✅ Created `archive/scripts/README.md` documenting archival rationale
- ✅ Git history preserved via `git mv` (not delete)

**Rationale:**
- `context_aware_rendering.py`: Superseded by Spatial AI orchestration (Phase 2, ADR-027)
- `premium_context_pipeline.py`: Depends on archived context_aware_rendering; superseded by production orchestrator
- `lux_render_pipeline_plus_v3.py`: Abandoned with POM stub (line 208); superseded by production PBR processor

**Location:** ~~`scripts/context_aware_rendering.py:354`~~ → `archive/scripts/context_aware_rendering.py`

```python
# TODO: Integrate with actual processing pipelines
# This would call:
# 1. depth_pipeline with depth_config
# 2. material_response with material_config
# 3. luxury_tiff_batch_processor with color_config
```

**Context:**
- Script generates context-aware rendering strategies
- **Standalone demonstration/POC** (never integrated)
- Last modified: 2026-02-07 (no functional changes, only formatting)
- **Superseded by:** Spatial AI orchestration (Phase 2, ADR-027)

**Architect Analysis:**
- Original vision: automated rendering parameter selection
- **Current reality:** Superseded by Spatial AI pipeline orchestration
- Spatial AI provides superior architecture for multi-stage workflows
- No evidence of user demand for standalone context-aware script

**Architect Decision:** **✅ ARCHIVED (2026-02-10)**

**Archival Completed:**
- [x] Moved to `archive/scripts/context_aware_rendering.py`
- [x] Added README explaining historical context and supersession (`archive/scripts/README.md`)
- [x] Marked as archived in TODO inventory (this document)
- [x] Preserved git history via `git mv` (not delete)

**Additional Files Archived:**
- [x] `premium_context_pipeline.py` → `archive/scripts/` (depends on context_aware_rendering)
- [x] `lux_render_pipeline_plus_v3.py` → `archive/scripts/pipelines/` (abandoned, POM stub)

**CHANGELOG Entry (to add in next commit):**
```
## [2.2.1] - 2026-02-10
### Removed
- Archived `context_aware_rendering.py` (superseded by Spatial AI orchestration)
- Archived `premium_context_pipeline.py` (depends on archived context_aware_rendering)
- Archived `lux_render_pipeline_plus_v3.py` (abandoned, superseded by production PBR processor)
- See `archive/scripts/README.md` for archival rationale
```

**Recommendation:** ~~Archive to `archive/` - no integration planned~~ → **COMPLETED**

**Effort:** ~~30 minutes~~ → 1 hour (actual, included 3 files + comprehensive README)

---

### 3.2 Sample Download URLs (3 instances)

**Status:** 🟡 READY FOR IMPLEMENTATION
**Priority:** P2 (User-facing feature)
**Effort:** 4 hours (upload + update)
**Owner:** DevOps/Docs

**Location:** `scripts/download_samples.py`

#### Affected Samples:

1. **demo_coastal_interior** (line 80)
   ```python
   "url": None,  # TODO: Upload to GitHub Release
   ```

2. **demo_pool_aerial** (line 100)
   ```python
   "url": None,  # TODO: Upload to GitHub Release
   ```

3. **sample_render_4k** (multiple entries, similar pattern)

**Current Behavior:**
```
⚠️ demo_coastal_interior: URL not yet available (TODO: upload to GitHub Release)
```

**Implementation Steps:**
- [ ] Create GitHub Release (e.g., `v2.0.0-samples`)
- [ ] Upload sample files:
  - `demo_coastal_interior.jpg` (5MB)
  - `demo_pool_aerial.jpg` (8MB)
  - `sample_render_4k.tif` (25MB)
  - Additional depth maps and test data
- [ ] Generate SHA256 hashes
- [ ] Update `download_samples.py` with URLs and hashes
- [ ] Update README with download instructions
- [ ] Test download script end-to-end

**Files Affected:**
- `scripts/download_samples.py`
- `README.md` (sample data section)
- GitHub Release creation

**Recommendation:** Schedule as quick win (4h effort, high user value)

---

### 3.3 Performance Regression Investigation — ✅ TODO COMMENTS REMOVED

**Status:** ✅ RESOLVED AS TODO GOVERNANCE ITEM
**Priority:** P4 (Low)
**Action:** Track in IMPROVEMENT_OPPORTUNITIES.md (already done)

**Former Location:** `tests/stress/test_stress_large_batch.py`

#### 3.3.1 Draft Preset Performance

The scanner-visible TODO comment was removed. The current test asserts that the
draft preset stays within the accepted tolerance of the standard preset.

**Context:**
- Draft preset sometimes slower than standard
- Non-deterministic, likely resource contention
- Does not affect correctness, only performance

**Status:** Already tracked in `docs/guides/IMPROVEMENT_OPPORTUNITIES.md:281` (TEST-006)

#### 3.3.2 Preset Performance Ordering

The scanner-visible TODO comment was removed. The current batch stress test
asserts bounded draft-vs-standard throughput behavior.

**Context:**
- Preset performance ordering inconsistent
- May be environment-dependent (CI vs local)
- Already documented in IMPROVEMENT_OPPORTUNITIES.md:287

**Decision:** No TODO-governance action remains. Broader performance optimization
can stay in [IMPROVEMENT_OPPORTUNITIES.md](../guides/IMPROVEMENT_OPPORTUNITIES.md).

---

## 4. Documentation TODOs

### 4.1 Binary File Cleanup — ✅ CLOSED 2026-05-02

**Status:** ✅ OBSOLETE (COMPLETED) — recommendation now executed
**Priority:** N/A
**Action:** None remaining

**Location:** `docs/fixes/BINARY_FILE_BEST_PRACTICES.md` (previously lines 133, 140)

**Original residue:**
```markdown
│   ├── *.png                 # ⚠️ TODO: Exclude PNG previews
│   └── *.jpg                 # TODO: Move to docs/examples/
```

**Evidence of Completion:**
- `docs/fixes/PRE_PUSH_AUDIT_REPORT.md:256` states "Minimal TODO comments (only in documentation)"
- Binary cleanup action plan executed
- Pre-push audits passing
- `git ls-files input_images/ processed_images/` (2026-05-02) shows no
  PNG/JPG/TIFF binaries tracked under either directory
- TODO markers in the directory-structure diagram replaced with ✅ status
  rows in an earlier sweep

**Closure (2026-05-02):**
- ✅ Status banner added to top of `docs/fixes/BINARY_FILE_BEST_PRACTICES.md`
  noting that the "🎯 Executive Decision: Current Push" remediation and the
  four steps under "🔧 Immediate Actions Required" are complete, so the
  historical "⚠️ Currently tracked" warnings under
  "📋 Binary File Guidelines → 2. What MUST Be Excluded" describe the
  2025-11-06 snapshot rather than the current state.
- ✅ QW-3 in `docs/deliverables/QUICK_WINS.md` marked completed (2026-05-02)
  with verification command captured.

---

### 4.2 PR #98 Action Items — ✅ ARCHIVED 2026-03-14

**Status:** ✅ OBSOLETE (PR MERGED) and ✅ ARCHIVED
**Priority:** N/A
**Action:** Archive tracking documents — done

**Original locations** *(both paths archived; canonical copies are at the archive location below)*:
- ~~`docs/pr_reports/PR98_ACTION_ITEMS.md`~~ → `docs/_archive/2026-03-legacy-prs/`
- ~~`docs/development/pr/PR98_ACTION_ITEMS.md`~~ → `docs/_archive/2026-03-legacy-prs/` (redirect stub also deleted 2026-05-01)

Multiple TODO items (resolved with PR archival):
- Line 52: Mark PR as Ready for Review
- Line 65: Request Review
- Line 82: Unblock CI
- Line 123: Merge PR

**Context:**
- PR #98 merged and deployed
- Tracking documents no longer relevant; both copies archived

---

### 4.3 ADR-023 Manifest Implementation — ✅ AUDIT COMPLETE (2026-05-01)

**Status:** ✅ COMPLETED
**Priority:** P3 (Low)
**Effort:** 2 hours (actual)
**Owner:** Specialist

**Outcome:** Audit confirmed all 8 implementation TODOs in the guide are fully shipped in production. Phase 2 (`tools/performance_ledger.py` v1.7, 752 lines): `parse_manifests`, `extract_timings`, `compute_statistics`, and `main` (baseline + comparison modes) all complete. Phase 3 (`src/transformation_portal/lux_depth_v3/manifest.py` and `orchestrator.py`): `BackendSelectionMetadata` dataclass at `manifest.py:164–241` with `to_dict`/`from_dict` and schema versioning; `CombinedManifest` integration at `manifest.py:448` / deserialization at `manifest.py:518–519`; `_capture_backend_metadata` at `orchestrator.py:1279–1301`; truth-line logging at `orchestrator.py:5160–5167`. Phase 3 covered by `tests/test_backend_selection.py` (215 lines, 10+ cases including schema validation, serialization roundtrip, fallback handling, and backward compatibility).

The implementation guide itself (`docs/_archive/2026-Q1-consolidation/ADR-023-implementation-guide.md`) was a planning artifact in a frozen archive directory whose own header (lines 20–21) acknowledged the TODOs were stale. With implementation complete and the parent `ADR-023-post-pr841-hardening.md` (same archive directory) plus active-docs successors (`docs/architecture/performance_ledger_v1.7_review.md`, `docs/decisions/ARCHITECT_RESPONSE_POST_PR841.md`) covering the authoritative record, the obsolete guide was deleted rather than updated in place. The single intra-archive link in `ADR-023-v1.7-amendment.md` was repointed to the parent hardening ADR. Tracked separately as QW-4 in `docs/deliverables/QUICK_WINS.md`.

---

### 4.4 V2.0.0 Release Checklist

**Status:** 🟢 **SUPERSEDED BY v2.2.0 REALITY**
**Priority:** ~~P0~~ → RETROSPECTIVE UPDATE REQUIRED
**Action:** Update to reflect actual v2.0.0 → v2.2.0 state

**Location:** `docs/architecture/V2_0_0_RELEASE_REVIEW.md`

**Original P0 Items vs Reality:**

| Item (2026-02-05 Checklist) | Actual Status (2026-02-13) | Evidence |
|------|--------|---------|
| Add code coverage to CI | ✅ **COMPLETED** | `ci.yml:424-438` dual-gate (25% floor + 80% diff) |
| Enforce 70% coverage threshold | 🟡 **PARTIAL** (25% floor, 80% diff ratchet) | Coverage improvement roadmap: docs/guides/coverage-improvement-plan.md |
| Add security scan to PR workflow | ✅ **COMPLETED** | Bandit, pip-audit, gitleaks, CodeQL, Safety (ci.yml + security-unified.yml) |
| Configure branch protection | ❓ **UNKNOWN** (repo setting, not in code) | Requires GitHub UI inspection |
| Document rollback procedures | ✅ **COMPLETED** | docs/operations/ROLLBACK_PROCEDURES.md (v1.0.0) |
| Define staging validation | ❌ **NOT DONE** | No docker-compose.staging.yml or staging workflow |

**Current Production Release:** v2.2.0-phase3-l1-foundation (2026-02-13)

**Architect Assessment:**
- **v2.0.0 was released WITHOUT all P0 items** (pragmatic decision)
- Critical items (coverage, security) were completed post-v2.0.0
- v2.2.0 is in **SIGNIFICANTLY BETTER** state than original v2.0.0 checklist
- Rollback procedures now fully documented (operational maturity achieved)
- Remaining gaps (staging) are **operational/optional**, not blocking

**Architect Decision:**

**ACCEPT current state as v2.2.0 baseline** with conditions:

1. **Coverage Goal:** Continue incremental improvement to 33% by Q2 2026 (roadmap exists)
2. ~~**Rollback Procedures:**~~ ✅ DONE - docs/operations/ROLLBACK_PROCEDURES.md exists (v1.0.0, 2025-01-28)
3. **Staging Environment:** Defer to v2.3.0 or v3.0.0 (P2, 16h effort, low ROI for current users)
4. **Branch Protection:** Verify via GitHub UI (Architect or admin to check)

**Updated Checklist (for v2.3.0 planning):**

- ✅ Code coverage CI enforcement (DONE)
- ✅ Security scanning (DONE)
- ✅ Contract/schema validation (DONE - ingest_contract_validation.yml)
- ✅ Nightly regression suite (DONE - nightly.yml)
- ✅ Rollback procedures documentation (DONE - docs/operations/ROLLBACK_PROCEDURES.md)
- ❌ Staging environment (DEFERRED to v2.3.0)
- ❓ Branch protection verification (ACTION REQUIRED)

**Action Items:**
- [x] Create rollback procedures doc (P1, 2h) → ✅ Already exists
- [ ] Verify branch protection settings (P1, 15min)
- [ ] Update V2_0_0_RELEASE_REVIEW.md with "SUPERSEDED BY v2.2.0 REALITY" notice
- [ ] Create V2_3_0_RELEASE_CHECKLIST.md for next cycle

**Effort:** ~~3 hours total~~ 1 hour remaining (rollback docs done)
**Deadline:** Before v2.3.0 planning (est. March 2026)

---

### 4.5 Repository Health Report Placeholders

**Status:** 🟢 OBSERVATIONAL
**Priority:** P4 (Low)
**Action:** None (status markers, not TODOs)

**Location:** `docs/historical/architecture/REPOSITORY_HEALTH_REPORT_2026-02-03.md:143`

```markdown
#### ⚠️ **Placeholder/TODO Markers**

- **Findings:** TODOs are legitimate placeholders for optional/future work:
```

**Context:**
- Document acknowledges TODOs exist
- Assesses them as "legitimate placeholders"
- Not actionable items, status assessment

**Recommendation:** No action (correct analysis)

---

## 5. CI/CD Gaps

**Updated Assessment (2026-02-13):** ✅ **MAJORITY IMPLEMENTED**

Original inventory (2026-02-05) identified 6 gaps. Current reality:
- ✅ **4 gaps CLOSED** (coverage, security, contract validation, nightly regression)
- ❌ **2 gaps REMAIN** (dependency pinning validation, ADR consistency checks)
- 🟡 **1 gap PARTIAL** (CLI test suite exists but incomplete)

---

### 5.1 Code Coverage Enforcement

**Status:** ✅ **IMPLEMENTED**
**Priority:** ~~P0~~ → DONE
**Effort:** 4 hours (actual: ~6 hours based on commit history)
**Completion Date:** 2026-02-01 (PR #832)

**Implementation Details:**
- **Dual-gate coverage system** (ci.yml:424-438):
  1. **Floor gate:** 25% combined coverage (prevents regression)
  2. **Diff gate:** 80% coverage on changed lines (ratcheting mechanism)
- **Combined coverage:** Core tests + ML tests merged
- **Artifacts:** HTML reports uploaded for PR review
- **Improvement roadmap:** docs/guides/coverage-improvement-plan.md (target: 33% by Q2 2026)

**Evidence:**
```yaml
# .github/workflows/ci.yml:424-438
- name: Generate combined coverage report
  run: |
    coverage combine .coverage.*
    coverage report --fail-under=25
    coverage html
- name: Diff coverage check (80% on changes)
  run: |
    diff-cover coverage.xml --fail-under=80
```

**Architect Assessment:** ✅ VERIFIED - Exceeds original specification with dual-gate approach

---

### 5.2 Security Scanning in PR Workflow

**Status:** ✅ **IMPLEMENTED**
**Priority:** ~~P0~~ → DONE
**Effort:** 4 hours (actual: ~8 hours based on commit history)
**Completion Date:** 2026-01-28 (multiple PRs)

**Implementation Details:**
- **Bandit:** Security linting (ci.yml:127-132)
- **pip-audit:** Dependency vulnerabilities (blocks HIGH/CRITICAL)
- **gitleaks:** Secret detection (pre-commit + CI)
- **CodeQL:** Static analysis (security-unified.yml)
- **Safety:** Dependency security scan with policy file

**Quality Gate Integration:**
- All security scans aggregate in `quality-summary` job
- PR merge blocked on security failures
- Badge in README showing security status

**Evidence:**
```yaml
# .github/workflows/ci.yml:127-132
- name: Bandit security scan
  run: bandit -r src/ -ll -ii -f json -o bandit-report.json
```

**Architect Assessment:** ✅ VERIFIED - Comprehensive multi-tool approach

---

### 5.3 Branch Protection Rules

**Status:** ❓ **VERIFICATION REQUIRED**
**Priority:** P1 (High)
**Effort:** 15 minutes (verification only, no code changes)
**Action:** Repository admin must verify via GitHub UI

**Expected Configuration:**
- Require PR before merge: ❓
- Require at least 1 approval: ❓
- Require status checks to pass (build, lint, tests): ❓
- Restrict force pushes: ❓
- Restrict deletions: ❓

**Verification Method:**
```
GitHub UI → Settings → Branches → Branch protection rules → main
```

**Architect Action Required:**
- [ ] Verify branch protection settings exist
- [ ] Document current settings in CONTRIBUTING.md
- [ ] If missing: Configure per specification
- [ ] Update this inventory with confirmed status

**Note:** Cannot be verified from code inspection (repository metadata)

---

### 5.4 CLI Test Suite

**Status:** 🟡 **PARTIAL IMPLEMENTATION**
**Priority:** P2 (Medium)
**Effort:** 4 hours remaining (8h original - 4h completed)
**Current Coverage:** Basic CLI parsing tested, end-to-end integration incomplete

**Completed:**
- ✅ Unit tests for CLI argument parsing (tests/test_cli_*.py)
- ✅ Help text validation
- ✅ Basic invocation smoke tests

**Missing:**
- ❌ End-to-end CLI integration tests with actual image processing
- ❌ Exit code validation for error cases
- ❌ Output validation (files created, manifests, logs)
- ❌ Preset selection end-to-end tests
- ❌ Backend selection CLI tests (--depth-backend flag)

**Implementation Plan:**
- [ ] Create `tests/cli/test_e2e_lux_depth_v3.py`
- [ ] Test full pipeline invocation: `lux-depth-v3 input/ output/ --preset standard`
- [ ] Validate output structure matches contract
- [ ] Test error cases: missing input, invalid preset, invalid backend
- [ ] Test backend selection: `--depth-backend depth_pro` vs `--depth-backend da3`

**Files to Create:**
- `tests/cli/test_e2e_lux_depth_v3.py` (new)
- `tests/cli/test_backend_selection_cli.py` (new)
- `tests/cli/fixtures/` (test images)

**Architect Recommendation:** Schedule for v2.3.0 (P2 priority, 4h effort)

---

### 5.5 Staging Environment

**Status:** ❌ **NOT IMPLEMENTED**
**Priority:** P2 (Medium - LOW ROI for current user base)
**Effort:** 16 hours
**Recommendation:** DEFER to v3.0.0 or until deployment demand increases

**Current State:**
- No staging environment
- No containerized deployment automation
- `Dockerfile` exists but not used in CI/CD
- `docker-compose.yml` exists but basic

**Rationale for Deferral:**
- **User base:** Primarily researchers/developers running locally
- **Deployment model:** Not a SaaS/hosted service
- **Risk:** Low (direct production deployment not applicable)
- **Cost/benefit:** 16h effort for minimal gain

**Architect Decision:** **DEFER** unless deployment model changes

**If reconsidered (v3.0.0):**
- [ ] Define staging environment requirements
- [ ] Create `docker-compose.staging.yml`
- [ ] Add staging deployment workflow
- [ ] Document validation procedures
- [ ] Estimate cloud hosting costs (if applicable)

**Alternative:** Focus on better local testing/validation (higher ROI)

---

### 5.6 Rollback Procedures Documentation

**Status:** ✅ **IMPLEMENTED**
**Priority:** ~~P1~~ → DONE
**Effort:** 2 hours (actual)
**Completion Date:** 2025-01-28

**Current State:**
- ✅ Comprehensive rollback procedures documented
- ✅ Git tag reversion process defined (Section 2)
- ✅ Dependency pinning rollback covered (Section 3)
- ✅ Communication templates included (Section 5)
- ✅ Post-rollback checklist provided (Section 6)

**Implementation:**
- File: `docs/operations/ROLLBACK_PROCEDURES.md`
- Version: 1.0.0, Production-Ready
- Authority: Transformation Portal Architect

**Coverage:**
- [x] Git tag reversion: How to roll back to previous version
- [x] PyPI package handling: Documented as N/A (not published)
- [x] Dependency pinning rollback: constraints.txt management
- [x] Data migration rollback: Documented as N/A (stateless tool)
- [x] Incident response: Integrated with post-rollback checklist

**Architect Assessment:** ✅ VERIFIED COMPLETE - Exceeds original specification with comprehensive coverage

---

### 5.7 NEW: Dependency Pinning Validation

**Status:** ✅ **IMPLEMENTED** (script + Make target landed 2026-05-03; dedicated CI workflow landed 2026-06-11)
**Priority:** P2 (Medium)
**Effort:** 4 hours (script: ~1h done; remaining: dedicated workflow)
**Impact:** Supply chain security, reproducibility

**Current State:**
- `dependency-submission.yml` submits dependencies to GitHub
- ✅ `scripts/validation/check_dependency_pinning.py` enforces that every
  requirement line in `requirements/*.txt` uses an exact `==` pin
  (`constraints.txt` is exempt — it intentionally uses `>=` for banned packages)
- ✅ Wired into `make ci` and exposed as `make check-dependency-pinning`
- ✅ Tests in `tests/validation/test_check_dependency_pinning.py`
- ✅ Standalone GitHub Actions workflow `dependency-pinning-check.yml` provides
  an isolated PR/push signal (in addition to the transitive `make ci` coverage)

**Proposed Implementation:**
- [x] Validate all `requirements/*.txt` use `==` pinning (not `>=`, `~=`)
  via `scripts/validation/check_dependency_pinning.py`
- [x] Wire into `make ci` so PR lanes catch drift
- [x] Create `.github/workflows/dependency-pinning-check.yml` for an isolated
  signal independent of the broader CI matrix
- [ ] Validate constraints.txt matches installed versions
- [ ] Check for floating transitive dependencies (separate analysis)

**Example Check (now codified in script):**
```python
# scripts/validation/check_dependency_pinning.py flags any line in
# requirements/*.txt that uses >=, <=, ~=, !=, >, < or has no version pin.
```

**Files Affected:**
- `scripts/validation/check_dependency_pinning.py` (new)
- `tests/validation/test_check_dependency_pinning.py` (new)
- `Makefile` (`check-dependency-pinning` target + wired into `ci`)
- `.github/workflows/dependency-pinning-check.yml` (landed 2026-06-11)

**Architect Recommendation:** Local enforcement landed for v2.2.x; the isolated
workflow now ships as a dedicated `dependency-pinning-check.yml` (chosen over
overloading an existing governance workflow so the supply-chain signal stays
focused and independently observable). The constraints-vs-installed audit and
floating-transitive analysis remain the open follow-ups under this item.

---

### 5.8 NEW: ADR Consistency Checks

**Status:** ❌ **NOT IMPLEMENTED**
**Priority:** P3 (Low - nice-to-have)
**Effort:** 6 hours
**Impact:** Architectural governance, documentation quality

**Current State:**
- No automated validation of ADR format
- No checks for ADR updates when architecture changes
- Manual governance only (Architect review)

**Proposed Implementation:**
- [ ] Create `.github/workflows/adr-consistency-check.yml`
- [ ] Validate ADR numbering (sequential, no duplicates)
- [ ] Check ADR template compliance (required sections)
- [ ] Detect architectural changes without ADR update:
  - Changes to `src/*/orchestrator.py` → Check for ADR update
  - New `src/*/backends/` → Require ADR
- [ ] Validate cross-references (ADR mentions existing ADRs correctly)

**Challenges:**
- Hard to detect "architectural significance" programmatically
- Risk of false positives (trivial changes triggering ADR requirement)
- May create bureaucratic overhead

**Architect Decision:** **DEFER** to v3.0.0 - Manual review sufficient for now

**Alternative:** Improve ADR template and CONTRIBUTING.md guidance instead

---

## 6. Test Infrastructure

### 6.1 Dependency-Conditional Test Skips

**Status:** ✅ CORRECT PATTERN
**Priority:** N/A
**Action:** None (working as intended)

**Pattern:**
```python
@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
@pytest.mark.skipif(not HAS_OPENCV, reason="opencv required")
```

**Locations:**
- `tests/smoke/test_ml_stack_smoke.py` (multiple)
- `tests/test_depth_writer.py`
- `tests/test_format_utils_enhancements.py`
- And many others

**Analysis:**
- All skips are conditional on optional dependencies
- No permanent `@pytest.mark.skip` without condition
- No `@pytest.mark.xfail` indicating incomplete work
- Pattern: `skipif(not DEPENDENCY_AVAILABLE, reason=...)`

**Recommendation:** No action (correct pattern for optional dependencies)

---

### 6.2 ML Test Isolation

**Status:** ✅ CORRECT PATTERN
**Priority:** N/A
**Action:** None (working as intended)

**Pattern:**
```python
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch required for ML smoke tests")
```

**Context:**
- ML tests marked with `ml` marker
- CI runs ML tests only on Python 3.11
- Prevents model downloads in all CI jobs
- TRANSFORMERS_OFFLINE=1 honored

**Recommendation:** No action (well-designed ML test isolation)

---

### 6.3 Golden Fixture Regression Tests

**Status:** 🟡 RECOMMENDED ENHANCEMENT
**Priority:** P3 (Medium)
**Effort:** 8 hours
**Impact:** Prevent visual regression

**Current State:**
- No golden fixture tests
- Output validation mostly structural (file exists, not corrupted)
- Visual quality regression possible

**Implementation:**
- [ ] Create `tests/fixtures/golden/` directory
- [ ] Generate golden outputs for key presets:
  - Standard depth estimation
  - Lux render with PBR
  - Material response output
- [ ] Write comparison tests (perceptual hash or SSIM)
- [ ] Allow tolerance for platform differences (CPU vs GPU)

**Files Affected:**
- `tests/fixtures/golden/` (new directory)
- `tests/regression/test_golden_outputs.py` (new)

**Reference:** IMPROVEMENT_OPPORTUNITIES.md:255 (TEST-005)

---

### 6.4 Property-Based Tests (Hypothesis)

**Status:** ✅ PARTIALLY IMPLEMENTED
**Priority:** P4 (Nice-to-have)
**Action:** None (sufficient coverage)

**Current State:**
- Hypothesis available and used in select tests
- Property-based tests for validation logic
- Not comprehensive but targeted

**Examples:**
- `tests/test_property_based_validation.py`

**Recommendation:** No action (appropriate level of property testing)

---

## 7. Obsolete Items

### 7.1 Obsolete Documentation Tracking

**Items for Archive/Removal:**

1. **PR #98 Action Items** — ✅ RESOLVED 2026-03-14
   - ~~`docs/pr_reports/PR98_ACTION_ITEMS.md`~~ → archived to `docs/_archive/2026-03-legacy-prs/`
   - ~~`docs/development/pr/PR98_ACTION_ITEMS.md`~~ → archived to `docs/_archive/2026-03-legacy-prs/`; redirect stub additionally deleted 2026-05-01
   - Reason: PR merged and deployed

2. **Binary Cleanup TODOs**
   - `docs/fixes/BINARY_FILE_BEST_PRACTICES.md` (lines 133, 140)
   - Reason: Cleanup completed per PRE_PUSH_AUDIT_REPORT.md

3. **Old Verification Reports** — ✅ RESOLVED 2026-05-01
   - ~~`docs/development/VERIFICATION_COMPLETE.md` (multiple PENDING markers)~~ — redirect stub deleted; canonical `docs/verification/VERIFICATION_COMPLETE.md` retained
   - Reason: Verification completed, report outdated

**Recommendation:**
- Move to `archive/docs/` for historical reference
- Update README to remove references
- Clean up cross-links

---

### 7.2 Superseded Technical Decisions

**Items for Documentation Update:**

1. **ADR-019 Deferral Decision**
   - `docs/architecture/decisions/ADR-019-IMPLEMENTATION-STATUS.md`
   - Status: Superseded by ADR-019-REVISED-DECISION.md
   - Action: Add prominent SUPERSEDED notice (already done)

2. **Depth Canonical Registry**
   - `src/transformation_portal/depth_canonical/`
   - Status: Likely superseded by `depth/backends/`
   - Action: Audit module, remove if unused

---

## Summary Statistics

**Updated for v2.0.0 (2026-02-13)**

### By Category

| Category | Total | ✅ Done | 🟡 Partial | 🟢 Correct | 🔴 Deferred | ❌ Missing | 📦 Obsolete |
|----------|-------|---------|------------|------------|-------------|-----------|------------|
| Architectural Decisions | 6 | 1 | 0 | 0 | 2 | 0 | 3 |
| Stub Implementations | 29 | 0 | 0 | 25 | 0 | 0 | 4 |
| Code TODOs | 11 | 0 | 0 | 4 | 0 | 0 | 7 |
| Documentation TODOs | 7 | 2 | 0 | 2 | 0 | 0 | 3 |
| CI/CD Gaps | 8 | 5 | 1 | 0 | 1 | 1 | 0 |
| Test Infrastructure | 4 | 2 | 1 | 1 | 0 | 0 | 0 |
| **TOTAL** | **65** | **10** | **2** | **32** | **3** | **1** | **17** |

**Change from v1.0.0:**
- Total items: 36 → 65 (+29, mostly new NotImplementedError instances from Phase 2)
- Completed: 8 → 10 (+2, ADR-019 + Rollback docs)
- Obsolete: 5 → 17 (+12, major cleanup identified)

### By Priority

| Priority | Count | v1.0.0 Count | Change | Examples |
|----------|-------|--------------|--------|----------|
| ~~P0~~ DONE | 3 | 5 | -2 | ADR-019, Coverage CI, Security scanning |
| P1 (High) | 2 | 2 | 0 | Branch protection, CLI e2e tests (rollback docs done) |
| P2 (Medium) | 6 | 6 | 0 | SAM2 auto mode, dependency pinning, staging |
| P3 (Low) | 8 | 8 | 0 | ADR checks, SLERP interpolation, RAW format |
| P4 (Nice-to-have) | 7 | 10 | -3 | Backend CLI commands, golden fixtures |
| P5 (Research) | 1 | 1 | 0 | Parallax occlusion mapping |
| N/A (Correct/Obsolete) | 38 | 4 | +34 | Most stubs are correct (25), obsolete items (12) |

### By Effort

| Effort Range | Count | v1.0.0 | Change | Cumulative Hours |
|--------------|-------|--------|--------|------------------|
| < 1 hour | 12 | 3 | +9 | ~8 hours |
| 1-4 hours | 17 | 5 | +12 | ~43 hours |
| 4-8 hours | 8 | 12 | -4 | ~48 hours |
| 1-2 days | 6 | 10 | -4 | ~80 hours |
| > 1 week | 4 | 6 | -2 | ~200 hours |
| N/A (Done/Obsolete) | 18 | 0 | +18 | 0 hours |

**Total Remaining Effort:** ~381 hours (was ~450 hours in v1.0.0)
**Effort Reduction:** 69 hours (15% reduction due to completions)

### By Status Detail

| Status | Count | Percentage | Action Required |
|--------|-------|------------|-----------------|
| ✅ Completed | 9 | 14% | None (celebrate!) |
| 🟢 Correct (No Action) | 32 | 49% | Document/validate only |
| 🟡 In Progress | 2 | 3% | Continue work |
| 🔴 Deferred (ADR'd) | 3 | 5% | Track, don't implement |
| ❌ Not Implemented | 2 | 3% | Schedule for v2.3.0 |
| 📦 Obsolete/Archive | 17 | 26% | Cleanup required |

**Key Insight:** 49% of "TODOs" are actually **correct as-is** (intentional stubs, phase gates, platform limitations)

### Health Metrics

**Completion Rate (since v1.0.0):**
- Items completed: 9 / 65 = **14% of inventory**
- High-priority completed: 3 / 3 P0 items = **100% of critical path**
- Technical debt removed: 17 obsolete items identified

**Risk Assessment:**
- **P0 Blockers:** 0 (all critical items resolved)
- **P1 Items:** 3 (rollback docs, branch protection, CLI tests)
- **Security Posture:** ✅ Strong (4/4 security items implemented)
- **Governance Posture:** ✅ Strong (coverage, nightly, contracts enforced)

**Architect Verdict:** **Repository health EXCELLENT** - 63% of items require no action (correct/completed/obsolete)

---

## Recommended Action Sequence

**Updated for v2.2.0 Reality (2026-02-13)**

---

### ✅ COMPLETED Since v1.0.0 (Celebrate!)

1. ✅ **ADR-019 Backend Integration** (2-3 days) - PR #906
2. ✅ **CI Coverage Enforcement** (4 hours) - PR #832
3. ✅ **Security Scanning** (4 hours) - ci.yml + security-unified.yml
4. ✅ **Contract Validation** (8 hours) - ingest_contract_validation.yml
5. ✅ **Nightly Regression Suite** (12 hours) - nightly.yml

**Total Delivered:** ~200 hours of critical infrastructure work

---

### 🔴 IMMEDIATE ACTIONS (Before v2.3.0 Planning)

**Deadline:** 2 weeks (by March 1, 2026)
**Owner:** Architect + DevOps
**Total Effort:** ~~4 hours~~ 2 hours remaining (rollback docs done)

1. ~~**Create Rollback Procedures** (P1, 2h)~~ ✅ DONE
   - File: `docs/operations/ROLLBACK_PROCEDURES.md` (v1.0.0, 2025-01-28)
   - Content: Git tag reversion, dependency rollback, incident response, communication templates
   - Status: Production-Ready

2. **Verify Branch Protection Settings** (P1, 15min)
   - Method: GitHub UI → Settings → Branches
   - Document in: `CONTRIBUTING.md`
   - Owner: Repository admin

3. **Update V2_0_0_RELEASE_REVIEW.md** (P1, 30min)
   - Add "SUPERSEDED BY v2.2.0 REALITY" notice
   - Link to this TODO inventory
   - Archive to `docs/architecture/archive/`

4. ~~**Archive Obsolete Modules** (P2, 1h)~~ ✅ DONE
   - ✅ `context_aware_rendering.py` → `archive/scripts/` (already moved)
   - ✅ `lux_render_pipeline_plus_v3.py` → `archive/scripts/` (already moved)
   - ✅ `depth_canonical/` → `archive/depth_canonical/` (already moved)
   - ✅ README explaining supersession exists (`archive/scripts/README.md`)

---

### 📋 Sprint 1: Repository Cleanup (Week 1, ~8 hours)

**Goal:** Remove technical debt, improve discoverability

1. ~~**Delete depth_canonical Module** (1h)~~ ✅ DONE
   - ✅ Moved to `archive/depth_canonical/`
   - ✅ No active imports in src/ (verified)
   - Update CHANGELOG.md with archival note

2. ~~**Archive Obsolete PR Tracking Docs** (30min)~~ ✅ DONE 2026-03-14
   - ✅ Moved `docs/pr_reports/PR98_ACTION_ITEMS.md` → `docs/_archive/2026-03-legacy-prs/`
   - ✅ Moved `docs/development/pr/PR98_ACTION_ITEMS.md` → `docs/_archive/2026-03-legacy-prs/`; redirect stub additionally deleted 2026-05-01

3. **Update Binary Cleanup Docs** (30min)
   - Remove TODOs from `docs/fixes/BINARY_FILE_BEST_PRACTICES.md`
   - Mark as complete or delete sections

4. **Complete CLI E2E Test Suite** (4h)
   - Create `tests/cli/test_e2e_lux_depth_v3.py`
   - Create `tests/cli/test_backend_selection_cli.py`
   - Test presets, backends, error cases
   - Add to CI workflow

5. **Audit ADR-023 Manifest Implementation** (2h) — ✅ COMPLETED 2026-05-01
   - Audit confirmed all 8 implementation TODOs shipped in production
   - Obsolete planning guide deleted from frozen archive
   - See Section 4.3 for full outcome

---

### 📋 Sprint 2: Phase 2 Foundation (Week 2-3, ~16 hours)

**Goal:** Enable Spatial AI Phase 2.1

6. **Implement SAM2 Auto Mask Generation** (P2, 3-4 days)
   - Location: `spatial_ai/segmentation/sam2_backend.py`
   - Remove NotImplementedError for auto mode
   - Add tests, documentation
   - **Blocker:** Requires SAM2 model weights + license review

7. **Implement Dependency Pinning Validation** (P2, 4h)
   - Create `.github/workflows/dependency-pinning-check.yml`
   - Validate all requirements.txt use `==` pinning
   - Fail on `>=` or `~=` (floating versions)

8. **Create V2_3_0_RELEASE_CHECKLIST.md** (P2, 2h)
   - Learn from v2.0.0 → v2.2.0 journey
   - Include rollback validation
   - Include branch protection verification
   - Add sample data upload checklist

---

### 📋 Sprint 3: Nice-to-Have (v2.3.0, ~12 hours)

**Goal:** Improve developer experience

9. **Implement Sample Data GitHub Release** (P2, 4h)
   - Upload demo_coastal_interior.jpg
   - Upload demo_pool_aerial.jpg
   - Upload sample_render_4k.tif
   - Generate SHA256 hashes
   - Update download_samples.py

10. **Add Golden Fixture Tests** (P3, 8h)
    - Create `tests/fixtures/golden/` directory
    - Generate baseline outputs for standard preset
    - Implement SSIM comparison tests
    - Allow tolerance for CPU/GPU differences

---

### 🔮 Future Sprints (v3.0.0+)

**Deferred (No Action Required for v2.x)**

- **Staging Environment** (P2, 16h) - LOW ROI for current user base
- **ADR Consistency Checks** (P3, 6h) - Manual governance sufficient
- **Backend CLI Commands** (P4, 1 week) - Nice-to-have, not demanded
- **Parallax Occlusion Mapping** (P5, 2-3 weeks) - Archive indefinitely

---

### 🚫 No Scanner-Governance Action Required

**Items That Are Correct As-Is:**

- **24 NotImplementedError instances** - Correct stubs (abstract methods, phase gates, platform limitations)
- **0 scanner-visible TODO comments** - Source/test/script/tool/web scanner scope is clean
- **2 Documentation TODOs** - Status markers, not tasks
- **2 Test patterns** - Dependency-conditional skips (correct pattern)
- **9 Completed items** - Celebrate and move on!

**Architect Directive:** Do not "fix" items marked 🟢 CORRECT - they are **working as intended**

---

## Effort Summary

| Phase | Effort | Items | Deadline |
|-------|--------|-------|----------|
| **Immediate Actions** | 4h | 4 | March 1, 2026 |
| **Sprint 1 (Cleanup)** | 8h | 5 | March 15, 2026 |
| **Sprint 2 (Phase 2)** | 16h | 3 | April 1, 2026 |
| **Sprint 3 (Nice-to-have)** | 12h | 2 | v2.3.0 release |
| **Total Scheduled** | **40h** | **14 items** | Q1 2026 |

**Remaining Backlog:** ~340 hours (deferred to v3.0.0 or later)

---

## Success Criteria (v2.3.0)

**Must Have:**
- ✅ Rollback procedures documented
- ✅ Branch protection verified
- ✅ depth_canonical module deleted
- ✅ CLI e2e tests passing
- ✅ Dependency pinning enforced

**Should Have:**
- ✅ SAM2 auto mode implemented
- ✅ Sample data available via GitHub Release
- ✅ Golden fixture tests baseline established

**Nice to Have:**
- ADR consistency checks (defer if time constrained)
- Staging environment (defer indefinitely)

---

---

## Critical Discrepancies & Architect Findings

**Assessment Date:** 2026-02-13
**Assessor:** Transformation Portal Architect

---

### 🔴 Critical Finding #1: V2.0.0 Release Checklist Mismatch

**Discrepancy:** Original v2.0.0 release checklist (dated 2026-02-05) marked 6 P0 items as "Must Complete Before Release" - but v2.0.0 was released **WITHOUT** completing all items.

**Timeline:**
- 2026-02-05: TODO inventory created, v2.0.0 checklist marked as P0 blockers
- 2026-02-07: v2.0.0 tag created (based on git tags)
- 2026-02-13: v2.2.0 is current production tag

**Reality:**
- Coverage enforcement: Implemented **after** v2.0.0 (PR #832)
- Security scanning: Implemented **after** v2.0.0
- Rollback docs: **Still not done**
- Staging environment: **Still not done**

**Architect Assessment:**
- **Pragmatic decision** to release without all P0 items (likely correct given user base)
- Demonstrates **checklist was aspirational**, not binding
- **Post-release hardening** successfully completed critical items

**Corrective Action:**
- ✅ Update TODO inventory to reflect reality (this document)
- ⏳ Complete remaining P1 items (rollback docs, branch protection verification)
- ⏳ Create V2_3_0_RELEASE_CHECKLIST.md with **validated** P0 criteria

**Severity:** MEDIUM (checklist accuracy issue, not technical debt)

---

### 🟡 Critical Finding #2: Inventory Scope Creep

**Discrepancy:** Original inventory (v1.0.0) tracked 36 items. Current reality: **65 items** (+80% growth in 8 days).

**Root Cause Analysis:**
- 18 new NotImplementedError instances added in Phase 2 (SAM2, materials-v3)
- Original inventory was **incomplete** (missed existing stubs)
- Phase 2/3 work introduced **intentional feature gates** (correct, not technical debt)

**Breakdown:**
- **True new work:** 7 items (SAM2 segmentation, materials v3 features)
- **Already existed (missed in v1.0.0):** 11 items (existing stubs, TODOs)
- **Completed items (removed from backlog):** 9 items (ADR-019, CI/CD work)

**Architect Assessment:**
- Growth is **healthy** (mostly intentional stubs for future features)
- 49% of items are **correct as-is** (no action required)
- **Inventory accuracy improved** (better discovery process)

**Corrective Action:**
- ✅ Categorize stubs as "CORRECT" vs "TODO" (this document)
- ⏳ Establish inventory update cadence: **Monthly** or **per-release**
- ⏳ Add to Definition of Done: "Update TODO inventory if NotImplementedError added"

**Severity:** LOW (process improvement opportunity)

---

### 🟢 Critical Finding #3: Obsolete Module Accumulation

**Discrepancy:** 3 obsolete modules/scripts identified with **no active usage**:
1. `depth_canonical/` (superseded by ADR-019 backends)
2. `context_aware_rendering.py` (superseded by Spatial AI orchestration)
3. `lux_render_pipeline_plus_v3.py` (abandoned, no commits in 5 months)

**Impact:**
- **Confusion:** Developers may waste time exploring dead code
- **Maintenance burden:** Tests may falsely pass on unused code
- **Documentation drift:** README/docs may reference obsolete features

**Architect Assessment:**
- **Normal technical debt** accumulation during rapid development
- **Cleanup required** before v2.3.0 release
- **Good discovery** process (inventory caught this)

**Corrective Action:**
- ⏳ Delete `depth_canonical/` module (1h, Sprint 1)
- ⏳ Archive `context_aware_rendering.py` and `lux_render_pipeline_plus_v3.py` (30min)
- ⏳ Update CHANGELOG.md with removal notices
- ⏳ Add to Definition of Done: "Archive superseded modules within 1 release cycle"

**Severity:** LOW (cleanup, not blocker)

---

### ✅ Positive Finding #4: CI/CD Maturity Acceleration

**Finding:** Original inventory identified **6 CI/CD gaps** (all marked P0/P1). Current reality: **4 of 6 implemented** within 8 days.

**Completed:**
- ✅ Coverage enforcement (dual-gate: floor + diff)
- ✅ Security scanning (5 tools: Bandit, pip-audit, gitleaks, CodeQL, Safety)
- ✅ Contract validation (ingest_contract_validation.yml)
- ✅ Nightly regression suite (benchmarks, stress tests, memory leak detection)

**Remaining:**
- ⏳ Dependency pinning validation (P2, defer to v2.3.0)
- ⏳ ADR consistency checks (P3, defer indefinitely)

**Architect Assessment:**
- **Exceptional progress** on critical infrastructure
- **Repository health improved significantly** since v2.0.0
- **Quality gates operational** and enforced in CI

**Recognition:** Commend team for rapid CI/CD hardening

**Severity:** N/A (positive finding)

---

### 🟡 Critical Finding #5: Sample Data URLs Unchanged Since 2026-02-05

**Discrepancy:** `scripts/download_samples.py` still contains `url: None # TODO` for 3 sample files:
- demo_coastal_interior.jpg
- demo_pool_aerial.jpg
- sample_render_4k.tif

**Impact:**
- **User experience:** New users cannot easily download sample data
- **Onboarding friction:** Manual sample creation required
- **Documentation quality:** README references unavailable samples

**Architect Assessment:**
- **Low priority** (workaround exists: users can create own samples)
- **High value** for user experience (4h effort, big UX win)
- **Release blocker?** No (defer to v2.3.0)

**Corrective Action:**
- ⏳ Create GitHub Release: `v2.2.0-samples`
- ⏳ Upload sample files with SHA256 hashes
- ⏳ Update download_samples.py URLs
- ⏳ Test end-to-end download workflow

**Deadline:** Before v2.3.0 release (March 2026)

**Severity:** LOW (UX improvement, not blocker)

---

## Governance Recommendations

Based on discrepancies identified, Architect recommends:

### 1. **Definition of Done Updates**

Add to CONTRIBUTING.md Definition of Done:
- [ ] "If NotImplementedError added, update TODO_INVENTORY.md and categorize as feature gate or technical debt"
- [ ] "If module superseded, archive to `archive/` within 1 release cycle"
- [ ] "Release checklists must be validated against actual blockers (not aspirational)"

### 2. **Inventory Maintenance Cadence**

Establish regular inventory review:
- **Frequency:** Monthly or per-release (whichever is more frequent)
- **Owner:** Architect (review) + Specialist (update)
- **Scope:** Validate status, identify new items, archive completed items
- **Duration:** 1 hour per review

### 3. **Release Checklist Governance**

Improve release checklist process:
- **Mandatory:** All P0 items must be **actually required** for release (not wishlist)
- **Validation:** Run checklist against current state 1 week before release
- **Retrospective:** Post-release review of what was actually blocking vs nice-to-have

### 4. **Stub Classification Standard**

Standardize NotImplementedError messages:
```python
# CORRECT: Feature gate (intentional)
raise NotImplementedError("RAW format is a Phase II feature (ADR-027)")

# CORRECT: Abstract method
@abstractmethod
def execute(self):
    raise NotImplementedError("Subclass must implement")

# CORRECT: Platform limitation
raise NotImplementedError("Windows not supported (Unix-only security features)")

# TECHNICAL DEBT: Missing implementation
raise NotImplementedError("Single-view reconstruction not yet implemented (TODO_INVENTORY.md)")
```

**Rule:** All new NotImplementedError must include context (feature gate/abstract/limitation/debt)

---

**Document Control:**
- **Maintained By:** Transformation Portal Architect
- **Update Frequency:** After each sprint or major milestone
- **Issue Tracking:** Link to GitHub Issues/Projects when created
- **Review Date:** Next review at v2.1.0 planning phase

## References

### Architecture & Decision Documents
- **ADR-019-REVISED-DECISION.md** - Backend integration approval (IMPLEMENTED)
- ~~**ADR-023-implementation-guide.md**~~ - Deleted 2026-05-01 (audit found all TODOs shipped; see §4.3)
- **ADR-026-APEX-governance-framework.md** - APEX Research Ultra governance
- **ADR-027-phase2-spatial-ai-extension.md** - Spatial AI Phase 2 roadmap
- **ADR-029-execution-graph-abstraction.md** - Phase 3 execution graphs
- **agent_governance.md** - Architect/Specialist governance policy

### Status & Planning Documents
- **V2_0_0_RELEASE_REVIEW.md** - Original release checklist (SUPERSEDED)
- **PHASE2_PR_SUMMARY.md** - Phase 2 implementation status
- **PHASE3_L1_IMPLEMENTATION_SUMMARY.md** - Phase 3 L1 cache work
- **REPOSITORY_HEALTH_REPORT_2026-02-03.md** - Repository health snapshot
- **coverage-improvement-plan.md** - Coverage roadmap to 33%

### Improvement Tracking
- **IMPROVEMENT_OPPORTUNITIES.md** - Existing improvement tracking (21 items)
- **QUICK_WINS.md** - Tasks < 4 hours (if exists)
- **OUTSTANDING_WORK_SUMMARY.md** - Executive summary (if exists)

### Technical Documentation
- **CLI_REFERENCE.md** - CLI documentation (backend selection, presets)
- **CHANGELOG.md** - Version history and feature tracking
- **CONTRIBUTING.md** - Development workflow and standards

### Workflow & CI/CD
- `.github/workflows/ci.yml` - Main CI pipeline (coverage, security, tests)
- `.github/workflows/nightly.yml` - Nightly regression suite
- `.github/workflows/ingest_contract_validation.yml` - Schema validation
- `.github/workflows/security-unified.yml` - Security scanning orchestration

### Dependency Management
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies
- `requirements-ci.txt` - CI-specific dependencies
- `requirements-lint.txt` - Linting and security tools

---

## Appendix A: NotImplementedError Audit Trail

**Methodology:** `grep -r "raise NotImplementedError" src/ tests/` (2026-02-13)

**Total Count:** 29 instances
- **Abstract Methods:** 5 (ComfyUI BaseNode, depth processors, scene builder)
- **Platform Limitations:** 2 (Windows security utils)
- **Phase Gates:** 8 (SAM2 modes, RAW format, Phase II preprocessing)
- **Research Features:** 2 (NVDIFFREC, MaterialGAN)
- **Legacy/Superseded:** 1 (depth_canonical registry)
- **Correct Error Handling:** 1 (depth_anything_v2 unknown backend)
- **Tests (validation):** 8 (verify NotImplementedError raised correctly)
- **Abandoned Features:** 1 (Parallax Occlusion Mapping)
- **Interpolation Limitation:** 1 (SLERP not implemented, linear only)

**Categorization:**
- ✅ **Correct as-is:** 25 (86%)
- ❌ **Technical debt:** 3 (10%)
- 🗑️ **Obsolete/archive:** 1 (3%)

---

## Appendix B: TODO Comment Audit Trail

**Methodology:** `python scripts/validation/scan_todo_inventory.py --write-snapshot` (2026-05-11)

**Total Count:** 0 scanner-visible TODO comments

**Source Code (`src/`):** 0 TODO comments ✅
- All source code TODOs have been cleaned up or addressed

**Tests (`tests/`):** 0 scanner-visible TODO comments ✅
- Historical performance notes in `tests/stress/test_stress_large_batch.py` were replaced by direct tolerance assertions.
- Test fixture text that resembles TODO prose is not a scanner-visible TODO comment.

**NotImplementedError Count:** 24 instances across `src/`, `tests/`, `scripts/`, `tools/` (all governed)

The canonical, line-accurate list is now machine-generated by the scanner:
```bash
python scripts/validation/scan_todo_inventory.py --json
python scripts/validation/scan_todo_inventory.py --write-snapshot
```
CI runs this on every PR (`.github/workflows/enforcement.yml`, `hf-revision-policy` job), so drift is caught at PR time rather than at the next manual review.

**Categorical breakdown** (as of 2026-05-11 scan):
- **Abstract methods (ABC pattern):** ~6 instances — `comfyui/custom_nodes.py` BaseNode, `execution_graph/nodes/base.py`, `depth/processors/base.py`
- **Phase gates / feature stubs:** ~4 instances — single-view reconstruction (`spatial_ai/orchestration/pipeline.py`), non-linear scene interpolation guard, unknown-backend guard, etc.
- **Platform limits:** 1 — `utils/security.py` Unix-only timeout
- **Technical debt (deferred):** ~2 — single-view reconstruction, gaussian_backend SHA verification
- **Test protocol stubs:** ~10 — protocol validation patterns under `tests/`
- **Scripts / one-shots:** ~3 — `scripts/apex_verify_contract.py`, `scripts/pipelines/unified_meta_pipeline.py`, `scripts/utilities/depth_anything_v2.py`

The previous v2.4.0 count of "12" tracked only `src/` and contained internal arithmetic errors (5+4+1≠12). The current 24-item scanner baseline is the full scanner scope and is the figure CI enforces against.

**Priority Recommendation:**
- No source code TODOs requiring action
- Scanner-visible test TODO comments are 0
- All NotImplementedError instances are governed; no governance violations as of 2026-05-11

---

## Appendix C: Obsolete Item Cleanup Manifest

**Items Identified for Removal/Archive:**

| Item | Type | Location | Action | Effort | Status |
|------|------|----------|--------|--------|--------|
| depth_canonical module | Module | ~~`src/transformation_portal/depth_canonical/`~~ | ~~DELETE~~ | ~~1h~~ | ✅ **DELETED** (verified 2026-03-25) |
| context_aware_rendering.py | Script | ~~`scripts/context_aware_rendering.py`~~ | ~~ARCHIVE~~ | ~~30min~~ | ✅ **ARCHIVED** 2026-02-10 |
| premium_context_pipeline.py | Script | ~~`scripts/premium_context_pipeline.py`~~ | ~~ARCHIVE~~ | ~~20min~~ | ✅ **ARCHIVED** 2026-02-10 |
| lux_render_pipeline_plus_v3.py | Script | ~~`scripts/pipelines/lux_render_pipeline_plus_v3.py`~~ | ~~ARCHIVE~~ | ~~30min~~ | ✅ **ARCHIVED** 2026-02-10 |
| PR #98 action items | Docs | ~~`docs/pr_reports/PR98_ACTION_ITEMS.md`~~ | ~~ARCHIVE~~ | ~~15min~~ | ✅ **ARCHIVED** (verified 2026-03-25) |
| PR #98 action items (dup) | Docs | ~~`docs/development/pr/PR98_ACTION_ITEMS.md`~~ | ~~ARCHIVE~~ | ~~15min~~ | ✅ **ARCHIVED** (verified 2026-03-25) |
| Binary cleanup TODOs | Docs | `docs/fixes/BINARY_FILE_BEST_PRACTICES.md` | N/A | 0min | ✅ **NO ACTION NEEDED** (action items, not code TODOs) |
| V2.0.0 release checklist | Docs | ~~`docs/architecture/V2_0_0_RELEASE_REVIEW.md`~~ | ~~UPDATE~~ | ~~30min~~ | ✅ **UPDATED** 2026-02-10 |

**Total Cleanup Effort:** ~3.5 hours
**Completed Effort:** 3.5 hours (all items verified complete)
**Remaining Effort:** 0 hours
**Recommended Sprint:** ✅ **COMPLETE**

**Completed Actions (2026-03-25 verification):**
- ✅ depth_canonical module: Directory does not exist in src/ (already deleted/archived)
- ✅ PR #98 docs: Verified archived to `archive/docs/pr_reports/` and `docs/_archive/2026-03-legacy-prs/`
- ✅ Binary cleanup TODOs: Not actual code TODOs, just action item descriptions in documentation

**Completed Actions (2026-02-10):**
- ✅ Archived 3 obsolete scripts with git history preservation (`git mv`)
- ✅ Created comprehensive `archive/scripts/README.md` documenting rationale
- ✅ Updated V2_0_0_RELEASE_REVIEW.md with SUPERSEDED notice
- ✅ Updated TODO_INVENTORY.md to mark items as archived
3. Add README.md in archive explaining why items were archived
4. Update CHANGELOG.md with removal notices
5. Search for references in active docs and remove/update
6. Verify no broken links with `markdown-link-check`

---

## 2026-05-04 Refresh — Monolithic-File TODO Audit

**Trigger:** Branch `claude/fix-monolithic-file-todos-63mEO` review.
**Scanner snapshot:** [`todo_scanner_snapshot.json`](todo_scanner_snapshot.json) (committed alongside this entry; reproducible via `python scripts/validation/scan_todo_inventory.py --json`).

**Findings:**
- `python scripts/validation/scan_todo_inventory.py --check-governance` → exit 0; **0 ungoverned TODOs** across 1,492 files scanned.
- **25** `NotImplementedError` instances, all governed (abstract methods, phase gates, platform limits) — unchanged from the 2026-05-01 baseline in `TODO_INVENTORY_QUICK_REF.md`.
- **Zero** `TODO|FIXME|XXX|HACK` markers in the five largest source modules: `app.py` (10,039 LOC), `lux_depth_v3/orchestrator.py` (7,257), `lux_depth_v3/segmentation_backend.py` (2,519), `pipelines/rendering_4k_pipeline.py` (2,380), `spatial_ai/orchestration/pipeline.py` (2,121 — only `TODO_INVENTORY.md` reference at line 1774, a governed phase gate for single-view reconstruction).
- ADR-043 orchestrator decomposition is complete; the open guidance is residual slimming, captured in `DEVELOPMENT_ROADMAP_2026_Q2.md` lines 174–210.

**Action taken:** No source changes. Authored a ranked seam target list at [`../architecture/MONOLITH_DECOMPOSITION_TARGETS.md`](../architecture/MONOLITH_DECOMPOSITION_TARGETS.md) and an ADR stub at [`../architecture/ADR-045-monolith-decomposition-residuals.md`](../architecture/ADR-045-monolith-decomposition-residuals.md) (status: Proposed). Future extraction PRs draw from that target list under the ADR-043 phased pattern.

**Diff vs prior baseline:** none — counts and governance posture identical to the 2026-05-01 review. This refresh replaces the manual narrative drift control with a committed JSON snapshot.

---

## 2026-05-11 Refresh — Scanner Snapshot And Schema Alignment

**Trigger:** Post-#1722 repo-wide documentation baseline refresh.
**Scanner snapshot:** [`todo_scanner_snapshot.json`](todo_scanner_snapshot.json) regenerated with `python scripts/validation/scan_todo_inventory.py --write-snapshot`.

**Findings:**
- `python scripts/validation/scan_todo_inventory.py --check-governance` → exit 0; **0 ungoverned TODOs** across 1,570 files scanned.
- **24** `NotImplementedError` instances, all governed (abstract methods, phase gates, platform limits, or test protocol stubs).
- **Zero** `TODO|FIXME|XXX|HACK` comment markers in scanner scope.
- The 25 → 24 delta comes from removal of the previously scanned `scripts/pipelines/tiff_enhancement_pipeline.py`; no new backlog item was opened or closed by this refresh.
- The scanner now owns snapshot generation through `--write-snapshot`, avoiding ad hoc shell redirection for the committed JSON baseline.

**Action taken:** Refreshed this inventory, the action plan, quick reference, scanner snapshot, and TODO priority schema so all current counts, commands, and governance language point at the same live scanner contract.

**Diff vs prior baseline:** scanner scope grew from 1,492 to 1,570 files because repo content expanded; governed item count decreased by one because a previously scanned script file is absent from the tracked tree. Governance posture remains unchanged: 0 ungoverned TODOs.

---

**END OF TODO INVENTORY v2.4.7**
