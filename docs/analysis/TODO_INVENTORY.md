# TODO Inventory - Transformation Portal

**Document Version:** 1.0.0  
**Date:** February 5, 2026  
**Last Updated:** 2026-02-05  

---

## Purpose

This document provides a **complete, categorized inventory** of all TODOs, NotImplementedError stubs, pending integrations, and deferred work across the Transformation Portal codebase.

**Usage:**
- Reference for sprint planning
- Audit trail for architectural decisions
- Integration with issue tracking systems

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

**Status:** ✅ APPROVED FOR IMMEDIATE IMPLEMENTATION  
**Priority:** P0 (Critical)  
**Effort:** 2-3 dev days  
**Owner:** Specialist (Architect-supervised)  

**Context:**
- Original decision: Defer to v2.1.0 (Depth Pro not operational)
- New information: Depth Pro checkpoint verified at `checkpoints/depth_pro.pt`
- Revised decision: Immediate integration approved

**Scope:**
- [ ] Create `DA3Backend` adapter wrapping `DA3InferenceEngine`
- [ ] Integrate `DepthBackendRegistry` into orchestrator
- [ ] Implement backend selection: `config.depth_backend or "depth_anything_v3"`
- [ ] Add fallback policy: `depth_pro → da3`
- [ ] Update metadata capture for backend tracking
- [ ] Write integration tests (orchestrator + both backends)
- [ ] Update CLI documentation (`--depth-backend` flag)

**Files Affected:**
- `src/transformation_portal/depth/backends/depth_anything_v3.py` (new)
- `src/transformation_portal/lux_depth_v3/orchestrator.py` (modify)
- `tests/integration/test_orchestrator_backend_selection.py` (new)
- `docs/CLI_REFERENCE.md` (update)

**References:**
- ADR-019-REVISED-DECISION.md
- ADR-019-INTEGRATION-APPROVAL.md
- ADR-019-IMPLEMENTATION-STATUS.md (superseded)

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
- Mentioned in: `docs/IMPROVEMENT_OPPORTUNITIES.md:315`

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

**Status:** 🔴 DEFERRED TO v3.x (EPIC)  
**Priority:** P5 (Research Required)  
**Effort:** 2-3 weeks  
**Impact:** Advanced rendering feature, unclear value proposition  

**Current State:**
- Stub exists in `scripts/pipelines/lux_render_pipeline_plus_v3.py:208`
- Raises `NotImplementedError` with clear message

**Location:**
```python
# scripts/pipelines/lux_render_pipeline_plus_v3.py:208
raise NotImplementedError(
    "Parallax Occlusion Mapping (POM) is not yet implemented in this pipeline. "
    "This is an advanced feature requiring additional research and validation."
)
```

**Scope (if pursued):**
- Research: Algorithm selection (POM vs relief mapping vs parallax mapping)
- Prototype: Shader implementation or Python approximation
- Validation: Real-world architectural render testing
- Integration: Lux render pipeline hooks
- Documentation: Theory, usage, limitations

**Recommendation:** Keep as `NotImplementedError` until user demand demonstrated.

---

## 2. Stub Implementations

### 2.1 Lux Depth V3 Preprocessing (2 functions)

**Status:** 🟢 LEGITIMATE STUBS  
**Priority:** P4 (Low)  
**Action:** Document as optional, remove if never implemented  

**Location:** `src/transformation_portal/lux_depth_v3/preprocessing.py`

#### 2.1.1 `normalize_exif_orientation()`

**Line:** 227  
**Purpose:** EXIF orientation normalization  
**Reason:** Not required for V3 depth inference (PIL handles automatically)  

```python
raise NotImplementedError(
    "normalize_exif_orientation() is a stub - full implementation pending. "
    "This module was created to enable package imports."
)
```

**Decision:**
- [ ] Option A: Implement if EXIF issues reported (2h effort)
- [ ] Option B: Remove stub and function signature (30m cleanup)
- [ ] Option C: Keep as documented limitation (status quo)

**Recommendation:** Option C (keep stub, document as "not required for current workflow")

#### 2.1.2 `validate_depth_image_alignment()`

**Line:** 248  
**Purpose:** Validate depth map and image dimension matching  
**Reason:** Not required for V3 (orchestrator handles validation)  

```python
raise NotImplementedError(
    "validate_depth_image_alignment() is a stub - full implementation pending. "
    "This module was created to enable package imports."
)
```

**Decision:** Same as 2.1.1 (keep as documented limitation)

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
- [ ] Add `@abstractmethod` decorators for clarity
- [ ] Document ComfyUI integration as experimental
- [ ] Create example subclass in `examples/comfyui/`
- [ ] Defer full integration to plugin system (v2.2.0)

**Recommendation:** Add documentation, keep stubs (correct pattern for abstract base class)

---

### 2.3 Depth Canonical Model Registry

**Status:** 🟢 SUPERSEDED (REMOVE)  
**Priority:** P3 (Cleanup)  
**Action:** Remove or consolidate with `depth/backends/registry.py`  

**Location:** `src/transformation_portal/depth_canonical/models/registry.py:31`

```python
raise NotImplementedError
```

**Context:**
- Older registry pattern from depth_canonical module
- Superseded by `src/transformation_portal/depth/backends/registry.py`
- `depth_canonical` module appears to be legacy/experimental

**Decision:**
- [ ] Audit `depth_canonical` module usage
- [ ] If unused: Remove entire module (1h effort)
- [ ] If used: Migrate to new backend registry pattern (4h effort)
- [ ] Update imports and tests

**Recommendation:** Audit + remove (likely unused given ADR-019 backend registry exists)

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

**Status:** 🟡 CLARIFICATION NEEDED  
**Priority:** P3 (Medium)  
**Effort:** 2h (decision) or 8h (integration)  
**Owner:** Architect (decision), Specialist (implementation)  

**Location:** `scripts/context_aware_rendering.py:354`

```python
# TODO: Integrate with actual processing pipelines
# This would call:
# 1. depth_pipeline with depth_config
# 2. material_response with material_config
# 3. luxury_tiff_batch_processor with color_config
```

**Context:**
- Script generates context-aware rendering strategies
- Currently standalone demonstration/POC
- Integration path unclear

**Decision Required:**
- [ ] Option A: Keep as demo script (move to `examples/`, document as POC)
- [ ] Option B: Integrate into orchestrator (8h effort, requires design)
- [ ] Option C: Remove if superseded by other features

**Questions for Architect:**
1. Is this feature still relevant? (context-aware rendering)
2. Does it overlap with material_response or lux_depth_v3?
3. Should it be a separate CLI command or orchestrator mode?

**Recommendation:** Architect decision required (likely move to examples/)

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

### 3.3 Performance Regression Investigation

**Status:** 🟢 OBSERVATIONAL (NOT A BLOCKER)  
**Priority:** P4 (Low)  
**Action:** Track in IMPROVEMENT_OPPORTUNITIES.md (already done)  

**Location:** `tests/stress/test_stress_large_batch.py`

#### 3.3.1 Draft Preset Performance (line 326)

```python
# TODO: Investigate why draft is slower than expected (expected: draft < standard)
```

**Context:**
- Draft preset sometimes slower than standard
- Non-deterministic, likely resource contention
- Does not affect correctness, only performance

**Status:** Already tracked in `docs/IMPROVEMENT_OPPORTUNITIES.md:281` (TEST-006)

#### 3.3.2 Preset Performance Ordering (line 372)

```python
# TODO: Investigate preset performance ordering regression
```

**Context:**
- Preset performance ordering inconsistent
- May be environment-dependent (CI vs local)
- Already documented in IMPROVEMENT_OPPORTUNITIES.md:287

**Decision:** No immediate action (performance optimization, not bug)

---

## 4. Documentation TODOs

### 4.1 Binary File Cleanup

**Status:** ✅ OBSOLETE (COMPLETED)  
**Priority:** N/A  
**Action:** Archive tracking document  

**Location:** `docs/fixes/BINARY_FILE_BEST_PRACTICES.md:133,140`

```markdown
│   ├── *.png                 # ⚠️ TODO: Exclude PNG previews
│   └── *.jpg                 # TODO: Move to docs/examples/
```

**Evidence of Completion:**
- `docs/fixes/PRE_PUSH_AUDIT_REPORT.md:256` states "Minimal TODO comments (only in documentation)"
- Binary cleanup action plan executed
- Pre-push audits passing

**Recommendation:** Remove TODO from BINARY_FILE_BEST_PRACTICES.md or mark as complete

---

### 4.2 PR #98 Action Items

**Status:** ✅ OBSOLETE (PR MERGED)  
**Priority:** N/A  
**Action:** Archive tracking documents  

**Location:** `docs/pr_reports/PR98_ACTION_ITEMS.md` and `docs/development/pr/PR98_ACTION_ITEMS.md`

Multiple TODO items:
- Line 52: Mark PR as Ready for Review
- Line 65: Request Review
- Line 82: Unblock CI
- Line 123: Merge PR

**Context:**
- PR #98 merged and deployed
- Tracking documents no longer relevant

**Recommendation:** Move to `archive/pr_reports/` or delete

---

### 4.3 ADR-023 Manifest Implementation

**Status:** 🟢 QUICK WIN  
**Priority:** P3 (Low)  
**Effort:** 2 hours  
**Owner:** Specialist  

**Location:** `docs/architecture/decisions/ADR-023-implementation-guide.md`

Multiple TODOs:
- Line 28: "TODO: Implement actual manifest parsing logic"
- Line 38: "TODO: Extract timing from actual manifest schema"
- Line 53: "TODO: Replace placeholder with actual extraction"
- Line 72: "TODO: Wire up the full workflow"

**Context:**
- These are **implementation guide placeholders**, not actual code TODOs
- Guide shows example of what to implement
- Actual manifest code may already exist in `src/transformation_portal/lux_depth_v3/manifest.py`

**Decision:**
- [ ] Audit `manifest.py` to see if TODOs already implemented
- [ ] If implemented: Update ADR-023 guide to reference actual code
- [ ] If not: Implement missing pieces (timing extraction, workflow wiring)

**Recommendation:** Quick audit + documentation update (likely already implemented)

---

### 4.4 V2.0.0 Release Checklist

**Status:** 🔴 CRITICAL  
**Priority:** P0 (Blocks Release)  
**Action:** IMMEDIATE REVIEW REQUIRED  

**Location:** `docs/architecture/V2_0_0_RELEASE_REVIEW.md`

**P0 Items (Must Complete Before Release):**

| Item | Status | Deadline | Blocker |
|------|--------|----------|---------|
| Add code coverage to CI | ❌ TODO | Day 1 | Yes |
| Enforce 70% coverage threshold | ❌ TODO | Day 1 | Yes |
| Add security scan to PR workflow | ❌ TODO | Day 1 | Yes |
| Configure branch protection | ❌ TODO | Day 1 | Yes |
| Document rollback procedures | ❌ TODO | Day 2 | Yes |
| Define staging validation | ❌ TODO | Day 2 | Yes |

**Estimated Effort:** 12-16 hours  
**Recommendation:** **NO-GO on v2.0.0 release until P0 items completed**

**Questions for Architect:**
1. Is v2.0.0 already released? (Git tags suggest yes)
2. Are P0 items actual blockers or aspirational?
3. Should checklist be updated to reflect current reality?

**Recommendation:** Architect clarification required (release already happened or update checklist)

---

### 4.5 Repository Health Report Placeholders

**Status:** 🟢 OBSERVATIONAL  
**Priority:** P4 (Low)  
**Action:** None (status markers, not TODOs)  

**Location:** `docs/architecture/REPOSITORY_HEALTH_REPORT_2026-02-03.md:143`

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

### 5.1 Code Coverage Enforcement

**Status:** 🔴 P0 (CRITICAL)  
**Effort:** 4 hours  
**Impact:** Quality gate, release blocker  

**Current State:**
- Coverage calculated but not enforced
- No CI gate to fail on coverage drop
- Target: 70% threshold (currently at ~78%)

**Implementation:**
- [ ] Add `coverage` report to CI workflow
- [ ] Add `pytest-cov` HTML report artifact
- [ ] Add threshold check: `--cov-fail-under=70`
- [ ] Update PR template to require coverage review

**Files Affected:**
- `.github/workflows/build.yml`
- `.github/workflows/pr-checks.yml`
- `requirements-ci.txt` (verify pytest-cov pinned)

**Reference:** V2_0_0_RELEASE_REVIEW.md:580

---

### 5.2 Security Scanning in PR Workflow

**Status:** 🔴 P0 (CRITICAL)  
**Effort:** 4 hours  
**Impact:** Supply chain security, compliance  

**Current State:**
- CodeQL analysis exists (scheduled)
- No PR-blocking security scan
- Dependency scanning via Dependabot (reactive)

**Implementation:**
- [ ] Add Bandit security linting to PR checks
- [ ] Add `safety check` for dependency vulnerabilities
- [ ] Configure failure threshold (high/critical only)
- [ ] Add security scan badge to README

**Files Affected:**
- `.github/workflows/pr-checks.yml`
- `requirements-lint.txt` (add bandit, safety)

**Reference:** V2_0_0_RELEASE_REVIEW.md:582

---

### 5.3 Branch Protection Rules

**Status:** 🔴 P0 (CRITICAL)  
**Effort:** 1 hour  
**Impact:** Prevent accidental force-pushes, require reviews  

**Current State:**
- No branch protection configured (or unclear from code inspection)
- Main branch potentially unprotected

**Implementation:**
- [ ] Configure `main` branch protection:
  - Require PR before merge
  - Require at least 1 approval
  - Require status checks to pass (build, lint, tests)
  - Restrict force pushes
  - Restrict deletions
- [ ] Document in `CONTRIBUTING.md`

**Note:** Requires repository admin permissions

**Reference:** V2_0_0_RELEASE_REVIEW.md:583

---

### 5.4 CLI Test Suite

**Status:** 🟡 P1 (HIGH)  
**Effort:** 8 hours  
**Impact:** Prevent CLI regressions  

**Current State:**
- Unit tests for CLI argument parsing exist
- No end-to-end CLI integration tests
- Manual testing only

**Implementation:**
- [ ] Create `tests/cli/` directory
- [ ] Write CLI invocation tests:
  - `lux-depth-v3 --help`
  - `lux-depth-v3 --list-presets`
  - `lux-depth-v3 --depth-backend depth_anything_v3 input/ output/`
- [ ] Add exit code validation
- [ ] Add output validation (files created, logs generated)
- [ ] Add error case testing (invalid args, missing inputs)

**Files Affected:**
- `tests/cli/test_lux_depth_v3_cli.py` (new)
- `tests/cli/test_cli_presets.py` (new)

**Reference:** V2_0_0_RELEASE_REVIEW.md:595

---

### 5.5 Staging Environment

**Status:** 🟡 P1 (HIGH)  
**Effort:** 16 hours  
**Impact:** Pre-production validation  

**Current State:**
- No staging environment
- Direct production deployment (or no deployment automation)

**Implementation:**
- [ ] Define staging environment requirements:
  - Containerized deployment (Docker)
  - Sample data preloaded
  - Environment variables configuration
- [ ] Create `docker-compose.staging.yml`
- [ ] Add staging deployment workflow
- [ ] Document staging validation procedures

**Files Affected:**
- `docker-compose.staging.yml` (new)
- `.github/workflows/deploy-staging.yml` (new)
- `docs/deployment/staging_guide.md` (new)

**Reference:** V2_0_0_RELEASE_REVIEW.md:596

---

### 5.6 Rollback Procedures Documentation

**Status:** 🔴 P0 (CRITICAL)  
**Effort:** 2 hours  
**Impact:** Incident response, production safety  

**Current State:**
- No documented rollback procedures
- Unclear how to revert a bad release

**Implementation:**
- [ ] Document rollback procedures:
  - Git tag reversion process
  - PyPI package yanking (if applicable)
  - Dependency pinning rollback
  - Data migration rollback (if applicable)
- [ ] Create `docs/deployment/rollback_procedures.md`
- [ ] Add to incident response runbook

**Files Affected:**
- `docs/deployment/rollback_procedures.md` (new)
- `docs/operations/incident_response.md` (update)

**Reference:** V2_0_0_RELEASE_REVIEW.md:584

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

1. **PR #98 Action Items**
   - `docs/pr_reports/PR98_ACTION_ITEMS.md`
   - `docs/development/pr/PR98_ACTION_ITEMS.md`
   - Reason: PR merged and deployed

2. **Binary Cleanup TODOs**
   - `docs/fixes/BINARY_FILE_BEST_PRACTICES.md` (lines 133, 140)
   - Reason: Cleanup completed per PRE_PUSH_AUDIT_REPORT.md

3. **Old Verification Reports**
   - `docs/development/VERIFICATION_COMPLETE.md` (multiple PENDING markers)
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

### By Category

| Category | Total | P0 | P1 | P2 | P3 | P4 | Obsolete |
|----------|-------|----|----|----|----|----|---------
| Architectural Decisions | 6 | 1 | 0 | 2 | 1 | 1 | 1 |
| Stub Implementations | 8 | 0 | 0 | 0 | 3 | 4 | 1 |
| Code TODOs | 5 | 0 | 0 | 2 | 1 | 2 | 0 |
| Documentation TODOs | 7 | 1 | 0 | 1 | 2 | 1 | 2 |
| CI/CD Gaps | 6 | 3 | 2 | 0 | 0 | 0 | 1 |
| Test Infrastructure | 4 | 0 | 0 | 1 | 1 | 2 | 0 |
| **TOTAL** | **36** | **5** | **2** | **6** | **8** | **10** | **5** |

### By Effort

| Effort | Count | Items |
|--------|-------|-------|
| < 2 hours | 8 | Sample URLs, ADR-023, stub cleanup, branch protection, etc. |
| 2-8 hours | 12 | Context-aware rendering, coverage CI, security scan, CLI tests, etc. |
| 1-2 days | 10 | ADR-019, YAML config, Depth Pro presets, staging, etc. |
| > 1 week | 6 | POM, backend commands, full staging environment, etc. |

### By Status

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Completed/Correct | 8 | 22% |
| 🟢 Quick Wins | 7 | 19% |
| 🟡 Medium Tasks | 10 | 28% |
| 🔴 Deferred/Epic | 6 | 17% |
| 📦 Obsolete | 5 | 14% |

---

## Recommended Action Sequence

### Sprint 1: Critical Items (Week 1)

1. ✅ **ADR-019 Backend Integration** (2-3 days) - APPROVED
2. 🔴 **V2.0.0 P0 Items Review** (4 hours) - Clarify status
3. 🔴 **CI Coverage Enforcement** (4 hours) - If P0 confirmed
4. 🔴 **Security Scanning** (4 hours) - If P0 confirmed

### Sprint 2: High-Value Items (Week 2)

5. 🟢 **Sample Data GitHub Release** (4 hours)
6. 🟢 **Context-Aware Rendering Decision** (2 hours)
7. 🟢 **ADR-023 Manifest Audit** (2 hours)
8. 🟡 **CLI Test Suite** (8 hours)

### Sprint 3: Cleanup & Documentation (Week 3)

9. 🟢 **Obsolete Doc Archive** (2 hours)
10. 🟢 **Stub Consolidation** (4 hours)
11. 🟡 **Depth Pro Presets** (4 hours) - Post ADR-019
12. 🟡 **Golden Fixture Tests** (8 hours)

### Future Sprints (v2.1.0+)

- YAML Config Loading (1-2 days)
- Backend Strict Enforcement (1-2 days)
- Staging Environment (2 weeks)
- Backend CLI Commands (1 week)

### No Action Required

- Dependency-conditional test skips (correct pattern)
- ML test isolation (working as designed)
- Platform restrictions (documented limitations)
- Abstract base class NotImplementedErrors (correct usage)

---

## References

- **OUTSTANDING_WORK_SUMMARY.md** - Executive summary
- **QUICK_WINS.md** - Tasks < 4 hours
- **IMPROVEMENT_OPPORTUNITIES.md** - Existing improvement tracking
- **ADR-019-REVISED-DECISION.md** - Backend integration approval
- **V2_0_0_RELEASE_REVIEW.md** - Release readiness assessment

---

**Document Control:**
- **Maintained By:** Transformation Portal Architect
- **Update Frequency:** After each sprint or major milestone
- **Issue Tracking:** Link to GitHub Issues/Projects when created
- **Review Date:** Next review at v2.1.0 planning phase
