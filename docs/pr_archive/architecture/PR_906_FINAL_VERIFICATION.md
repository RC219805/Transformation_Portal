# PR #906 Final Architectural Verification Report

**Date:** 2026-02-11
**Reviewer:** Transformation Portal Architect
**PR:** #906 - APEX Research Ultra Phase 1
**Commit Verified:** `c7c1deb4`
**Branch:** `feat/apex-research-ultra-phase1`

---

## Executive Summary

**VERDICT: ⚠️ APPROVE WITH CONDITIONS**

All three critical bugs (P1-A, P1-B, P2-C) identified in the original architectural review **have been correctly fixed** and are verified present in commit `c7c1deb4`. The fixes are mathematically sound, properly tested, and correctly implemented.

However, **four additional issues** have been identified during this verification review:
- 1 MEDIUM severity issue (test coverage gap)
- 1 MEDIUM severity issue (non-functional API parameter)
- 1 MEDIUM-LOW issue (HDR preservation fallback violation)
- 1 LOW severity issue (docstring mismatch)

**Recommendation:** Merge PR #906 as-is to unblock Phase 1 work, but **immediately open a follow-up PR** to address the four additional issues before Phase 2 begins.

---

## Part 1: Critical Bug Fix Verification

### ✅ Bug P1-A: FIXED (Metric Alignment Runtime Crash)

**Original Issue:**
Runtime crash when ensemble contains metric depth models due to `.values()` instead of `.items()` in loop.

**Fix Verification:**
```python
# File: src/transformation_portal/depth/backends/ensemble.py
# Lines: 418, 430

# Metric branch:
for name, result in model_results.items():  # ✅ CORRECT (was .values())
    if result.is_metric:
        aligned[name] = result.depth_map

# Relative branch:
for name, result in model_results.items():  # ✅ CORRECT (was .values())
    depth = result.depth_map.astype(np.float32)
```

**Status:** ✅ **FIXED AND VERIFIED**
**Evidence:**
- Code inspection confirms `.items()` on lines 418 and 430
- Test `test_metric_alignment_with_mixed_backends` passes
- Test creates mixed metric/relative models and verifies no crash

**Test Coverage:** Adequate. Test explicitly exercises the bug path.

---

### ✅ Bug P1-B: FIXED (Variance-Weighted Fusion Algebraic Cancellation)

**Original Issue:**
Variance-weighted fusion computed a single `inv_variance` map and multiplied all models by the same value, causing algebraic cancellation (variance terms cancel out in numerator/denominator).

**Fix Verification:**
```python
# File: src/transformation_portal/depth/backends/ensemble.py
# Lines: 335-365

# Step 1: Compute per-pixel variance
depth_stack = np.stack(list(aligned_depths.values()), axis=0)  # (N, H, W)
variance_map = np.var(depth_stack, axis=0)  # (H, W)

# Step 2: Compute inverse variance (per-pixel)
epsilon = 1e-6
inv_variance = 1.0 / (variance_map + epsilon)

# Step 3: Compute per-model variance-weighted contributions (ELEMENT-WISE)
weighted_depths = []
effective_weights_list = []

for model_name, depth_map in aligned_depths.items():
    model_weight = model_weights.get(model_name, 0.0)
    # Effective weight is model_weight × inverse_variance (per-pixel array)
    effective_weight = model_weight * inv_variance  # (H, W) ✅ Element-wise array

    weighted_depths.append(depth_map * effective_weight)  # ✅ Element-wise
    effective_weights_list.append(effective_weight)

# Step 4: Fuse using element-wise array summation (NO CANCELLATION)
fused_depth = np.sum(weighted_depths, axis=0)  # ✅ Sum arrays element-wise
total_effective_weight = np.sum(effective_weights_list, axis=0)  # ✅ Sum arrays

# Normalize per-pixel
total_effective_weight = np.maximum(total_effective_weight, epsilon)
fused_depth /= total_effective_weight  # ✅ Element-wise division
```

**Mathematical Verification:**
The algorithm now correctly implements variance-weighted fusion:

```
w_i(x,y) = model_weight_i / variance(x,y)
fused(x,y) = Σ[w_i(x,y) * depth_i(x,y)] / Σ[w_i(x,y)]
```

Each pixel gets a **different** weight based on local variance. No algebraic cancellation.

**Status:** ✅ **FIXED AND VERIFIED**
**Evidence:**
- Code uses `np.sum()` with element-wise array operations
- Test `test_variance_weighted_fusion_actually_uses_variance` passes
- Test creates local disagreement regions and verifies variance affects output

**Test Coverage:** Adequate. Test creates spatially-varying variance and validates fusion output is variance-weighted.

---

### ✅ Bug P2-C: FIXED (ADR-023 Isolation Regex Pattern Typo)

**Original Issue:**
Isolation check regex pattern had space after dots (`"from ... spatial_ai"`) which would never match real imports (`from ...spatial_ai`).

**Fix Verification:**
```python
# File: scripts/security/verify_pipeline_isolation.py
# Lines: 67-73

forbidden_patterns = [
    "from transformation_portal.spatial_ai",
    "import transformation_portal.spatial_ai",
    "from ..spatial_ai",
    "from ...spatial_ai",  # ✅ FIXED: No space after dots (was "from ... spatial_ai")
    "from ....spatial_ai",  # ✅ ADDED: 4-dot pattern for completeness
]
```

**Status:** ✅ **FIXED AND VERIFIED**
**Evidence:**
- Code inspection confirms no space: `"from ...spatial_ai"`
- 4-dot pattern added for completeness: `"from ....spatial_ai"`
- All 6 tests in `TestIsolationCheckRegex` pass
- Tests explicitly verify the buggy pattern (`"from ... spatial_ai"`) would NOT match

**Test Coverage:** Excellent. 6 dedicated tests cover all regex patterns, false positives, and explicitly test the bug fix.

---

## Part 2: New Test Coverage Assessment

### Test Count Verification

**Claimed:** 9 new tests
**Actual:** 8 new tests

**Breakdown:**
1. `test_metric_alignment_with_mixed_backends` - 1 test (P1-A coverage)
2. `test_variance_weighted_fusion_actually_uses_variance` - 1 test (P1-B coverage)
3. `TestIsolationCheckRegex` suite - 6 tests (P2-C coverage):
   - `test_pattern_matching_absolute_imports`
   - `test_pattern_matching_two_dot_relative_imports`
   - `test_pattern_matching_three_dot_relative_imports`
   - `test_pattern_matching_four_dot_relative_imports`
   - `test_safe_imports_not_flagged`
   - `test_comprehensive_pattern_coverage`

**Total:** 8 tests (commit message claims 9, likely a counting error)

**Assessment:** Test coverage is adequate for the three critical bugs. All tests pass.

---

## Part 3: Additional Issues Identified

### ❌ Issue #4: Ensemble Test Uses Duplicate Model Names (MEDIUM)

**Location:** `tests/depth/backends/test_ensemble.py:126-128`

**Problem:**
```python
def test_variance_weighted_fusion_synthetic(self):
    # ...
    models = [
        ModelConfig(name="synthetic", weight=0.5),
        ModelConfig(name="synthetic", weight=0.5),  # ❌ Same name as above
    ]
```

**Impact:**
When `_run_models()` executes, it stores results as:
```python
results[model_config.name] = result  # Line 256 of ensemble.py
```

With duplicate names, the second model **overwrites** the first in the dict. The test is actually testing **one model**, not two.

**Severity:** **MEDIUM**
This is a test coverage gap. The test claims to test variance-weighted fusion with two models, but actually only tests with one model. This reduces confidence in the fusion algorithm.

**Recommendation:**
Fix in follow-up PR by using distinct model names:
```python
models = [
    ModelConfig(name="synthetic", weight=0.5),
    ModelConfig(name="synthetic_2", weight=0.5),  # ✅ Distinct name
]
```

Or mock two different backends.

**Blocking for Phase 1:** No. The P1-B test (`test_variance_weighted_fusion_actually_uses_variance`) correctly tests with distinct model names (`"model_a"`, `"model_b"`), so the fusion algorithm is adequately tested.

---

### ❌ Issue #5: LinearDecoder `validate_contract=False` Is Non-Functional (MEDIUM)

**Location:** `src/transformation_portal/spatial_ai/ingest/linear_decoder.py`

**Problem:**
`LinearDecoder.__init__` accepts `validate_contract=False` to bypass gamma enforcement:

```python
# Line 130-134
if validate_contract and abs(gamma - 1.0) > 1e-6:
    raise ValueError(...)  # Only raised if validate_contract=True
```

However, `LinearIngestResult.__post_init__` **always** enforces gamma==1.0:

```python
# Line 79-82
def __post_init__(self):
    if abs(self.gamma - 1.0) > 1e-6:
        raise ValueError(...)  # ❌ Always raises, ignores validate_contract
```

**Impact:**
The `validate_contract=False` parameter is **non-functional**. Even if you bypass the decoder check, the result's `__post_init__` will still raise an error when the result is created.

**Example:**
```python
# This will NOT work:
decoder = LinearDecoder(gamma=1.5, validate_contract=False)  # ✅ Doesn't raise
result = decoder.decode("image.tiff")  # ❌ Raises in LinearIngestResult.__post_init__
```

**Severity:** **MEDIUM**
This is an API contract violation. The `validate_contract` parameter is documented as an override mechanism, but it doesn't work. This could mislead users or cause unexpected failures in edge cases.

**Recommendation:**
Fix in follow-up PR by either:

**Option A (Recommended):** Remove `validate_contract` parameter entirely if gamma==1.0 is non-negotiable:
```python
class LinearDecoder:
    def __init__(self, gamma: float = 1.0, bit_depth: int = 32):
        if abs(gamma - 1.0) > 1e-6:
            raise ValueError("Linear ingest requires gamma=1.0")
        # ... rest of init
```

**Option B:** Thread `validate_contract` through to the result:
```python
@dataclass
class LinearIngestResult:
    # ... existing fields ...
    _validate_contract: bool = True

    def __post_init__(self):
        if self._validate_contract and abs(self.gamma - 1.0) > 1e-6:
            raise ValueError(...)
```

**Blocking for Phase 1:** No. The gamma==1.0 enforcement is actually correct for the SpatialCaptureV1 contract. The issue is just that the API is confusing. Phase 1 doesn't rely on gamma overrides.

---

### ⚠️ Issue #6: EXR Fallback Clips HDR Values (MEDIUM-LOW)

**Location:** `src/transformation_portal/spatial_ai/ingest/linear_decoder.py:442`

**Problem:**
When OpenEXR is not installed, `_save_exr` falls back to 16-bit TIFF and **clips HDR values**:

```python
# Line 442
img_uint16 = np.clip(linear_rgb * 65535, 0, 65535).astype(np.uint16)
```

This violates the stated HDR preservation claim in the docstring and README.

**Impact:**
If a user runs the decoder without OpenEXR installed:
1. They get a TIFF file instead of EXR (expected fallback)
2. Values >1.0 are **clipped** (unexpected data loss)
3. A warning is printed, but data is already corrupted

**Example:**
```python
linear_rgb = np.array([[[2.0, 0.5, 0.3]]])  # HDR value 2.0
# Fallback: 2.0 * 65535 = 131070 → clipped to 65535 → normalized back to 1.0
# Result: Value 2.0 becomes 1.0 (data loss)
```

**Severity:** **MEDIUM-LOW**
This is a **documentation/honesty issue**, not a critical bug:
- The warning message is clear: "HDR values >1.0 clipped"
- The fallback is explicitly labeled as lossy
- Users are told to install OpenEXR for proper HDR support

However, it violates the spirit of "HDR preservation" claims. Research users may not read warnings and could unknowingly train on clipped data.

**Recommendation:**
Fix in follow-up PR by either:

**Option A (Recommended):** Fail loudly instead of silently clipping:
```python
if linear_rgb.max() > 1.0 and OpenEXR not available:
    raise RuntimeError(
        "HDR data detected (values >1.0) but OpenEXR not installed. "
        "Install OpenEXR to preserve HDR range: pip install OpenEXR"
    )
```

**Option B:** Use 32-bit TIFF instead of 16-bit:
```python
# Save as 32-bit float TIFF (preserves HDR, but less portable)
img = Image.fromarray(linear_rgb, mode="F")  # Float mode
img.save(output_path, format="TIFF", compression="lzw")
```

**Option C:** Document this limitation explicitly in ADR-026 and README.

**Blocking for Phase 1:** No. Phase 1 requires OpenEXR installation (in `requirements.txt`), so this fallback path is not expected to execute in normal operation. This is more of a defensive coding issue.

---

### 📝 Issue #7: `required_packages()` Docstring Mismatch (LOW)

**Location:** `src/transformation_portal/depth/backends/ensemble.py:503-512`

**Problem:**
```python
@classmethod
def required_packages(cls) -> list[str]:
    """Return required import modules for ensemble.

    Ensemble requires at least torch + transformers (for DA3).  # ❌ Docstring
    Depth Pro is optional (graceful degradation).

    Returns:
        List of required module names.
    """
    return ["transformers"]  # ❌ Code only returns transformers, not torch
```

**Impact:**
Docstring says "torch + transformers", code returns only `["transformers"]`.

**Severity:** **LOW**
This is a documentation inconsistency. The code is likely correct (torch is handled by the APEX runner, per protocol documentation line 219), but the docstring is misleading.

**Recommendation:**
Fix in follow-up PR by updating docstring:
```python
"""Return required import modules for ensemble.

Ensemble requires transformers (for DA3 backend).
torch is handled by the APEX runner and not listed here.
Depth Pro is optional (graceful degradation).

Returns:
    List of required module names.
"""
return ["transformers"]
```

**Blocking for Phase 1:** No. This is a minor documentation issue with no functional impact.

---

## Part 4: Final Verdict and Recommendations

### Overall Assessment

**Critical Bugs:** ✅ All 3 fixed and verified
**Test Coverage:** ✅ Adequate (8 tests, all passing)
**New Issues:** ⚠️ 4 identified (0 blocking, 2 medium, 1 medium-low, 1 low)

### Merge Decision

**✅ APPROVE PR #906 FOR MERGE**

**Rationale:**
1. All critical bugs (P1-A, P1-B, P2-C) are **correctly fixed** and verified
2. Fixes are mathematically sound and properly tested
3. No new blocking issues introduced
4. Additional issues are non-critical and can be addressed in follow-up

### Conditions for Merge

**REQUIRED BEFORE MERGE:**
- Add PR comment documenting the 4 additional issues
- Link to this verification report in PR description

**REQUIRED IMMEDIATELY AFTER MERGE:**
- Open follow-up issue/PR to address issues #4-7
- Target completion before Phase 2 begins

### Follow-Up Work Required

Create a new issue with these tasks:

```markdown
## Follow-Up: Address PR #906 Verification Findings

**Priority:** MEDIUM (complete before Phase 2)

### Tasks:
- [ ] Issue #4: Fix ensemble test to use distinct model names
- [ ] Issue #5: Remove or fix `validate_contract` parameter in LinearDecoder
- [ ] Issue #6: Either fail loudly on HDR+no-OpenEXR or use 32-bit TIFF fallback
- [ ] Issue #7: Update `required_packages()` docstring in ensemble backend

**Reference:** docs/pr_archive/architecture/PR_906_FINAL_VERIFICATION.md
```

---

## Part 5: Reconciliation with Independent Review

The independent review claimed bugs P1-A, P1-B, and P2-C were **still present**. This verification proves they are **fixed**.

**Likely explanation:**
The independent review was looking at an **older state** of the PR (before commit `c7c1deb4`), or at the initial PR diff without the fix commit.

**Evidence:**
- Commit `c7c1deb4` is present on `origin/feat/apex-research-ultra-phase1`
- Git log shows it was pushed on 2026-02-11 11:12:56
- All fixes are present in current branch state
- All tests pass

**Conclusion:**
The Specialist's claim that "all bugs are fixed in c7c1deb4" is **accurate**. The independent review was likely stale.

---

## Appendices

### Appendix A: Test Execution Evidence

```bash
# P1-A test
$ pytest tests/depth/backends/test_ensemble.py::TestVarianceFusion::test_metric_alignment_with_mixed_backends -xvs
PASSED [100%]

# P1-B test
$ pytest tests/depth/backends/test_ensemble.py::TestVarianceFusion::test_variance_weighted_fusion_actually_uses_variance -xvs
PASSED [100%]

# P2-C tests (6 total)
$ pytest tests/security/test_pipeline_isolation.py -xvs
PASSED [100%] (6/6 tests)
```

### Appendix B: Code Review Methodology

1. Verified current branch and commit state
2. Inspected actual source code for each claimed fix
3. Traced fix through algorithm to verify mathematical correctness
4. Executed tests to verify they pass
5. Checked for test coverage gaps
6. Performed holistic review to identify new issues

### Appendix C: Architectural Sign-Off

As Transformation Portal Architect, I certify:

✅ All critical bugs are fixed
✅ Fixes are mathematically sound
✅ Test coverage is adequate
✅ No blocking issues introduced
⚠️ Four non-blocking issues identified for follow-up

**Recommendation:** Merge with conditions outlined in this report.

---

**Report Author:** Transformation Portal Architect
**Report Date:** 2026-02-11
**Commit Verified:** c7c1deb4
**Next Review:** Follow-up PR for issues #4-7
