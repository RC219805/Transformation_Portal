# Performance Ledger v1.7 Upgrade: Architect Verdict

**Date:** 2026-02-05  
**Status:** ⚠️ CONDITIONAL APPROVAL - BREAKING CHANGES REQUIRE MITIGATION

---

## TL;DR

**CAN THIS BE A DROP-IN REPLACEMENT?** 
❌ **NO** - Multiple breaking changes require:
1. CLI compatibility shims
2. Test suite expansion (16 → ~35 tests)
3. ADR-023 amendment
4. Staged rollout plan

**SHOULD WE PROCEED?**
✅ **YES** - But with mandatory conditions (see §6 below)

---

## Key Findings

### Breaking Changes Identified

1. **CLI Flag Rename** (HIGH IMPACT)
   - `--version` → `--baseline-version`
   - **Fix:** Add `--version` as deprecated alias
   
2. **Exit Code Expansion** (MEDIUM IMPACT)
   - v1.0: `0=success, 1=regression`
   - v1.7: `0=success, 1=regression, 2=backend_mismatch, 3=insufficient_data`
   - **Risk:** Exit code 0 may hide "potential regressions" without `--strict`
   - **Fix:** Make `--strict` default in CI, lenient for exploratory use
   
3. **NumPy Optional** (MEDIUM IMPACT)
   - Pure Python fallbacks add 50-100x performance penalty
   - **Risk:** Subtle math errors in pure Python percentile/bootstrap
   - **Fix:** Comprehensive property-based testing (Hypothesis)
   
4. **JSON Schema Extensions** (LOW IMPACT)
   - Additive only (forensics, backend_compliance)
   - **Risk:** Baseline files grow 2-3x with forensics data
   - **Fix:** Keep forensics in separate output file

### Current Test Coverage Analysis

**Existing (v1.0):** 16 unit tests  
**Coverage:** Function-level only, no CLI integration tests

**What's NOT tested:**
- ❌ CLI argument parsing
- ❌ Exit code behavior
- ❌ End-to-end workflows (capture → compare)
- ❌ Backend mismatch detection
- ❌ `--emit-json` output format

**Result:** Existing tests WILL pass with v1.7, but that's insufficient validation.

---

## Test Impact Assessment

### Tests That Will Break
**Direct:** NONE (no tests invoke CLI)  
**Behavioral:** Hypothetical CLI tests would break due to `--version` rename

### Required New Tests (Priority Order)

**P0: Compatibility (before merge)**
```python
def test_version_flag_backward_compatibility()  # --version still works
def test_exit_codes_all_paths()                # 0, 1, 2, 3 reachable
def test_strict_mode_changes_behavior()        # --strict affects exit code
def test_baseline_v1_0_load_compatibility()    # Load old baselines
```

**P1: Pure Python Mode (before merge)**
```python
@pytest.mark.parametrize("has_numpy", [True, False])
def test_statistics_numpy_vs_pure_python()     # Exact match required

def test_bootstrap_ci_without_numpy()          # Fallback works
def test_percentile_edge_cases()               # Empty, single value, etc.
```

**P2: New Features (within 1 sprint)**
```python
def test_error_taxonomy_classification()       # Failure bucketing
def test_backend_mismatch_detection()          # Exit code 2
def test_histogram_rendering()                 # ASCII art
def test_outlier_detection()                   # Top slowest, p95 contributors
```

**P3: Property-Based (within 1 sprint)**
```python
@given(st.lists(st.floats(0.1, 100), min_size=10))
def test_percentile_pure_python_correctness()  # Hypothesis validation
```

**Total new tests required:** ~19 (16 existing + 19 new = 35 total)

---

## Integration Recommendations

### Phased Rollout (Recommended)

**Week 1-2: Internal Testing**
- Deploy v1.7 with `--version` alias (backward compat)
- Test in dev/staging environments
- Validate pure Python mode performance

**Week 3-4: Gradual Migration**
- Update CI scripts to `--baseline-version`
- Add `--strict` to production regression checks
- Monitor for unexpected failures

**Month 2: Deprecation Notices**
- Warn on `--version` usage
- Document migration guide
- Update all examples in README/docs

**Month 3-6: Cleanup**
- Remove `--version` alias
- Bump to v2.0.0 (semantic versioning)
- Remove compatibility shims

### Alternative: Big Bang Migration

If coordination overhead is low (< 5 users), could bump to v2.0.0 immediately with:
- Update all CI scripts in one PR
- Announce breaking changes
- Provide migration script

**Not recommended** - gradual rollout is safer.

---

## Risk Assessment

### Critical Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Pure Python math errors | HIGH | MEDIUM | Property-based testing, cross-validate 10K datasets |
| Bootstrap CI false positives | MEDIUM | MEDIUM | Tune defaults conservatively, document tuning |
| Backend mismatch false alarms | MEDIUM | LOW | Canonical name normalization, alias whitelist |
| Performance degradation (no NumPy) | LOW | HIGH | Document clearly, fail fast if > 1000 samples |

### Security Implications

✅ **POSITIVE:** Reduces dependency attack surface (NumPy optional)  
⚠️ **CAUTION:** Validate new input bounds (bootstrap iterations, histogram bins)

**Required validation:**
```python
parser.add_argument("--bootstrap-iterations", type=int, 
                    default=1000, 
                    choices=range(100, 10001))  # Prevent DoS
```

---

## Conditions for Merge Approval

### Mandatory (before merge)

1. ✅ **Backward Compatibility Shims**
   - Add `--version` as deprecated alias for `--baseline-version`
   - Preserve exact v1.0 baseline JSON schema in `load_baseline()`
   
2. ✅ **Pure Python Validation**
   - Property-based tests (Hypothesis) for percentile correctness
   - Cross-validate against NumPy for 1000+ random datasets
   - Document any intentional deviations
   
3. ✅ **CLI Integration Tests**
   - Minimum 10 tests covering: capture, compare, exit codes, flags
   - Test all 4 exit code paths (0, 1, 2, 3)
   - Test `--strict` mode behavior
   
4. ✅ **ADR-023 Amendment**
   - Document breaking changes
   - Migration guide for existing users
   - Exit code contract definition
   
5. ✅ **Performance Regression Tests**
   - Benchmark NumPy mode (should be ~same as v1.0)
   - Benchmark pure Python mode (document acceptable slowdown)
   - Fail if NumPy mode degrades > 2x
   
6. ✅ **Input Validation**
   - Bound all user inputs (iterations, bins, samples)
   - Test rejection of extreme values
   - Prevent DoS via resource exhaustion

### Recommended (within 1 sprint)

7. 🔵 **Forensics Isolation**
   - Move forensics data to separate `--emit-forensics` file
   - Keep baseline JSON compact for version control
   
8. 🔵 **Schema Versioning**
   - Add `"schema_version": "1.0"` field to baselines
   - Implement forward/backward compatibility checks
   
9. 🔵 **Documentation Updates**
   - Update README examples
   - Add migration guide to docs/
   - Document pure Python performance characteristics

---

## Estimated Effort

| Task | Effort | Owner |
|------|--------|-------|
| Compatibility shims | 4 hours | Specialist |
| Pure Python tests | 8 hours | Specialist |
| CLI integration tests | 8 hours | Specialist |
| Property-based tests | 4 hours | Specialist |
| ADR amendment | 2 hours | Architect/Specialist |
| Performance tests | 4 hours | Specialist |
| Input validation | 2 hours | Specialist |
| **Total** | **32 hours (4 days)** | |

**Timeline:** 1 week for implementation + 1 week for review/testing = 2 weeks total

---

## Architect Decision

**CONDITIONAL APPROVAL** granted subject to:
1. All 6 mandatory conditions met
2. Test coverage >= 80% for new code paths
3. ADR-023 amendment reviewed by Architect

**Specialist is authorized to proceed** with implementation of conditions.

**Return for final review when:**
- All conditions implemented
- Test suite expanded to ~35 tests
- Pure Python mode validated
- Compatibility shims in place

**Escalate back to Architect if:**
- Pure Python math deviates > 1% from NumPy
- Performance degradation > 2x in NumPy mode
- New security concerns discovered

---

## What This Means for PR #845

The prompt mentioned "tests we just fixed in PR #845" - those 4 tests are likely:
1. `test_detect_regressions_p95_threshold`
2. `test_detect_regressions_mean_threshold`
3. `test_detect_regressions_failure_rate`
4. `test_detect_regressions_no_regressions`

**Status:** These tests will continue to pass with v1.7 (function-level, not CLI).

**But:** They are insufficient for v1.7 validation. Need CLI integration tests.

---

## Final Verdict

**This is NOT a drop-in replacement.**

**But it CAN be made safe** with 4 days of focused work on:
1. Compatibility layer
2. Test expansion
3. Validation rigor

**Proceed with conditions - do not merge as-is.**

---

**Reviewed by:** Transformation Portal Architect  
**Date:** 2026-02-05  
**Next Action:** Specialist to implement mandatory conditions

**Full architectural analysis:** `docs/architecture/performance_ledger_v1.7_review.md`
