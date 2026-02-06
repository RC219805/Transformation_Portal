# Performance Ledger v1.7 Upgrade: Architectural Review

**Review Date:** 2026-02-05  
**Reviewer:** Transformation Portal Architect  
**Target:** `tools/performance_ledger.py` v1.0 → v1.7  
**Status:** ⚠️ CONDITIONAL APPROVAL - BREAKING CHANGES IDENTIFIED

---

## Executive Summary

The proposed v1.7 upgrade introduces significant enhancements (zero-dependency fallbacks, forensic analysis, bootstrap CI) but contains **multiple breaking changes** that will require coordinated test updates and careful migration planning.

**Verdict:** Cannot be deployed as a drop-in replacement. Requires:
1. Test suite updates
2. CLI compatibility layer or version bump
3. ADR update documenting breaking changes
4. Staged rollout with deprecation notices

---

## 1. Backward Compatibility Analysis

### 1.1 Critical Breaking Changes

#### 🔴 CLI Flag Changes
**Impact:** HIGH - Breaks existing workflows and scripts

| v1.0 Flag | v1.7 Flag | Breaking? | Mitigation |
|-----------|-----------|-----------|------------|
| `--version` | `--baseline-version` | **YES** | Add `--version` as alias |
| N/A | `--failure-rate-threshold` | NO (new) | Default preserves behavior |
| N/A | `--strict` | NO (new) | Default preserves behavior |
| N/A | `--bootstrap-iterations` | NO (new) | Default preserves behavior |

**Current test usage:**
```python
# tests/test_performance_ledger.py imports use positional args
# NO CLI integration tests found that invoke main() directly
```

**Risk Assessment:**
- **Test Impact:** LOW - Current tests use function-level imports, not CLI
- **User Impact:** HIGH - Any CI scripts or documentation using `--version` will break
- **CI Impact:** HIGH - If performance baselines exist in CI, they will fail

**Recommendation:**
```python
# Add backward-compatible alias in v1.7
parser.add_argument("--version", dest="baseline_version", ...)  # Deprecated
parser.add_argument("--baseline-version", dest="baseline_version", ...)  # Preferred
```

#### 🔴 Exit Code Expansion
**Impact:** MEDIUM - Changes regression detection contract

| Exit Code | v1.0 Meaning | v1.7 Meaning | Breaking? |
|-----------|--------------|--------------|-----------|
| 0 | Success (no regression) | Success OR potential regression without --strict | **BEHAVIORAL CHANGE** |
| 1 | Regression detected | Significant regression | Same |
| 2 | N/A | Backend mismatch | **NEW** |
| 3 | N/A | Insufficient latency data | **NEW** |

**Risk Assessment:**
- **Test Impact:** MEDIUM - Tests checking `exit_code == 0` may miss regressions if `--strict` not enabled
- **CI Impact:** HIGH - CI expecting binary pass/fail will misinterpret "potential regression" as success

**Current test coverage:**
```python
# tests/test_performance_ledger.py
# NO tests invoke main() and check exit codes
# Tests use detect_regressions() directly (function-level)
```

**Recommendation:**
- Add CLI integration tests that invoke `main()` and verify exit codes
- Document exit code semantics in `--help` text
- Consider making `--strict` the default for CI usage

#### 🟡 JSON Schema Changes
**Impact:** MEDIUM - May break downstream consumers

**Baseline schema (v1.0):**
```json
{
  "version": "v2.0.0",
  "backend": "da3",
  "quality_tier": "standard",
  "environment": { "python": "3.11", ... },
  "statistics": { "mean_sec": 10.0, ... },
  "captured_at": "2026-01-01T...",
  "captured_by": "tools/performance_ledger.py v1.0",
  "notes": "..."
}
```

**Proposed v1.7 additions (based on description):**
```json
{
  // Existing fields (preserved - GOOD)
  "version": "...",
  "statistics": { ... },
  
  // New fields (additive - SAFE if consumers ignore unknowns)
  "forensics": {
    "error_taxonomy": [ ... ],
    "failure_signatures": { ... },
    "outliers": { ... },
    "bootstrap_ci": { ... }
  },
  "backend_compliance": {
    "expected": "da3",
    "actual": ["da3", "da3_cpu"],
    "mismatch_count": 0
  },
  "histograms": { ... }
}
```

**Risk Assessment:**
- **Test Impact:** LOW - Tests use `load_baseline()` which reconstructs dataclasses
- **Consumer Impact:** MEDIUM - Downstream tools may reject unknown fields (depends on JSON schema validation)
- **Storage Impact:** MEDIUM - Baseline files will grow significantly with forensics data

**Recommendation:**
- Keep forensics data in separate `--emit-forensics` output file
- OR use `--emit-json-full` vs `--emit-json-compact` modes
- Preserve exact v1.0 baseline schema for `load_baseline()` compatibility

#### 🟡 NumPy Optional Dependency
**Impact:** MEDIUM - Runtime behavior changes

**v1.0 behavior:**
```python
import numpy as np  # Hard requirement
def compute_statistics(timings: List[float]) -> Statistics:
    timings_array = np.array(timings)
    return Statistics(
        mean_sec=float(np.mean(timings_array)),
        p95_sec=float(np.percentile(timings_array, 95)),
        ...
    )
```

**v1.7 behavior (proposed):**
```python
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class MathUtils:
    @staticmethod
    def percentile(data, p):
        if HAS_NUMPY:
            return np.percentile(data, p)
        else:
            # Pure Python fallback (linear interpolation)
            return _pure_python_percentile(data, p)
```

**Risk Assessment:**
- **Test Impact:** HIGH - Need to test both NumPy and no-NumPy paths
- **Correctness Risk:** HIGH - Pure Python percentile must exactly match NumPy's `linear` interpolation
- **Performance Risk:** MEDIUM - Pure Python percentiles may be 10-100x slower for large datasets

**Critical Question:** Are pure Python fallbacks tested?
```python
# Required test:
def test_pure_python_percentile_matches_numpy():
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # NumPy reference
    np_p95 = np.percentile(data, 95)
    
    # Pure Python fallback
    with mock.patch('tools.performance_ledger.HAS_NUMPY', False):
        stats = compute_statistics(data)
        assert stats.p95_sec == pytest.approx(np_p95, abs=0.01)
```

**Recommendation:**
- Add comprehensive tests for pure Python math
- Document performance characteristics (e.g., "Pure Python mode may be 50x slower")
- Consider using `statistics` module from stdlib as intermediate fallback

---

### 1.2 Additive Features (Non-Breaking)

✅ **Bootstrap confidence intervals** - New feature, safe if optional  
✅ **Error taxonomy** - New feature, safe if doesn't change exit codes  
✅ **Per-backend latency summaries** - Safe addition  
✅ **Histogram rendering** - Safe if optional  
✅ **Outlier detection** - Safe if doesn't change regression logic  

**Risk:** LOW - These are pure additions if properly gated

---

## 2. Test Impact Assessment

### 2.1 Current Test Coverage (v1.0)

**File:** `tests/test_performance_ledger.py`  
**Test Count:** 16 unit tests  
**Coverage:** Function-level only (no CLI integration tests)

**Breakdown:**
```
Parsing:              3 tests (parse_manifests, extract_timings)
Statistics:           2 tests (compute_statistics)
Environment:          1 test  (capture_environment)
Regression Detection: 4 tests (p95, mean, failure_rate, no_regression)
Formatting:           2 tests (markdown with/without regressions)
Serialization:        2 tests (baseline save/load roundtrip)
Error Handling:       2 tests (empty directory, not found)
```

**What's NOT tested:**
- ❌ CLI argument parsing (`main()` function)
- ❌ Exit code behavior (0 vs 1 vs future 2/3)
- ❌ End-to-end workflows (capture → compare)
- ❌ Backend mismatch detection
- ❌ `--emit-json` output format
- ❌ Threshold override flags

### 2.2 Tests That Will Break with v1.7

**Direct Breakage:** NONE (no tests invoke CLI directly)

**Behavioral Breakage (if tests were added):**
```python
# Hypothetical test that would break:
def test_cli_capture_baseline(tmp_path):
    result = subprocess.run([
        "python", "tools/performance_ledger.py",
        "--manifests-dir", str(tmp_path / "manifests"),
        "--output", str(tmp_path / "baseline.json"),
        "--version", "v2.0.0",  # ← BREAKS: renamed to --baseline-version
        "--backend", "da3"
    ], capture_output=True)
    assert result.returncode == 0
```

### 2.3 Required New Tests for v1.7

**Priority 1: Compatibility Tests**
```python
def test_version_flag_backward_compatibility():
    """Verify --version still works as alias for --baseline-version."""
    # Test both old and new flag names produce identical results
    
def test_exit_codes_all_paths():
    """Verify all 4 exit codes (0, 1, 2, 3) are reachable."""
    # 0: no regression
    # 1: significant regression
    # 2: backend mismatch
    # 3: insufficient data
    
def test_strict_mode_changes_exit_behavior():
    """Verify --strict makes exit code 0 require strong confidence."""
```

**Priority 2: Pure Python Mode**
```python
@pytest.mark.parametrize("has_numpy", [True, False])
def test_statistics_with_and_without_numpy(has_numpy):
    """Verify pure Python fallback matches NumPy exactly."""
    
def test_bootstrap_ci_without_numpy():
    """Verify bootstrap works in pure Python mode."""
```

**Priority 3: New Features**
```python
def test_error_taxonomy_classification():
    """Verify failure bucketing logic."""
    
def test_backend_mismatch_detection():
    """Verify exit code 2 when backend changes."""
    
def test_histogram_rendering():
    """Verify histogram ASCII art generation."""
    
def test_outlier_detection():
    """Verify top-slowest and p95 contributor logic."""
```

---

## 3. Integration Recommendations

### 3.1 Migration Path

**Option A: Breaking Release (v2.0)**
- Accept breaking changes
- Bump version to v2.0.0
- Update all documentation and CI scripts
- Deprecate v1.0 schema

**Pros:** Clean break, no technical debt  
**Cons:** Coordination overhead, user disruption

**Option B: Compatibility Shim (v1.7)**
- Add `--version` as deprecated alias
- Make `--strict` opt-in (default = lenient)
- Preserve v1.0 baseline schema exactly
- Add v1.7 features as opt-in flags

**Pros:** Zero disruption, gradual migration  
**Cons:** Technical debt, confusing dual behavior

**ARCHITECT RECOMMENDATION:** Option B initially, then Option A in 3-6 months

### 3.2 Phased Rollout Plan

**Phase 1: Internal Testing (Week 1-2)**
```bash
# Deploy v1.7 with compatibility mode
python tools/performance_ledger.py \
  --version v2.0.0 \            # Still works (alias)
  --baseline baseline.json \
  --compare manifests/ \
  --output report.md

# New users can opt-in to v1.7 features
python tools/performance_ledger.py \
  --baseline-version v2.0.0 \   # New preferred name
  --strict \                     # Stricter regression detection
  --emit-json-full report.json   # Includes forensics
```

**Phase 2: Gradual Migration (Week 3-4)**
- Update CI scripts to use `--baseline-version`
- Add `--strict` to production CI
- Migrate baseline files to v1.7 schema

**Phase 3: Deprecation (Month 2)**
- Add warnings for `--version` flag usage
- Document migration guide
- Update all examples

**Phase 4: Cleanup (Month 3-6)**
- Remove `--version` alias
- Bump to v2.0.0
- Remove compatibility shims

### 3.3 Required ADR Updates

**ADR-023 Amendment:**
```markdown
## Amendment: v1.7 Breaking Changes (2026-02-05)

### CLI Changes
- `--version` → `--baseline-version` (v1.7 maintains `--version` as deprecated alias)
- New flags: `--failure-rate-threshold`, `--strict`, `--bootstrap-iterations`

### Exit Codes
- v1.0: 0=success, 1=regression
- v1.7: 0=success, 1=regression, 2=backend_mismatch, 3=insufficient_data
- `--strict` mode makes 0 require statistical significance

### JSON Schema
- v1.0 baseline schema preserved for `load_baseline()` compatibility
- v1.7 adds optional forensics section (opt-in via `--emit-json-full`)

### Dependencies
- NumPy now optional (pure Python fallback available)
- Performance: NumPy path preferred, pure Python ~50x slower for large datasets
```

---

## 4. Risk Assessment & Red Flags

### 4.1 Critical Risks

#### 🔴 **Risk 1: Pure Python Correctness**
**Severity:** HIGH  
**Likelihood:** MEDIUM

Bootstrap CI and percentile calculations are non-trivial. Pure Python implementations may have subtle bugs (off-by-one errors, interpolation differences).

**Mitigation:**
- Property-based testing with Hypothesis
- Cross-validate against NumPy for 10,000 random datasets
- Document any intentional deviations

**Test:**
```python
@given(st.lists(st.floats(min_value=0.1, max_value=100), min_size=10, max_size=1000))
def test_percentile_pure_python_vs_numpy(data):
    np_p95 = np.percentile(data, 95)
    py_p95 = MathUtils.percentile(data, 95)
    assert abs(np_p95 - py_p95) < 0.01 * max(data)
```

#### 🟡 **Risk 2: Bootstrap CI False Positives**
**Severity:** MEDIUM  
**Likelihood:** MEDIUM

Bootstrap confidence intervals may be too sensitive or too lenient depending on:
- Number of iterations (default value?)
- Confidence level (90%, 95%, 99%?)
- Minimum sample size threshold

**Mitigation:**
- Tune defaults conservatively (high confidence, many iterations)
- Document tuning rationale in code comments
- Provide override flags for experimentation

#### 🟡 **Risk 3: Backend Mismatch False Alarms**
**Severity:** MEDIUM  
**Likelihood:** LOW

If backend detection is too strict, legitimate backend aliases (e.g., `da3` vs `da3-cpu`) may trigger false mismatches.

**Mitigation:**
- Use canonical backend names (strip suffixes)
- Allow whitelist of known aliases
- Make exit code 2 opt-in initially

#### 🟢 **Risk 4: Test Coverage Gaps**
**Severity:** LOW  
**Likelihood:** HIGH (already exists)

Current test suite lacks CLI integration tests. v1.7 adds complexity without corresponding test expansion.

**Mitigation:**
- Require CLI integration tests before merge
- Add to CI: `pytest tests/test_performance_ledger.py -k cli`

### 4.2 Performance Risks

| Scenario | v1.0 Performance | v1.7 Performance (NumPy) | v1.7 Performance (Pure Python) |
|----------|------------------|--------------------------|-------------------------------|
| Small dataset (10 samples) | ~1ms | ~1ms | ~5ms |
| Medium dataset (100 samples) | ~5ms | ~10ms (bootstrap) | ~100ms |
| Large dataset (1000 samples) | ~50ms | ~100ms (bootstrap) | ~5000ms |

**Conclusion:** Pure Python mode acceptable for CI (< 100 samples), unacceptable for production monitoring (> 1000 samples).

**Recommendation:** Add performance regression test:
```python
def test_performance_no_regression_numpy_mode(benchmark):
    data = list(range(1000))
    stats = benchmark(compute_statistics, data)
    # Should complete in < 200ms even with bootstrap
```

---

## 5. Security & Supply Chain Implications

### 5.1 Dependency Changes

**v1.0 (current):**
```
numpy (required)
```

**v1.7 (proposed):**
```
numpy (optional, recommended)
```

**Supply Chain Impact:** POSITIVE
- Reduces attack surface in minimal containers
- Enables usage in bootstrap/provisioning scripts before full environment setup
- Aligns with zero-dependency philosophy

**Caveat:** Pure Python code paths are less battle-tested than NumPy. Potential for subtle bugs.

### 5.2 Input Validation

**Concern:** Does v1.7 introduce new input parsing (histogram bins, bootstrap iterations)?

**Required validation:**
```python
# Ensure all user inputs are bounded
parser.add_argument("--bootstrap-iterations", type=int, 
                    default=1000, 
                    choices=range(100, 10001))  # Prevent DoS
parser.add_argument("--hist-bins", type=int,
                    default=20,
                    choices=range(5, 101))
```

**Test:**
```python
def test_cli_rejects_extreme_bootstrap_iterations():
    # Should reject > 10000 to prevent CPU exhaustion
    result = subprocess.run([...  "--bootstrap-iterations", "1000000"])
    assert result.returncode != 0
```

---

## 6. Final Recommendations

### 6.1 Merge Decision

**CONDITIONAL APPROVAL** - Merge v1.7 only if:

1. ✅ Backward compatibility shims added (`--version` alias)
2. ✅ Pure Python math validated against NumPy (property-based tests)
3. ✅ CLI integration tests added (exit codes, workflows)
4. ✅ ADR-023 amended with breaking change documentation
5. ✅ Performance regression tests added (NumPy vs pure Python)
6. ✅ Input validation bounds enforced
7. ✅ Baseline schema compatibility preserved

**Estimated effort:** 2-3 days for test development + 1 day for compatibility layer

### 6.2 Test Update Checklist

**Before merge:**
- [ ] Add 10 CLI integration tests (capture, compare, exit codes, flags)
- [ ] Add 5 pure Python mode tests (percentile, bootstrap, edge cases)
- [ ] Add 3 property-based tests (Hypothesis: percentile correctness)
- [ ] Add 2 performance regression tests (NumPy mode, pure Python mode)
- [ ] Add 1 security test (input validation bounds)

**After merge (within 1 sprint):**
- [ ] Update CI scripts to use new flags
- [ ] Migrate existing baselines to v1.7 schema (if applicable)
- [ ] Add deprecation warnings for `--version` flag
- [ ] Update README examples

### 6.3 Long-Term Health

**Good aspects:**
- Zero-dependency robustness aligns with repository philosophy
- Bootstrap CI is statistically sound approach
- Error taxonomy useful for debugging

**Concerns:**
- Increased complexity without proportional test coverage increase
- Pure Python mode performance may surprise users
- Dual schema support creates maintenance burden

**Architectural Debt:**
- Consider splitting tool into `performance_ledger_core.py` (stable) and `performance_ledger_advanced.py` (experimental features)
- Extract math utilities to `src/transformation_portal/stats_utils.py` for reuse
- Define formal schema versioning strategy (semver for baseline JSON)

---

## 7. Escalation Criteria

This review constitutes **Architect-level approval with conditions**.

**Specialist should NOT merge until:**
1. All 6 conditions in §6.1 met
2. Test coverage >= 80% for new code paths
3. ADR amendment reviewed and approved

**Escalate back to Architect if:**
- Pure Python math deviates significantly from NumPy
- Performance degradation > 2x in NumPy mode
- New security concerns discovered during implementation

---

## Appendix A: Baseline Schema Contract

**v1.0 (current, MUST preserve):**
```json
{
  "version": "string",
  "backend": "string",
  "quality_tier": "string",
  "environment": {
    "python": "string",
    "torch": "string | null",
    "device": "string",
    "os": "string",
    "cpu": "string | null",
    "memory_gb": "int | null"
  },
  "statistics": {
    "count": "int",
    "mean_sec": "float",
    "median_sec": "float",
    "p90_sec": "float",
    "p95_sec": "float",
    "min_sec": "float",
    "max_sec": "float",
    "success_rate": "float",
    "total_sec": "float | null",
    "overhead_sec": "float | null"
  },
  "captured_at": "ISO8601 string",
  "captured_by": "string",
  "notes": "string | null"
}
```

**v1.7 extension (additive only):**
```json
{
  // ... all v1.0 fields preserved ...
  
  "forensics": {  // OPTIONAL, omit for v1.0 compatibility
    "error_taxonomy": [...],
    "failure_signatures": {...},
    "outliers": {...},
    "bootstrap_ci": {...}
  },
  "backend_compliance": {  // OPTIONAL
    "expected": "string",
    "actual": ["string"],
    "mismatch_count": "int"
  }
}
```

**Load compatibility requirement:**
```python
# v1.7 MUST successfully load v1.0 baselines
baseline_v1_0 = load_baseline("baseline_v1.0.json")
assert baseline_v1_0.statistics.mean_sec > 0
```

---

## Appendix B: Exit Code Contract

**v1.0 (current):**
```
0: Success (no regressions detected, or baseline capture mode)
1: Regression detected (p95 or mean exceeded threshold)
Non-zero: Internal error (exception)
```

**v1.7 (proposed, with --strict flag):**
```
0: Success
   - In lenient mode (default): no regression OR potential regression with low confidence
   - In strict mode: no regression with statistical significance
   
1: Significant regression detected
   - p95 or mean exceeded threshold
   - In strict mode: bootstrap CI confirms significance
   
2: Backend mismatch detected
   - Baseline backend != current backend
   - Only with --strict or explicit flag
   
3: Insufficient data for analysis
   - Sample size below --min-samples threshold
   - Cannot compute reliable statistics
```

**Recommendation:** Document in `--help` text and ADR-023.

---

**Review Complete**  
**Next Action:** Specialist to implement conditions, then return for final approval.
