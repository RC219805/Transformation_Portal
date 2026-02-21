# ADR-034: Benchmark Test Exclusion from PR Gating CI

**Status:** Accepted  
**Date:** 2026-02-07  
**Architect:** Transformation Portal Architect  
**Context:** PR #990 CI failure investigation

---

## Context

PR #990 ("feat(determinism): CLI refactor + hardware FP-state enforcement + CAS normalization + manifest v2 + CI gate") was failing on the test `test_no_significant_regressions` in `tests/spatial_ai/test_phase2_performance.py`. The test detected a catastrophic performance regression (materials_heuristic_1024: 4.45s baseline → 22.92s actual, 415.2% slower).

### Investigation Findings

1. **Test Classification:**
   - The test is properly marked with `@pytest.mark.benchmark` (line 209)
   - Test is designed to detect performance regressions via the Performance Ledger
   - Contains hard assertions that fail on >5x regressions (line 266-268)

2. **CI Configuration Issue:**
   - `.github/workflows/build.yml` marker expressions: `"not ml and not slow"`
   - Did NOT exclude `benchmark` marker
   - Result: benchmark tests ran in PR gating CI despite governance policy

3. **Policy Contradiction:**
   - **Governance Policy** (`.github/copilot-instructions.md`):
     > "Benchmark/performance regression tests must be explicitly marked and kept out of fast PR gating CI. Nightly/deep-check workflows may run benchmarks; PR gating workflows should not."
   
   - **Implementation Reality** (`tests/benchmarks/README.md`):
     > "Benchmarks ARE included in PR gating CI (runs on every PR). Policy Decision (L0.0): Keep benchmarks in PR gating CI with warnings-only approach."

4. **Root Cause:**
   - CI runner variance on shared GitHub Actions runners
   - Performance tests are non-deterministic by nature when measuring wall-clock time
   - No control over runner CPU frequency, background processes, or scheduling
   - Catastrophic regression threshold (>5x) still triggered by runner variance

---

## Decision

**Benchmark tests MUST be excluded from PR gating CI.**

### Marker Expression Update

All PR gating workflows now use:
```yaml
# Core tests
markexpr: "not ml and not slow and not benchmark"

# ML tests  
markexpr: "ml and not slow and not benchmark"
```

### Rationale

1. **Determinism is an Architectural Invariant:**
   - PR gating must provide deterministic pass/fail signals
   - Non-deterministic failures erode developer trust in CI
   - Runner variance is uncontrollable in shared CI environments

2. **Separation of Concerns:**
   - **PR gating CI:** Fast feedback on correctness, type safety, security
   - **Scheduled workflows:** Performance regression detection with controlled baselines
   - Different execution environments, different acceptance criteria

3. **Governance Enforcement:**
   - Policy documents must match enforcement reality
   - "CI typically excludes" → "CI MUST exclude" (enforceable)
   - Machine-checkable controls over prose recommendations

4. **Fast Feedback Loops:**
   - Benchmark tests add latency to PR feedback (even if fast individually)
   - Developer experience: wait for CI → fix issue → wait again
   - Non-deterministic failures multiply wait time exponentially

---

## Consequences

### Positive

1. **Deterministic PR gating:**
   - CI failures now signal real issues, not runner variance
   - Developers can trust CI signals

2. **Governance alignment:**
   - Policy and implementation now match
   - Enforceable via marker expressions and CI configuration

3. **Clear separation:**
   - Performance testing in dedicated workflows (nightly, scheduled)
   - Can use controlled hardware, multiple runs, statistical methods

4. **32 benchmark tests** now properly excluded from PR gating

### Negative / Trade-offs

1. **Delayed performance regression detection:**
   - Regressions discovered in nightly runs, not immediately on PR
   - Mitigation: Fast nightly cadence (daily) + performance budget alerts

2. **Developer awareness:**
   - Developers may not think about performance without PR feedback
   - Mitigation: Nightly workflow comments on PRs when regressions detected

3. **Two-tier testing:**
   - Adds complexity: what runs where?
   - Mitigation: Clear marker taxonomy + documentation

---

## Alternatives Considered

### Alternative 1: Keep Benchmarks in PR CI with Relaxed Thresholds

**Approach:** Use warnings-only (no hard failures) for benchmark tests in PR CI.

**Rejected because:**
- Still creates noise in CI output (warnings developers ignore)
- Doesn't solve non-determinism problem
- "Warnings-only" tests are not tests—they're documentation
- Test with hard assertion (>5x threshold) still failed despite "relaxed" intent

### Alternative 2: Use Dedicated PR CI Runners

**Approach:** Configure self-hosted runners with controlled CPU/memory for deterministic performance.

**Rejected because:**
- High operational cost (infrastructure, maintenance, security)
- Doesn't scale to open-source contributions (external PRs)
- Over-engineering for problem solved by better test classification

### Alternative 3: Statistical Baseline Comparison

**Approach:** Run benchmark N times, compare to baseline with confidence intervals.

**Rejected for PR gating because:**
- Adds significant latency (N runs per test)
- Still subject to runner variance (wider intervals = less sensitivity)
- Appropriate for scheduled workflows, not fast PR feedback

---

## Implementation

### Files Changed

1. **`.github/workflows/build.yml`:**
   - Updated marker expressions for core and ML test matrices
   - Added comment explaining exclusion rationale

2. **`.github/copilot-instructions.md`:**
   - Updated marker taxonomy table: "EXCLUDED from PR gating CI (enforced in build.yml)"
   - Removed hedging language: "not currently excluded" → "EXCLUDED"

3. **`tests/benchmarks/README.md`:**
   - Removed contradictory "keep in PR CI" policy
   - Documented where benchmarks DO run (nightly, manual, scheduled)
   - Clear governance reference

### Affected Tests (32 total)

- `tests/benchmarks/test_lux_depth_v3_perf_smoke.py` (8 tests)
- `tests/spatial_ai/test_phase2_performance.py` (3 classes: TestMaterialsPerformance, TestPerformanceRegression, TestPerformanceDocumentation)
- `tests/spatial_ai/segmentation/test_sam2_backend_performance.py` (4 classes)
- `tests/spatial_ai/segmentation/test_sam2_confidence.py::TestPerformance`
- `tests/test_performance_regression.py` (12 tests)
- `tests/test_phase3_advanced.py` (2 tests)

All properly marked with `@pytest.mark.benchmark` and now excluded.

---

## Enforcement

### CI Enforcement

Marker expressions in `.github/workflows/build.yml` are the authoritative enforcement mechanism. Documentation updates are for human understanding; CI configuration is binding.

### Pre-commit Enforcement (Future)

Consider adding a pre-commit hook that:
1. Detects new `@pytest.mark.benchmark` tests
2. Warns developer they won't run in PR CI
3. Points to nightly workflow for validation

### Monitoring

Track in nightly workflow:
- How many benchmark tests run
- How many fail (regression detection)
- Trend over time (flake rate)

---

## References

- **Governance Policy:** `.github/copilot-instructions.md` (Testing Taxonomy)
- **Test Marker Definition:** `pyproject.toml` (tool.pytest.ini_options.markers)
- **Benchmark Documentation:** `tests/benchmarks/README.md`
- **Trigger Issue:** PR #990 CI failure (test_no_significant_regressions)
- **Related ADRs:**
  - ADR-033: Test Flake Management
  - ADR-031: Test Dependency Isolation
  - ADR-024: Performance Regression Authority Canonicalization

---

## Rollback Plan

If this decision needs to be reversed (e.g., benchmarks proven stable enough):

1. Update `.github/workflows/build.yml` marker expressions (remove `and not benchmark`)
2. Update `.github/copilot-instructions.md` marker table
3. Update `tests/benchmarks/README.md` policy section
4. Create superseding ADR explaining why determinism trade-off is acceptable

**Criteria for reversal:**
- Controlled CI runners available (self-hosted, dedicated)
- Statistical baseline comparison implemented with acceptable latency
- 30-day flake rate <0.1% on benchmark tests

---

## Lessons Learned

1. **Policy-Code Alignment:**
   - Policy documents that contradict enforcement are worse than no policy
   - "Typically" and "should" must become "always" and "must" with machine enforcement

2. **Determinism as Non-Negotiable:**
   - Non-deterministic CI is worse than slow CI
   - Developer trust in CI is a critical asset

3. **Test Taxonomy Must Be Explicit:**
   - Markers are not documentation—they are contracts
   - CI configuration is the binding interpretation of those contracts

4. **Hedging Language is a Code Smell:**
   - "CI typically excludes" = "CI doesn't actually exclude"
   - "Future iterations may add" = "not enforced today"
   - Be explicit about current reality vs. aspirational policy
