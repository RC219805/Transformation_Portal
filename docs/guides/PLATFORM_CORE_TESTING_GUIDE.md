# Platform Core Testing Guide

**Status**: 🔒 **LOCKED** - These standards are non-negotiable  
**Version**: 1.0  
**Date**: December 10, 2025  
**Context**: Established after Architecture Hardening completion

---

## Purpose

This guide codifies testing standards for Platform Core modules to prevent regression of critical guarantees. These standards were established after identifying and fixing weakened tests that compromised security and stability.

**Key Principle**: Tests are not just for green CI. They are **executable specifications** of critical behavior.

---

## Critical Test Guarantees (DO NOT WEAKEN)

### 🔒 Security Tests - Guarded Territory

These tests protect against path traversal, symlink attacks, and arbitrary file access. **Never weaken these assertions.**

#### `tests/test_fallbacks.py::test_path_traversal_prevention`

**What it guards**:
- PathValidator correctly rejects paths that escape allowed roots
- safe_resolve_path() raises ValueError on traversal attempts
- Both soft failures (returns False) and hard failures (raises) work

**Required assertions**:
```python
# ✅ CORRECT: Test actual API behavior
validator = PathValidator(allowed_roots=[tmp_path])
is_valid = validator.validate(dangerous_path)

# Check if path actually escapes
try:
    resolved.relative_to(tmp_path)
    assert is_valid  # Inside root - should be valid
except ValueError:
    assert not is_valid  # Outside root - must be invalid

# ✅ CORRECT: Test hard failure mode
with pytest.raises(ValueError, match="escapes allowed root"):
    safe_resolve_path(tmp_path / ".." / "outside.txt", root=tmp_path)
```

**❌ NEVER DO THIS**:
```python
# ❌ WRONG: Generic exception catching
with pytest.raises(Exception):  # Too broad!
    validator.validate(path)

# ❌ WRONG: Only checking resolve() semantics
assert not dangerous_path.resolve().is_relative_to(tmp_path)  # Doesn't test our API!

# ❌ WRONG: Vacuous assertion
assert True  # Or any assertion that can't fail
```

#### `tests/test_fallbacks.py::test_symlink_attack_prevention`

**What it guards**:
- Symlinks pointing outside allowed roots are detected and rejected
- Both PathValidator.validate() and safe_resolve() handle symlinks correctly

**Required assertions**:
```python
# ✅ CORRECT: Test actual security API
validator = PathValidator(allowed_roots=[allowed_dir])

# validate() should return False
assert not validator.validate(symlink), \
    "Symlink pointing outside allowed root should be rejected"

# safe_resolve() should raise
with pytest.raises(ValueError, match="escapes allowed root"):
    validator.safe_resolve(symlink, root=allowed_dir)
```

**Why this matters**:
- Symlink attacks are a real security vector (MITRE ATT&CK T1574.008)
- Phase 1 hardening (PR-1) specifically addresses CVE-2024-27763 class vulnerabilities
- Weak tests here could hide regressions that expose client data

#### Test Coverage Requirements for Security

**Minimum coverage** (already implemented):
- ✅ Basic path traversal (`../`)
- ✅ Nested traversal (`../../`)
- ✅ Symlinks outside root

**Recommended additions** (future work):
- [ ] Symlinks inside root (should be allowed)
- [ ] Symlink chains (symlink → dir → symlink → outside)
- [ ] Mixed separators for portability
- [ ] Unicode path attacks
- [ ] Null byte injection attempts

---

### 🔒 Checkpoint/Retry Tests - Stability Core

These tests protect Phase 1 stability guarantees. Checkpoint correctness is the foundation for all batch operations.

#### `tests/core/test_batch.py::test_batch_processor_retry_failed`

**What it guards**:
- Failed jobs move to completed after successful retry
- Checkpoint state is preserved and reloadable
- Status transitions are correct (FAILED → COMPLETED)
- Timing metadata is captured

**Required assertions**:
```python
# ✅ CORRECT: Assert initial state explicitly
assert len(loaded_job.get_failed_items()) == 1
assert len(loaded_job.get_completed_items()) == 0

# ✅ CORRECT: Assert post-retry state precisely
retried_job = batch_processor.retry_failed(loaded_job)
completed = retried_job.get_completed_items()
failed = retried_job.get_failed_items()

assert len(completed) == 1, "Failed item should complete after retry"
assert len(failed) == 0, "No items should fail with working processor"
assert completed[0].status == JobStatus.COMPLETED
assert completed[0].duration_ms > 0, "Must capture timing data"
```

**❌ NEVER DO THIS**:
```python
# ❌ WRONG: Vacuous assertion (can't fail)
assert len(completed) >= 0  # Always true!

# ❌ WRONG: Not checking status transitions
assert len(completed) > 0  # But did they come from failed items?

# ❌ WRONG: Not verifying checkpoint semantics
# (Just checking final state without confirming it came from checkpoint)
```

**Why this matters**:
- Checkpoint/retry is the foundation of Phase 1 stability
- Large batch failures cost hours if restart required
- Incorrect checkpoint state → data loss or duplicate work
- 27/27 Phase 1 tests depend on this working correctly

#### Test Coverage Requirements for Checkpoints

**Minimum coverage** (already implemented):
- ✅ Single failed item retry
- ✅ Checkpoint save/load round-trip
- ✅ Status transitions (PENDING/FAILED/COMPLETED/SKIPPED)

**Recommended additions** (future work):
- [ ] Multi-item retry (2 completed, 1 failed → retry only failed)
- [ ] Partial batch failures (some succeed, some fail mid-run)
- [ ] Corrupt checkpoint recovery (invalid JSON)
- [ ] Missing checkpoint file (should error clearly)
- [ ] Concurrent checkpoint access (thread safety)
- [ ] Checkpoint size limits (1000+ item batches)

---

## Testing Patterns for Core Modules

### Pattern 1: Test Public APIs, Not Implementation

**✅ CORRECT**:
```python
# Test the contract, not how it's implemented
from transformation_portal.core.security.path import PathValidator

def test_path_validator_rejects_traversal():
    validator = PathValidator(allowed_roots=[Path("/safe")])
    assert not validator.validate(Path("/safe/../evil"))
```

**❌ WRONG**:
```python
# Don't test internal implementation details
def test_path_validator_calls_resolve():
    validator = PathValidator(allowed_roots=[Path("/safe")])
    # Don't mock internal methods or check call counts
    with mock.patch.object(Path, 'resolve') as mock_resolve:
        validator.validate(Path("/safe/file"))
        assert mock_resolve.called  # Fragile! Implementation detail
```

**Why**: Testing public APIs allows refactoring internals without breaking tests.

---

### Pattern 2: Assert Pre-Conditions and Post-Conditions

**✅ CORRECT**:
```python
def test_batch_retry_moves_failed_to_completed():
    # GIVEN: A job with 1 failed item
    job = create_job_with_failed_item()
    assert len(job.get_failed_items()) == 1  # Pre-condition
    
    # WHEN: We retry with a working processor
    retried = processor.retry_failed(job)
    
    # THEN: Failed item moves to completed
    assert len(retried.get_completed_items()) == 1  # Post-condition
    assert len(retried.get_failed_items()) == 0
```

**❌ WRONG**:
```python
def test_batch_retry():
    job = create_job()
    retried = processor.retry_failed(job)
    # Missing pre-condition checks!
    assert len(retried.get_completed_items()) >= 0  # Vacuous
```

**Why**: Pre-conditions document assumptions; post-conditions document guarantees.

---

### Pattern 3: Use Specific Exception Types and Messages

**✅ CORRECT**:
```python
def test_safe_resolve_raises_on_traversal():
    with pytest.raises(ValueError, match="escapes allowed root"):
        safe_resolve_path(Path("/safe/../evil"), root=Path("/safe"))
```

**❌ WRONG**:
```python
def test_safe_resolve_raises():
    with pytest.raises(Exception):  # Too broad!
        safe_resolve_path(Path("/safe/../evil"), root=Path("/safe"))
```

**Why**: Specific exceptions catch incorrect error handling; message matching catches API changes.

---

### Pattern 4: Parametrize Similar Test Cases

**✅ CORRECT**:
```python
@pytest.mark.parametrize("dangerous_path,reason", [
    ("../outside.txt", "parent traversal"),
    ("../../outside.txt", "nested traversal"),
    ("subdir/../../outside.txt", "relative traversal"),
])
def test_path_validator_rejects_traversal(dangerous_path, reason):
    validator = PathValidator(allowed_roots=[Path("/safe")])
    assert not validator.validate(Path("/safe") / dangerous_path), reason
```

**❌ WRONG**:
```python
def test_path_validator_rejects_parent_traversal():
    # ... test ../
    
def test_path_validator_rejects_nested_traversal():
    # ... test ../../
    
# Lots of duplication!
```

**Why**: Parametrization reduces duplication and makes coverage explicit.

---

## Performance Regression Tests (Future)

**Status**: 🔜 Planned for Objective 3 (Weeks 3-5)

### Pattern: Baseline Comparison with Tolerance

**Template**:
```python
@pytest.mark.performance
def test_depth_processing_no_regression():
    """Ensure depth processing doesn't regress below baseline."""
    # Baseline from Phase 1 validation: 45-65ms on M4 Max
    baseline_ms = 200  # Conservative for CI (CPU)
    tolerance = 1.05   # Allow 5% variance
    
    pipeline = LuxDepthPipeline.from_preset("interior_luxury")
    test_image = load_test_image("sample_interior.jpg")
    
    start = time.perf_counter()
    result = pipeline.process(test_image)
    duration_ms = (time.perf_counter() - start) * 1000
    
    assert duration_ms < baseline_ms * tolerance, \
        f"Performance regression: {duration_ms}ms > {baseline_ms * tolerance}ms"
```

**Why**: Performance tests prevent "death by a thousand cuts" where small slowdowns accumulate.

**Guidelines**:
- Mark with `@pytest.mark.performance` (can skip in fast runs)
- Use realistic baselines from actual hardware (document source)
- Allow reasonable tolerance (5-10%) for variance
- Run on CI with consistent environment (same instance type)

---

## Fixtures and Test Data

### Core Module Fixtures

**Location**: `tests/conftest.py` (shared) or `tests/core/conftest.py` (core-specific)

**Recommended fixtures**:
```python
@pytest.fixture
def tmp_safe_dir(tmp_path):
    """Safe directory for path security tests."""
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    return safe_dir

@pytest.fixture
def sample_batch_job(tmp_path):
    """Sample batch job with mixed status items."""
    items = [
        JobItem("input1.jpg", "output1.jpg", status=JobStatus.COMPLETED),
        JobItem("input2.jpg", "output2.jpg", status=JobStatus.FAILED),
        JobItem("input3.jpg", "output3.jpg", status=JobStatus.PENDING),
    ]
    checkpoint = tmp_path / "checkpoints" / "job.json"
    checkpoint.parent.mkdir(parents=True)
    return BatchJob("test-job", items, checkpoint, created_at="2025-01-01T00:00:00Z")

@pytest.fixture
def mock_working_processor():
    """Processor that always succeeds."""
    def processor(path):
        class Result:
            def save(self, output_path):
                Path(output_path).write_text("processed")
        return Result()
    return processor
```

**Why**: Fixtures reduce test setup duplication and ensure consistency.

---

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── core/                          # Core module tests
│   ├── conftest.py                # Core-specific fixtures
│   ├── test_batch.py              # 🔒 Checkpoint/retry (GUARDED)
│   ├── test_profiler.py
│   ├── test_tiling.py
│   └── validation/
│       ├── test_comparison.py
│       ├── test_metrics.py
│       └── test_report.py
├── test_fallbacks.py              # 🔒 Security fallbacks (GUARDED)
├── test_edge_cases.py
└── stage_graph/                   # Stage graph tests
    ├── test_graph.py
    ├── test_policy.py
    └── test_stage.py
```

### Naming Conventions

**Test files**: `test_<module>.py`  
**Test functions**: `test_<behavior>_<expected_outcome>`

**Examples**:
- ✅ `test_path_validator_rejects_traversal`
- ✅ `test_batch_retry_moves_failed_to_completed`
- ✅ `test_profiler_overhead_under_5_percent`
- ❌ `test_1` (no context)
- ❌ `test_batch` (too vague)

---

## Running Tests

### Fast Tests (Development)
```bash
make test-fast
# Runs core tests, skips slow ML model tests
```

### Full Suite (Pre-commit)
```bash
make test-full
# All tests including ML, performance, integration
```

### Focused Tests (Debugging)
```bash
# Security tests only
pytest tests/test_fallbacks.py::test_path_traversal_prevention -xvs

# Core batch tests
pytest tests/core/test_batch.py -v

# Performance tests only
pytest -m performance
```

### Coverage Requirements

**Minimum coverage** (enforced in CI):
- Overall: 85%
- Core modules: 90%
- Security module: 95%

**Check coverage**:
```bash
pytest --cov=src/transformation_portal/core --cov-report=html
open htmlcov/index.html
```

---

## CI/CD Integration

### Current Workflows

**`.github/workflows/build.yml`**:
- Runs on every PR and push to `main`
- Tests on Python 3.10, 3.11, 3.12
- Linting (flake8, pylint)
- Full test suite
- Must pass before merge

**`.github/workflows/architecture-hardening.yml`**:
- Security scanning (bandit, pip-audit)
- Dependency ban guard (CVE-2024-27763)
- Hardening layer tests

### Adding Performance Tests to CI

**Future** (Objective 3, Week 3):
```yaml
# .github/workflows/performance-regression.yml
name: Performance Regression Tests
on:
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Run performance tests
        run: pytest -m performance --benchmark-only
      
      - name: Compare to baseline
        run: |
          python scripts/compare_performance.py \
            --current .benchmark/current.json \
            --baseline .benchmark/baseline.json \
            --max-regression 0.05  # 5% tolerance
```

---

## Adding New Core Modules

When adding a new module to `src/transformation_portal/core/`:

### Checklist

1. **Write tests first** (TDD)
   - [ ] Public API coverage
   - [ ] Error cases
   - [ ] Edge cases
   - [ ] Integration with existing core

2. **Security review** (if handling user input)
   - [ ] Input validation
   - [ ] Path traversal protection
   - [ ] Resource limits

3. **Performance baseline**
   - [ ] Measure overhead (<5% for instrumentation, <1% for validation)
   - [ ] Document in module docstring
   - [ ] Add performance regression test

4. **Documentation**
   - [ ] Module-level docstring with examples
   - [ ] Update `PLATFORM_CORE_API.md`
   - [ ] Add to this testing guide if critical

5. **Integration**
   - [ ] Add to `core/__init__.py` exports
   - [ ] Update compatibility adapters if needed
   - [ ] Write parity tests vs legacy

---

## Breaking Glass: When to Relax Assertions

**Short answer**: Almost never.

**Acceptable reasons**:
1. **API contract change** (documented in migration guide)
   - Update test to match new contract
   - Add deprecation period for old behavior
   - Ensure no security/stability regression

2. **False positive from external dependency**
   - Document the issue (link to upstream bug)
   - Add `pytest.mark.xfail` with reason
   - Remove when upstream fixed

3. **Flaky test due to timing**
   - Add explicit waits or retries
   - Document timing assumptions
   - Consider marking with `@pytest.mark.flaky(reruns=3)`

**Unacceptable reasons**:
- ❌ "CI is red and I need to merge"
- ❌ "This test is annoying"
- ❌ "It works on my machine"
- ❌ "We'll fix it later"

**Process for relaxing assertions**:
1. Open issue documenting why assertion is problematic
2. Propose alternative that maintains same guarantee
3. Get approval from architect or lead
4. Update test with clear comment linking to issue
5. Add TODO with deadline to restore strictness

---

## Review Checklist

When reviewing PRs that touch tests:

### Security/Stability Tests
- [ ] Are PathValidator/safe_resolve calls preserved?
- [ ] Are checkpoint state transitions still verified?
- [ ] Are exception types specific (not generic Exception)?
- [ ] Are pre/post conditions explicitly asserted?

### New Tests
- [ ] Do they test public APIs, not internals?
- [ ] Are parametrize opportunities identified?
- [ ] Is test data in fixtures, not hardcoded?
- [ ] Are failure messages descriptive?

### Test Removals
- [ ] Is there a documented reason (issue link)?
- [ ] Is equivalent coverage maintained elsewhere?
- [ ] Has architect approved the removal?

---

## References

- **Architecture Hardening Plan**: `docs/architecture/ARCHITECTURE_HARDENING_PLAN.md`
- **Strategic Action Plan**: `docs/guides/STRATEGIC_ACTION_PLAN_DEC2025.md`
- **Phase 1 Completion**: Phase 1 stability docs in `docs/`
- **Security Guidelines**: `lux_depth_v2/SECURITY.md`

---

## Appendix: Common Mistakes and Fixes

### Mistake 1: Catching Generic Exceptions

**❌ Before**:
```python
with pytest.raises(Exception):
    validator.validate(bad_path)
```

**✅ After**:
```python
with pytest.raises(ValueError, match="specific error message"):
    validator.validate(bad_path)
```

### Mistake 2: Vacuous Assertions

**❌ Before**:
```python
assert len(items) >= 0  # Can't fail!
```

**✅ After**:
```python
assert len(items) == expected_count
assert items[0].status == JobStatus.COMPLETED
```

### Mistake 3: Testing Implementation Details

**❌ Before**:
```python
# Testing internal method
assert obj._internal_counter == 5
```

**✅ After**:
```python
# Testing observable behavior
assert obj.get_count() == 5
```

### Mistake 4: No Pre-Condition Checks

**❌ Before**:
```python
def test_process():
    result = processor.process(data)
    assert result.success
```

**✅ After**:
```python
def test_process():
    # Assert starting state
    assert processor.is_ready()
    assert data.is_valid()
    
    result = processor.process(data)
    
    # Assert ending state
    assert result.success
    assert result.processed_items == len(data)
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-10  
**Approved By**: Transformation Portal Architect  
**Status**: 🔒 **LOCKED** - These standards are non-negotiable
