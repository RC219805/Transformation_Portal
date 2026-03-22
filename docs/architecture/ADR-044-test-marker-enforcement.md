# ADR-044: Test Marker Enforcement Policy

**Status:** ACCEPTED
**Date:** 2026-03-20
**Implemented:** 2026-03-21
**Decision Makers:** Architect
**Replaces:** None

---

## Context

The repository has a well-documented testing strategy (`docs/testing/STRATEGY.md`) that defines:
- Test tiers (Core, ML-Fast, ML-Slow, Integration, Benchmarks)
- Pytest markers (`@pytest.mark.unit`, `@pytest.mark.ml`, etc.)
- CI execution patterns

However, **~51% of tests (2,244 of 4,381 functions) lack markers**, making it impossible to:
- Run targeted test suites efficiently
- Parallelize CI jobs by test type
- Provide fast PR feedback

### Current State (as of 2026-03-21 implementation)

| Marker | Count | Expected | Status |
|--------|-------|----------|--------|
| No marker | ~210 (4.9%) | <5% | ✅ Target Met |
| `@pytest.mark.unit` | 3,100+ | 2,500+ | ✅ Complete |
| `@pytest.mark.ml` | 293 | 300+ | ✅ Near Target |
| `@pytest.mark.security` | 157+ | 150+ | ✅ Complete |
| `@pytest.mark.integration` | 50+ | 200+ | In Progress |

**Retrofit Summary (2026-03-21):**
- Automated script `scripts/validation/retrofit_test_markers.py` applied markers to 137 files
- Module-level `pytestmark` pattern used for consistency
- Coverage improved from 48.6% to 95.1%

---

## Decision

### 1. Retrofit Markers to All Existing Tests

All existing tests will be tagged with appropriate markers within Q2 2026:

| Test Location | Default Marker |
|--------------|----------------|
| `tests/unit/` | `@pytest.mark.unit` |
| `tests/security/` | `@pytest.mark.security` + `@pytest.mark.unit` |
| `tests/integration/` | `@pytest.mark.integration` |
| `tests/benchmarks/` | `@pytest.mark.benchmark` |
| `tests/stress/` | `@pytest.mark.stress` + `@pytest.mark.slow` |
| `tests/smoke/` | `@pytest.mark.unit` |

**Note on tests/smoke/:** The `tests/smoke/` directory contains quick sanity checks. While "smoke test" has a specific meaning in testing terminology, we map it to `@pytest.mark.unit` because: (1) `smoke` is not currently registered in `pyproject.toml`, and (2) smoke tests in this repository are fast, isolated checks that fit the `unit` marker semantics. If a distinct `smoke` marker becomes needed, it should be added to `pyproject.toml` first.

| Root-level tests | Analyze individually, default to `@pytest.mark.unit` |

### 2. Enforce Markers on New Tests (Fail-Closed Guard)

A pre-commit hook requires all new test functions to have at least one **category** marker. This is a **fail-closed** guard: any new test file or test function without proper markers is blocked from commit.

**Implementation:** `scripts/validation/check_test_markers.py`

The script supports two modes:
1. **Pre-commit mode**: Validates specific files passed as arguments (fail-closed)
2. **Audit mode**: Scans entire `tests/` directory for comprehensive coverage report

**Pre-commit configuration** (`.pre-commit-config.yaml`):
```yaml
- repo: local
  hooks:
    - id: check-test-markers
      name: Check Test Markers (ADR-044)
      entry: python scripts/validation/check_test_markers.py
      language: system
      files: ^tests/.*\.py$
      types: [python]
```

**Fail-Closed Behavior:**
- New `test_*.py` files without category markers are rejected
- Tests with only built-in markers (`skip`, `skipif`, etc.) are rejected
- Clear error messages guide remediation

**Usage:**
```bash
# Pre-commit mode (validate specific files)
python scripts/validation/check_test_markers.py tests/test_foo.py

# Full audit mode (scan entire tests/ directory)
python scripts/validation/check_test_markers.py --audit

# Show detailed report
python scripts/validation/check_test_markers.py --audit --verbose
```

### 3. CI Alignment

CI has been updated to leverage markers for efficiency:

```yaml
# Fast PR gate with parallel execution (<3 min)
pytest -n auto -m "unit and not slow" tests/

# Extended PR gate with parallel execution (<10 min)
pytest -n auto -m "(unit or integration) and not slow" tests/

# Nightly full suite
pytest tests/
```

**Implementation Note (2026-03-21):**
- Added `pytest-xdist>=3.5,<4` for parallel test execution
- CI workflows now use `-n auto` to parallelize tests across available CPU cores
- Pinned all GitHub Actions to commit SHAs for supply chain security
- Made mypy type checking hard-fail for critical modules

---

## Marker Taxonomy

### Category Markers vs Built-in Markers

**Important distinction:** Only *category markers* satisfy coverage requirements. Built-in markers like `skip`, `skipif`, `xfail`, `parametrize`, `usefixtures`, `filterwarnings`, and `timeout` do NOT count toward coverage.

A test with only `@pytest.mark.skipif(...)` is considered **unmarked** and will be rejected by the pre-commit hook. You must add a category marker (e.g., `@pytest.mark.unit`) alongside any built-in markers.

### Category Markers (Required for Coverage)

| Marker | Semantics | CI Inclusion |
|--------|-----------|--------------|
| `unit` | Fast, isolated, no I/O | PR gate |
| `integration` | Cross-module, may use I/O | PR gate (optional) |
| `ml` | Requires torch/ML stack | ML tier CI |
| `slow` | >10 seconds execution | Nightly only |
| `benchmark` | Performance measurement | Scheduled only |
| `security` | Security-critical paths | PR gate |
| `stress` | Resource-intensive | Manual/scheduled |
| `regression` | Regression tests with known fixtures | PR gate |
| `golden` | Golden master regression tests | PR gate |

### When to Use Which Marker

| Scenario | Recommended Marker |
|----------|-------------------|
| Fast, isolated test (<1s) | `unit` |
| Test requires torch/diffusers/ML stack | `ml` |
| Test spans multiple modules/systems | `integration` |
| Test validates security invariants | `security` |
| Test uses baseline fixtures for regression | `regression` |
| Long-running test (>10s) | `slow` (add to another category) |

### Module-Level vs Per-Test Markers

**Prefer module-level `pytestmark` declarations** for consistency:

```python
import pytest

pytestmark = pytest.mark.unit  # All tests in this file are unit tests

def test_one():
    pass

def test_two():
    pass
```

For mixed-category modules or when combining with built-in markers:

```python
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform == "win32", reason="Unix only"),
]
```

**Note:** This taxonomy is derived from the markers registered in `pyproject.toml` (which enforces `--strict-markers`). Any new marker must be added to `pyproject.toml` before use, otherwise pytest will fail collection.

---

## Consequences

### Positive
- Enable `pytest -m unit` execution in <3 minutes
- Allow CI parallelization by marker
- Align actual tests with documented strategy
- Prevent marker regression going forward

### Negative
- One-time retrofit effort (10-15 hours)
- Pre-commit hook adds friction (minimal)

---

## Implementation Plan

### Week 1
1. Tag all `tests/unit/` with `@pytest.mark.unit` (batch script)
2. Tag all `tests/security/` with `@pytest.mark.security`
3. Tag all `tests/integration/` with `@pytest.mark.integration`

### Week 2
1. ~~Analyze and tag root-level tests (181 files)~~ ✅ Complete (2026-03-21)
2. ~~Implement pre-commit hook~~ ✅ Complete
3. ~~Update CI workflow to use markers~~ ✅ Complete

---

## Enforcement

- [x] Pre-commit hook blocks unmarked tests (`check-test-markers` hook)
- [x] Audit script verifies marker coverage (`--audit` mode)
- [x] Automated retrofit script (`retrofit_test_markers.py`)
- [x] Contract tests for audit semantics (`tests/test_check_test_markers.py`)
- [ ] CI runs marker-specific jobs (in progress)
- [ ] Weekly automated audit (future enhancement)

### Audit Contract Tests (2026-03-22)

The `tests/test_check_test_markers.py` file provides 54 fixture-based contract tests that pin the audit semantics:

| Test Class | Purpose |
|------------|---------|
| `TestBuiltinMarkersDoNotSatisfyCoverage` | Verifies `skip`, `skipif`, etc. alone are violations |
| `TestCategoryMarkersSatisfyCoverage` | Verifies `unit`, `ml`, etc. satisfy coverage |
| `TestModuleLevelMarkerDetection` | Verifies `pytestmark` declarations work |
| `TestSmokePathMapping` | Verifies `smoke/` → `unit` requirement |
| `TestStressDirectoryRequirements` | Verifies `stress/` requires both `stress` + `slow` |

---

## Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Unmarked tests | 0% | <5% | ✅ Achieved (100% coverage) |
| `pytest -m unit` time | TBD | <3 min | Pending CI validation |
| `pytest -m "unit or integration"` time | TBD | <10 min | Pending CI validation |

---

## References

- [Testing Strategy](../testing/STRATEGY.md)
- [Q2 2026 Development Roadmap](DEVELOPMENT_ROADMAP_2026_Q2.md)

---

**Author:** Transformation Portal Architect
**Review Required:** Yes
**Effective Date:** Upon merge
