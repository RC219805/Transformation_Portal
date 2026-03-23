# ADR-044: Test Marker Enforcement Policy

**Status:** ACCEPTED
**Date:** 2026-03-20
**Implementation Status:** IMPLEMENTED
- Enforcement infrastructure: ✅ Complete (2026-03-21)
- Marker retrofit: ✅ Complete (2026-03-21)
- CI marker-specific jobs: ✅ Complete (2026-03-23) — positive marker selection deployed
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

**Retrofit Summary (2026-03-22):**
- Automated script `scripts/validation/retrofit_test_markers.py` applied markers to 137 files
- Module-level `pytestmark` pattern used for consistency
- Coverage improved from 48.6% to 100% (verified by `--audit` mode)

---

## Decision

### 1. Retrofit Markers to All Existing Tests

All existing tests will be tagged with appropriate markers within Q2 2026:

| Test Location | Default Marker |
|--------------|----------------|
| `tests/unit/` | `@pytest.mark.unit` |
| `tests/security/` | `@pytest.mark.security` |
| `tests/integration/` | `@pytest.mark.integration` |
| `tests/benchmarks/` | `@pytest.mark.benchmark` |
| `tests/stress/` | `@pytest.mark.stress` + `@pytest.mark.slow` |
| `tests/smoke/` | `@pytest.mark.unit` |

**Note on tests/security/:** Files in `tests/security/` require the `@pytest.mark.security` marker. Authors may optionally add `@pytest.mark.unit` for tests that are fast and fit the unit tier semantics, but the enforcement script requires only `security`.

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

**Known Limitations:**
- Files with Python syntax errors emit a warning and are skipped (not hard-failed). Fix syntax errors before relying on marker validation.
- The AST parser handles common patterns but does not replicate pytest's full collector semantics. Edge cases include:
  - Multiple `pytestmark = ...` assignments at module scope (later assignment wins, not merged)
  - Deeply nested `Test*` class patterns may not be fully traversed

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

CI workflows leverage markers for test selection. The implementation now uses **positive marker selection** (selecting specific tiers).

#### Current Implementation (build.yml, ci.yml, ci-quality-firewall.yml)

All quality-control workflows use positive marker selection for core tests:

```yaml
# Core tests (build.yml, ci.yml, ci-quality-firewall.yml)
# Positive selection: explicitly select core test categories
pytest -v tests/ -m "(unit or security or regression or golden or integration) and not ml and not slow and not benchmark" --maxfail=1

# ML tests (unchanged - already uses positive selection)
pytest -v tests/ -ra -m "ml and not slow and not integration and not benchmark" --maxfail=1
```

#### Parallel Execution (ci.yml)

```yaml
# Core tests with parallelization
pytest -v tests/ -n auto -m "(unit or security or regression or golden or integration) and not ml and not slow and not benchmark and not stress"

# ML tests with parallelization
pytest -v tests/ -n auto -m "ml and not slow and not benchmark and not stress"
```

#### Future Simplification

Once marker discipline is fully established and all core tests are correctly tagged, CI can simplify to:

```yaml
# Target: Fast PR gate with parallel execution (<3 min)
pytest -n auto -m "unit and not slow" tests/

# Target: Extended PR gate with parallel execution (<10 min)
pytest -n auto -m "(unit or integration) and not slow" tests/

# Nightly full suite
pytest tests/
```

**Implementation History:**
- 2026-03-21: Added `pytest-xdist>=3.5,<4` for parallel test execution
- 2026-03-22: Action pinning complete (SHA refs in all quality-control workflows)
- 2026-03-23: Typecheck policy normalized (hard-fail in build.yml, ci.yml; soft-fail in ci-quality-firewall.yml)
- 2026-03-23: **Marker selection migrated from negative to positive selection** (build.yml, ci.yml, ci-quality-firewall.yml)

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
    pytest.mark.timeout(60),
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
- [x] CI runs marker-specific jobs (2026-03-23 — positive marker selection deployed in build.yml, ci.yml, ci-quality-firewall.yml)
- [ ] Weekly automated audit (future enhancement)

### Audit Contract Tests (2026-03-22)

The `tests/test_check_test_markers.py` file provides 54 fixture-based contract tests that pin the audit semantics:

| Test Class | Purpose |
|------------|---------|
| `TestMarkerConstants` | Verifies constant definitions match ADR-044 |
| `TestDirectoryMarkerRequirements` | Verifies directory-based requirements |
| `TestBuiltinMarkersDoNotSatisfyCoverage` | Verifies `skip`, `skipif`, etc. alone are violations |
| `TestCategoryMarkersSatisfyCoverage` | Verifies `unit`, `ml`, etc. satisfy coverage |
| `TestModuleLevelMarkerDetection` | Verifies `pytestmark` declarations work |
| `TestSmokePathMapping` | Verifies `smoke/` → `unit` requirement |
| `TestClassLevelMarkerDetection` | Verifies class-level markers propagate |
| `TestPreCommitMode` | Verifies fail-closed pre-commit behavior |
| `TestAuditMode` | Verifies full directory audit functionality |
| `TestEdgeCases` | Async functions, nested classes, syntax variations |
| `TestDirectoryTypeDetection` | Verifies directory type classification |
| `TestStressDirectoryRequirements` | Verifies `stress/` requires both `stress` + `slow` |

---

## Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Unmarked tests | 0% | <5% | ✅ Achieved (100% coverage) |
| `pytest -m unit` time | TBD | <3 min | Pending CI validation |
| `pytest -m "unit or integration"` time | TBD | <10 min | Pending CI validation |

---

## Authoritative Enforcement Order

When marker governance rules appear in multiple documents, resolve conflicts using this precedence (highest to lowest):

1. **`pyproject.toml` marker registry** — The canonical list of valid markers. `--strict-markers` enforces this at pytest collection time.
2. **`src/transformation_portal/dev/check_test_markers.py`** — The enforcement logic. Pre-commit hook and audit mode derive their semantics from this module.
3. **Workflow marker expressions** — CI workflows (`build.yml`, `ci.yml`) use marker expressions for test selection. These must align with the taxonomy.
4. **Strategy documentation** (`docs/testing/STRATEGY.md`) — Guidance for test authors. Updated to align with this ADR but considered secondary if conflicts arise.

This hierarchy ensures that machine-enforced rules take precedence over prose documentation.

---

## References

- [Testing Strategy](../testing/STRATEGY.md)
- [Q2 2026 Development Roadmap](DEVELOPMENT_ROADMAP_2026_Q2.md)

---

**Author:** Transformation Portal Architect
**Review Required:** Yes
**Revised:** 2026-03-22 (governance alignment)
