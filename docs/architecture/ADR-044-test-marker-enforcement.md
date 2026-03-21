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

### Current State (as of implementation)

| Marker | Count | Expected |
|--------|-------|----------|
| No marker | 2,244 (51%) | <5% |
| `@pytest.mark.unit` | 1,555 | 2,500+ |
| `@pytest.mark.ml` | 293 | 300+ |
| `@pytest.mark.security` | 157 | 150+ |
| `@pytest.mark.integration` | 21 | 200+ |

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

### 2. Enforce Markers on New Tests

A pre-commit hook requires all new test functions to have at least one marker.

**Implementation:** `scripts/validation/check_test_markers.py`

The script supports two modes:
1. **Pre-commit mode**: Validates specific files passed as arguments
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

CI will be updated to leverage markers for efficiency:

```yaml
# Fast PR gate (<3 min)
pytest -m "unit and not slow" tests/

# Extended PR gate (<10 min)
pytest -m "(unit or integration) and not slow" tests/

# Nightly full suite
pytest tests/
```

---

## Marker Taxonomy

Canonical markers as registered in `pyproject.toml` under `[tool.pytest.ini_options].markers`:

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
1. Analyze and tag root-level tests (181 files)
2. Implement pre-commit hook
3. Update CI workflow to use markers

---

## Enforcement

- [x] Pre-commit hook blocks unmarked tests (`check-test-markers` hook)
- [x] Audit script verifies marker coverage (`--audit` mode)
- [ ] CI runs marker-specific jobs (in progress)
- [ ] Weekly automated audit (future enhancement)

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Unmarked tests | 51% | <5% |
| `pytest -m unit` time | N/A | <3 min |
| `pytest -m "unit or integration"` time | N/A | <10 min |

---

## References

- [Testing Strategy](../testing/STRATEGY.md)
- [Q2 2026 Development Roadmap](DEVELOPMENT_ROADMAP_2026_Q2.md)

---

**Author:** Transformation Portal Architect
**Review Required:** Yes
**Effective Date:** Upon merge
