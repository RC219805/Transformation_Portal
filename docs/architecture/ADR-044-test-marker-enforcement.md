# ADR-044: Test Marker Enforcement Policy

**Status:** PROPOSED  
**Date:** 2026-03-20  
**Decision Makers:** Architect  
**Replaces:** None  

---

## Context

The repository has a well-documented testing strategy (`docs/testing/STRATEGY.md`) that defines:
- Test tiers (Core, ML-Fast, ML-Slow, Integration, Benchmarks)
- Pytest markers (`@pytest.mark.unit`, `@pytest.mark.ml`, etc.)
- CI execution patterns

However, **75% of tests (3,168 of 4,221 functions) lack markers**, making it impossible to:
- Run targeted test suites efficiently
- Parallelize CI jobs by test type
- Provide fast PR feedback

### Current State

| Marker | Count | Expected |
|--------|-------|----------|
| No marker | 3,168 (75%) | <5% |
| `@pytest.mark.unit` | 10 | 2,500+ |
| `@pytest.mark.integration` | 5 | 200+ |
| `@pytest.mark.ml` | 50 | 150+ |
| `@pytest.mark.security` | 18 | 50+ |

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
| `tests/smoke/` | `@pytest.mark.smoke` |
| Root-level tests | Analyze individually, default to `@pytest.mark.unit` |

### 2. Enforce Markers on New Tests

A pre-commit hook will require all new test functions to have at least one marker.

**Implementation:**
```yaml
# .pre-commit-config.yaml addition
- repo: local
  hooks:
    - id: check-test-markers
      name: Ensure test functions have markers
      entry: scripts/validation/check_test_markers.py
      language: python
      files: ^tests/.*\.py$
      types: [python]
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

Canonical markers aligned with `docs/testing/STRATEGY.md`:

| Marker | Semantics | CI Inclusion |
|--------|-----------|--------------|
| `unit` | Fast, isolated, no I/O | PR gate |
| `integration` | Cross-module, may use I/O | PR gate (optional) |
| `ml` | Requires torch/ML stack | ML tier CI |
| `slow` | >10 seconds execution | Nightly only |
| `benchmark` | Performance measurement | Scheduled only |
| `security` | Security-critical paths | PR gate |
| `stress` | Resource-intensive | Manual/scheduled |
| `smoke` | Quick sanity check | PR gate |

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

- [ ] Pre-commit hook blocks unmarked tests
- [ ] CI runs marker-specific jobs
- [ ] Weekly audit script verifies marker coverage

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Unmarked tests | 75% | <5% |
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
