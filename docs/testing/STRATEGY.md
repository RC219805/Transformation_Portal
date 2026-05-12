# Test Strategy

**Document Status:** Active
**Last Updated:** 2026-03-23
**Version:** 1.2.0
**Related ADRs:** ADR-034 (Benchmark Exclusion), ADR-044 (Test Marker Enforcement)

> **Note:** ADR-044 is the authoritative source for marker enforcement policy.
> This document provides guidance for test authors. If conflicts arise between
> this strategy document and ADR-044, ADR-044 takes precedence.

---

## Overview

This document defines the testing strategy for the Transformation Portal repository. It establishes a tiered test architecture designed to balance coverage, execution speed, and CI resource constraints.

---

## Test Tier Architecture

### Layer 1: Core (torch-free)

**Purpose:** Fast validation of core logic without ML dependencies.

| Attribute | Value |
|-----------|-------|
| **Markers** | `unit`, `security`, `regression`, `golden`, `integration` (any non-ML category marker) |
| **CI Selection** | `(unit or security or regression or golden or integration) and not ml and not slow and not benchmark` |
| **Duration** | < 2 minutes |
| **Dependencies** | Standard library, numpy, PIL, pyyaml |
| **Coverage Target** | 25% minimum for core modules |

**Note:** Per ADR-044, all tests must have a category marker. CI uses **positive marker selection** to explicitly select core test categories.

**Included Tests:**
- Config parsing and validation
- IO utilities and path handling
- Schema validation
- Security functions (sanitization, validation)
- Orchestration logic (mocked backends)

**CI Command (PR Gating):**
```bash
pytest -v tests/ -ra -m "(unit or security or regression or golden or integration) and not ml and not slow and not benchmark" --maxfail=1
```

---

### Layer 2: ML-Fast

**Purpose:** Backend wiring, inference shape validation, and device placement tests.

| Attribute | Value |
|-----------|-------|
| **Markers** | `ml` (without `slow` or `integration`) |
| **CI Selection** | `ml and not slow and not integration and not benchmark` |
| **Duration** | < 10 minutes |
| **Dependencies** | torch (CPU), transformers (offline mode) |
| **Test Ceiling** | 75 tests (enforced by `tests/enforcement/test_ml_fast_collection_contract.py`) |

**Requirements:**
- Must operate in offline mode (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`)
- No model downloads during test execution
- Use small fixtures or mocks for inference tests

**CI Command (PR Gating):**
```bash
pytest -v tests/ -ra -m "ml and not slow and not integration and not benchmark" --maxfail=1
```

---

### Layer 3: ML-Slow / Integration

**Purpose:** Full pipeline integration, large fixtures, and stress tests.

| Attribute | Value |
|-----------|-------|
| **Markers** | `ml and slow`, `integration`, `stress` |
| **Duration** | Nightly/scheduled only |
| **Dependencies** | Full ML stack, model downloads |

**Execution:** Manual or scheduled workflows only. Excluded from PR gating CI.

---

### Layer 4: Benchmarks

**Purpose:** Performance regression detection and Quality Firewall enforcement.

| Attribute | Value |
|-----------|-------|
| **Markers** | `benchmark` |
| **Duration** | Scheduled (nightly/weekly) |
| **Execution** | Excluded from default pytest runs |

**Gate Thresholds (Quality Firewall):**
- Block if p95 latency increases by > 10%
- Block if mean latency increases by > 15%
- Block if failure rate > 0% for Golden Path stages

**CI Exclusion:** Per ADR-034, benchmark tests are excluded from PR gating via:
- Marker expression: `not benchmark`
- Environment variable opt-in: `TP_RUN_BENCHMARKS=1`

---

## Marker Taxonomy

| Marker | Purpose | Default Included | CI Tier |
|--------|---------|------------------|---------|
| `unit` | Fast unit tests (<1s each) | Yes | Core |
| `regression` | Regression tests with known fixtures | Yes | Core |
| `security` | Security hardening and trust boundary tests | Yes | Core |
| `golden` | Golden master regression tests | Yes | Core |
| `ml` | Tests requiring ML models/dependencies | No | ML-Fast |
| `slow` | Long-running tests | No | ML-Slow |
| `integration` | Multi-component integration tests | No | Integration |
| `stress` | Resource limit and large batch tests | No | Manual |
| `benchmark` | Performance benchmarks | No | Scheduled |

---

## Test Organization

### Directory Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── TEST_STATUS.md        # Test run summary
├── unit/                 # Pure unit tests (torch-free)
│   ├── depth/
│   ├── determinism/
│   └── lux_depth_v3/
├── security/             # Security-focused tests
├── golden/               # Golden master regression tests
├── integration/          # Multi-component integration
├── benchmarks/           # Performance benchmarks
├── spatial_ai/           # Spatial AI module tests
├── enforcement/          # CI/governance enforcement tests
└── *.py                  # Root-level tests (legacy, being migrated)
```

### Migration Plan

Root-level test files should be progressively migrated to appropriate subdirectories:
1. Pure unit tests → `unit/<module>/`
2. Security tests → `security/`
3. Integration tests → `integration/`

---

## Fixture Strategy

### Tier 1: Pure Fixtures (no IO, no heavy deps)

```python
@pytest.fixture
def deterministic_rng() -> np.random.Generator:
    """Provide deterministic RNG for reproducible tests."""
    return np.random.default_rng(seed=42)

@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Minimal valid config dictionary."""
    return {"model_variant": "DA3-Large", ...}
```

### Tier 2: IO Fixtures (temp files, small assets)

```python
@pytest.fixture
def temp_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create structured temporary workspace."""
    ...

@pytest.fixture
def sample_image_file(temp_workspace, sample_rgb_pil) -> Path:
    """Save sample RGB image to temp file."""
    ...
```

### Tier 3: Optional/ML Fixtures (guarded by importorskip)

```python
@pytest.fixture
def mock_depth_model(deterministic_rng):
    """Mock depth model for testing without ML deps."""
    pytest.importorskip("unittest.mock")
    ...
```

---

## CI Integration

> **Current State (2026-03-23):** CI uses **positive marker selection** for core tests.
> This explicitly selects test categories (unit, security, regression, golden, integration)
> rather than excluding unwanted tiers. ML tests use positive selection (`ml and not slow...`).

### Canonical Workflow

**`build.yml`** is the canonical PR gating workflow. It is the only workflow required by branch protection.

| Workflow | Role | Marker Semantics | Typecheck Policy |
|----------|------|------------------|------------------|
| `build.yml` | **Canonical PR gate** | ✅ Positive selection | Hard-fail mypy |
| `ci.yml` | Post-merge validation | ✅ Positive selection | Hard-fail mypy |
| `ci-quality-firewall.yml` | Post-CI verification | ✅ Positive selection | Soft-fail mypy |

> **Note:** `quality-gate.yml` is part of the broader quality-control plane but is intentionally
> excluded from this canonical CI workflow table. It runs pre-commit style checks but is not
> a branch-protection requirement. Its scope is governed by the Quality Firewall documentation.

### PR Gating Jobs

| Job | Python | Requirements | Markers |
|-----|--------|--------------|---------|
| Lint | 3.12 | `requirements-lint.txt` | N/A |
| Typecheck | 3.12 | `mypy`, `types-PyYAML` | N/A |
| Core Tests | 3.11, 3.12 | `requirements-ci.txt` | `(unit or security or regression or golden or integration) and not ml and not slow and not benchmark` |
| ML Tests | 3.11 | CPU torch + CI deps | `ml and not slow and not integration and not benchmark` |

### Excluded from PR Gating

- `benchmark` marked tests
- `stress` marked tests
- Full ML integration tests requiring model downloads

---

## Determinism Requirements

For tests validating deterministic behavior:

1. **Seed RNGs:** Use fixed seeds for `random`, `numpy`, `torch`
2. **Avoid nondeterministic ops:** Use deterministic kernels where available
3. **Use tolerant metrics:** PSNR/SSIM thresholds instead of exact byte comparison
4. **Document exceptions:** If a test cannot be deterministic, document why

---

## Adding New Tests

### Checklist

1. [ ] Choose appropriate tier (Core/ML-Fast/ML-Slow/Benchmark)
2. [ ] Add required markers (`@pytest.mark.unit`, `@pytest.mark.security`, etc.)
3. [ ] Use fixtures from `conftest.py` where appropriate
4. [ ] Ensure offline operation for ML tests
5. [ ] Update `FAST_ML_SELECTED_CEILING` if adding ML-fast tests
6. [ ] Run targeted tests locally before pushing

### Marker Application

```python
# Pure unit test (Core tier)
@pytest.mark.unit
def test_config_validation():
    ...

# Security test in tests/security/ (requires @pytest.mark.security)
# Optionally add @pytest.mark.unit for fast security tests
@pytest.mark.security
def test_path_traversal_blocked():
    ...

# ML test requiring torch
@pytest.mark.ml
def test_depth_inference_shape():
    ...

# Performance benchmark
@pytest.mark.benchmark
def test_pipeline_latency():
    ...
```

---

## Running Tests Locally

> **Note:** The commands below are convenience aliases for local development.
> For the exact PR-gating expressions used in CI, see the [PR Gating Jobs](#pr-gating-jobs) table above.

### Quick Validation (Core)

```bash
make test-fast
# or (matches PR-gating expression)
pytest -v tests/ -ra -m "(unit or security or regression or golden or integration) and not ml and not slow and not benchmark" --maxfail=1
```

### Broad Non-Video Suite

```bash
make test-novideo
# or
pytest -q -k "not video_master_grader"
```

This convenience target excludes the luxury video master grader tests. It is
not an ML-only selector.

### Full Suite

```bash
make test-full
# or
pytest -v tests/ -ra --maxfail=5
```

### Benchmarks (opt-in)

```bash
TP_RUN_BENCHMARKS=1 pytest -v tests/benchmarks/ -m "benchmark"
```

---

## Quality Gates

### Coverage Thresholds

**CI-Enforced (Current):**

| Tier | Minimum | Notes |
|------|---------|-------|
| Core tests | 20% | `--cov-fail-under=20` in build.yml |
| ML tests | — | Coverage disabled for faster PR feedback |

**Aspirational Per-Module Targets (Not Yet Enforced):**

These targets guide new test development. Per-module enforcement
will be added once tooling supports granular thresholds.

| Module | Target Coverage |
|--------|-----------------|
| `core/config` | 80% |
| `core/security` | 90% |
| `streaming` | 70% |

### Performance Gates (Quality Firewall)

- p95 latency regression: > 10% blocks merge
- Mean latency regression: > 15% blocks merge
- Failure rate: > 0% blocks merge (for required stages)

---

## References

- [pyproject.toml](../../pyproject.toml) - Pytest marker definitions
- [conftest.py](../conftest.py) - Shared fixtures
- [ADR-034](../architecture/adr/) - Benchmark exclusion policy
- [CODEBASE_AUDIT_2026_Q1.md](../architecture/CODEBASE_AUDIT_2026_Q1.md) - Testing assessment
