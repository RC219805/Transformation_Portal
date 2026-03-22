# Test Strategy

**Document Status:** Active
**Last Updated:** 2026-03-22
**Version:** 1.1.0
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
| **Markers** | Default (no marker), or `not ml and not slow` |
| **Duration** | < 2 minutes |
| **Dependencies** | Standard library, numpy, PIL, pyyaml |
| **Coverage Target** | 25% minimum for core modules |

**Included Tests:**
- Config parsing and validation
- IO utilities and path handling
- Schema validation
- Security functions (sanitization, validation)
- Orchestration logic (mocked backends)

**CI Command:**
```bash
pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
```

---

### Layer 2: ML-Fast

**Purpose:** Backend wiring, inference shape validation, and device placement tests.

| Attribute | Value |
|-----------|-------|
| **Markers** | `ml and not slow and not integration` |
| **Duration** | < 10 minutes |
| **Dependencies** | torch (CPU), transformers (offline mode) |
| **Test Ceiling** | 70 tests (enforced by `test_ml_fast_collection_contract.py`) |

**Requirements:**
- Must operate in offline mode (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`)
- No model downloads during test execution
- Use small fixtures or mocks for inference tests

**CI Command:**
```bash
pytest -v tests/ -ra -m "ml and not slow" --maxfail=1
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

> **Current State:** CI uses **negative marker selection** (e.g., `not ml and not slow`)
> to exclude unwanted test tiers. ADR-044 defines a target state using positive marker
> selection (e.g., `unit and not slow`). The transition will occur after full marker
> retrofit is validated in production CI.

### PR Gating Jobs

| Job | Python | Requirements | Markers |
|-----|--------|--------------|---------|
| Lint | 3.12 | `requirements-lint.txt` | N/A |
| Core Tests | 3.11, 3.12 | `requirements-ci.txt` | `not ml and not slow and not benchmark` |
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

### Quick Validation (Core)

```bash
make test-fast
# or
pytest -v tests/ -ra -m "not ml and not slow" --maxfail=1
```

### With ML Dependencies

```bash
make test-novideo
# or
pytest -v tests/ -ra -m "ml and not slow" --maxfail=1
```

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
