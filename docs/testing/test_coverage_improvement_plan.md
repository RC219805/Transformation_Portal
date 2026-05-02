# Test Coverage Improvement Plan

**Status**: Phase 0 complete (infrastructure setup)
**Last Updated**: 2026-04-16

## Executive Summary

The repository has substantial test volume, but high-risk areas remain under-covered. This plan establishes **trustworthy CI coverage gates**, closes **highest-risk functional gaps**, and ratchets coverage upward without destabilizing delivery velocity.

### Guiding Principles

1. **Risk over raw percentage** — Coverage increases should track operational risk, not just line counts
2. **PR-required lanes must be deterministic** — No network, model downloads, or GPU assumptions in required CI
3. **Large modules by behavior slices** — `app.py` and `orchestrator.py` tested by feature surface, not monolithic file targeting
4. **Coverage must ratchet, not oscillate** — Once a package floor is raised, it should not drop
5. **Golden and benchmark suites need governance** — Explicit, reviewable, no silent mutation

## CI Lane Model

### Required on Every PR (blocking)

| Check | Tool | Threshold |
|-------|------|-----------|
| Unit tests | pytest | Pass |
| Offline integration tests | pytest | Pass |
| Contract tests | pytest | Pass |
| Security boundary tests | pytest | Pass |
| **Diff coverage** | diff-cover | ≥85% |
| **Global floor** | coverage | ≥25% |

### Optional / Nightly Lane (non-blocking)

- ML-dependent tests
- GPU-dependent tests
- Large orchestration smokes
- Benchmark trend jobs
- Large golden/regression validation

### Manual / Release Lane

- Full pipeline end-to-end validation
- High-cost golden refresh verification
- Benchmark acceptance checks
- Model-backed regression suites

## Coverage Gate Strategy

### Immediate Gate (Active)

```bash
# Enforced in CI today
diff-cover coverage.xml --compare-branch=origin/main --fail-under=85
coverage report --fail-under=25
```

### Ratcheted Package Gates (Future Milestones)

#### Milestone 1

| Package | Target |
|---------|--------|
| `events/` | 60% |
| `storage/` | 50% |
| `runtime/` | 35% |
| `app.py` | 50% |
| `lux_depth_v3/` | 55% |

#### Milestone 2

| Package | Target |
|---------|--------|
| `events/` | 70% |
| `storage/` | 65% |
| `runtime/` | 45% |
| `app.py` | 60% |
| `lux_depth_v3/` | 60% |
| `hardening/` | 65% |

#### Long-Horizon Target

- Overall repository coverage: **70%**

## Phase Plan

### Phase 0 — Baseline and Governance Setup ✅ COMPLETE

**Duration**: 2-3 days
**Status**: Complete

#### Deliverables

- [x] `pytest-cov` HTML, XML, terminal, and branch reports configured
- [x] `make coverage-report` — comprehensive local coverage
- [x] `make coverage-diff` — diff coverage vs main branch
- [x] `make coverage-package` — package-level baseline report
- [x] CI artifact upload for coverage outputs
- [x] Diff coverage reporting in CI (85% threshold)
- [x] `diff-cover` added to dev dependencies

### Phase 1 — Highest-Gap, Highest-Yield Coverage

**Duration**: ~1 week
**Status**: In progress

#### Priority Targets

- `events/store.py`
- `events/replay.py`
- `events/decorators.py`
- `storage/merkle_dag.py`
- `storage/cas_store.py`
- `app.py` feature-flag helpers, auth enforcement, typed error envelopes
- `core/cas_dag_executor.py` (new — 739 LOC executor with no direct tests)
- `core/security/torch_security.py` (new — 512 LOC with no direct tests)

#### Deliverables

- [ ] New `tests/events/` suite
- [x] `tests/events/test_store_malformed.py` — malformed/corrupt event-file paths
- [x] Expanded `tests/storage/` — `test_cas_store_hash_mismatch.py`
- [x] `tests/test_app_feature_flags.py`
- [x] `tests/test_app_security.py`
- [x] `tests/core/test_cas_dag_executor.py`
- [x] `tests/core/test_torch_security.py`
- [x] CI lint banning `assert True` in `tests/` (`scripts/ci/check_no_tautological_tests.py`)

### Coverage Source Pruning

The following packages are excluded from `[tool.coverage.run].source` until they
grow a smoke suite. They were dragging the global floor down with no tested
behavior. Re-include a package by removing its entry from `pyproject.toml` and
adding a `tests/<pkg>/` suite:

- `depth_intelligence/`
- `diffusion/`
- `dwm/`
- `interfaces/`
- `pfm/`

### Phase 2 — Business-Critical Orchestration Coverage

**Duration**: 1.5-2 weeks
**Status**: Pending

#### Focus

- `lux_depth_v3/orchestrator.py` (split by behavior slices)
- Dispatch lifecycle
- Run card generation
- Manifest creation
- Preview/config normalization
- Batch partial-failure behavior
- Backend-selection / fallback logic

#### Test File Structure

```
tests/lux_depth_v3/
├── test_orchestrator_preview.py
├── test_orchestrator_dispatch.py
├── test_orchestrator_manifests.py
├── test_orchestrator_partial_failure.py
└── test_orchestrator_backend_resolution.py
```

### Phase 3 — Runtime and Execution-Path Hardening

**Duration**: 1-1.5 weeks
**Status**: Pending

#### Focus

- `sandbox_executor.py`
- `process_executor.py`
- `engine.py`
- `autoscaler.py`
- `worker.py`
- `ledger.py`

### Phase 4 — Security and Boundary Coverage

**Duration**: ~1 week
**Status**: Pending

#### Focus

- `hardening/universal.py`
- `utils/security.py`
- `utils/input_validation.py`
- Path traversal tests
- Hostile input normalization
- Injection prevention

### Phase 5 — ML, Optional, and Benchmark Lanes

**Duration**: 1-2 weeks
**Status**: Pending (non-blocking)

#### Focus

- `spatial_ai/`
- `evals/`
- `vlm/`
- `inference.py` mocked-backend tests
- Benchmark and golden governance

## Local Developer Workflow

```bash
# Generate comprehensive coverage report
make coverage-report
# Opens htmlcov/index.html for visual exploration

# Check diff coverage before PR
make coverage-diff
# Requires coverage.xml from coverage-report

# Package-level baseline for ratcheting decisions
make coverage-package

# Quick scoped coverage for specific work
make coverage-fast-scope
```

## Test Design Standards

### Markers (ADR-044)

- `unit` — fast unit tests (<1s each)
- `integration` — tests requiring multiple components
- `security` — security-focused tests
- `ml` — tests requiring ML models
- `benchmark` — performance benchmarks
- `slow` — slow tests
- `regression` — regression tests with fixtures

### Fixture Policy

- Use smallest deterministic fixture that proves the behavior
- No network in required lanes
- No model downloads in required lanes
- No GPU assumption in required lanes
- Avoid giant binary fixtures

### Large-Module Testing Policy

For files like `app.py` (7,311 lines) and `orchestrator.py` (6,237 lines), split tests by behavior slice:

- auth
- rollout / feature flags
- preview
- dispatch
- manifests
- partials / batch failures
- telemetry
- storage consistency
- executor lifecycle

## Golden Fixture Policy

### Storage Layout

```
tests/golden/
├── lux_depth_v3/
│   ├── run_cards/
│   └── manifests/
├── provenance/
├── archive_gates/
└── security/
```

### Update Policy

- Updates must be explicit, never automatic
- Dedicated `--accept` or helper script flow
- Human review of fixture diffs in PRs
- README per directory describing provenance

## Benchmark Policy

### PR-Required Budgets (p95)

| Category | Budget |
|----------|--------|
| Pure validation / path / serialization | ≤50ms |
| Route-support helpers, envelope builders | ≤150ms |
| Storage and DAG on small fixtures | ≤250ms |
| Offline preview/config normalization | ≤1s |
| Small non-ML smoke scenarios | ≤5s |

### ML/Orchestration Benchmarks

- Regression-vs-baseline thresholds (fail on >20% slowdown)
- Run only in nightly/manual lanes
- Collect on stable hardware

## Success Criteria

A phase is complete when:

- [x] Required CI enforces diff coverage and no unintended regression
- [ ] New tests are correctly marked under ADR-044 conventions
- [ ] Touched high-risk modules have success-path, failure-path, and boundary tests
- [ ] Tests run offline unless explicitly marked otherwise
- [ ] Package coverage ratchets are updated only after stable baselines
- [ ] Golden fixtures, if added, have explicit update path and review expectations

## References

- `docs/guides/coverage-improvement-plan.md` — Quick reference
- `docs/guides/QUALITY_FIREWALL.md` — Quality gates overview
- `pyproject.toml` — Coverage configuration
- `.github/workflows/ci.yml` — CI coverage gate implementation
