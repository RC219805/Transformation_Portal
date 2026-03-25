# Quality Firewall - Transformation Portal

**Status**: ✅ **FULLY OPERATIONAL**
**Last Verified**: March 2026

---

## Overview

The Quality Firewall is a multi-layer enforcement system that prevents regressions in code quality, performance, and security from reaching production. It combines:

1. **CI Quality Gates** - 9 jobs blocking PR merges
2. **Performance Regression Detection** - Scene-aware thresholds
3. **Security Scanning** - Multi-tool vulnerability detection
4. **Coverage Enforcement** - Ratcheting floor + diff coverage

---

## Quick Reference

### CI Jobs (All Required for Merge)

| Job | Purpose | Blocking |
|-----|---------|----------|
| `lint` | flake8 + black + isort | ✅ |
| `typecheck` | mypy on critical modules | ⚠️ Non-blocking |
| `security` | bandit + pip-audit + gitleaks | ✅ |
| `test-core` | Python 3.10 & 3.12 | ✅ |
| `test-ml` | ML-specific tests (3.11) | ✅ |
| `coverage-gate` | Diff coverage 80%+ | ✅ |
| `build` | Package build + install check | ✅ |
| `repo-hygiene` | Root cleanliness | ✅ |
| `quality-summary` | Aggregate gate | ✅ |

### Performance Thresholds (Scene-Dependent)

| Bucket | p50 (sec) | p95 (sec) | Filters |
|--------|-----------|-----------|---------|
| aerial_large_mps | 8.5 | 12.0 | aerial, ≥20M pixels, MPS |
| pool_medium_mps | 11.0 | 15.0 | pool, ≥10M pixels, MPS |
| interior_standard_mps | 7.0 | 10.0 | interior, ≤15M pixels, MPS |
| generic_large | 10.0 | 15.0 | ≥20M pixels (fallback) |
| generic_medium | 6.0 | 10.0 | 5M–20M pixels (fallback) |

**Verdict Logic:**
- `total > p95` → **BLOCK** (regression)
- `total > p50 × 1.5` → **WARN** (investigate)
- `total ≤ p50` → **PASS** (nominal)

---

## Local Verification

### Before Committing

```bash
# Format
black --line-length=127 src/ tests/
isort --profile=black --line-length=127 src/ tests/

# Lint
flake8 src/ tests/ --max-line-length=127

# Test (fast)
pytest -v tests/ -m "not ml and not slow"
```

### Full Pre-PR Check

```bash
# Security
bandit -r src/ -ll

# Coverage
pytest -v tests/ -m "not slow" \
  --cov=src/transformation_portal \
  --cov-report=html

# Build
python -m build
twine check dist/*
```

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [QUALITY_FIREWALL_QUICK_REF.md](implementation_notes/QUALITY_FIREWALL_QUICK_REF.md) | Detailed implementation reference |
| [QUALITY_FIREWALL_IMPLEMENTATION.md](guides/QUALITY_FIREWALL_IMPLEMENTATION.md) | Bug fixes and feature additions |
| [QUALITY_FIREWALL_VALIDATED.md](guides/QUALITY_FIREWALL_VALIDATED.md) | Validation status and evidence |
| [QUALITY_FIREWALL_BIT_DEPTH_CONTRACT.md](contracts/QUALITY_FIREWALL_BIT_DEPTH_CONTRACT.md) | Bit-depth preservation contract |
| [ci.yml](../.github/workflows/ci.yml) | CI workflow definition |
| [nightly.yml](../.github/workflows/nightly.yml) | Nightly deep checks |

---

## Nightly Deep Checks (5 Jobs)

1. **stress-tests** - Large batches, memory growth, endurance
2. **performance-benchmarks** - Regression detection with budgets
3. **memory-leak-detection** - Repeated operations profiling
4. **dependency-audit-deep** - SBOM generation, banned deps
5. **integration-tests-full** - End-to-end validation

---

## Firewall Rules Summary

### Code Quality
- **Block** if lint/format fails
- **Block** if diff coverage < 80% on changed lines
- **Block** if package build fails

### Security
- **Block** if bandit HIGH severity issues detected
- **Block** if pip-audit finds known vulnerabilities (with exceptions)
- **Block** if secrets detected in diff

### Performance
- **Block** if p95 latency increases by > 10%
- **Block** if mean latency increases by > 15%
- **Block** if failure rate > 0% for required stages

---

## Troubleshooting

### CI Job Fails
1. Check job logs in GitHub Actions
2. Run same commands locally
3. Fix issues and push updates

### Coverage Decrease Blocked
1. Run `pytest --cov` locally
2. Add tests for changed code
3. Verify diff-cover passes

### Security Scan Flags Code
1. Review bandit/gitleaks output
2. Fix if real issue
3. Add exception if false positive (document why)

### Performance Regression Detected
1. Check `nightly.yml` job logs
2. Review capsule comparison output
3. Profile with `cProfile` or `py-spy`
4. Optimize hot path or adjust threshold (with ADR)

---

**Document Version**: 1.0.0
**Last Updated**: March 2026
**Owner**: Transformation Portal Architect
