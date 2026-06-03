# Production Readiness Status

## Current Status: Conditionally Ready

Transformation Portal v2.0.0 has achieved "**conditional go**" status with production-ready contracts and preset governance, but systematic quality enforcement is still being established.

### ✅ Production Ready
- API contracts versioned and stable
- Preset stability taxonomy enforced
- Core processing pipelines validated
- Depth estimation and PBR generation functional
- Docker containerization available

### ⚠️ In Progress
- **Test Coverage**: Currently ~33% overall
  - Target: 60%+ overall, 80%+ for critical modules
  - CLI coverage: 0% → 80%+ (tests implemented, pending verification)
  - Critical module coverage: Variable (orchestrator, preprocessing need improvement)
- **CI Quality Gates**: Partially enforced
  - Linting: ✅ Enforced
  - Security scans: ✅ Active
  - Test gates: ✅ Running
  - Coverage ratcheting: 🟡 Diff and cold-zone gates active where scoped
  - Branch protection: ✅ Current required check is `CI Gate` with admin enforcement and conversation resolution enabled

### Production Readiness Checklist

Before deploying to production, verify:

- [ ] **Tests passing** on clean environment (Python 3.11 and 3.12; Python 3.10 is retired)
- [ ] **Coverage gates** met (global minimum + diff coverage)
- [ ] **Security scans** clean (bandit, gitleaks, pip-audit)
- [ ] **Build verification** (wheel builds and installs successfully)
- [ ] **Staging smoke test** (docker-compose or equivalent)
- [ ] **Rollback procedure** validated (documented and tested)
- [ ] **Monitoring baseline** established (logs + key metrics)
- [ ] **Dependency audit** current (no critical vulnerabilities)

### Quality Enforcement Strategy

We use a **ratcheting quality approach**:

1. **Diff Coverage**: New/changed code must be 80%+ covered (enforced in CI)
2. **Global Minimum**: Coverage never decreases vs main branch
3. **Critical Path Floors**: Key modules (CLI, orchestrator, preprocessing) have minimum thresholds
4. **Incremental Improvement**: Coverage targets increase on schedule (+5% per sprint)

### CI/CD Pipeline

Our CI pipeline enforces quality gates on every PR:

```text
PR → Lint → Type Check → Security Scan → Tests → Coverage Gate → Build Check → Repo Hygiene → CI Gate → Merge
```

All required checks must pass before merge to `main`. See [CONTRIBUTING.md](../../CONTRIBUTING.md) and [Branch Protection Setup](../ci/BRANCH_PROTECTION_SETUP.md) for details.

### Nightly Deep Checks

Additional quality assurance runs nightly:
- **Stress tests**: Large batches, memory growth, long-run stability
- **Performance benchmarks**: Regression detection with budgets
- **Integration tests**: Full pipeline validation
- **Dependency audit**: Supply chain security monitoring

---

## Quick Status Check

Check current quality metrics:

```bash
# Run core test suite
make test-fast

# Check coverage
make coverage-report

# Security scan
make validate-ci

# Build verification
make ci-quick
```

---

## Migration from Pre-v2.0.0

If upgrading from earlier versions:

1. **Contracts**: API payloads are now versioned. Update client code to use schema-aligned requests.
2. **Presets**: Use `--list-stable` to discover production-ready presets. Experimental presets are clearly marked.
3. **Dependencies**: Check `requirements/constraints.txt` for banned dependencies (e.g., `realesrgan`).
4. **Environment**: Validate on Python 3.11+; the supported CI matrix covers Python 3.11 and 3.12.

---

## Support & Escalation

- **Issues**: Report bugs via GitHub Issues
- **Security**: See [SECURITY.md](../../SECURITY.md) for vulnerability reporting
- **Contribution**: See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines
- **Architecture**: See [Architecture docs](../architecture/) for design decisions (ADRs)

---

**Note**: This status reflects our commitment to transparency. "Conditionally ready" means the core functionality works reliably, but systematic quality enforcement is being strengthened. We will update this status as coverage and CI gates reach production thresholds.

---
