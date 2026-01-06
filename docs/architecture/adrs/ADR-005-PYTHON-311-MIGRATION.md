# ADR-005: Python 3.11 Migration Strategy

**Status**: Proposed
**Date**: 2026-01-06
**Authors**: Transformation Portal Architect
**Related Issues**: #660 (scipy 1.16), #661 (Pillow 12)
**Related Config**: `.github/dependabot.yml`, `requirements/base.in`

---

## Context

### Current State (2026-01-06)

The Transformation Portal maintains **Python 3.10-3.12 compatibility** with the following critical dependency constraints:

```python
# requirements/base.in
Pillow>=10.0.0,<12       # Pillow 12+ requires Python 3.11+
scipy>=1.15,<1.16        # scipy 1.16+ requires Python 3.11+
```

**Python Version Support Timeline**:
- **Python 3.10**: EOL **October 4, 2026** (~9 months)
- **Python 3.11**: EOL October 2027 (21 months)
- **Python 3.12**: EOL October 2028 (33 months)

### Blocked Dependency Updates

Recent Dependabot PRs were **closed** due to Python 3.11+ requirements:

| PR | Package | Version | Python Req | Status | Impact |
|----|---------|---------|------------|--------|--------|
| #660 | scipy | 1.16.3 | 3.11+ | ❌ Closed | Performance improvements, new algorithms |
| #661 | Pillow | 12.1.0 | 3.11+ | ❌ Closed | Security fixes, format support |

**Dependabot Configuration** (2026-01-06):
```yaml
ignore:
  - dependency-name: "scipy"
    versions: [">=1.16"]
  - dependency-name: "pillow"
    versions: [">=12.0"]
```

### Benefits of Python 3.11 Migration

**1. Performance Improvements**:
- 10-60% faster CPython execution (PEP 659: Specialized Adaptive Interpreter)
- Faster startup time (~10-15%)
- Lower memory overhead for large codebases
- Improved asyncio performance (relevant for FastAPI service mode)

**2. Unlocked Dependency Updates**:
- **scipy 1.16+**: New sparse matrix algorithms, improved signal processing
- **Pillow 12+**: JPEG XL support, improved WebP encoding, security patches
- Future ML dependencies (transformers, torch) increasingly targeting 3.11+

**3. Language Features** (nice-to-have):
- PEP 673: Self type annotation (cleaner generic classes)
- PEP 680: `tomllib` in stdlib (reduce `tomli` dependency)
- Improved error messages (better debugging)
- Exception groups and `except*` syntax

**4. Ecosystem Alignment**:
- Ubuntu 24.04 LTS ships with Python 3.11 as default
- GitHub Actions `ubuntu-24.04` images use Python 3.11
- AWS Lambda supports Python 3.11 (better cloud deployment)

### User Impact Analysis

**Current User Base** (estimated from download patterns):
- **Docker deployments**: 60% (controlled environment, easy migration)
- **Local development**: 30% (varied Python versions)
- **CI/CD integrations**: 10% (controlled environment)

**Migration Friction**:
- ✅ Low for Docker users (Dockerfile update)
- ⚠️ Medium for local dev (requires `pyenv` or manual upgrade)
- ✅ Low for CI/CD (GitHub Actions matrix update)

---

## Decision

We will **migrate to Python 3.11 as the minimum supported version** with the following phased approach:

### Phase 1: Preparation (Q1 2026 - Months 1-2)
**Target**: February-March 2026

1. **Audit Python 3.10-specific code**:
   - Search for workarounds targeting 3.10 (e.g., `sys.version_info` checks)
   - Identify compatibility shims or backport dependencies

2. **Update documentation**:
   - Create user migration guide: `docs/PYTHON_311_MIGRATION_GUIDE.md`
   - Update README.md with new Python version requirement
   - Add deprecation notice to CHANGELOG.md

3. **CI/CD preparation**:
   - Add Python 3.11 to test matrix (already supported, verify passing)
   - Add Python 3.13 (alpha) to test matrix for future-proofing
   - Create pre-migration validation job

4. **Communication**:
   - GitHub Discussions announcement
   - CHANGELOG entry for deprecation notice
   - Update issue templates with new Python requirement

### Phase 2: Migration (Q2 2026 - Month 3)
**Target**: April 2026 (6 months before Python 3.10 EOL)

1. **Update dependency constraints**:
   ```diff
   # requirements/base.in
   - Pillow>=10.0.0,<12
   + Pillow>=12.0.0,<14
   - scipy>=1.15,<1.16
   + scipy>=1.16,<1.18
   ```

2. **Update `pyproject.toml`**:
   ```diff
   [project]
   - requires-python = ">=3.10,<4"
   + requires-python = ">=3.11,<4"
   ```

3. **Update Dockerfile**:
   ```diff
   - FROM python:3.10-slim
   + FROM python:3.11-slim
   ```

4. **Update CI/CD**:
   ```diff
   # .github/workflows/ci-consolidated.yml
   strategy:
     matrix:
   -   python-version: ["3.10", "3.11", "3.12"]
   +   python-version: ["3.11", "3.12", "3.13"]
   ```

5. **Remove Dependabot ignore rules**:
   ```diff
   # .github/dependabot.yml
   - ignore:
   -   - dependency-name: "scipy"
   -     versions: [">=1.16"]
   -   - dependency-name: "pillow"
   -     versions: [">=12.0"]
   ```

6. **Regenerate lockfiles**:
   ```bash
   make requirements-compile
   pip install -r requirements/all.txt
   make test-full
   ```

### Phase 3: Validation & Release (Q2 2026 - Month 4)
**Target**: May 2026

1. **Comprehensive testing**:
   - Full test suite on Python 3.11, 3.12, 3.13
   - Benchmark suite validation (ensure performance gains realized)
   - Docker image smoke tests
   - Production-like integration tests

2. **Performance validation**:
   - Re-run `bench/bench_phase2.py` on Python 3.11
   - Document performance improvements in CHANGELOG
   - Update throughput baselines if significant gains

3. **Release**:
   - Version bump: `v2.0.0` (breaking change: Python version requirement)
   - Tag: `v2.0.0-python311`
   - Release notes with migration guide link
   - GitHub Release with migration checklist

4. **Post-release monitoring**:
   - Monitor issue tracker for migration problems
   - Prepare hotfix branch for critical issues
   - Update FAQ with common migration questions

---

## Consequences

### Positive

1. **Unlocked Dependency Updates**:
   - scipy 1.16+ enables better numerical stability
   - Pillow 12+ provides security patches and new formats
   - Future-proofs dependency chain

2. **Performance Gains**:
   - 10-25% faster depth estimation (CPython improvements)
   - Reduced memory footprint for batch processing
   - Faster API response times (asyncio improvements)

3. **Ecosystem Alignment**:
   - Matches Ubuntu 24.04 LTS default Python
   - Simplifies CI/CD (fewer Python versions to test)
   - Better alignment with ML ecosystem (PyTorch, Transformers)

4. **Technical Debt Reduction**:
   - Remove Python 3.10 compatibility shims
   - Cleaner code with modern Python features
   - Simplified dependency management

### Negative

1. **Breaking Change**:
   - Users on Python 3.10 must upgrade or stay on older release
   - Requires communication and documentation effort
   - Potential user churn if migration is difficult

2. **Migration Effort**:
   - ~4-6 hours of engineering time (testing, documentation)
   - Risk of edge-case breakage during migration
   - Requires coordinated release and communication

3. **Support Burden**:
   - Need to maintain old release branch for critical fixes (optional)
   - Issue tracker may see migration-related questions
   - Documentation must be clear to minimize support burden

### Mitigation Strategies

1. **Clear Communication**:
   - 3-month advance notice via CHANGELOG and GitHub Discussions
   - Prominent README badge during transition period
   - Comprehensive migration guide with examples

2. **Backward Compatibility Option**:
   - Tag final Python 3.10-compatible release as `v1.x-lts`
   - Document how to pin to this version in installation guide
   - Optionally maintain critical security fixes for 6 months

3. **Validation Safety Net**:
   - Keep Python 3.11 in CI matrix during Phase 1 (already done)
   - Run parallel testing with 3.10 and 3.11 before migration
   - Create rollback plan if critical issues discovered

4. **User Support**:
   - Create migration checklist in GitHub Discussions
   - Proactively respond to migration issues
   - Update FAQ with common problems and solutions

---

## Timeline

| Phase | Duration | Target Date | Key Milestone |
|-------|----------|-------------|---------------|
| Phase 1: Preparation | 2 months | Feb-Mar 2026 | Migration guide published |
| Phase 2: Migration | 1 month | April 2026 | `v2.0.0-beta1` released |
| Phase 3: Validation | 1 month | May 2026 | `v2.0.0` released |
| Python 3.10 EOL | - | October 2026 | End of official support |

**Buffer**: 5 months between migration and Python 3.10 EOL provides safety margin.

---

## Alternatives Considered

### Alternative 1: Maintain Python 3.10 Compatibility Indefinitely

**Pros**:
- No breaking changes for users
- Simpler short-term maintenance

**Cons**:
- Blocked dependency updates accumulate technical debt
- Miss performance improvements from newer Python versions
- Ecosystem divergence (most ML tools targeting 3.11+)
- Python 3.10 security support ends Oct 2026 anyway

**Decision**: ❌ Rejected - postpones inevitable migration, accumulates debt

### Alternative 2: Support Multiple Python Versions (3.10-3.13)

**Pros**:
- Maximum compatibility
- No forced user migration

**Cons**:
- Requires maintaining parallel dependency lockfiles
- CI matrix complexity (4 Python versions)
- Testing burden multiplied
- Cannot use Python 3.11+ only dependencies (blocks scipy 1.16+)

**Decision**: ❌ Rejected - high maintenance cost, doesn't solve dependency blockers

### Alternative 3: Immediate Migration (January 2026)

**Pros**:
- Fastest access to new dependencies
- Simplest implementation

**Cons**:
- No user preparation time
- Risk of surprising users with breaking change
- Insufficient testing period

**Decision**: ❌ Rejected - insufficient communication and validation time

---

## References

- **Python 3.11 Release Notes**: https://docs.python.org/3/whatsnew/3.11.html
- **Python 3.10 End of Life**: https://devguide.python.org/versions/
- **scipy 1.16 Release Notes**: https://scipy.github.io/devdocs/release/1.16.0-notes.html
- **Pillow 12 Release Notes**: https://pillow.readthedocs.io/en/stable/releasenotes/12.0.0.html
- **PEP 659 (Specialized Adaptive Interpreter)**: https://peps.python.org/pep-0659/
- **Ubuntu 24.04 LTS Python Version**: https://packages.ubuntu.com/noble/python3

---

## Implementation Checklist

### Phase 1: Preparation
- [ ] Create `docs/PYTHON_311_MIGRATION_GUIDE.md`
- [ ] Add deprecation notice to CHANGELOG.md
- [ ] Update README.md with Python 3.11 recommendation
- [ ] Audit codebase for Python 3.10-specific workarounds
- [ ] Announce migration plan in GitHub Discussions
- [ ] Add Python 3.13 (alpha) to CI matrix

### Phase 2: Migration
- [ ] Update `pyproject.toml` requires-python to `>=3.11,<4`
- [ ] Update `requirements/base.in` to allow Pillow 12+, scipy 1.16+
- [ ] Update `Dockerfile` to use `python:3.11-slim`
- [ ] Update `.github/workflows/ci-consolidated.yml` Python matrix
- [ ] Remove Dependabot ignore rules
- [ ] Regenerate all lockfiles with `make requirements-compile`
- [ ] Run full test suite on Python 3.11, 3.12, 3.13

### Phase 3: Validation & Release
- [ ] Run benchmark suite and document performance improvements
- [ ] Create `v2.0.0-beta1` release for early adopters
- [ ] Collect feedback from beta testers
- [ ] Create `v2.0.0` release with migration guide
- [ ] Tag final Python 3.10-compatible release as `v1.x-lts`
- [ ] Monitor issue tracker for migration problems
- [ ] Update documentation website (if exists)

---

## Approval

**Proposed by**: Transformation Portal Architect
**Date**: 2026-01-06
**Status**: ⏳ Awaiting maintainer approval

**Next Steps**:
1. Review this ADR with project maintainers
2. Approve or request revisions
3. Begin Phase 1 implementation upon approval
