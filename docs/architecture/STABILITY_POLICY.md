# Stability Policy

**Version**: 1.0  
**Effective Date**: December 23, 2025  
**Review Cycle**: Quarterly

---

## Overview

This document defines the long-term stability guarantees for Transformation Portal components and provides governance for changes that impact production systems.

---

## Stability Tiers

### ✅ Tier 1: Production (Feature-Frozen)

**Scope**: `lux_depth_v2/`

**Guarantees**:
- ❄️ **Feature-frozen** - No new features, parameters, or presets
- 🔒 **API stability** - No breaking changes to CLI or service API
- 📦 **Dependency lock** - Only security updates to dependencies
- 📖 **Documentation frozen** - Behavior docs match reality exactly

**Allowed Changes**:
- Security fixes (CVE remediation)
- Bug fixes (correctness issues)
- Performance optimizations (no behavior changes)
- Documentation improvements (clarification only)
- Test enhancements (coverage, stability)

**Change Process**:
1. Open GitHub issue with `tier1-change` label
2. Provide justification (security, bug, performance)
3. Architect review required
4. Must include test coverage
5. Document in CHANGELOG

**SLA**:
- Zero critical linting errors
- 100% test pass rate
- < 0.1% production error rate
- Zero critical CVEs
- 99.9% uptime (service mode)

---

### 🔧 Tier 2: Advanced (Community-Supported)

**Scope**: `docs/advanced/`, `src/transformation_portal/streaming/`, `src/transformation_portal/context_aware_rendering/`, `material_response.py`, `luxury_video_master_grader.py`

**Guarantees**:
- ✅ **Stable** - Breaking changes rare and announced
- 📝 **Documented** - Changes documented in CHANGELOG
- 🧪 **Tested** - Moderate test coverage required
- 🔔 **Migration guides** - Provided for breaking changes

**Allowed Changes**:
- New features (with justification)
- Breaking changes (with migration guide)
- Experimental integrations (clearly labeled)
- Performance optimizations
- Refactoring (with tests)

**Change Process**:
1. Open GitHub issue or PR
2. Include tests for new functionality
3. Document breaking changes
4. Update relevant docs
5. Community review (no architect gate)

**SLA**:
- 90%+ test coverage for new code
- Zero critical linting errors
- Breaking changes announced 2 weeks in advance
- Migration guides provided

---

### ⚠️ Tier 3: Research (Experimental)

**Scope**: `docs/research/`, `experimental/` (if exists)

**Guarantees**:
- ❌ **NO stability guarantees**
- ❌ **APIs may change without notice**
- ❌ **May be removed without warning**
- ❌ **Community support only (best-effort)**

**Allowed Changes**:
- Anything (within legal/ethical bounds)
- No review required
- Minimal documentation
- Breaking changes anytime

**Requirements**:
- Clear "EXPERIMENTAL" labeling
- Cannot be imported by Tier 1 or Tier 2 code (CI-enforced)
- Basic safety checks (no security holes)

**Graduation Path**:
- 6+ months stability → Tier 2 (Advanced)
- Exceptional cases → Tier 1 (requires governance review)

---

## Change Classification

### Security Fixes

**Priority**: Critical  
**Timeline**: 48 hours for critical CVEs, 1 week for high  
**Allowed in**: All tiers  
**Process**: Expedited review, can bypass feature freeze

**Examples**:
- CVE remediation
- Input validation improvements
- Authentication/authorization fixes

### Bug Fixes

**Priority**: High (if production-blocking), Medium (otherwise)  
**Timeline**: 1 week (blocking), 2-4 weeks (non-blocking)  
**Allowed in**: All tiers  
**Process**: Standard review, requires test coverage

**Examples**:
- Incorrect output (wrong results)
- Crashes/exceptions
- Memory leaks
- Resource cleanup failures

### Performance Optimizations

**Priority**: Medium  
**Timeline**: 2-4 weeks  
**Allowed in**: All tiers  
**Process**: Benchmark before/after, no behavior changes

**Examples**:
- Throughput improvements
- Memory usage reduction
- GPU utilization optimization
- Cache efficiency

**Requirements**:
- Quantify improvement (e.g., "30% faster")
- Benchmark on representative workload
- No change to output (bit-exact preferred)

### Feature Additions

**Priority**: Low (Tier 1), Medium (Tier 2), N/A (Tier 3)  
**Timeline**: Blocked (Tier 1), 2-6 weeks (Tier 2)  
**Allowed in**: Tier 2, Tier 3 only (Tier 1 frozen)  
**Process**: Design review, test coverage, documentation

**Examples**:
- New presets (Tier 2 only)
- New processing stages (Tier 2 only)
- Experimental algorithms (Tier 3 only)

### Breaking Changes

**Priority**: Low  
**Timeline**: 4-8 weeks notice  
**Allowed in**: Tier 2, Tier 3 only (never Tier 1)  
**Process**: Migration guide required, announced in advance

**Examples**:
- API signature changes
- Default parameter changes
- Removed functionality

**Requirements** (Tier 2):
- 2 weeks advance notice (GitHub discussion)
- Migration guide with code examples
- Deprecation warnings (1 release prior if possible)
- Document rationale

---

## Monitoring Metrics

### Tier 1 (Production)

Track daily:
- ✅ Test pass rate (target: 100%)
- ✅ Linting errors (target: 0 critical)
- ✅ Security scan (target: 0 critical CVEs)
- ✅ Production error rate (target: < 0.1%)
- ✅ Service uptime (target: 99.9%)

Alert thresholds:
- Any test failure → Immediate investigation
- Critical lint error → Block merge
- Critical CVE → 48-hour remediation
- Error rate > 1% → On-call alert

### Tier 2 (Advanced)

Track weekly:
- ✅ Test coverage (target: 90%+)
- ✅ Linting errors (target: 0 critical)
- ✅ Documentation drift (target: 0 outdated docs)

Alert thresholds:
- Coverage drop > 5% → Review required
- Critical lint error → Address before next release

### Tier 3 (Research)

No formal monitoring. Community-driven quality.

---

## Enforcement

### CI/CD Gates

**Pre-merge checks** (all tiers):
- [ ] Tests passing
- [ ] No critical lint errors
- [ ] Security scan passing

**Additional for Tier 1**:
- [ ] Feature freeze compliance check
- [ ] No new parameters/presets
- [ ] Architect review approved

**Additional for Tier 2**:
- [ ] Migration guide (if breaking change)
- [ ] Documentation updated
- [ ] Test coverage maintained

### Import Guards

**CI rule**: Tier 1 code cannot import from Tier 3 code

```python
# .github/workflows/ci-import-guard.yml
# Check that lux_depth_v2/ and src/ don't import from experimental/
```

**Implementation**: See `.github/workflows/ci-consolidated.yml` (import guard job)

---

## Exception Process

### Tier 1 Feature Freeze Exception

**Criteria** (ALL must be met):
1. Security vulnerability (CVSS ≥ 7.0) OR production blocker OR data loss risk
2. Cannot wait until freeze ends (March 2026)
3. Risk assessment completed
4. Rollback plan documented

**Process**:
1. Open issue with `freeze-exception` label
2. Fill out exception template
3. Architect review + approval
4. Document decision in issue
5. Update CHANGELOG

**Approval Authority**: Transformation Portal Architect

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial stability policy |

---

## Related Documentation

- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines
- [lux_depth_v2/FEATURE_FREEZE.md](../../lux_depth_v2/FEATURE_FREEZE.md) - Feature freeze policy
- [MISSION_STATEMENT.md](../../MISSION_STATEMENT.md) - Strategic direction

---

*Stability is not about resisting change—it's about controlling change.*
