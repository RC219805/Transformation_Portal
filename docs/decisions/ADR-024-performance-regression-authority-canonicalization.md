# ADR-024: Performance Regression Authority Canonicalization

**Date:** 2026-02-08
**Status:** Approved
**Decision:** APEX system is the authoritative performance regression judge
**Architect:** Transformation Portal Architect

---

## Context

The repository currently has **two** performance regression detection systems:

1. **Legacy: `tools/performance_ledger.py`**
   - Mentioned in README Quality Firewall section
   - Thresholds: p95 > 10%, mean > 15%, failure_rate > 0%
   - Manifest-based analysis
   - Used for historical baseline tracking

2. **New: APEX Performance Observability Platform (PR #867)**
   - SQLite-backed ledger with bucketing
   - Same threshold defaults (p95 > 10%, mean > 15%)
   - Multi-zone aggregation
   - CI-integrated gating
   - Scene-aware classification

This creates a **canonicalization risk**:
- Which system is authoritative?
- What happens when they disagree?
- How do we migrate historical baselines?

## Decision

**APEX becomes the single authoritative performance regression judge.**

### Migration Strategy

**Phase 1 (Immediate):**
- ✅ APEX is authoritative for PR gating (already in CI)
- ✅ `tools/performance_ledger.py` remains available for historical queries
- ✅ README documents APEX as "the system"
- ✅ No breaking changes to existing workflows

**Phase 2 (Future - Issue #869):**
- Migrate `tools/performance_ledger.py` to use APEX ledger as backend
- Maintain CLI compatibility (backward compatibility)
- Add adapter to read legacy manifest format into APEX ledger
- Historical baselines converted to APEX schema

**Phase 3 (Long-term):**
- Deprecate direct manifest parsing in favor of ledger-first workflow
- `tools/performance_ledger.py` becomes a thin CLI wrapper over APEX APIs

## Rationale

### Why APEX wins:

1. **Superior architecture:**
   - Scene-aware bucketing (pools ≠ interiors)
   - Multi-zone aggregation (detects regional variance)
   - Structured schema (SQLite vs ad-hoc manifest parsing)

2. **CI integration:**
   - Already wired into GitHub Actions
   - Automated PR comments
   - Blocking gates with shadow mode

3. **Observability:**
   - Dashboard support (Phase 3)
   - Trend analysis (apex_trends view)
   - Regression comparisons (V1 vs V2)

4. **Contract stability:**
   - Explicit schema versioning (v3)
   - Minimum sample size protection (n >= 20)
   - Insufficient-data buckets never block

### Why preserve `tools/performance_ledger.py`:

1. **Historical baselines exist:**
   - `docs/performance/baselines/*.json` contain valuable data
   - Immediate migration would require conversion scripts

2. **Simple queries useful:**
   - Ad-hoc manifest analysis
   - Quick local validation
   - Bootstrap CI mode (no NumPy)

3. **Backward compatibility:**
   - Existing scripts may depend on it
   - Migration should be gradual, not disruptive

## Consequences

### Immediate (Phase 1):

✅ **Pros:**
- Clear authority: APEX gates PRs
- No breaking changes
- Preserves historical data

⚠️ **Cons:**
- Two tools exist (temporary duplication)
- README must clarify boundaries
- Users must know which to use

### Long-term (Phase 2-3):

✅ **Pros:**
- Single source of truth
- Unified query interface
- Simplified onboarding

⚠️ **Migration cost:**
- Convert historical baselines
- Update dependent scripts
- Test backward compatibility

## Enforcement

### Documentation updates required:

- [x] README: APEX is authoritative judge
- [x] This ADR: Migration plan documented
- [ ] `tools/performance_ledger.py` docstring: Note "see APEX for CI gating"
- [ ] Add migration script stub: `tools/migrate_legacy_baselines_to_apex.py`

### CI enforcement:

- [x] APEX workflow runs on every PR
- [x] PR comments show APEX results
- [ ] (Future) Performance ledger v1 deprecated in CI

### Tests required:

- [x] APEX contract tests pass
- [x] Gate evaluation respects insufficient-data protection
- [ ] (Future) Migration script preserves baselines accurately

## Related Decisions

- **ADR-023:** Performance ledger v1.7 (legacy system improvements)
- **ADR-019:** CLI-to-package delegation (context for enhance_image.py passthrough)

## References

- PR #867: APEX scaffolding (merged)
- Issue #868: Real pipeline integration (next step)
- `QUALITY_FIREWALL_QUICK_REF.md`: Mentions both tools (needs update)

---

**Architect Decision:** Approved 2026-02-08
**Effective immediately:** APEX is authoritative for performance gating
**Migration timeline:** Phase 2 tracked in Issue #869 (to be created)
