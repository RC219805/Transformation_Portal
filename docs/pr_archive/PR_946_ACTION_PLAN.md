# PR #946 Action Plan — Pre-Merge Hardening

**Status:** Ready for hardening work
**Target:** Merge-ready within 8-11 hours
**Authority:** Transformation Portal Architect

---

## Quick Summary

PR #946 is **production-ready** but needs **2-3 hardening items** before merge to prevent Phase II integration issues.

**Current state:** ✅ 73 tests passing, excellent architecture, comprehensive docs
**Blocking issues:** None (but see P1 items below)
**Recommended fixes:** 2 high-priority, 1 medium-priority

---

## Pre-Merge Tasks

### P1 (High Priority) — Required Before Merge

#### Task 1: Color Space Normalization Boundary
**Effort:** 4-5 hours
**Risk if skipped:** High — Phase II integration failure, dataset invalidation

**What to do:**

1. Add `color_space` field to `LinearIngestResult` dataclass
2. Update `LinearDecoder._decode_raw()` to validate camera color matrix exists
3. Return actual color space from decode (not hardcoded default)
4. Make `color_space` required in manifest schema validation
5. Add test: `test_raw_color_space_validation()`

**Files to modify:**
- `src/transformation_portal/spatial_ai/ingest/linear_decoder.py`
- `src/transformation_portal/spatial_ai/ingest/manifest_schema.py`
- `src/transformation_portal/spatial_ai/ingest/exceptions.py` (add `ColorSpaceError`)
- `tests/spatial_ai/ingest/test_linear_decoder.py`

**Acceptance criteria:**
- RAW files without color matrix raise `ColorSpaceError`
- `LinearIngestResult.color_space` is validated, not default
- Manifest schema enforces valid color space enum
- Test verifies rejection of invalid color spaces

---

#### Task 2: RAW Demosaic Determinism Pinning
**Effort:** 2-3 hours
**Risk if skipped:** Medium-High — Non-reproducible datasets, debugging nightmares

**What to do:**

1. Capture `rawpy.__version__` and `libraw_version` in provenance
2. Add `demosaic_library` field to `TransformMetadata`
3. Pin rawpy version in `requirements/raw.txt`
4. Add test: `test_raw_demosaic_determinism()`

**Files to modify:**
- `src/transformation_portal/spatial_ai/ingest/provenance.py`
- `src/transformation_portal/spatial_ai/ingest/linear_decoder.py`
- `requirements/raw.txt`
- `tests/spatial_ai/ingest/test_linear_decoder.py`

**Acceptance criteria:**
- Provenance captures rawpy/LibRaw versions
- Test verifies identical RAW → identical content hash (pixel-level)
- `requirements/raw.txt` pins rawpy minor version
- Test uses downloadable DNG sample (e.g., Adobe DNG spec samples)

---

### P2 (Medium Priority) — Strongly Recommended

#### Task 3: Manifest DAG Forward Compatibility
**Effort:** 2-3 hours
**Risk if skipped:** Medium — Schema v2.0.0 required for Phase II (migration complexity)

**What to do:**

1. Add optional `parent_artifact_hash` field to `ImageMetadataV1`
2. Add optional `pipeline_stage` field (default: `"linear_ingest"`)
3. Update `ImageManifestEntry` builder
4. Add test: `test_manifest_with_dag_fields_validates()`

**Files to modify:**
- `src/transformation_portal/spatial_ai/ingest/manifest_schema.py`
- `tests/spatial_ai/ingest/test_manifest_schema.py`

**Acceptance criteria:**
- Fields are **optional** (Phase I ignores them)
- Pydantic validation accepts DAG fields
- Test verifies forward compatibility
- Documentation notes DAG fields for Phase II

---

## Task Execution Order

**Recommended sequence:**

1. **Task 2** (demosaic determinism) — Low risk, foundational
2. **Task 1** (color space) — Depends on Task 2 provenance changes
3. **Task 3** (DAG fields) — Independent, can be parallel or last

**Parallel execution possible:**
- Task 2 and Task 3 can run in parallel (no dependencies)
- Task 1 should wait for Task 2 (shares provenance module)

---

## Testing Checklist

After completing tasks, verify:

- [ ] All 73 existing tests still pass
- [ ] New tests added for each task (3 new tests minimum)
- [ ] `pytest tests/spatial_ai/ingest/ -v` shows 76+ tests passing
- [ ] No new linter warnings
- [ ] Documentation updated (if interfaces changed)

---

## File Change Summary

**Expected diff stats after hardening:**

```
Files changed: ~6-8
Lines added: ~150-250
Lines removed: ~10-20
```

**Modules affected:**
- `linear_decoder.py` — color space validation + library versioning
- `provenance.py` — capture rawpy/LibRaw versions
- `manifest_schema.py` — DAG fields + color space enum
- `exceptions.py` — new `ColorSpaceError`
- Test files (3 modules) — new tests
- `requirements/raw.txt` — version pin

---

## Merge Criteria

**Ready to merge when:**

✅ P1 Task 1 complete (color space validation)
✅ P1 Task 2 complete (demosaic determinism)
✅ All tests passing (76+ tests)
✅ No regressions in existing functionality
✅ Code review approved by peer

**P2 Task 3 (DAG fields):**
- Strongly recommended but not blocking
- Architect preference: include to avoid schema v2.0.0 in Phase II
- If time-constrained: defer to Phase I.1

---

## Questions & Clarifications

**Q: Why is color space validation P1 (required)?**
A: Phase II (SuGaR/3DGS) requires known color space. Wrong color space → silent rendering errors → must re-ingest all data.

**Q: Can we skip demosaic determinism and fix later?**
A: Technically yes, but debugging non-deterministic data is expensive. Fix now while context is fresh.

**Q: What if Task 1 takes longer than 5 hours?**
A: Escalate to Architect. May need to simplify validation or accept documented limitation.

**Q: Can we merge without Task 3 (DAG fields)?**
A: Yes. Architect prefers including it (clean schema evolution) but not blocking.

---

## Escalation Path

**If blocked or time estimate off by >50%:**
1. Document blocker in PR comments
2. Tag `@transformation-portal-architect` for guidance
3. Consider split: merge P1-only, defer P2 to Phase I.1

**If integration issues discovered:**
1. Stop work immediately
2. Document issue with reproduction steps
3. Escalate to Architect for design decision

---

## Success Criteria

**Phase I complete when:**
- P1 tasks implemented and tested ✅
- All tests passing ✅
- Documentation updated ✅
- PR approved by Architect ✅
- Merged to `main` ✅

**Post-merge:**
- Tag release: `v1.0.0-spatial-ai-phase1`
- Update CHANGELOG.md
- Close Issue #890 (Spatial AI Foundation Phase I)

---

## Timeline Estimate

**Conservative estimate:**
- Task 1 (color space): 4-5 hours
- Task 2 (demosaic): 2-3 hours
- Task 3 (DAG fields): 2-3 hours
- Testing + integration: 1-2 hours
- **Total: 9-13 hours** (including contingency)

**Aggressive estimate (experienced developer):**
- Task 1: 3 hours
- Task 2: 2 hours
- Task 3: 2 hours
- Testing: 1 hour
- **Total: 8 hours**

**Recommendation:** Plan for 10-12 hours to allow for unexpected issues.

---

## Contact

**Architect:** transformation-portal-architect
**Specialist:** transformation-portal-specialist (for implementation questions)
**Escalation:** Per governance policy (docs/architecture/agent_governance.md)

---

**Document Version:** 1.0
**Date:** 2025-02-15
**Status:** Active — awaiting task execution
