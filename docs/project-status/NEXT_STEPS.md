# APEX Next Steps (Post Phase 2.1)

## Current State (2026-02-08)

✅ **Main is green** - all Phase 1/1.1/2/2.1 work merged
✅ **CI runs synthetic mode** - deterministic, fast, no ML deps required
✅ **Real execution path exists** - local/manual runs work with `--no-dry-run`

## Phase 3 Tracking Items

### Backend-Aware Dependency Validation

**Issue:** Current dependency check treats torch+transformers as universally required.
**Reality:** Some backends may not need transformers (future state).

**Proposed change:**
- Always require `torch` for real runs
- Require `transformers` only when backend uses HF/DA3/etc.
- Backend registry enforces backend-specific deps via `ensure_available()`

**Acceptance criteria:**
- [ ] Import failures (ImportError/OSError) resolve to single, actionable RuntimeError
- [ ] torch required for all real runs (always)
- [ ] transformers only required when backend needs it
- [ ] Unit tests cover:
  - [ ] broken torch install → clear error
  - [ ] backend without transformers → doesn't demand it

**Priority:** Low (Phase 2.1 policy is correct for current backends)

### Nightly Real Runs (Optional)

**Goal:** Run real pipeline on committed fixtures nightly to detect regressions.

**Prerequisites:**
- [ ] Self-hosted runner OR GitHub-hosted with ML deps cached
- [ ] Model caching strategy (HF cache, torch cache)
- [ ] Stable device config (CPU pinning for determinism)

**Priority:** Medium (nice-to-have for continuous performance monitoring)

---

**Note:** No urgent work required. Main is production-ready in hybrid mode.
