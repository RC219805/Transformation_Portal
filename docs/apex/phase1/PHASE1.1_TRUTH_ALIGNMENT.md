# APEX Phase 1.1: Truth Alignment

## Context

Phase 1 (#869) successfully landed platform hardening, but introduced truth drift between:
- Code behavior vs documentation
- CI configuration vs README claims
- Schema reality vs auto-detection assumptions

This Phase 1.1 fixes those mismatches to make APEX trustworthy while in "shadow + synthetic" mode.

## Issues Identified (Agent Review)

### A) Synthetic Labeling Not Bulletproof
**Problem:** PR comment tries to auto-detect synthetic via `apex_runs.is_synthetic`, but schema doesn't have that field.

**Fix:**
- Make `--synthetic` flag explicit in CLI
- Never auto-detect; callers must pass flag
- CI always passes `--synthetic` when using `--dry-run`

### B) README Claims Don't Match Operating Mode
**Problem:** README says thresholds "block" but workflow is `shadow + dry-run`.

**Fix:** Update README to clearly state:
- "Current: shadow + synthetic signals (informational)"
- "Future: enforce + real runs once ML deps + caching in place"

### C) validate_single_run_capsules() Semantics Mismatch
**Problem:**
- Name/doc: "multi-run contamination detection"
- Implementation: "mixed workflow_version per zone" (proxy)
- Comments say "logs warning" but strict default can raise

**Fix:** Either:
1. Rename to match implementation: `validate_workflow_version_consistency_by_zone()`
2. Or implement real run scoping (commit_sha/run_id)

Choosing (1) for Phase 1.1 since schema doesn't have run_id/commit_sha yet.

### D) Comment Lies in Code
**Problem:** Comments like "logs warning if violated" when strict mode raises.

**Fix:** Update all comments to match actual strict/non-strict behavior.

### E) Technical Debt Items
1. **pipeline_version="2.0.0" hardcoded** → source from package version or git SHA
2. **sha256(read_bytes()) loads whole file** → use chunked hashing

## Implementation Completed

### ✅ Task 1: Explicit Synthetic Flag
- Added `--synthetic` flag to `apex_matrix_runner.py`
- Added `--synthetic` flag to `apex_pr_comment.py`
- Updated workflow to pass `--synthetic` when `--dry-run` used
- Removed auto-detection logic that queries non-existent schema field

### ✅ Task 2: README Truth Alignment
- Updated README "Performance Monitoring" section
- Clarified current state: shadow + synthetic
- Clarified future state: enforce + real runs (Phase 2)

### ✅ Task 3: Rename validate_single_run_capsules
- Renamed to `validate_workflow_version_consistency`
- Updated docstring to match implementation
- Fixed "logs warning" → "raises ValueError (strict) or logs warning (non-strict)"
- Updated all call sites
- Updated tests

### ✅ Task 4: Fix Code Comments
- Audited all APEX module comments for truth drift
- Fixed validate_* docstrings
- Fixed workflow comments

### ✅ Task 5: Technical Debt
- Sourced pipeline_version from `__version__` or git SHA (via `_get_pipeline_version()`)
- Implemented chunked hashing for input_hash
- Cached pipeline_version outside hot loop (performance fix)
- Removed deprecated `expected_run_context` parameter

## Outcomes Achieved

✅ CI workflow explicitly passes `--synthetic` with `--dry-run`
✅ No auto-detection of synthetic status (explicit only)
✅ README accurately describes shadow + synthetic current state
✅ Function names match their implementations
✅ All comments match actual behavior (strict vs non-strict)
✅ pipeline_version sourced from version metadata
✅ Hashing uses chunked reads (not full file load)
✅ All tests pass
✅ No "truth drift" between code, docs, and CI config

## Non-Goals (Deferred to Phase 2)

- Real pipeline integration (still dry-run)
- Enforce mode (still shadow)
- Schema addition of run_id/commit_sha fields
- ML dependency installation in CI

## Success Metric

"A new contributor can read README + workflow + code and understand exactly what APEX does today vs what it will do in Phase 2, with no confusion or false confidence."

**Status: ✅ ACHIEVED** — All truth drift eliminated; APEX is now honest about its current capabilities.
