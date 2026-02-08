# APEX Next Steps: Phase 3 – Backend-Aware Validation & Nightly Real Runs

## Overview

Phase 2.1 successfully hardened dependency validation by requiring both `torch` and `transformers` for all real pipeline runs. This was the correct **blunt instrument** to ensure stability and prevent false negatives during the integration phase.

**Phase 3** evolves this into **backend-aware dependency validation** and enables **scheduled nightly real runs** with full ML inference.

---

## Phase 3 Goals

### 1. Backend-Aware Dependency Validation

**Current State (Phase 2.1):**
- `check_ml_dependencies()` requires **both** `torch` + `transformers` for real runs
- Fails fast with actionable error messages
- Catches `Exception` (not just `ImportError`) to handle broken installs

**Phase 3 Evolution:**
- `torch` remains **always required** for real runs
- `transformers` becomes **backend-conditional** (required only for HF/DA3/etc.)
- Backend registry enforces backend-specific requirements via `ensure_available()` style checks
- Validation remains fail-fast with clear guidance

**Acceptance Criteria:**
- [ ] `torch` always required for non-dry-run execution
- [ ] `transformers` only required when backend uses HuggingFace models
- [ ] Import failures (`ImportError`/`OSError`) produce single, actionable `RuntimeError`
- [ ] Unit tests cover:
  - [ ] Broken `torch` install → "missing/broken torch"
  - [ ] Backend not requiring `transformers` → doesn't demand it
  - [ ] Backend requiring `transformers` → enforces it

---

### 2. Nightly Real Runs (Scheduled CI)

**Current State:**
- PR lane: `--dry-run --synthetic` (fast, deterministic)
- Real execution: manual `workflow_dispatch` only

**Phase 3 Target:**
- **Nightly scheduled workflow** (`schedule: cron`) runs real pipeline
- Uses self-hosted runner OR GitHub-hosted with ML deps cached
- Produces **real performance capsules** for trend analysis
- Still runs in **shadow mode** (informational, non-blocking)

**Infrastructure Requirements:**
- [ ] Model caching strategy (HuggingFace cache, pinned revisions)
- [ ] Either:
  - [ ] Self-hosted runner with GPU + stable environment, OR
  - [ ] GitHub-hosted with Actions cache for models (~5GB)
- [ ] Deterministic CPU/thread pinning for reproducible timings

**Acceptance Criteria:**
- [ ] Nightly workflow runs real execution on committed fixtures
- [ ] Produces **non-synthetic** performance capsules
- [ ] Run completes in < 10 minutes (3-image fixture set)
- [ ] Ledger artifact uploaded and retained for 90 days
- [ ] Dashboard auto-deploys with real trend data

---

### 3. Performance Threshold Calibration

**Objective:** Use 2–4 weeks of nightly real run data to establish credible performance budgets.

**Method:**
1. Collect p50, p95, p99 latencies from real runs
2. Compute stability metrics (variance, outliers)
3. Set initial thresholds at **worst observed + 20% safety margin**
4. Document per-zone/per-workflow baselines

**Acceptance Criteria:**
- [ ] 2+ weeks of nightly data collected
- [ ] Performance budgets documented in `docs/apex/phase3/PERFORMANCE_BUDGETS.md`
- [ ] Thresholds encoded in `scripts/apex_gate_evaluator.py` (or config)
- [ ] At least one "would-block" scenario tested in shadow mode

---

## Migration Path (Sequencing)

### Step 1: Backend-Aware Validation
- Small, focused PR
- Refactor `check_ml_dependencies()` to accept backend context
- Backend registry implements `get_required_packages()` or similar
- Update user-facing error messages

### Step 2: Nightly Workflow (Shadow)
- Add `.github/workflows/apex_nightly_real.yml`
- Start with GitHub-hosted + HF cache
- Keep gate in shadow mode (`--mode shadow`)
- Monitor for stability/flakiness

### Step 3: Threshold Calibration
- After 2+ weeks of nightly data:
  - Analyze ledger
  - Set budgets
  - Add enforcement logic (but keep disabled via flag)

### Step 4: Enforcement (Phase 4)
- Flip `--mode enforce` in nightly workflow
- PR lane stays synthetic + shadow
- Nightly lane becomes **blocking regression gate**

---

## Success Metrics

✅ Phase 3 is complete when:
1. Backend validation is conditional and testable
2. Nightly real runs produce non-synthetic data for 2+ weeks without failure
3. Performance budgets are documented and encoded
4. The system is **ready** to enforce (but enforcement is opt-in)

---

## Related Documentation

- [Phase 2 Completion Report](../apex/phase2/COMPLETION_REPORT.md)
- [Phase 2.1 Hardening Summary](../apex/phase2/PHASE2.1_HARDENING_SUMMARY.md)
- [APEX Real Pipeline Integration](../apex/phase2/REAL_PIPELINE_INTEGRATION.md)
- [Performance Contract](../apex/PERFORMANCE_CONTRACT.md)

---

**Status:** 📋 Planning (not yet started)
**Estimated Effort:** 3–4 PRs over 2–4 weeks (backend refactor + nightly CI + calibration analysis)
**Blocking Dependencies:** None (Phase 2.1 complete)
