# APEX Phase 4 Execution Plan: From Shadow to Authority
**Date:** 2026-02-09
**Status:** Post-Phase 3 merge (PR #878)
**Goal:** Move APEX from synthetic shadow mode to production performance judge

---

## Current State Assessment

✅ **What's working:**
- Matrix runner executes v1/v2 workflows across zones
- Performance ledger (SQLite) + aggregation pipeline operational
- PR comment generation (idempotent) with synthetic labeling
- Backend-aware dependency validation (Phase 3: merged)
- Hybrid CI strategy: PR lane stays synthetic, real lane is opt-in

⚠️ **What's blocking production use:**
- Still running `--dry-run --synthetic` in all PR CI
- No real-run lane established (manual/nightly)
- Registry lookup couples to private `_backends` internals
- Nightly Deep Checks workflow has tooling failures (not product failures)
- Performance baselines not deterministic/reproducible
- Depth Pro integration incomplete

---

## Phase 4 Priorities (Ordered by Impact)

### Tier 1: Fix Papercuts (Low Effort / High Leverage)
**Goal:** Eliminate future confusion + unlock next phase work

#### 1.1 Make Registry Lookup a Public API
**Impact:** Stop coupling runner to registry internals
**Effort:** 20 min
**Files:** `src/transformation_portal/depth/backends/__init__.py`, `scripts/apex_matrix_runner.py`

**Action:**
```python
# Add to DepthBackendRegistry:
def get_backend_class(self, backend_id: str) -> type[DepthBackend] | None:
    """Public API for backend class lookup."""
    return self._backends.get(backend_id)

def keys(self) -> list[str]:
    """List registered backend IDs."""
    return list(self._backends.keys())
```

Then update `check_ml_dependencies()` to use `registry.get_backend_class(backend_id)`.

**Acceptance:**
- No direct `._backends` access in user code
- `pytest tests/test_apex_backend_deps.py -xvs` still passes

---

#### 1.2 Fix Phase 3 Docs Examples
**Impact:** Prevent doc rot
**Effort:** 10 min
**File:** `docs/apex/phase3/README.md`

**Issues:**
- Shell commands fenced as `python` (wrong syntax highlighting)
- Missing required CLI flags (`--output-dir`, `--ledger-db`)
- `--input-dir` points to broad fixtures dir instead of specific subset

**Action:**
Update examples to be copy/paste runnable:
```bash
python scripts/apex_matrix_runner.py \
  --workflow-versions v1 v2 \
  --zones local \
  --backend-id da3 \
  --input-dir ./tests/fixtures/apex_images \
  --sample-size 3 \
  --output-dir ./apex_results \
  --ledger-db ./apex_performance.db \
  --device cpu
```

---

### Tier 2: Make Phase 3 Operational in CI
**Goal:** Backend-aware deps become actual CI behavior

#### 2.1 Split Integration Test Markers
**Impact:** Stop hard-installing transformers in lanes that don't need it
**Effort:** 30 min
**Files:** `tests/conftest.py`, `.github/workflows/nightly_deep_checks.yml`

**Current problem:**
Nightly Deep Checks "Full Integration Tests" unconditionally installs `transformers` even for non-HF backends.

**Action:**
```python
# In conftest.py / test files
pytest.mark.integration_hf  # requires transformers
pytest.mark.integration_nonhf  # does not require transformers
pytest.mark.integration_depth_pro  # requires depth_pro
```

Update workflow:
```yaml
- name: Install ML deps (backend-aware)
  run: |
    if pytest --collect-only -m "integration_hf" tests/ -q; then
      pip install -e ".[ml]"
    elif pytest --collect-only -m "integration_depth_pro" tests/ -q; then
      pip install depth-pro
    fi
```

---

#### 2.2 Add Real Registry Integration Smoke Test
**Impact:** Catch future wiring breaks
**Effort:** 20 min
**File:** `tests/test_apex_backend_deps.py`

**Missing:** Test that exercises `registry → backend → required_packages() → import check` end-to-end.

**Action:**
```python
def test_real_registry_backend_dependency_chain():
    """Integration test: registry → backend class → dependency check."""
    from transformation_portal.depth.backends import get_registry
    from scripts.apex_matrix_runner import check_ml_dependencies

    registry = get_registry()
    # Pick a known backend (da3)
    result, missing = check_ml_dependencies("da3", registry)

    # In CI without ML deps, should get clear message
    if not result:
        assert "transformers" in missing or "torch" in missing
```

---

### Tier 3: Stabilize Nightly Deep Checks
**Goal:** Turn failures into actionable signals

#### 3.1 Fix Deep Dependency Audit Job
**Impact:** Make SBOM + vulnerability audit actually run
**Effort:** 15 min
**File:** `.github/workflows/nightly_deep_checks.yml`

**Current failure:** `cyclonedx-bom: command not found` (exit 127)

**Root cause:** Package installs `cyclonedx-py` but CLI is `cyclonedx-bom`.

**Action:**
```yaml
- name: Deep Dependency Audit
  run: |
    pip install pip-audit safety cyclonedx-bom
    python -m pip_audit --format json > pip-audit-report.json
    safety check --json > safety-report.json
    cyclonedx-py --format json --output sbom.json
```

Then ensure artifacts upload those exact files.

---

#### 3.2 Fix Artifact Upload Paths
**Impact:** "Always upload logs" actually uploads something
**Effort:** 10 min
**Files:** `.github/workflows/nightly_deep_checks.yml`

**Issues:**
- Stress tests upload `logs/*.log` but those files don't exist
- Memory leak job uploads `*.dat` but script doesn't generate them

**Action:**
- Ensure test runners write logs to expected paths, OR
- Change artifact paths to match reality:
```yaml
- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: stress-test-outputs
    path: |
      pytest-results/
      .coverage
```

---

#### 3.3 Make Performance Baselines Deterministic
**Impact:** Perf regression checks become reproducible
**Effort:** 45 min (merge existing PR work)
**Status:** You already have refactoring in progress toward ledger/subprocess approach

**Action:**
- Merge performance regression refactor PR (if not already)
- Align nightly job to use same ledger-based comparison model as APEX
- Store baseline in artifact/cache keyed by commit SHA or tag

---

### Tier 4: Real-Run Lane (The Big Move)
**Goal:** Collect real data without slowing PRs

#### 4.1 Add `workflow_dispatch` Inputs to APEX Workflow
**Impact:** Enable on-demand real runs
**Effort:** 20 min
**File:** `.github/workflows/apex_performance.yml`

**Action:**
```yaml
on:
  pull_request:
  workflow_dispatch:
    inputs:
      mode:
        description: 'Execution mode'
        required: true
        default: 'synthetic'
        type: choice
        options:
          - synthetic
          - real
      backend_id:
        description: 'Backend to use (da3, depth_pro, mock)'
        required: false
        default: 'da3'
      sample_size:
        description: 'Number of images per workflow'
        required: false
        default: '3'
      device:
        description: 'Device (cpu, cuda, mps)'
        required: false
        default: 'cpu'
```

Then wire inputs into runner invocation (use `${{ inputs.mode }}` to conditionally add `--dry-run`).

---

#### 4.2 Add Scheduled Real-Run Lane
**Impact:** Start collecting real shadow data
**Effort:** 30 min
**File:** New workflow `.github/workflows/apex_real_nightly.yml` (or extend existing)

**Strategy:**
```yaml
name: APEX Real Nightly

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  real-run-cpu-da3:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install ML deps (backend-aware)
        run: |
          pip install -e ".[ml]"
          python -c "import torch; import transformers; print('✅ ML deps OK')"

      - name: Run APEX (real, shadow mode)
        run: |
          python scripts/apex_matrix_runner.py \
            --workflow-versions v1 v2 \
            --zones nightly \
            --backend-id da3 \
            --input-dir ./tests/fixtures/apex_images \
            --sample-size 3 \
            --output-dir ./apex_results \
            --ledger-db ./apex_performance.db \
            --device cpu

      - name: Aggregate & Report
        run: |
          python scripts/apex_rebuild_ledger.py \
            --input-dir ./apex_results \
            --ledger-db ./apex_performance.db

          python scripts/apex_pr_comment.py \
            --ledger-db ./apex_performance.db \
            --output apex_nightly_report.md

      - uses: actions/upload-artifact@v4
        with:
          name: apex-nightly-ledger
          path: apex_performance.db
          retention-days: 90
```

**Key constraints:**
- Start with CPU torch only (no GPU runner complexity yet)
- Use small sample size (3 images) to keep runtime < 5 min
- Stay in shadow mode (no blocking) for 2+ weeks
- Upload ledger artifact for trend analysis

---

#### 4.3 Calibration → Gradual Enforcement
**Impact:** Turn shadow data into actionable thresholds
**Effort:** Ongoing (2–4 weeks of data collection)

**Process:**
1. **Week 1–2:** Collect nightly shadow data
2. **Week 3:** Analyze p95 distributions, set conservative thresholds (+20% margin)
3. **Week 4:** Switch nightly to `--mode enforce` (still not blocking PRs)
4. **Week 5+:** If stable, consider enforce mode for PR lane (or keep PRs synthetic, enforce only on main-merge)

**Success criteria from Phase 2:**
- p95 latency increase threshold: **10%**
- Mean latency increase threshold: **15%**
- Failure rate threshold: **0%**
- Minimum sample size: **n ≥ 20** (for statistical validity)

---

### Tier 5: Depth Pro First-Class Integration
**Goal:** Make Depth Pro a real APEX backend lane

#### 5.1 Expose Depth Pro in Registry + APEX
**Impact:** Backend selection is deterministic
**Effort:** 15 min
**Status:** Backend exists, needs registry wiring + CLI validation

**Action:**
- Ensure `DepthProBackend` is registered in `get_registry()`
- Add contract test:
```python
def test_depth_pro_backend_in_registry():
    from transformation_portal.depth.backends import get_registry
    registry = get_registry()
    assert "depth_pro" in registry.keys()
    backend_cls = registry.get_backend_class("depth_pro")
    assert backend_cls.required_packages() == ["depth_pro"]
```

---

#### 5.2 Make Depth Pro CI Lane Explicit (or Gracefully Skip)
**Impact:** Avoid silent failures in Linux CI
**Effort:** 10 min

**Decision point:** Can `depth-pro` install on Linux CI runners?
- If **yes:** add a nightly lane that installs it + runs APEX with `--backend-id depth_pro`
- If **no (macOS-only):** make APEX gracefully skip with clear message:

```python
# In check_ml_dependencies()
if backend_id == "depth_pro" and platform.system() != "Darwin":
    logger.warning("Depth Pro backend requires macOS; skipping on %s", platform.system())
    return True, []  # Don't block; just skip
```

---

## Summary Execution Timeline

| Phase | Effort | Impact | Blocking? |
|-------|--------|--------|-----------|
| **Tier 1: Papercuts** | 30 min | High (unlock next work) | No |
| **Tier 2: Phase 3 → CI** | 50 min | High (real backend-aware behavior) | No |
| **Tier 3: Nightly Stability** | 70 min | Medium (reduce noise) | No |
| **Tier 4: Real-Run Lane** | 50 min setup + 2–4 weeks data | **Highest** (production readiness) | No (gradual) |
| **Tier 5: Depth Pro** | 25 min | Medium (completeness) | No |

**Total hands-on effort:** ~3.5 hours
**Calendar time to production:** 4–6 weeks (due to shadow data collection)

---

## Recommended Immediate Next Steps (Today)

1. **Tier 1.1:** Make registry lookup public (20 min)
   → Opens PR: `refactor(apex): expose registry public API for backend lookup`

2. **Tier 1.2:** Fix Phase 3 docs (10 min)
   → Same PR or separate micro-PR

3. **Tier 4.1:** Add `workflow_dispatch` inputs (20 min)
   → Opens PR: `feat(apex): add workflow_dispatch for manual real runs`

**Why this order:**
- Tier 1 unblocks future work and is low-risk
- Tier 4.1 is the "gateway" to real execution and can be tested immediately via manual dispatch
- Tier 2/3 are important but not blocking (can happen in parallel)

---

## Anti-Patterns to Avoid

❌ **Don't:** Enable enforce mode before 2+ weeks of shadow data
✅ **Do:** Collect, analyze, tune thresholds, then enforce

❌ **Don't:** Install all ML deps in PR lane "just in case"
✅ **Do:** Keep PR lane synthetic; real execution = manual/nightly only

❌ **Don't:** Couple to registry internals (`._backends`)
✅ **Do:** Use public API (`registry.get_backend_class()`)

❌ **Don't:** Let docs claim "tests need refinement" when they're green
✅ **Do:** Keep docs aligned with repo reality

---

## Success Metrics (Phase 4 Complete)

- [ ] APEX can run real workflows via `workflow_dispatch` (manual trigger works)
- [ ] Nightly real-run lane collects data for 2+ weeks without failures
- [ ] Performance thresholds calibrated based on real data (p95 + mean + failure rate)
- [ ] PR lane stays fast (<2 min) and synthetic
- [ ] Depth Pro backend is either fully integrated or gracefully skipped on Linux
- [ ] No `._backends` direct access in user code
- [ ] Nightly Deep Checks reports actionable signals (not tooling errors)

---

**Meta-goal:** APEX becomes a **trustworthy, authoritative performance judge** that developers respect (not bypass).
