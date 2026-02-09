# APEX Performance Governance Orchestration Plan

**Status:** Systems Engineering Implementation Plan
**Created:** 2025-02-09
**Authority:** Transformation Portal Architect
**Phase:** Post-Phase 2 (Real Pipeline Integration Complete)

---

## Executive Summary

This document provides a systems-level assessment and orchestration plan for merging and implementing the APEX performance governance system. The system is designed to be:

- **Fast enough for CI** (synthetic mode for PRs, real mode for scheduled runs)
- **Honest enough to trust** (clear synthetic vs real distinction, airtight gating)
- **Statistical enough to avoid gaslighting** (distribution-aware comparisons, effect sizes, changepoint detection)

---

## Current State Assessment

### ✅ Phase 2: COMPLETED (Real Pipeline Integration)

**Branch:** `feat/apex-real-pipeline-integration` (ready to merge)

**Implementation Status:**
- ✅ Hybrid CI strategy (synthetic for PR/push, real for workflow_dispatch/schedule)
- ✅ Event-based mode gating (airtight logic in workflow)
- ✅ Dependency gating (ML deps only for real mode)
- ✅ Metadata/provenance capture (commit SHA, workflow version, backend, zone, device, runner, mode, sample size)
- ✅ Artifact durability (capsules + ledger uploaded with retention policies)
- ✅ Infrastructure scripts (matrix runner, ledger rebuild, aggregation, PR comment generation)

**Workflow File:** `.github/workflows/apex_performance.yml`
- Lines 105-111: Event type determines execution mode
- Lines 76-80: Conditional ML dependency installation
- Lines 126-132: Mode-specific command building
- Lines 264-272: Shadow mode gate enforcement

**Key Scripts:**
- `scripts/apex_matrix_runner.py`: Orchestrates parallel runs
- `scripts/apex_enforce_gate.py`: Gate enforcement with shadow/enforce modes
- `scripts/apex_pr_comment.py`: Generates PR comments with synthetic flag
- `scripts/apex_rebuild_ledger.py`: Aggregates results into SQLite ledger
- `scripts/apex_aggregate_ledger.py`: Computes statistics

**Artifacts:**
- Performance capsules (JSON): 3 days retention
- Ledger database (SQLite): 90 days retention
- Weekly ledger backups via GitHub Releases

### 🟡 Phase 3: IN PROGRESS (Governance Framework)

**Branch:** `origin/copilot/best-outcome-roadmap-apex` (requires review + split)

**Implementation Status:**
- ✅ Policy-as-code infrastructure
  - `docs/apex/policy/enforcement_policy.yaml`: Statistical methods, evidence gates
  - `docs/apex/policy/performance_budgets.yaml`: Per-bucket thresholds
  - `docs/apex/policy/governance_rules.yaml`: Waiver system, incident automation
  - `docs/apex/policy/workload_suites.yaml`: Golden/canary/deep suite definitions
  - `docs/apex/policy/apex_policy_schema.yaml`: Schema validation rules
- ✅ Policy validator: `scripts/apex_validate_policy.py`
- ✅ ADR-026: Governance framework architecture decision record
- ✅ User guide: `docs/apex/GOVERNANCE_USER_GUIDE.md`

**Missing Components:**
- ❌ CI workflow for policy validation (only runs when policy files change)
- ❌ Integration with `apex_enforce_gate.py` (currently simple threshold checks)
- ❌ Waiver workflow implementation
- ❌ Incident automation (auto-create issues on failures)
- ❌ Unit tests for policy validator

### 🔴 Dependency Updater: BROKEN

**Workflow File:** `.github/workflows/dependency-update.yml`

**Current Issues:**
1. ❌ No actual requirement diffs in PR body (only template text)
2. ❌ Safety report attached but not summarized in PR
3. ❌ No validation of updated requirements before PR creation
4. ❌ No testing of updated dependencies
5. ❌ Relies on `make update` in `requirements/` dir which may not exist

**Required Fixes:**
- Generate diff summary showing old → new versions
- Parse safety report and include summary in PR body
- Run basic smoke tests (import checks) before creating PR
- Add validation that core dependencies remain installable

### 🔴 Performance Regression Tests: INCOMPLETE

**Workflow File:** `.github/workflows/performance-monitor.yml`

**Current Issues:**
1. ❌ Duplicate/overlapping with APEX (both measure performance)
2. ❌ Uses pytest-benchmark but no actual benchmark tests exist
3. ❌ Baseline management is artifact-based (fragile, loses history)
4. ❌ No integration with APEX ledger
5. ❌ Memory profiling creates noise without structured analysis

**Recommendation:** **CLOSE or SUBSUME into APEX**
- APEX already provides comprehensive performance regression detection
- pytest-benchmark is redundant with APEX capsule system
- If memory profiling is needed, integrate into APEX as optional metrics

---

## Step A: Land Phase 2 (#884 equivalent)

### Truth Properties Checklist

#### 1. Event Gating ✅ VERIFIED

**Location:** `.github/workflows/apex_performance.yml:105-111`

```yaml
if [[ "${{ github.event_name }}" == "pull_request" ]] || [[ "${{ github.event_name }}" == "push" ]]; then
  MODE="synthetic"
  echo "ℹ️ PR/push lane: forcing synthetic mode (fast validation)"
elif [[ "${{ github.event_name }}" == "schedule" ]]; then
  MODE="real"
  echo "ℹ️ Scheduled run: using real execution (nightly monitoring)"
fi
```

**Verdict:** ✅ **AIRTIGHT** - Event type deterministically controls mode, cannot be bypassed in PR/push events.

#### 2. Dependency Gating ✅ VERIFIED

**Location:** `.github/workflows/apex_performance.yml:76-80`

```yaml
- name: Install dependencies (ML tier)
  if: github.event.inputs.mode == 'real' || github.event_name == 'schedule'
  run: |
    python -m pip install -e .[ml]
```

**Verdict:** ✅ **AIRTIGHT** - ML dependencies only installed for real mode, synthetic mode uses core-only.

#### 3. Metadata/Provenance ✅ VERIFIED

**Location:** Multiple

```yaml
env:
  APEX_RUN_ID: ${{ github.run_id }}-${{ github.run_attempt }}
  APEX_COMMIT_SHA: ${{ github.sha }}
```

**Captured Fields:**
- ✅ run_id
- ✅ commit_sha
- ✅ workflow_version (v1/v2)
- ✅ zone (local/cloud)
- ✅ backend_id (da3/depth_pro/mock)
- ✅ device (cpu/cuda/mps)
- ✅ mode (synthetic/real)
- ✅ sample_size
- ✅ runner (implicit in GitHub Actions context)
- ⚠️ MISSING: warmup behavior, seeds (not currently captured in capsule schema)

**Verdict:** ✅ **COMPLETE** for current scope, ⚠️ **NEEDS ENHANCEMENT** for full calibration (Step B).

#### 4. Semantic Honesty in PR Comments ✅ VERIFIED

**Location:** `scripts/apex_pr_comment.py` (with --synthetic flag)

```bash
if [[ "${MODE}" == "synthetic" ]]; then
  CMD+=(--synthetic)
fi
```

**Verdict:** ✅ **HONEST** - PR comments clearly distinguish synthetic vs real data.

#### 5. Artifact & Ledger Durability ✅ VERIFIED

**Artifacts:**
- Performance capsules: 3 days retention (line 143)
- Ledger database: 90 days retention (line 280)
- Weekly backups: GitHub Releases (lines 328-364)

**Verdict:** ✅ **DURABLE** - Multi-tier retention strategy with long-term backup.

### Step A Recommendation: **MERGE NOW**

**Merge Target:** `main`
**Branch:** `feat/apex-real-pipeline-integration`
**Squash Strategy:** No - preserve commit history for audit trail

**Post-Merge Actions:**
1. Trigger manual `workflow_dispatch` with `mode=real`
2. Verify end-to-end execution
3. Download and inspect ledger artifact
4. Validate metadata completeness
5. Check capsule JSON schema alignment

---

## Step B: Make Real-Run Lane Trustworthy

### Measurement Protocol Definition

**ADR Required:** Yes - defines statistical contract for APEX

**Components:**

#### 1. Warmup Behavior

**Current State:** ❌ Not implemented
**Required:**
- Discard first N runs per bucket (cold start effects)
- N = 2 for synthetic, N = 5 for real
- Capture warmup metadata in capsule

**Implementation:**
- Add `warmup_runs` field to capsule schema
- Modify `apex_matrix_runner.py` to execute warmup runs
- Exclude warmup samples from statistics

#### 2. Repetitions Per Bucket

**Current State:** ⚠️ Sample size is configurable but not enforced
**Required:**
- Minimum sample size per bucket for statistical validity
- Default: 10 for p50, 20 for p95, 50 for p99 (from governance policy)
- Fail with clear error if insufficient samples

**Implementation:**
- Validate sample size in `apex_enforce_gate.py`
- Add sample size metadata to ledger schema
- Report "insufficient_data" verdict when below threshold

#### 3. Fixed Workload Inputs

**Current State:** ✅ Test images in `tests/fixtures/apex_images/`
**Required:**
- Version-controlled fixture images
- Checksum validation before runs
- Document scene characteristics (complexity, size, format)

**Implementation:**
- Add `.checksums.txt` to fixtures directory
- Validate checksums in `apex_matrix_runner.py`
- Fail fast if fixtures are corrupted

#### 4. Locked Dependencies

**Current State:** ⚠️ Dependencies pinned in requirements but not verified
**Required:**
- Capture exact dependency versions in ledger
- Flag when dependency versions change between runs
- Enable dependency-aware baseline comparisons

**Implementation:**
- Run `pip freeze` during real execution
- Store as artifact alongside capsule
- Compare against baseline dependency snapshot

#### 5. CPU Affinity (Optional)

**Current State:** ❌ Not implemented
**Required (if feasible on GitHub Actions):**
- Pin to specific CPU cores to reduce variance
- If not feasible: document runner variance in metadata

**Implementation:**
- Use `taskset` if available on runner
- Document runner CPU info in capsule metadata
- Mark as best-effort (runner heterogeneity is a fact of life)

### Performance Noise Control

**Current State:** ⚠️ Runs on shared GitHub runners (high variance)

**Mitigation Strategies:**

1. **Cache State Tracking**
   - Record cold vs warm cache state
   - Document cache hits/misses if measurable
   - Consider cache-clear step before each run

2. **Avoid Shared Runners (Ideal)**
   - Use self-hosted runners for real mode if available
   - Document runner characteristics in ledger

3. **Record Timestamps**
   - ✅ Already captured (GitHub Actions metadata)
   - Use for temporal drift analysis

### Scheduled Run Frequency

**Current State:**
```yaml
schedule:
  - cron: "0 0 * * 0" # Weekly on Sunday 00:00 UTC
```

**Recommendation:**
- **Golden Suite:** Daily at 03:00 UTC (low-load time)
- **Deep Suite:** Weekly (current schedule)
- **Canary Suite:** On-demand (manual dispatch)

**Implementation:**
- Split workflow into three separate jobs with different schedules
- Use matrix strategy to select suite based on trigger
- Document suite definitions in `workload_suites.yaml`

---

## Step C: Merge Governance Scaffolding

### Recommended Split: TWO PRs

#### PR 1: Policy Infrastructure (Non-Blocking)

**Contents:**
- Policy YAML files (read-only, informational)
- Policy validator script (`apex_validate_policy.py`)
- ADR-026
- User guide
- CI workflow for policy validation (runs on policy file changes only)

**Enforcement Mode:** Shadow (informational only)

**Success Criteria:**
- Policy files validate against schema
- CI fails if policy files are malformed
- No impact on merge decisions

#### PR 2: Enforcement Integration (Blocking, Future)

**Contents:**
- Integration with `apex_enforce_gate.py`
- Statistical methods implementation
- Waiver workflow
- Incident automation

**Dependencies:** PR 1 + Step D (calibration pipeline)

**Success Criteria:**
- Baseline data established
- Statistical methods validated
- Waiver system tested

### Backward Compatibility Guarantee

**Contract:**
- Policy validator fails gracefully (warning, not error) if policy files missing
- `apex_enforce_gate.py` degrades to simple threshold checks if policy not loaded
- Default mode is `shadow` (never blocks without explicit opt-in)

**Implementation:**
- Add `--require-policy` flag (default: False)
- Log warnings when policy files not found
- Fail only in enforce mode with `--require-policy`

### Validator Unit Tests

**Test Cases:**

```python
def test_valid_policy_passes():
    # Valid policy files should validate without errors
    assert validate_policy(valid_policy_dir) == []

def test_missing_bucket_fails():
    # Policy missing a bucket definition should fail
    errors = validate_policy(incomplete_policy_dir)
    assert "Missing bucket definition for 'pool_large_mps'" in errors

def test_schema_mismatch_fails():
    # Policy with wrong schema version should fail
    errors = validate_policy(wrong_schema_dir)
    assert "schema_version" in errors[0].lower()

def test_circular_dependency_fails():
    # Waiver referencing nonexistent bucket should fail
    errors = validate_policy(circular_policy_dir)
    assert "circular" in errors[0].lower() or "unknown bucket" in errors[0].lower()
```

**Implementation:** Add to `tests/test_apex_policy_validator.py`

### Policy Workflow Scope

**Trigger:**
```yaml
on:
  pull_request:
    paths:
      - 'docs/apex/policy/**'
      - 'scripts/apex_validate_policy.py'
  push:
    branches: [main]
    paths:
      - 'docs/apex/policy/**'
```

**Job:**
```yaml
validate-policy:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - name: Install dependencies
      run: pip install pyyaml jsonschema
    - name: Validate policy files
      run: python scripts/apex_validate_policy.py --policy-dir docs/apex/policy/
```

### Waiver System Anti-Abuse

**Design Principles:**

1. **Scoped to Specific Buckets**
   - Waiver applies to one bucket + commit range only
   - No global "bypass all checks" waivers

2. **Time-Boxed**
   - Expiration date required (max 30 days)
   - Auto-revoke after expiration

3. **Tracked in Git**
   - Waivers stored in `docs/apex/waivers/YYYY-MM-DD-<bucket>.yaml`
   - Requires PR review + approval

4. **Auditable**
   - Waiver file includes: reason, owner, start/end commits, expiration, approver
   - Dashboard shows active waivers

5. **No Automatic Renewal**
   - Expired waivers require new PR to extend

**Schema:**
```yaml
waiver_id: "2025-02-09-pool-large-mps-001"
bucket_name: "pool_large_mps"
reason: "Known issue #1234: MPS backend memory leak investigation"
owner: "github_username"
approver: "github_username"
start_commit: "abc123def"
end_commit: "xyz789pqr"  # Optional: applies until this commit
expiration_date: "2025-03-11"  # Max 30 days from creation
created_at: "2025-02-09T12:00:00Z"
status: "active"  # active | expired | revoked
```

---

## Step D: Calibration Pipeline

**Status:** Not Started
**Dependency:** Step A (Phase 2 merged) + Step B (measurement protocol)

### Baseline Semantics

**Three Approaches (choose one or hybrid):**

#### Option 1: Rolling Window (Recommended)

**Definition:** Baseline = last N successful real runs (e.g., N=30)

**Pros:**
- Adapts to gradual performance changes
- Simple to implement
- No manual intervention

**Cons:**
- Can silently drift if all recent runs are slow
- Vulnerable to "boiling frog" effect

**Implementation:**
```sql
SELECT p95
FROM apex_runs
WHERE bucket_name = ?
  AND mode = 'real'
  AND pass_fail != 'fail'
  AND timestamp > datetime('now', '-30 days')
ORDER BY timestamp DESC
LIMIT 30
```

#### Option 2: Last-N-Good (Conservative)

**Definition:** Baseline = last N runs that passed enforcement

**Pros:**
- Never drifts to worse performance
- Conservative (protects against regressions)

**Cons:**
- Can become stale if many failures
- Requires manual baseline reset after intentional slowdowns

**Implementation:**
```sql
SELECT p95
FROM apex_runs
WHERE bucket_name = ?
  AND mode = 'real'
  AND pass_fail = 'pass'
ORDER BY timestamp DESC
LIMIT 30
```

#### Option 3: Changepoint-Aware (Advanced, Future)

**Definition:** Baseline = last stable regime before detected changepoint

**Pros:**
- Automatically detects intentional performance changes
- Adapts to both improvements and regressions

**Cons:**
- Complex algorithm (PELT, Bayesian changepoint detection)
- Requires sufficient historical data
- May have false positives

**Implementation:** Use `ruptures` library (add to dependencies)

**Recommendation for MVP:** Start with **Option 1** (rolling window), migrate to **Option 3** in Phase 5.

### Distribution-Aware Comparisons

**Current State:** Simple threshold checks (p95 > limit → fail)
**Required:** Statistical significance testing

**Proposed Method (from governance policy):**

```python
from scipy.stats import mannwhitneyu

def compare_distributions(baseline_samples, current_samples, alpha=0.05):
    """
    Test if current distribution is significantly slower than baseline.

    Returns:
        (is_significant, p_value, effect_size)
    """
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = mannwhitneyu(
        baseline_samples,
        current_samples,
        alternative='less'  # Test if current > baseline
    )

    # Cliff's Delta effect size (non-parametric)
    effect_size = cliffs_delta(baseline_samples, current_samples)

    is_significant = (p_value < alpha)

    return is_significant, p_value, effect_size

def cliffs_delta(x, y):
    """Non-parametric effect size."""
    n_x, n_y = len(x), len(y)
    dominance = sum(1 for a in x for b in y if a < b)
    return (dominance / (n_x * n_y)) * 2 - 1
```

**Integration Point:** `scripts/apex_enforce_gate.py`

### Changepoint Detection

**Library:** `ruptures` (add to `requirements/ml.txt`)

**Implementation:**

```python
import ruptures as rpt

def detect_changepoints(time_series, min_size=10):
    """
    Detect performance regime changes in time series.

    Args:
        time_series: List of (timestamp, p95) tuples
        min_size: Minimum segment size

    Returns:
        List of changepoint indices
    """
    signal = [p95 for _, p95 in time_series]

    # PELT algorithm with RBF kernel
    algo = rpt.Pelt(model="rbf", min_size=min_size)
    algo.fit(signal)

    # Detect changepoints with penalty parameter (controls sensitivity)
    changepoints = algo.predict(pen=3)

    return changepoints
```

**Usage:**
- Run on weekly schedule
- Flag buckets with recent changepoints
- Exclude from enforcement until new baseline stabilizes (N=30 runs after changepoint)

### Two-Tier Workload Suite

**From:** `docs/apex/policy/workload_suites.yaml` (governance branch)

#### Golden Suite (Enforcement)
- **Frequency:** Daily
- **Purpose:** Fast feedback, blocking enforcement
- **Workload:** 3-5 small/medium images per bucket
- **Sample Size:** 20 runs (sufficient for p95)
- **Backends:** DA3 (stable), Depth Pro (canary)
- **Devices:** CPU only (deterministic)

#### Deep Suite (Insight)
- **Frequency:** Weekly
- **Purpose:** Comprehensive profiling, non-blocking
- **Workload:** 20+ images per bucket (all scene types)
- **Sample Size:** 50 runs (sufficient for p99)
- **Backends:** All (including experimental)
- **Devices:** CPU + GPU (MPS/CUDA)

#### Canary Suite (Experimental)
- **Frequency:** On-demand
- **Purpose:** Test new backends/features before promotion
- **Workload:** Full deep suite
- **Sample Size:** 10 runs (exploration)
- **Backends:** Experimental only
- **Devices:** All available

**Implementation:**
- Add `--suite` flag to `apex_matrix_runner.py`
- Define suites in YAML config
- Use matrix strategy in workflow to select suite

### Minimum Sample Sizes for Enforcement

**From:** `enforcement_policy.yaml`

```yaml
sample_size_requirements:
  p50: 10   # Median: relatively stable
  p95: 20   # 95th percentile: needs more data
  p99: 50   # 99th percentile: very sensitive
```

**Implementation:**

```python
def validate_sample_size(bucket_stats, percentile):
    """Check if sample size is sufficient for enforcement."""
    required_sizes = {"p50": 10, "p95": 20, "p99": 50}

    if bucket_stats.sample_size < required_sizes[percentile]:
        return {
            "verdict": "insufficient_data",
            "reason": f"Need {required_sizes[percentile]} samples for {percentile}, got {bucket_stats.sample_size}"
        }

    return None  # Sufficient data
```

---

## Step E: Gradual Enforcement Rollout

### Phase 1: Shadow Only (Weeks 1-4)

**Goal:** Establish baseline, calibrate thresholds, validate statistical methods

**Configuration:**
```yaml
# In enforcement_policy.yaml
default_mode: shadow
```

**Behavior:**
- Generate PR comments with verdicts
- Populate dashboard with historical data
- Log all enforcement decisions
- Never block merges

**Success Criteria:**
- 30 days of daily real runs completed
- Baseline established for all buckets
- No false positives in shadow mode (manual review)
- Changepoint detection validated

### Phase 2: Soft Enforcement (Weeks 5-8)

**Goal:** Add friction for regressions, allow overrides

**Configuration:**
```yaml
default_mode: soft_enforce
require_acknowledgment: true
```

**Behavior:**
- Add "performance-regression" label to PR
- Require comment acknowledgment (e.g., "I acknowledge the regression is acceptable because...")
- Track repeated violations per contributor
- Auto-escalate after 3 violations (ping team lead)

**Success Criteria:**
- No complaints about false positives
- Regressions acknowledged or fixed
- Violation tracking working

### Phase 3: Hard Enforcement on Golden Buckets (Weeks 9-12)

**Goal:** Block merges for golden buckets only

**Configuration:**
```yaml
buckets:
  pool_large_mps:
    enforcement_mode: enforce  # Blocks merge
  aerial_large_cuda:
    enforcement_mode: soft_enforce  # Warning only
```

**Behavior:**
- Golden buckets: Block merge if regression detected AND statistically significant AND not drifting
- Non-golden buckets: Soft enforcement only
- Waiver system available for emergencies

**Conditions for Blocking:**
1. ✅ Real data exists (not synthetic)
2. ✅ Sample size meets minimum (n≥20 for p95)
3. ✅ Regression is statistically significant (p < 0.05)
4. ✅ Effect size is large (Cliff's delta > 0.5)
5. ✅ Not explained by changepoint (no recent drift)

**Success Criteria:**
- Zero false blocks (all regressions are real)
- Waiver system used < 5% of the time
- No merge delays due to noisy enforcement

### Phase 4: Full Enforcement (Week 13+)

**Goal:** All buckets enforced

**Configuration:**
```yaml
default_mode: enforce
```

**Behavior:**
- All buckets enforce on p95 threshold + regression detection
- Incident automation enabled (auto-create issues)
- Performance SLAs tracked in dashboard

### Incident Automation

**Trigger:** Failure in enforce mode

**Action:**
```yaml
- name: Create Performance Incident
  if: ${{ failure() && steps.enforce.outputs.verdict == 'fail' }}
  uses: actions/github-script@v8
  with:
    script: |
      const issue = await github.rest.issues.create({
        owner: context.repo.owner,
        repo: context.repo.repo,
        title: `[APEX] Performance regression in ${bucket_name}`,
        body: `## Performance Incident

**Bucket:** ${bucket_name}
**Commit:** ${commit_sha}
**P95 Observed:** ${p95_observed}s
**P95 Threshold:** ${p95_threshold}s
**Regression:** +${regression_pct}%

**Evidence:**
- Run ID: ${run_id}
- Dashboard: [View](https://example.com/apex)
- Capsule: [Download](artifact_url)

**Owner:** @${commit_author}
**SLA:** Fix within 48 hours or file waiver

/cc @performance-team
        `,
        labels: ['performance', 'incident', 'automated'],
        assignees: [commit_author]
      });
```

**SLA:**
- **Severity 1** (>25% regression): 24 hours
- **Severity 2** (15-25% regression): 48 hours
- **Severity 3** (<15% regression): 1 week

---

## Odd Ducks

### Dependency Updater (#883): FIX OR CLOSE

**Recommendation:** **FIX**

**Required Changes:**

1. **Add Actual Diff Generation**
   ```bash
   # Before update
   pip freeze > before.txt

   # After update
   pip-compile requirements/base.in
   pip freeze > after.txt

   # Generate diff
   diff -u before.txt after.txt > requirements.diff
   ```

2. **Parse Safety Report**
   ```python
   import json

   with open('safety-report.json') as f:
       report = json.load(f)

   vulnerabilities = report.get('vulnerabilities', [])
   if vulnerabilities:
       summary = f"⚠️ {len(vulnerabilities)} vulnerabilities found:\n"
       for vuln in vulnerabilities[:5]:  # Top 5
           summary += f"- {vuln['package']}: {vuln['title']}\n"
   else:
       summary = "✅ No known vulnerabilities detected"
   ```

3. **Add Smoke Tests**
   ```bash
   # Test that core imports work
   python -c "import transformation_portal; print('✅ Core imports OK')"
   python -c "import torch; import transformers; print('✅ ML imports OK')"
   ```

4. **Validate Requirements**
   ```bash
   # Check that requirements are installable
   pip install --dry-run -r requirements/all.txt
   ```

**Timeline:** 1 day of work

### Performance Monitor (#845): CLOSE

**Recommendation:** **CLOSE and link to APEX**

**Rationale:**
1. APEX already provides comprehensive performance regression testing
2. pytest-benchmark is redundant with performance capsule system
3. Artifact-based baseline management is fragile (APEX uses SQLite ledger)
4. Memory profiling is noisy without structured analysis
5. Maintaining two performance systems creates confusion

**Migration Path:**
1. Document memory profiling as future APEX enhancement
2. Close PR with link to APEX documentation
3. Add memory metrics to performance capsule schema (future work)

**Alternative (if memory profiling is critical):**
- Integrate `memory_profiler` into APEX as optional metric
- Add memory fields to capsule schema
- Track memory baselines in ledger

---

## Blockers and Missing Components

### Blockers

1. ❌ **No unit tests for policy validator**
   - **Impact:** Cannot confidently merge governance PR
   - **Effort:** 4 hours
   - **Owner:** Should be included in governance PR

2. ❌ **No CI workflow for policy validation**
   - **Impact:** Policy files can drift without validation
   - **Effort:** 1 hour
   - **Owner:** Should be included in governance PR

3. ❌ **Statistical methods not implemented in apex_enforce_gate.py**
   - **Impact:** Cannot move beyond simple threshold checks
   - **Effort:** 2 days
   - **Owner:** Step D (calibration pipeline)

4. ❌ **Baseline management not defined**
   - **Impact:** Regression detection is unreliable
   - **Effort:** 1 day
   - **Owner:** Step D (calibration pipeline)

### Missing Components

1. **Changepoint Detection Library**
   - **Dependency:** Add `ruptures` to `requirements/ml.txt`
   - **Risk:** Adds ~10MB to install size
   - **Mitigation:** Only install for real mode runs

2. **Statistical Testing Library**
   - **Dependency:** `scipy` already installed (part of ML stack)
   - **Risk:** None

3. **Waiver Workflow**
   - **Implementation:** GitHub Actions workflow + issue template
   - **Effort:** 1 day
   - **Owner:** Step C PR 2

4. **Incident Automation**
   - **Implementation:** GitHub Actions step in apex_performance.yml
   - **Effort:** 4 hours
   - **Owner:** Step E Phase 3

5. **Dashboard Enhancements**
   - **Current:** Basic HTML dashboard
   - **Needed:** Waiver status, changepoint markers, baseline visualization
   - **Effort:** 2 days
   - **Owner:** Post-Step D

---

## Actionable Next Steps

### Immediate (This Week)

#### 1. Merge Phase 2 (Step A)
- [ ] Final review of `feat/apex-real-pipeline-integration` branch
- [ ] Merge to `main` (preserve commit history)
- [ ] Trigger manual real run via workflow_dispatch
- [ ] Verify end-to-end execution
- [ ] Download and inspect ledger artifact
- [ ] Document any runtime issues

**Owner:** Architect (me)
**Success Criteria:** Green workflow run, readable ledger, complete metadata

#### 2. Split Governance PR (Step C PR 1)
- [ ] Create new branch from `origin/copilot/best-outcome-roadmap-apex`
- [ ] Cherry-pick policy files + validator + ADR + user guide
- [ ] Add unit tests for policy validator
- [ ] Add CI workflow for policy validation
- [ ] Remove enforcement integration (save for PR 2)
- [ ] Open PR with "Shadow Mode Only" title
- [ ] Review and merge

**Owner:** Architect (me) or delegate to Specialist
**Success Criteria:** Policy validator green in CI, no merge blocking

#### 3. Fix Dependency Updater (Odd Duck #1)
- [ ] Implement diff generation
- [ ] Parse safety report
- [ ] Add smoke tests
- [ ] Test workflow locally (act or similar)
- [ ] Open PR with fixes
- [ ] Trigger manual run to validate

**Owner:** Delegate to Specialist
**Success Criteria:** PR includes actual diffs + safety summary

### Short-Term (Weeks 2-4)

#### 4. Implement Measurement Protocol (Step B)
- [ ] Add warmup behavior to apex_matrix_runner.py
- [ ] Implement sample size validation
- [ ] Add checksum validation for fixtures
- [ ] Capture dependency versions in capsule
- [ ] Document CPU affinity limitations
- [ ] Update APEX_CONTRACT.md with protocol details

**Owner:** Specialist
**Success Criteria:** Real runs include warmup metadata, checksums validated

#### 5. Increase Scheduled Run Frequency (Step B)
- [ ] Split workflow into golden/deep/canary jobs
- [ ] Configure daily golden suite (03:00 UTC)
- [ ] Keep weekly deep suite (Sunday 00:00 UTC)
- [ ] Add manual canary dispatch
- [ ] Monitor disk usage (daily runs = more artifacts)

**Owner:** Architect
**Success Criteria:** 7 consecutive days of successful golden suite runs

#### 6. Close Performance Monitor PR (Odd Duck #2)
- [ ] Document rationale in issue comment
- [ ] Link to APEX documentation
- [ ] Add memory profiling to APEX roadmap (future)
- [ ] Close PR #845

**Owner:** Architect
**Success Criteria:** PR closed with clear justification

### Medium-Term (Weeks 5-8)

#### 7. Implement Calibration Pipeline (Step D)
- [ ] Choose baseline semantics (rolling window for MVP)
- [ ] Implement distribution comparison (Mann-Whitney U)
- [ ] Add Cliff's Delta effect size calculation
- [ ] Integrate into apex_enforce_gate.py
- [ ] Add `ruptures` to dependencies
- [ ] Implement changepoint detection (optional for MVP)
- [ ] Define two-tier workload suites
- [ ] Update policy files with suite definitions

**Owner:** Specialist (implementation), Architect (review)
**Success Criteria:** Statistical enforcement working in shadow mode

#### 8. Shadow Mode Data Collection (Step E Phase 1)
- [ ] Run daily golden suite for 30 days
- [ ] Monitor baseline stability
- [ ] Validate changepoint detection (if implemented)
- [ ] Review shadow mode verdicts for false positives
- [ ] Calibrate thresholds if needed

**Owner:** Architect (monitoring), Specialist (threshold tuning)
**Success Criteria:** Zero false positives in shadow mode review

### Long-Term (Weeks 9+)

#### 9. Soft Enforcement Rollout (Step E Phase 2)
- [ ] Enable soft enforcement mode
- [ ] Implement acknowledgment requirement
- [ ] Track violations per contributor
- [ ] Monitor acknowledgment quality
- [ ] Iterate on thresholds

**Owner:** Architect
**Success Criteria:** No complaints about false positives

#### 10. Hard Enforcement on Golden Buckets (Step E Phase 3)
- [ ] Enable hard enforcement for golden buckets only
- [ ] Implement waiver workflow
- [ ] Add incident automation
- [ ] Monitor false positive rate (target: 0%)
- [ ] Iterate on enforcement conditions

**Owner:** Architect
**Success Criteria:** Zero false blocks in first 2 weeks

#### 11. Full Enforcement Rollout (Step E Phase 4)
- [ ] Enable enforcement for all buckets
- [ ] Document SLAs
- [ ] Train team on waiver process
- [ ] Monitor incident resolution times
- [ ] Declare performance governance "production ready"

**Owner:** Architect
**Success Criteria:** Performance SLAs met for 4 consecutive weeks

---

## Success Criteria Summary

### Systems Correctness
- ✅ Event gating is airtight (no way to run real in PR lane)
- ✅ Dependency gating is airtight (no ML deps in synthetic mode)
- ✅ Metadata is complete and auditable
- ✅ Artifacts are durable (multi-tier retention)
- ⚠️ Baseline management is reliable (not yet implemented)
- ⚠️ Statistical methods are sound (not yet implemented)

### Statistical Sanity
- ⚠️ Distribution-aware comparisons (not yet implemented)
- ⚠️ Effect sizes calculated (not yet implemented)
- ⚠️ Changepoint detection (not yet implemented)
- ⚠️ Sample size requirements enforced (not yet implemented)
- ✅ Non-parametric methods chosen (design complete)

### Governance Auditability
- ✅ Policy-as-code infrastructure exists
- ⚠️ Policy validation in CI (not yet implemented)
- ✅ Waiver system designed
- ⚠️ Waiver workflow implemented (not yet)
- ⚠️ Incident automation (not yet)
- ✅ ADR documenting governance decisions

---

## Risk Assessment

### High Risk
1. **Baseline Drift (Boiling Frog)**
   - **Mitigation:** Changepoint detection + manual review cadence
   - **Owner:** Architect

2. **False Positives in Enforcement**
   - **Mitigation:** Shadow mode data collection, statistical rigor, waiver system
   - **Owner:** Architect

3. **Measurement Noise on Shared Runners**
   - **Mitigation:** Larger sample sizes, non-parametric methods, warmup
   - **Owner:** Accept as limitation, document in metadata

### Medium Risk
1. **Dependency Updater Breakage**
   - **Mitigation:** Add smoke tests, validate before PR creation
   - **Owner:** Specialist

2. **Disk Space with Daily Runs**
   - **Mitigation:** Reduce capsule retention to 1 day, compress ledger backups
   - **Owner:** Architect

3. **Team Pushback on Enforcement**
   - **Mitigation:** Gradual rollout, clear communication, waiver system
   - **Owner:** Architect

### Low Risk
1. **Policy File Drift**
   - **Mitigation:** CI validation, schema checks
   - **Owner:** Policy validator

2. **Dashboard Complexity**
   - **Mitigation:** Start simple, iterate based on usage
   - **Owner:** Specialist

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| **Step A:** Merge Phase 2 | 2 days | Week 1 |
| **Step C PR 1:** Policy infrastructure | 3 days | Week 1 |
| **Odd Duck #1:** Fix dependency updater | 1 day | Week 1 |
| **Odd Duck #2:** Close performance monitor | 1 hour | Week 1 |
| **Step B:** Measurement protocol | 1 week | Week 2 |
| **Step D:** Calibration pipeline | 2 weeks | Week 4 |
| **Step E Phase 1:** Shadow mode (30 days) | 4 weeks | Week 8 |
| **Step E Phase 2:** Soft enforcement | 2 weeks | Week 10 |
| **Step E Phase 3:** Hard enforcement (golden) | 2 weeks | Week 12 |
| **Step E Phase 4:** Full enforcement | 2 weeks | Week 14 |

**Total Time to Production:** ~14 weeks (3.5 months)

---

## Conclusion

The APEX performance governance system is architecturally sound and well-positioned for production deployment. Phase 2 (Real Pipeline Integration) is complete and ready to merge. The governance scaffolding exists but requires splitting into two PRs for safe rollout.

Key success factors:
1. **Gradual rollout** (shadow → soft → hard enforcement)
2. **Statistical rigor** (distribution-aware, effect sizes, changepoint detection)
3. **Measurement discipline** (warmup, sample sizes, fixed inputs)
4. **Escape hatches** (waiver system, manual overrides)
5. **Auditability** (policy-as-code, ledger retention, incident tracking)

This is a **production-grade performance governance system** that will prevent performance regressions without gaslighting engineers.

---

## Appendix: Related Documentation

- **APEX Contract:** `docs/apex/APEX_CONTRACT.md`
- **Phase 2 Completion:** `docs/apex/phase2/COMPLETION_REPORT.md`
- **Governance ADR:** `docs/architecture/decisions/ADR-026-APEX-governance-framework.md` (on governance branch)
- **Governance User Guide:** `docs/apex/GOVERNANCE_USER_GUIDE.md` (on governance branch)
- **Policy Schema:** `docs/apex/policy/apex_policy_schema.yaml` (on governance branch)

---

**Architect Signature:** Transformation Portal Architect
**Date:** 2025-02-09
**Review Cadence:** Monthly during rollout, quarterly after production
