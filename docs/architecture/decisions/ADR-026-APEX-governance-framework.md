# ADR-026: APEX Performance Governance Framework

**Status:** Accepted  
**Date:** 2026-02-09  
**Deciders:** Transformation Portal Architect  
**Extends:** ADR-025 (APEX End-to-End Workflow Architecture)

---

## Context and Problem Statement

APEX has evolved from performance instrumentation to an observability platform. However, to become a true **performance governance system**, it needs:

1. **Explicit, versioned policy** (not implicit thresholds buried in code)
2. **Statistical rigor** (not single-run p95 panic)
3. **Accountability mechanisms** (waivers, incident tracking, budget evolution)
4. **Workload governance** (canonical test suites, not ad-hoc inputs)
5. **Enforcement that earns trust** (deterministic, reproducible, auditable)

**Current gaps:**
- Performance budgets are hardcoded in `DEFAULT_BUCKETS`
- No waiver mechanism exists
- No statistical distribution analysis
- No workload governance
- No explicit policy change review process
- Enforcement mode exists but lacks governance infrastructure

**Business Impact:**
- Cannot transition from shadow to enforce mode safely
- Budget changes have no audit trail
- False positives erode developer trust
- No mechanism for justified exceptions
- No clear ownership of performance regressions

---

## Decision Drivers

1. **Policy-as-Code:** Budgets must be versioned, reviewable, and auditable
2. **Statistical Validity:** Avoid false positives from measurement noise
3. **Governance Transparency:** Every decision must be traceable and explainable
4. **Developer Trust:** System must be fair, consistent, and escapable (with friction)
5. **Operational Safety:** Gradual rollout from shadow → enforce with evidence gates
6. **Contract Stability:** Changes must not break existing APEX contracts

---

## Considered Options

### Option 1: Full Governance Platform (Chosen)

**Approach:**
- Policy-as-code in versioned YAML files under `docs/apex/policy/`
- Statistical enforcement layer with distribution analysis
- Explicit governance workflows (incident response, waivers, budget evolution)
- Canonical workload suites with governed fixture sets
- Multi-tier enforcement gates (evidence quality → enforcement authority)

**Pros:**
- True governance (not just monitoring)
- Auditable and explainable
- Developer trust through transparency
- Enables safe shadow-to-enforce transition
- Research-aligned (PERUN-inspired workload governance)

**Cons:**
- Implementation complexity
- Requires cultural adoption (policy review discipline)
- More moving parts to maintain

### Option 2: Incremental Extensions Only

**Approach:**
- Add waiver labels to PR workflow
- Keep budgets in code with version comments
- Add simple outlier detection
- No formal policy files

**Pros:**
- Minimal disruption
- Faster to implement

**Cons:**
- Doesn't solve governance problem
- Budgets remain opaque
- No audit trail for policy changes
- Perpetuates "CI theater" risk

**Rejected:** Insufficient to earn enforcement authority

### Option 3: External Governance Platform

**Approach:**
- Integrate with external policy engine (Open Policy Agent, etc.)
- Use external service for decision logging

**Pros:**
- Industry-standard tooling
- Rich policy language

**Cons:**
- External dependency
- Offline operation broken
- Data governance boundary violation
- Complexity overhead for this use case

**Rejected:** Violates self-contained principle

---

## Decision Outcome

**Chosen Option:** Option 1 (Full Governance Platform)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     APEX Governance Stack                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Policy Layer (docs/apex/policy/)                      │  │
│  │ - performance_budgets.yaml (versioned thresholds)    │  │
│  │ - workload_suites.yaml (canonical test sets)         │  │
│  │ - enforcement_policy.yaml (gates, sample sizes)      │  │
│  │ - governance_rules.yaml (waiver, escalation)         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Statistical Engine (src/...metrics/statistical.py)   │  │
│  │ - Distribution storage (per-sample timings)          │  │
│  │ - Mann-Whitney U / bootstrap CI                      │  │
│  │ - Median + MAD outlier detection                     │  │
│  │ - Sample size validation                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Governance Engine (scripts/apex_governance.py)       │  │
│  │ - Incident workflow (regression → issue creation)    │  │
│  │ - Waiver tracking (labels, expiry, scope)            │  │
│  │ - Budget evolution (evidence, attribution, review)   │  │
│  │ - Policy validation (schema, consistency checks)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Existing APEX Stack                                   │  │
│  │ - apex_matrix_runner.py                              │  │
│  │ - apex_enforce_gate.py (enhanced with governance)    │  │
│  │ - Performance ledger (extended schema)               │  │
│  │ - PR comment generator (governance context)          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tier 6: Policy-as-Code

### Performance Budget Schema

**File:** `docs/apex/policy/performance_budgets.yaml`

```yaml
schema_version: "1.0.0"
effective_date: "2026-02-15"
review_date: "2026-05-15"  # Quarterly review
policy_owner: "transformation-portal-architect"

# Baseline source (required for budget validation)
baseline:
  source: "ledger"  # or "synthetic" during shadow
  minimum_sample_size: 20
  confidence_level: 0.95

# Budget definitions per workflow × bucket × zone × device/backend
budgets:
  - workflow_version: "v2"
    bucket_name: "pool_large_mps"
    stability_tier: "stable"  # stable/canary/experimental
    thresholds:
      p50_sec: 6.0
      p95_sec: 10.0
      max_regression_pct: 10.0
      warn_regression_pct: 5.0
    enforcement:
      mode: "shadow"  # shadow/enforce/disabled
      effective_from: "2026-02-15"
      review_required: true
    notes: |
      Baseline established from 30-day nightly runs (n=45).
      Budget set at p95 + 15% margin for measurement variance.

  - workflow_version: "v2"
    bucket_name: "aerial_large_mps"
    stability_tier: "stable"
    thresholds:
      p50_sec: 6.0
      p95_sec: 10.0
      max_regression_pct: 10.0
      warn_regression_pct: 5.0
    enforcement:
      mode: "shadow"
      effective_from: "2026-02-15"
      review_required: true

# Stability tier definitions
tier_policies:
  stable:
    description: "Production-safe, backward-compatible workflows"
    required_baseline_days: 30
    min_sample_size: 20
    max_change_frequency: "quarterly"
    
  canary:
    description: "Production trial workflows"
    required_baseline_days: 14
    min_sample_size: 10
    max_change_frequency: "monthly"
    
  experimental:
    description: "Research workflows"
    required_baseline_days: 7
    min_sample_size: 5
    max_change_frequency: "weekly"
```

### Enforcement Policy Schema

**File:** `docs/apex/policy/enforcement_policy.yaml`

```yaml
schema_version: "1.0.0"

# Evidence quality gates
evidence_gates:
  shadow_mode:
    description: "Informational only, never blocks"
    min_sample_size: 1
    allow_synthetic_data: true
    
  enforce_mode:
    description: "Blocks merge on failure"
    min_sample_size: 20
    allow_synthetic_data: false
    required_baseline_age_days: 30
    confidence_level: 0.95

# Statistical methods
statistical_methods:
  distribution_comparison:
    method: "mann_whitney_u"  # or "bootstrap_ci"
    alpha: 0.05
    
  outlier_detection:
    method: "median_mad"  # median absolute deviation
    threshold: 3.0  # MAD units

# Sample size requirements per percentile
sample_size_requirements:
  p50: 10
  p95: 20
  p99: 50
```

### Governance Rules Schema

**File:** `docs/apex/policy/governance_rules.yaml`

```yaml
schema_version: "1.0.0"

# Waiver mechanism
waivers:
  allowed_scopes:
    - "single_bucket"  # Scope waiver to specific bucket
    - "single_zone"    # Scope waiver to specific zone
  
  required_fields:
    - justification
    - expiry_date
    - scope_definition
    - approver
    
  labels:
    pr_label: "apex-waiver"
    tracking_label: "apex-waiver-tracking"
    
  expiry:
    default_days: 30
    max_days: 90
    auto_close_on_expiry: true

# Budget evolution workflow
budget_changes:
  required_evidence:
    - ledger_query_results
    - baseline_shift_analysis
    - attribution_investigation
    
  required_review:
    label: "apex-policy-change"
    min_approvers: 1  # Architect role
    
  documentation:
    changelog_required: true
    adr_required_for_major_changes: true

# Incident workflow
incidents:
  triggers:
    - condition: "regression > max_regression_pct"
      severity: "high"
      auto_create_issue: true
      
    - condition: "p95 > threshold_p95 * 1.15"
      severity: "medium"
      auto_create_issue: true
      
  issue_template:
    title_format: "[APEX] Performance Regression: {bucket_name} in {zone}"
    labels: ["performance", "apex-incident", "needs-triage"]
    assignees: ["transformation-portal-architect"]
```

---

## Tier 7: Statistical Enforcement

### Distribution Storage

**Ledger schema extension:**

```sql
-- New table for per-sample timings (enables distribution analysis)
CREATE TABLE IF NOT EXISTS apex_run_samples (
    run_id TEXT NOT NULL,
    workflow_version TEXT NOT NULL,
    zone TEXT NOT NULL,
    bucket_name TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    total_sec REAL NOT NULL,
    captured_at TEXT NOT NULL,
    PRIMARY KEY (run_id, workflow_version, zone, bucket_name, sample_index),
    FOREIGN KEY (run_id, workflow_version, zone, bucket_name) 
        REFERENCES apex_runs(run_id, workflow_version, zone, bucket_name)
);

CREATE INDEX IF NOT EXISTS idx_samples_bucket_zone_time 
    ON apex_run_samples(bucket_name, zone, captured_at DESC);
```

### Comparison Methods

**Implementation:** `src/transformation_portal/metrics/statistical.py`

```python
from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import stats

@dataclass
class DistributionComparison:
    """Result of statistical distribution comparison."""
    current_samples: List[float]
    baseline_samples: List[float]
    test_statistic: float
    p_value: float
    effect_size: float  # Cliff's Delta or Cohen's d
    verdict: str  # "no_change", "regression", "improvement"
    confidence: float

def compare_distributions(
    current: List[float],
    baseline: List[float],
    alpha: float = 0.05,
    method: str = "mann_whitney_u"
) -> DistributionComparison:
    """
    Compare two performance distributions.
    
    Returns statistical comparison result with confidence level.
    Requires minimum sample size (enforced by caller).
    """
    if method == "mann_whitney_u":
        statistic, p_value = stats.mannwhitneyu(
            current, baseline, alternative='greater'
        )
        # Calculate effect size (Cliff's Delta)
        effect_size = calculate_cliffs_delta(current, baseline)
        
    elif method == "bootstrap_ci":
        # Bootstrap confidence interval on median difference
        effect_size, p_value = bootstrap_median_diff(current, baseline, alpha)
        statistic = effect_size
        
    # Determine verdict
    if p_value > alpha:
        verdict = "no_change"
    elif effect_size > 0:
        verdict = "regression"
    else:
        verdict = "improvement"
        
    confidence = 1.0 - p_value
    
    return DistributionComparison(
        current_samples=current,
        baseline_samples=baseline,
        test_statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        verdict=verdict,
        confidence=confidence
    )
```

### Outlier Detection

```python
def detect_outliers_mad(
    samples: List[float],
    threshold: float = 3.0
) -> tuple[List[float], List[int]]:
    """
    Detect outliers using Median Absolute Deviation (MAD).
    
    More robust than mean+stddev for skewed distributions.
    
    Returns:
        clean_samples: Samples with outliers removed
        outlier_indices: Indices of detected outliers
    """
    arr = np.array(samples)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    
    # Modified z-score
    modified_z = 0.6745 * (arr - median) / mad
    
    outlier_mask = np.abs(modified_z) > threshold
    clean = arr[~outlier_mask]
    outlier_idx = np.where(outlier_mask)[0].tolist()
    
    return clean.tolist(), outlier_idx
```

---

## Tier 8: Governance Workflows

### Performance Incident Workflow

**Trigger:** APEX gate detects regression in enforce mode

**Automated Actions:**

1. **Create GitHub Issue** (via GitHub API or `gh` CLI):
   ```markdown
   Title: [APEX] Performance Regression: pool_large_mps in us-west-2a
   
   **Severity:** High
   **Detected:** 2026-02-15 14:32 UTC
   **Run ID:** gh-123456-1
   **Commit:** abc123def
   
   ## Regression Summary
   - Bucket: `pool_large_mps`
   - Zone: `us-west-2a`
   - Workflow: `v2`
   - Current p95: 12.3s
   - Baseline p95: 10.0s
   - Regression: +23% (threshold: 10%)
   
   ## Evidence
   - Ledger query: [link to query results]
   - Distribution comparison: Mann-Whitney U test, p=0.003
   - Sample size: current n=25, baseline n=30
   
   ## Top Contributing Stages
   (from instrumentation, if available)
   1. depth_inference: +15% (+1.8s)
   2. tone_mapping: +5% (+0.4s)
   
   ## Suggested Actions
   - [ ] Reproduce locally: `python scripts/apex_matrix_runner.py ...`
   - [ ] Profile stage X with cProfile
   - [ ] Check for recent dependency version changes
   - [ ] Bisect commits if needed
   
   ## Resolution Workflow
   - [ ] Investigation assigned
   - [ ] Root cause identified
   - [ ] Fix implemented or waiver requested
   - [ ] Verification run clean
   ```

2. **Block PR merge** (if in enforce mode)
3. **Post PR comment** with incident summary and resolution steps

### Waiver Mechanism

**Request Process:**

1. Developer adds label `apex-waiver` to PR
2. Developer fills waiver template in PR description:
   ```markdown
   ## APEX Waiver Request
   
   **Justification:** [Required - why is this regression acceptable?]
   
   **Scope:** 
   - Bucket: `pool_large_mps`
   - Zone: `us-west-2a`
   - Workflow: `v2`
   
   **Expiry Date:** 2026-03-15 (30 days from now)
   
   **Mitigation Plan:** [What will be done to address this?]
   
   **Evidence:** [Link to analysis, profiling, investigation]
   ```

3. CI detects waiver label and validates template
4. Architect reviews and approves/rejects
5. If approved:
   - PR allowed to merge
   - Tracking issue auto-created with expiry date
   - Waiver logged in ledger with `override=true` flag

**Enforcement:**

```python
# In apex_enforce_gate.py
def check_waiver_status(pr_number: int, regression_scope: dict) -> bool:
    """Check if valid waiver exists for this regression scope."""
    # Check PR labels for 'apex-waiver'
    # Parse waiver template from PR description
    # Validate scope matches regression
    # Validate expiry date is future
    # Validate approver is authorized
    # Return True if valid waiver exists
```

### Budget Evolution Workflow

**Trigger:** Maintainer wants to adjust performance budgets

**Required Process:**

1. **Evidence Collection:**
   ```bash
   # Query ledger for baseline shift
   python scripts/apex_query_baseline.py \
     --bucket pool_large_mps \
     --zone us-west-2a \
     --days 30 \
     --output baseline_shift_analysis.json
   ```

2. **Attribution Investigation:**
   - Which commit(s) caused the shift?
   - Is it a real algorithmic change or measurement variance?
   - Is the change justified (quality improvement, new feature)?

3. **Create Policy Change PR:**
   - Edit `docs/apex/policy/performance_budgets.yaml`
   - Add `apex-policy-change` label
   - Link to evidence and attribution analysis
   - Update review_date

4. **Review & Approval:**
   - Requires Architect approval
   - CI validates policy schema
   - ADR created if major change

5. **Baseline Reset:**
   - New budget becomes effective
   - Baseline recomputed from recent runs
   - Old budget archived for historical queries

---

## Workload Governance

### Canonical Workload Suites

**File:** `docs/apex/policy/workload_suites.yaml`

```yaml
schema_version: "1.0.0"

# Golden suite: small, stable, never changes without explicit review
golden_suite:
  description: "Stable, representative images for baseline establishment"
  fixture_dir: "tests/fixtures/apex_golden"
  images:
    - "pool_luxury_4k.jpg"
    - "aerial_estate_6k.jpg"
    - "interior_livingroom_3k.jpg"
  change_policy:
    requires_adr: true
    min_notice_days: 30
  usage:
    - "PR lane (synthetic or minimal real)"
    - "Nightly baseline establishment"

# Canary suite: broader, more realistic, can evolve
canary_suite:
  description: "Representative real-world workload"
  fixture_dir: "tests/fixtures/apex_canary"
  size: 20  # images
  selection_criteria:
    - "Diverse scene types"
    - "Realistic dimensions (3-8K)"
    - "Various EXIF metadata"
  change_policy:
    requires_pr: true
    approval: "transformation-portal-specialist"
  usage:
    - "Nightly deep runs"
    - "Weekly comprehensive analysis"

# Fuzz suite: synthetic stressors, pathological inputs
fuzz_suite:
  description: "Edge cases and stress tests"
  generation_script: "scripts/generate_fuzz_workload.py"
  scenarios:
    - "max_dimensions"  # 16K+ images
    - "weird_exif"      # Unusual metadata
    - "high_frequency"  # Texture-heavy images
    - "minimal_content" # Solid colors
  usage:
    - "Weekly stress tests"
    - "Pre-release validation"
```

### Workload-to-Lane Mapping

```yaml
# CI lane configuration
ci_lanes:
  pr_fast:
    workload: "golden_suite"
    mode: "synthetic"  # Or minimal real (3 images)
    max_duration_min: 2
    
  nightly_baseline:
    workload: "golden_suite"
    mode: "real"
    sample_size: "full"
    backends: ["da3"]
    
  nightly_comprehensive:
    workload: "canary_suite"
    mode: "real"
    sample_size: "full"
    backends: ["da3", "depth_pro"]
    
  weekly_stress:
    workload: "fuzz_suite"
    mode: "real"
    sample_size: "full"
    backends: ["da3"]
```

---

## Consequences

### Positive

1. **Auditability:** Every policy change has a paper trail
2. **Transparency:** Budgets are explicit, versioned, reviewable
3. **Statistical Rigor:** Reduces false positives from measurement noise
4. **Developer Trust:** Fair, consistent, escapable enforcement
5. **Safe Rollout:** Evidence gates prevent premature enforcement
6. **Alignment with Research:** Workload governance inspired by PERUN

### Negative

1. **Complexity:** More governance infrastructure to maintain
2. **Cultural Change:** Requires discipline around policy review
3. **Initial Overhead:** Setup and calibration time
4. **Learning Curve:** Developers must understand governance workflows

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Policy drift (files out of sync with code) | CI validation of policy schema + consistency checks |
| Governance becomes bureaucracy | Keep waiver process lightweight; automate where possible |
| False positives erode trust | Require strong statistical evidence before enforcement |
| Budget changes become rubber-stamp | Require evidence link; automated validation |

---

## Required Enforcement

### CI Gates

1. **Policy Schema Validation:**
   ```yaml
   - name: Validate APEX Policy Files
     run: |
       python scripts/apex_validate_policy.py \
         --policy-dir docs/apex/policy/ \
         --schema-version 1.0.0
   ```

2. **Statistical Evidence Check:**
   ```python
   # In apex_enforce_gate.py (enforce mode only)
   if mode == "enforce":
       evidence_quality = check_evidence_quality(run_id, ledger)
       if evidence_quality < REQUIRED_THRESHOLD:
           logger.warning("Insufficient evidence quality; downgrading to shadow")
           mode = "shadow"
   ```

3. **Waiver Validation:**
   ```python
   # In CI, check if apex-waiver label exists
   if has_waiver_label(pr_number):
       validate_waiver_template(pr_number)
       # If valid, allow merge despite failure
   ```

### Contract Tests

```python
def test_policy_schema_valid():
    """Ensure policy files conform to schema."""
    from apex_governance import validate_policy_files
    errors = validate_policy_files("docs/apex/policy/")
    assert not errors, f"Policy validation failed: {errors}"

def test_budget_consistency():
    """Ensure budgets match DEFAULT_BUCKETS (until migrated)."""
    from apex_governance import check_budget_consistency
    mismatches = check_budget_consistency()
    assert not mismatches

def test_statistical_sample_size():
    """Enforce minimum sample size for enforcement mode."""
    from apex_governance import check_sample_size_requirement
    assert check_sample_size_requirement(n=19, mode="enforce") == False
    assert check_sample_size_requirement(n=20, mode="enforce") == True
```

---

## Migration Plan

### Phase 1: Shadow Mode with Governance (Weeks 1-4)

- Deploy policy files in shadow mode
- Collect real data via nightly runs
- Run statistical analysis offline (manual)
- Refine budgets based on observed distributions

### Phase 2: Gradual Enforcement (Weeks 5-8)

- Enable enforce mode for nightly runs only
- Collect incident workflow data
- Test waiver mechanism
- Calibrate thresholds

### Phase 3: Production Enforcement (Week 9+)

- Enable enforce mode for merge-to-main gate
- Keep PR lane in shadow (or minimal real)
- Full governance workflows operational

---

## Success Metrics

- **False Positive Rate:** < 5% of enforcement failures are invalid
- **Time-to-Detection:** Regressions detected within 24 hours (nightly) or immediately (PR with real execution)
- **Time-to-Reproduce:** < 5 minutes with provided command
- **Budget Change Frequency:** < 1 per month for stable workflows
- **Waiver Approval Rate:** < 10% of PRs require waivers (indicates thresholds are well-calibrated)

---

## Related Work

- **PERUN:** Performance regression testing framework with workload governance and performance fuzzing ([arXiv:2207.12900](https://arxiv.org/pdf/2207.12900))
- **OpenTelemetry:** Observability standard (inspiration for trace/metric governance)
- **Open Policy Agent:** Policy-as-code approach (not used here but conceptually aligned)

---

## References

- ADR-025: APEX End-to-End Workflow Architecture
- APEX_CONTRACT.md: v1.0.0 baseline contract
- Issue #879: Phase 4 Execution Plan
- Issue (current): APEX Governance Roadmap

---

## Signature

This ADR establishes the governance framework for APEX performance management.

**Authority:** Transformation Portal Architect  
**Effective Date:** 2026-02-09  
**Next Review:** 2026-05-09 (quarterly)
