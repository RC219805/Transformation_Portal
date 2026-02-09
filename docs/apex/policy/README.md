# APEX Policy Directory

**Purpose:** This directory contains the governance policy files for APEX performance management.

**Authority:** These files are **binding configuration** for APEX enforcement. Changes require explicit review and approval.

---

## Directory Structure

```
docs/apex/policy/
├── README.md                       # This file
├── apex_policy_schema.yaml         # Schema definitions
├── performance_budgets.yaml        # Performance thresholds (versioned)
├── enforcement_policy.yaml         # Statistical methods and gates
├── governance_rules.yaml           # Waivers, incidents, budget evolution
└── workload_suites.yaml            # Canonical test workloads
```

---

## Policy Files

### `performance_budgets.yaml`

**Purpose:** Defines performance thresholds for each workflow × bucket × zone combination.

**Schema:** See `apex_policy_schema.yaml`

**Change Process:**
1. Create PR with budget changes
2. Add `apex-policy-change` label
3. Include evidence (ledger query, baseline analysis)
4. Require Architect approval
5. Update `review_date`

**Example:**
```yaml
budgets:
  - workflow_version: "v2"
    bucket_name: "pool_large_mps"
    stability_tier: "stable"
    thresholds:
      p50_sec: 6.0
      p95_sec: 10.0
      max_regression_pct: 10.0
```

---

### `enforcement_policy.yaml`

**Purpose:** Defines statistical methods, sample size requirements, and evidence quality gates.

**Change Process:**
1. Changes require ADR for methodology shifts
2. Sample size changes require statistical justification
3. Architect approval required

**Key Settings:**
- Minimum sample sizes per percentile
- Statistical test methods (Mann-Whitney U, bootstrap CI)
- Outlier detection thresholds
- Evidence quality gates (shadow vs enforce mode)

---

### `governance_rules.yaml`

**Purpose:** Defines governance workflows (waivers, incidents, budget evolution).

**Change Process:**
1. Changes require consensus review
2. Waiver mechanism changes need broad approval (impacts all developers)
3. Incident workflow changes documented in CHANGELOG

**Key Workflows:**
- Waiver request and approval
- Incident creation and tracking
- Budget evolution process

---

### `workload_suites.yaml`

**Purpose:** Defines canonical test workloads (Golden, Canary, Fuzz suites).

**Change Process:**
1. Golden suite changes require ADR + 30-day notice
2. Canary suite changes require Specialist approval
3. Fuzz suite generation logic changes require tests

**Workload Tiers:**
- **Golden:** Stable, never changes, baseline establishment
- **Canary:** Representative, evolves slowly, comprehensive testing
- **Fuzz:** Stress tests, pathological inputs, edge cases

---

## Policy Versioning

### Schema Version

All policy files include `schema_version` field (e.g., `"1.0.0"`).

**Breaking changes require schema version bump:**
- v1.0.0 → v2.0.0: Major structural changes
- v1.0.0 → v1.1.0: Backward-compatible additions
- v1.0.0 → v1.0.1: Clarifications, typo fixes

### Policy Review Dates

Each policy file includes:
- `effective_date`: When policy becomes active
- `review_date`: When policy must be reviewed (quarterly recommended)

**Review Process:**
1. Architect schedules quarterly review
2. Review evidence from recent runs
3. Adjust budgets if justified
4. Update `review_date`

---

## Enforcement Modes

APEX supports three enforcement modes (configured in `performance_budgets.yaml`):

### Shadow Mode (Default)

- Reports violations but **does not block**
- Used for calibration and baseline establishment
- Minimum sample size: 1 (informational)

### Enforce Mode

- **Blocks merge** on policy violations
- Requires strong evidence:
  - Minimum sample size: 20
  - Real (non-synthetic) data
  - Baseline age: ≥30 days
  - Statistical confidence: 0.95

### Disabled Mode

- Gate not executed
- Used for temporary suspension (requires justification)

---

## Validation

### Schema Validation

Run before committing policy changes:

```bash
python scripts/apex_validate_policy.py \
  --policy-dir docs/apex/policy/ \
  --schema-version 1.0.0
```

**CI enforces schema validation automatically.**

### Consistency Checks

Ensures policy files are internally consistent:

```bash
python scripts/apex_validate_policy.py \
  --check consistency \
  --policy-dir docs/apex/policy/
```

Checks:
- Budget buckets exist in `DEFAULT_BUCKETS`
- Workflow versions are valid (v1, v2)
- Stability tiers match tier_policies definitions
- Sample size requirements are met

---

## Anti-Patterns

❌ **Don't:**
- Change budgets without evidence
- Raise thresholds to "fix" CI failures (investigate root cause)
- Skip review dates (drift leads to stale policy)
- Use enforce mode without 30-day baseline

✅ **Do:**
- Link ledger queries in budget change PRs
- Keep review dates current
- Use shadow mode for new workflows
- Document rationale in policy files

---

## Migration from Hardcoded Thresholds

**Status:** In progress (v1.0.0 uses `DEFAULT_BUCKETS` in code)

**Migration Plan:**
1. Create `performance_budgets.yaml` matching `DEFAULT_BUCKETS`
2. Add validation ensuring sync
3. Migrate code to read from YAML
4. Deprecate `DEFAULT_BUCKETS` (keep as fallback)
5. Remove hardcoded thresholds (v2.0.0+)

---

## Contact

**Policy Owner:** transformation-portal-architect

**Questions:**
- Policy interpretation: Open issue with `apex-policy` label
- Budget adjustment requests: PR with `apex-policy-change` label
- Waiver requests: PR with `apex-waiver` label

---

**Last Updated:** 2026-02-09  
**Schema Version:** 1.0.0
