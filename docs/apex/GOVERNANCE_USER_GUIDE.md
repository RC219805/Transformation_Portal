# APEX Governance User Guide

**Version:** 1.0.0
**Last Updated:** 2026-02-09
**Audience:** Developers, Maintainers, Architects

---

## Purpose

This guide explains how to interact with the APEX performance governance system. APEX is not just a monitoring dashboard—it's a **performance judge** that enforces quality standards through explicit policy and governance workflows.

---

## Table of Contents

1. [Understanding APEX Governance](#understanding-apex-governance)
2. [When APEX Blocks Your PR](#when-apex-blocks-your-pr)
3. [Requesting a Waiver](#requesting-a-waiver)
4. [Proposing Budget Changes](#proposing-budget-changes)
5. [Responding to Performance Incidents](#responding-to-performance-incidents)
6. [Understanding the Verdict](#understanding-the-verdict)
7. [FAQ](#faq)

---

## Understanding APEX Governance

### What is APEX?

**APEX** (Architectural Photo Enhancement eXecution) is the performance governance platform for the Transformation Portal. It:

- **Measures** performance across workflows, zones, and backends
- **Compares** current performance to established baselines
- **Judges** whether performance changes are acceptable
- **Enforces** performance budgets (in enforce mode)
- **Tracks** governance decisions (waivers, incidents, budget changes)

### Governance Philosophy

APEX treats performance as a **first-class requirement**:

- Performance budgets are **policy-as-code** (versioned, reviewable, auditable)
- Enforcement requires **strong statistical evidence** (not single-run panics)
- Exceptions are **allowed with justification** (waivers add friction, not barriers)
- Decisions are **logged and traceable** (audit trail for learning)

### Enforcement Modes

APEX operates in three modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Shadow** | Reports violations, never blocks | Calibration, baseline establishment |
| **Enforce** | Blocks merge on policy violations | Production enforcement (after calibration) |
| **Disabled** | Gate not executed | Temporary suspension (requires justification) |

**Current status:** Shadow mode (as of 2026-02-09)
**Transition plan:** See [enforcement_policy.yaml](../policy/enforcement_policy.yaml)

---

## When APEX Blocks Your PR

### Step 1: Understand the Verdict

Check the APEX PR comment. It will show:

```markdown
# 🎯 APEX Performance Report

## Verdict: ❌ FAILED

**Failing Buckets:**
- `pool_large_mps` (zone: us-west-2a)
  - Current p95: 12.3s
  - Baseline p95: 10.0s
  - Regression: +23% (threshold: 10%)

## Statistical Evidence
- Mann-Whitney U test: p=0.003 (significant)
- Effect size (Cliff's Delta): 0.42 (medium)
- Sample size: current n=25, baseline n=30
```

### Step 2: Reproduce Locally

Use the command from the PR comment:

```bash
python scripts/apex_matrix_runner.py \
  --workflow-versions v2 \
  --zones local \
  --backend-id da3 \
  --input-dir ./tests/fixtures/apex_images \
  --output-dir ./apex_results \
  --ledger-db ./apex_performance.db \
  --device cpu
```

### Step 3: Investigate

**Common causes:**

1. **Algorithmic change:** Did you intentionally modify the pipeline?
   - Check: `git diff HEAD~1 src/transformation_portal/`

2. **Dependency update:** Did dependencies change?
   - Check: `git diff HEAD~1 requirements/`

3. **Measurement noise:** Is this a false positive?
   - Re-run locally multiple times
   - Check variance in measurements

4. **Accidental regression:** Did you introduce inefficiency?
   - Profile with: `python -m cProfile -s cumtime scripts/apex_matrix_runner.py ...`

### Step 4: Choose Your Path

You have three options:

#### Option A: Fix the Regression

1. Identify root cause
2. Optimize code
3. Re-run APEX to verify fix
4. Commit and push

APEX will re-run automatically and (hopefully) pass.

#### Option B: Request a Waiver

See [Requesting a Waiver](#requesting-a-waiver) below.

#### Option C: Propose Budget Change

See [Proposing Budget Changes](#proposing-budget-changes) below.

---

## Requesting a Waiver

### When to Request a Waiver

Waivers are for **justified exceptions**, not shortcuts:

**Valid reasons:**
- ✅ Performance regression is intentional (trading speed for quality)
- ✅ Temporary regression with committed fix plan
- ✅ Regression only affects edge case (low-priority bucket)
- ✅ False positive confirmed by investigation

**Invalid reasons:**
- ❌ "CI is annoying, let me merge"
- ❌ "I don't have time to fix this"
- ❌ "It's probably fine"

### How to Request a Waiver

**Step 1:** Add label to PR

```
apex-waiver
```

**Step 2:** Add waiver template to PR description

```markdown
## APEX Waiver Request

**Justification:** [Required - why is this regression acceptable?]

**Scope:**
- Bucket: `pool_large_mps`
- Zone: `us-west-2a`
- Workflow: `v2`

**Expiry Date:** 2026-03-15 (30 days from now, max 90 days)

**Mitigation Plan:** [What will be done to address this?]

**Evidence:** [Link to analysis, profiling, investigation]

**Approver:** @transformation-portal-architect
```

**Step 3:** Wait for review

- Architect reviews justification
- Approver comments on PR (approval or rejection)
- If approved: PR can merge, tracking issue auto-created
- If rejected: Fix regression or re-negotiate

**Step 4:** Post-merge tracking

- Waiver tracking issue created automatically
- Expiry date set (default 30 days, max 90 days)
- Issue auto-closes on expiry or manual resolution

### Waiver Scope

Waivers are **scoped** to prevent blanket exceptions:

- ✅ **single_bucket:** Waive failure for one bucket only
- ✅ **single_zone:** Waive failure for one zone only
- ✅ **workflow_version:** Waive failure for v1 or v2 only
- ❌ **global:** Not allowed (fix root cause instead)

### Waiver Expiry

All waivers expire:

- **Default:** 30 days
- **Maximum:** 90 days
- **Warning:** 7 days before expiry
- **Auto-close:** Tracking issue closes on expiry

If regression is permanent, propose a budget change instead.

---

## Proposing Budget Changes

### When to Propose Budget Changes

Budget changes are for **permanent shifts in baseline performance**:

**Valid reasons:**
- ✅ Algorithm improvement (higher quality, higher cost)
- ✅ Infrastructure change (new hardware, OS update)
- ✅ Dependency shift (library upgrade with known perf impact)
- ✅ Baseline recalibration (after 30+ days of new data)

**Invalid reasons:**
- ❌ "Raise threshold to fix CI" (anti-pattern)
- ❌ Temporary regression (use waiver instead)
- ❌ No evidence or investigation

### How to Propose Budget Changes

**Step 1:** Gather evidence

```bash
# Query ledger for baseline shift
python scripts/apex_query_baseline.py \
  --bucket pool_large_mps \
  --zone us-west-2a \
  --days 30 \
  --output baseline_shift_analysis.json
```

**Step 2:** Investigate attribution

- Which commits caused the shift?
- Is it intentional or accidental?
- Is the change justified?

**Step 3:** Create PR

Edit: `docs/apex/policy/performance_budgets.yaml`

```yaml
budgets:
  - workflow_version: "v2"
    bucket_name: "pool_large_mps"
    thresholds:
      p50_sec: 6.5  # Changed from 6.0
      p95_sec: 11.0  # Changed from 10.0
      max_regression_pct: 10.0
    notes: |
      Budget updated 2026-02-15 due to algorithm enhancement (PR #123).
      Evidence: ledger query shows consistent +8% shift over 30 days.
      Attribution: New depth refinement stage adds quality at cost of speed.
      Trade-off accepted per ADR-027.
```

**Step 4:** Add PR label and evidence

- Label: `apex-policy-change`
- Link evidence in PR description:
  - Ledger query results
  - Baseline shift analysis
  - Attribution investigation

**Step 5:** Wait for review

- Architect reviews evidence
- Approves or requests changes
- Update `review_date` in policy file

**Step 6:** Post-merge

- New budget becomes effective
- Baseline recomputed from recent runs
- Old budget archived for historical queries

---

## Responding to Performance Incidents

### What are Performance Incidents?

When APEX detects a regression in **enforce mode**, it automatically creates a GitHub issue:

```markdown
Title: [APEX] Performance Regression: pool_large_mps in us-west-2a (High)

## Regression Summary
- Bucket: `pool_large_mps`
- Current p95: 12.3s
- Baseline p95: 10.0s
- Regression: +23%

## Evidence
- Statistical test: Mann-Whitney U, p=0.003
- Effect size: Cliff's Delta = 0.42 (medium)

## Resolution Checklist
- [ ] Investigation assigned
- [ ] Root cause identified
- [ ] Fix implemented OR waiver requested
- [ ] Verification run clean
- [ ] Incident closed
```

### Incident Severity Levels

| Severity | Regression | SLA | Escalation |
|----------|-----------|-----|------------|
| **Critical** | >50% or complete failure | 4 hours | Architect |
| **High** | >15% (blocking) | 24 hours | Architect |
| **Medium** | 10-15% (warning) | 72 hours | None |
| **Low** | Minor issues | 1 week | None |

### Incident Response Workflow

1. **Triage:** Assigned to responsible team/person
2. **Investigate:** Follow suggested actions in issue
3. **Resolution:** Fix regression OR request waiver
4. **Verification:** Re-run APEX to confirm fix
5. **Close:** Document root cause, close issue

---

## Understanding the Verdict

### APEX Verdicts

APEX produces exactly three verdicts:

#### ✅ PASS

**Definition:** All buckets meet thresholds AND no bucket regresses >10% vs baseline.

**Action:** Merge allowed (in enforce mode).

**Example:**
```
V2 Gate: PASSED ✅
All buckets within limits. Max regression: +2.3%
```

#### ⚠️ WARN

**Definition:** At least one bucket exceeds threshold by ≤15% OR regression 10-15%.

**Action:** Merge allowed (informational only).

**Example:**
```
V2 Gate: WARNING ⚠️
Bucket 'pool_large_mps' p95: 11.2s (limit: 10.0s, +12%)
Consider investigating before merge.
```

#### ❌ FAIL

**Definition:** At least one bucket exceeds threshold by >15% OR regression >15%.

**Action:** Merge **blocked** (in enforce mode).

**Example:**
```
V2 Gate: FAILED ❌
Bucket 'aerial_large_mps' p95: 14.5s (limit: 10.0s, +45%)
BLOCKING: Performance regression detected.
```

### Statistical Interpretation

APEX uses **robust statistical methods**:

- **Mann-Whitney U test:** Non-parametric comparison (no normality assumption)
- **Cliff's Delta:** Effect size measure (small/medium/large)
- **Median + MAD:** Outlier detection (robust to skewed distributions)
- **Bootstrap CI:** Confidence intervals for uncertainty quantification

**Sample size requirements:**
- p50: minimum 10 samples
- p95: minimum 20 samples
- p99: minimum 50 samples

If insufficient samples: verdict is `insufficient_data` (shadow mode only).

---

## FAQ

### Q: Why is APEX blocking my PR if I haven't touched performance code?

**A:** Regressions can be caused by:
- Dependency updates (library version changes)
- Unintentional algorithmic changes (refactoring side effects)
- Infrastructure shifts (OS updates, hardware)

Investigate using the reproduce command and profiling.

### Q: Can I just raise the threshold to fix my CI failure?

**A:** No. This is explicitly an anti-pattern.

Budget changes require:
- Evidence (ledger query showing shift)
- Attribution (which commits caused it)
- Justification (why is the shift acceptable)
- Architect approval

If temporary, use a waiver instead.

### Q: How long does a waiver last?

**A:** Default 30 days, maximum 90 days.

Waivers are time-boxed to ensure:
- Temporary regressions get fixed
- Permanent shifts trigger budget changes
- Policy doesn't drift from reality

### Q: What if I disagree with APEX's verdict?

**A:** Escalation paths:

1. **Investigate:** Reproduce locally, check variance
2. **Request waiver:** If regression is justified
3. **Dispute methodology:** Create issue with `apex-policy` label
4. **Escalate to Architect:** Tag `@transformation-portal-architect`

APEX is not infallible. If you find a false positive, report it.

### Q: When will enforce mode be enabled?

**A:** Transition plan (see `enforcement_policy.yaml`):

1. **Phase 1 (Shadow mode):** 30 days of data collection (current)
2. **Phase 2 (Enforce nightly):** 14 days of nightly enforcement testing
3. **Phase 3 (Enforce PR):** Optional (may stay shadow for PRs)

Enforce mode requires:
- Stable baselines (30+ days)
- Sufficient samples (n≥20)
- Real (non-synthetic) data
- 95% confidence

### Q: How do I update workload suites?

**A:** Depends on suite tier:

- **Golden suite:** Requires ADR + 30-day notice (changes invalidate baselines)
- **Canary suite:** Requires PR + Specialist approval (monthly changes allowed)
- **Fuzz suite:** Requires PR + tests (scenario changes reviewed)

See `workload_suites.yaml` for details.

### Q: Can I run APEX locally?

**A:** Yes! Use the reproduce command from PR comments:

```bash
python scripts/apex_matrix_runner.py \
  --workflow-versions v2 \
  --zones local \
  --backend-id da3 \
  --input-dir ./tests/fixtures/apex_images \
  --output-dir ./apex_results \
  --ledger-db ./apex_performance.db \
  --device cpu
```

Add `--dry-run --synthetic` for fast validation (no ML deps).

---

## Contact

**Governance Owner:** transformation-portal-architect

**Questions:**
- Policy interpretation: Open issue with `apex-policy` label
- Budget adjustment requests: PR with `apex-policy-change` label
- Waiver requests: PR with `apex-waiver` label
- Incident response: Respond to auto-created incident issues

**Documentation:**
- [APEX Contract](../contracts/APEX_CONTRACT.md)
- [Performance Budgets](../policy/performance_budgets.yaml)
- [Enforcement Policy](../policy/enforcement_policy.yaml)
- [Governance Rules](../policy/governance_rules.yaml)
- [Workload Suites](../policy/workload_suites.yaml)
- [ADR-026: Governance Framework](../../architecture/decisions/ADR-026-APEX-governance-framework.md)

---

**Last Updated:** 2026-02-09
**Version:** 1.0.0
**Next Review:** 2026-05-09 (quarterly)
