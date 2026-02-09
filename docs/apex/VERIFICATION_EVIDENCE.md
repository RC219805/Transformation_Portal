# APEX Phase 2 Verification Evidence

**Verification Date:** 2026-02-09T19:42:00Z
**Branch:** `feat/apex-real-pipeline-integration`
**Commit:** `d71dd2091816c31b53935f02d046b6be7d58d9a3`
**Verifier:** Human + transformation-portal-architect agent

---

## What "Verified" Means in This Document

**"Verified"** = reproducible command + log/output + commit SHA

This is NOT marketing. This is audit-grade evidence.

---

## Test Suite Evidence

### Fast CI Lane (PR Gating Equivalent)

**Command:**
```bash
pytest -q -m "not ml and not slow" --tb=no
```

**Output:**
```
1504 passed, 132 skipped, 125 deselected in 38.56s
```

**Verdict:** ✅ **GREEN** (100% pass rate for eligible tests)

**Evidence Location:** Local execution log, 2026-02-09

### Contract Tests (Subset)

**Command:**
```bash
pytest tests/test_apex_contracts.py tests/test_apex_contract_verification.py -q
```

**Output:**
```
27 passed, 1 skipped in 3.25s
```

**Verdict:** ✅ **GREEN**

---

## Truth Properties Verification

### 1. Event Gating

**Method:** Code review of workflow YAML
**Location:** `.github/workflows/apex_performance.yml:105-111`

**Evidence:**
```yaml
if [[ "${{ github.event_name }}" == "pull_request" ]] || [[ "${{ github.event_name }}" == "push" ]]; then
  MODE="synthetic"
  echo "ℹ️ PR/push lane: forcing synthetic mode (fast validation)"
elif [[ "${{ github.event_name }}" == "schedule" ]]; then
  MODE="real"
  echo "ℹ️ Scheduled run: using real execution (nightly monitoring)"
fi
```

**Verdict:** ✅ **Airtight by design** (not yet runtime-verified)

**Runtime verification needed:** Manual workflow_dispatch trigger post-merge

---

### 2. Dependency Gating

**Method:** Code review of workflow YAML
**Location:** `.github/workflows/apex_performance.yml:76-80`

**Evidence:**
```yaml
- name: Install dependencies (ML tier)
  if: github.event.inputs.mode == 'real' || github.event_name == 'schedule'
  run: |
    python -m pip install -e .[ml]
```

**Verdict:** ✅ **Airtight by design** (not yet runtime-verified)

**Runtime verification needed:** Check workflow logs show conditional install behavior

---

### 3. Metadata/Provenance

**Method:** Code review + schema inspection
**Location:** `scripts/apex_matrix_runner.py`, workflow inputs

**Fields Captured:**
- ✅ run_id (GitHub Actions context)
- ✅ commit_sha (GitHub Actions context)
- ✅ workflow_version (matrix: v1, v2)
- ✅ zone (matrix: local)
- ✅ backend_id (input: da3 default)
- ✅ device (input: cpu default)
- ✅ mode (computed: synthetic/real)
- ✅ sample_size (input: 3 default)

**Missing (future enhancement):**
- ⚠️ Warmup behavior documentation
- ⚠️ RNG seeds
- ⚠️ Dependency snapshot/lockfile

**Verdict:** ✅ **Complete for Phase 2 scope**

---

### 4. Semantic Honesty

**Method:** Code review of PR comment generator
**Location:** `scripts/apex_pr_comment.py` + workflow line 217

**Evidence:**
```bash
if [[ "${MODE}" == "synthetic" ]]; then
  CMD+=(--synthetic)
fi
```

**Verdict:** ✅ **Honest** (--synthetic flag passed when appropriate)

**Runtime verification needed:** Generate actual PR comment and inspect

---

### 5. Artifact Durability

**Method:** Code review of workflow artifact retention
**Location:** `.github/workflows/apex_performance.yml`

**Retention Tiers:**
- Performance capsules: 3 days
- SQLite ledger: 90 days
- Automated backups: Weekly

**Verdict:** ✅ **Multi-tier retention implemented**

**Runtime verification needed:** Post-merge artifact download and inspection

---

## Diff Analysis

**Command:**
```bash
git diff --stat origin/main..feat/apex-real-pipeline-integration
```

**Summary:**
- 16 files changed
- 721 insertions
- 1312 deletions
- **Net: -591 lines (cleanup)**

**Key Changes:**
- `.github/workflows/apex_performance.yml`: Hybrid CI implementation
- `scripts/apex_matrix_runner.py`: Simplified (removed 101 lines)
- Deleted obsolete docs: `phase3/README.md`, `phase4/EXECUTION_PLAN.md`
- New docs: Phase 2 completion + implementation summary

**Verdict:** ✅ **Clean, focused, net reduction**

---

## What Is NOT Yet Verified

### Runtime Behavior (Post-Merge Required)

1. **Actual synthetic run in PR lane**
   - Verify no ML deps installed
   - Verify fast execution (<5 min)
   - Verify PR comment generated correctly

2. **Actual real run via workflow_dispatch**
   - Verify ML deps installed
   - Verify depth inference executes
   - Verify artifacts uploaded
   - Verify ledger written correctly

3. **Scheduled run behavior**
   - Wait for first Sunday 00:00 UTC run
   - Verify automatic mode=real selection

### CI Status

**Blocker:** GitHub API rate-limited during verification

**Needed:** Confirm CI checks GREEN on commit d71dd209

---

## Governance Theater Removed

**Changes Made:**

1. ✅ Removed "SIGNED: Transformation Portal Architect"
2. ✅ Changed "APPROVED FOR MERGE" → "RECOMMENDATION: MERGE-READY"
3. ✅ Corrected "77/78 passing" → "1504/1504 passing"
4. ✅ Added explicit "Not Yet Verified" section
5. ✅ Quarantined merge script → `scripts/runbooks/`
6. ✅ Added "Human verification required" disclaimer

**Principle Established:**

> "Verified" means there exists a reproducible command + log/URL + commit SHA.

---

## Recommendation

**Status:** ✅ **MERGE-READY** (subject to human approval)

**Confidence Level:** **High** for code quality, **Medium** for runtime behavior (needs post-merge validation)

**Merge Safety:** Shadow mode prevents enforcement risk

**Next Action:** Human decision + CI check + merge + post-merge validation

---

**Generated:** 2026-02-09T19:42:00Z
**Evidence Location:** This document + local test logs
**Commit Under Review:** d71dd2091816c31b53935f02d046b6be7d58d9a3
