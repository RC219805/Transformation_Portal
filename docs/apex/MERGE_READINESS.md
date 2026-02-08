# APEX Merge Readiness Checklist

**Contract Version:** 1.0.0
**Schema Version:** 3.0.0
**Last Verified:** 2026-02-08

## Machine-Verifiable Requirements

This document defines the **hard invariants** that must be true before APEX changes can merge.
Each item is checkable via `scripts/apex_verify_contract.py`.

### 1. Execution Mode Enforcement

- [x] **Requirement:** `--dry-run` flag is REQUIRED unless `REAL_EXECUTION_ENABLED=1` env var is set
- **Verification:** Run `apex_matrix_runner.py` without `--dry-run` → exit code 2 with helpful message
- **Evidence:** `scripts/apex_matrix_runner.py` lines 95-105
- **Test:** `tests/test_apex_contract_verification.py::test_dry_run_enforced`

### 2. Synthetic Data Labeling

- [x] **Requirement:** PR comment MUST contain literal marker `[SYNTHETIC DATA]` when using dry-run
- **Verification:** Parse PR comment artifact → assert marker present if any capsule has `is_synthetic=True`
- **Evidence:** `scripts/apex_pr_comment.py` lines 145-155
- **Test:** `tests/test_apex_contract_verification.py::test_synthetic_label_enforced`

### 3. Aggregation Scoping

- [x] **Requirement:** Aggregation queries MUST filter by `run_id` AND `commit_sha`
- **Verification:** SQL query inspection + end-to-end test with multi-run DB
- **Evidence:** `scripts/apex_aggregate_ledger.py` lines 67-72
- **Test:** `tests/test_apex_contract_verification.py::test_aggregation_scoped`

### 4. Minimum Sample Size Protection

- [x] **Requirement:** `n < 20` MUST produce `insufficient_data` verdict (never blocks)
- **Verification:** Unit test with capsule sets of size 1, 19, 20, 21
- **Evidence:** `src/transformation_portal/metrics/gate.py` lines 85-92
- **Test:** `tests/test_apex_contract_verification.py::test_min_samples_enforced`

### 5. Structural Synthetic Protection

- [x] **Requirement:** `PerformanceCapsule` carries `is_synthetic` field
- [x] **Requirement:** Ledger stores `is_synthetic` column
- [x] **Requirement:** Aggregator ignores synthetic capsules for baseline regression
- **Verification:** Schema introspection + data flow tests
- **Evidence:**
  - `src/transformation_portal/metrics/performance_capsule.py` line 45
  - `src/transformation_portal/metrics/ledger.py` schema v3
  - `src/transformation_portal/metrics/aggregator.py` lines 120-125
- **Test:** `tests/test_apex_contract_verification.py::test_synthetic_isolation`

## Human-Required Actions

These items are not machine-checkable but are part of the contract:

- [ ] **Human Approval:** At least one maintainer review
- [ ] **Contract Review Scheduled:** Next review date = 2026-05-08
  - Calendar entry: "APEX Contract Review Q2 2026"
  - Owner: RC219805
- [ ] **Documentation Updated:** All examples/READMEs reflect current contract
- [ ] **Breaking Changes:** CHANGELOG entry if contract version incremented

## Running Verification

```bash
# Verify all contract invariants
python scripts/apex_verify_contract.py

# CI check (returns 0 only if all pass)
pytest tests/test_apex_contract_verification.py -v --maxfail=1
```

## Compliance Table

| Requirement ID | Status | Evidence Commit | Verification Method |
|---------------|--------|----------------|---------------------|
| EXEC-1 (dry-run) | ✅ COMPLIANT | `8f218ea` | Unit test + CLI smoke |
| LABEL-1 (synthetic marker) | ✅ COMPLIANT | `8f218ea` | Integration test |
| SCOPE-1 (run scoping) | ✅ COMPLIANT | `8f218ea` | SQL audit + test |
| SAMPLE-1 (min n) | ✅ COMPLIANT | `8f218ea` | Parametric test |
| STRUCT-1 (is_synthetic) | ✅ COMPLIANT | `8f218ea` | Schema test |

## Failure Modes

### If verification fails:

1. **DO NOT MERGE** until the specific check passes
2. Update evidence links above after fix
3. Re-run full verification suite
4. Update "Last Verified" timestamp

### If requirements change:

1. Increment contract version (major if breaking)
2. Update this checklist
3. Add migration tests
4. Document in CHANGELOG

---

**Principle:** Text is vibes. Checks are reality.

Every strong claim in this file is traceable to a test or evidence artifact.
