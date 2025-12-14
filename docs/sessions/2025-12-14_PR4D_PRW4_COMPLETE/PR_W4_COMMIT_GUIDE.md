# PR-W4: Exact Files to Stage for Merge

**Purpose:** Clean PR with only essential files; no session artifacts or unrelated docs.

---

## Files to Stage (PR-W4 Only)

### Core Implementation
```bash
git add scripts/prw_water_validation.py
git add tests/test_prw_water_validation.py
git add tests/test_prw_water_validation_deterministic.py
git add scripts/check_regression.py
```

### Documentation (Essential Only)
```bash
git add docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md
git add docs/PR_W4_COMPLETION_ACCURATE.md
git add docs/PR_W4_DESCRIPTION.md
```

### Infrastructure
```bash
# .gitignore changes (if modified)
git add .gitignore

# Dataset scaffold (if exists and needed)
git add data/water_v0/README.md
git add data/water_v0/ci_subset.txt
```

### Verify Before Commit
```bash
# Check what you're staging
git status --short

# Should see only:
# M  scripts/prw_water_validation.py
# A  tests/test_prw_water_validation.py
# A  tests/test_prw_water_validation_deterministic.py
# A  scripts/check_regression.py
# A  docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md
# A  docs/PR_W4_COMPLETION_ACCURATE.md
# A  docs/PR_W4_DESCRIPTION.md
# M  .gitignore (if modified)
```

---

## Files to EXCLUDE (Session Artifacts)

### Root-Level Session Files
```bash
# Do NOT stage these:
PR_W4_ACCEPTANCE_CRITERIA_AUDIT.md
PR_W4_HONEST_STATUS.md
PR_W4_MERGE_READY_DESCRIPTION.md
WATER_DETECTION_ADVANCEMENT_PACKAGE.md
```

### Redundant Documentation
```bash
# Keep only essential docs; exclude:
docs/PR_W4_CORRECTNESS_FIXES.md  # Covered in COMPLETION_ACCURATE
docs/PR_W4_MERGE_READY.md        # Covered in PR_W4_DESCRIPTION
docs/WATER_DETECTION_72HR_PLAN.md       # Next-step docs, not PR scope
docs/WATER_DETECTION_72HR_PLAN_V2.md
docs/WATER_DETECTION_QUICK_REFERENCE.md
docs/WATER_DETECTION_README.md
docs/WATER_DETECTION_STRATEGIC_ASSESSMENT.md
docs/WATER_GROUND_TRUTH_FINAL.md  # Superseded by SCHEMA_FINAL
docs/WATER_GROUND_TRUTH_SCHEMA.md # Superseded by SCHEMA_FINAL
```

### Test/Validation Reports
```bash
# Do NOT commit:
docs/PR4D_FINAL_STATUS.md
docs/PR4D_VALIDATION_COMPLETE.md
docs/PR4D_VERIFICATION_COMPLETE.md
docs/cleanup_reports/
docs/validation_reports/
```

### Build Artifacts
```bash
# Do NOT commit:
.coverage
```

---

## Recommended Commit Message

```
PR-W4: Water Validation Harness (Pool + Ocean)

Added validation infrastructure for water detection quality measurement.

Core changes:
- scripts/prw_water_validation.py: Validation harness with edge alignment,
  stability, false trigger tracking, and JSON reporting
- tests/test_prw_water_validation*.py: 16 tests covering schema, metrics,
  determinism, and end-to-end validation
- scripts/check_regression.py: CI-ready regression checker (warning mode)

Schema & docs:
- docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md: Finalized schema with FT semantics
- docs/PR_W4_COMPLETION_ACCURATE.md: Detailed status and limitations
- docs/PR_W4_DESCRIPTION.md: Clean PR description

Infrastructure:
- .gitignore: Water dataset patterns (recursive, with metadata allow-list)
- data/water_v0/: Dataset scaffold (ci_subset.txt, README placeholders)

Known limitations:
- Detector is stub (PR-W1 pending)
- Thresholds uncalibrated (dataset v0 pending)
- Edge alignment defaults to 0.0 when mask unavailable

Tests: 16/16 passing
Linting: Clean (flake8)
Status: Harness complete; detector stub; integration-ready
```

---

## Pre-Merge Checklist

- [ ] Only PR-W4 files staged (no session artifacts)
- [ ] Tests passing: `pytest tests/test_prw_water_validation*.py -v`
- [ ] Linting clean: `flake8 scripts/prw_water_validation.py --max-line-length=127`
- [ ] Compilation clean: `python -m py_compile scripts/prw_water_validation.py`
- [ ] Commit message accurate and concise
- [ ] No overclaiming in PR description
- [ ] Known limitations explicitly stated

---

## Command Sequence (Copy-Paste)

```bash
# Stage core files
git add scripts/prw_water_validation.py
git add tests/test_prw_water_validation.py
git add tests/test_prw_water_validation_deterministic.py
git add scripts/check_regression.py

# Stage documentation
git add docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md
git add docs/PR_W4_COMPLETION_ACCURATE.md
git add docs/PR_W4_DESCRIPTION.md

# Stage infrastructure (if modified)
git add .gitignore

# Verify
git status --short

# Run tests
pytest tests/test_prw_water_validation.py tests/test_prw_water_validation_deterministic.py -v

# Lint check
flake8 scripts/prw_water_validation.py --max-line-length=127 --extend-ignore=E203,W503

# Commit
git commit -m "PR-W4: Water Validation Harness (Pool + Ocean)

Added validation infrastructure for water detection quality measurement.

Core changes:
- scripts/prw_water_validation.py: Harness with edge alignment, stability, false trigger tracking
- tests/test_prw_water_validation*.py: 16 tests (schema, metrics, determinism)
- scripts/check_regression.py: CI-ready regression checker

Schema & docs:
- docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md: Finalized schema
- docs/PR_W4_COMPLETION_ACCURATE.md: Detailed status
- docs/PR_W4_DESCRIPTION.md: PR description

Known limitations: Detector stub (PR-W1 pending), thresholds uncalibrated

Tests: 16/16 passing | Linting: Clean | Status: Harness complete, integration-ready"
```

---

## After Merge

### Immediate Next Steps
1. Start dataset v0 collection (Path A, Hour 0)
2. Implement PR-W1 detector (failure-driven)
3. Calibrate thresholds from dataset v0
4. Add CI regression job

### Session Cleanup
```bash
# After successful merge, clean up session files:
rm PR_W4_*.md WATER_DETECTION_*.md
rm docs/PR_W4_CORRECTNESS_FIXES.md
rm docs/PR_W4_MERGE_READY.md
rm docs/WATER_DETECTION_72HR_PLAN*.md
rm docs/WATER_DETECTION_QUICK_REFERENCE.md
rm docs/WATER_DETECTION_README.md
rm docs/WATER_GROUND_TRUTH_FINAL.md
rm docs/WATER_GROUND_TRUTH_SCHEMA.md  # Keep SCHEMA_FINAL only
rm -rf docs/cleanup_reports docs/validation_reports
rm .coverage
```

---

## Quality Verification

Before pushing, verify:
```bash
# Tests pass
pytest tests/test_prw_water_validation*.py -v
# Expected: 16 passed

# Linting clean
flake8 scripts/prw_water_validation.py --max-line-length=127 --extend-ignore=E203,W503
# Expected: no output (0 errors)

# Script runs
python scripts/prw_water_validation.py --help
# Expected: help text with no errors

# Regression checker works
python scripts/check_regression.py --help
# Expected: help text with no errors
```

---

**Ready to merge:** ✅ Yes (if checklist complete)  
**Post-merge action:** Start dataset v0 collection (Path A)
