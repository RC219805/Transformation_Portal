# Quick Reference: Update Branch Protection Settings

## ✅ COMPLETED (2026-02-03)

**Status:** Branch protection successfully updated and verified via PR #804.

**Validated:**
- ✅ Only `CI Gate` is required (single stable check)
- ✅ No matrix-expanded checks blocking merges
- ✅ Strict mode enabled (branches must be up to date)
- ✅ No phantom "Expected" checks

**Documentation:** See `docs/pr_archive/architecture/PR_804_GOVERNANCE_ANALYSIS.md` for full validation report.

---

## ⚡ Quick Steps

### 1. Navigate to Branch Protection Settings
```
https://github.com/RC219805/Transformation_Portal/settings/branches
```

### 2. Edit the `main` Branch Protection Rule
Click **Edit** button next to the `main` branch rule.

### 3. Update Required Status Checks

**In the "Require status checks to pass before merging" section:**

#### ✅ REQUIRE ONLY THIS:
- `CI Gate`

#### ❌ REMOVE ALL MATRIX-EXPANDED CHECKS:
- `test (3.11, cpu, core)` ← No longer needed (aggregated by CI Gate)
- `test (3.12, cpu, core)` ← No longer needed (aggregated by CI Gate)
- `test (3.11, cpu, ml)` ← No longer needed (aggregated by CI Gate)
- `test (3.10, cpu, core)` ← Legacy check that was blocking PRs
- `lint` ← No longer needed (aggregated by CI Gate)
- `generate-manifest` ← No longer needed (aggregated by CI Gate)

### 4. Save Changes
Click **Save changes** button at the bottom.

### 5. Verify
- Open any PR
- Check that only `CI Gate` appears in required checks
- CI Gate aggregates all upstream jobs (lint, test matrix, manifest)
- Future matrix changes won't require admin intervention

---

## 📋 CI Gate Pattern (Stable Aggregator)

From `.github/workflows/build.yml`:

### What is CI Gate?
`CI Gate` is a **stable aggregator job** that:
- Depends on all critical jobs: `lint`, `test`, `generate-manifest`
- Runs even if upstream jobs fail (using `if: always()`)
- Reports a single green/red status based on all upstream results
- **Has a stable name** that doesn't change when the test matrix evolves

### Current Upstream Jobs (Auto-Aggregated)
| Job Name | Python | Device | Test Type | Purpose |
|----------|--------|--------|-----------|---------|
| `lint` | 3.12 | n/a | n/a | Code linting |
| `test (3.11, cpu, core)` | 3.11 | cpu | core | Core tests (no ML) |
| `test (3.12, cpu, core)` | 3.12 | cpu | core | Core tests (no ML) |
| `test (3.11, cpu, ml)` | 3.11 | cpu | ml | ML/AI tests |
| `generate-manifest` | 3.12 | n/a | n/a | Montecito manifest |

**Branch protection requires:** `CI Gate` only
**CI Gate requires:** All jobs above must succeed

---

## 🔍 Why This Change?

**Problem:** Branch protection tied to matrix-expanded job names
- When Python 3.10 was removed, PRs got stuck on "Expected — Waiting for status"
- Every matrix change (Python versions, test types, devices) required manual admin updates
- Job names like `test (3.11, cpu, core)` are unstable and change with the matrix

**Solution:** CI Gate pattern
- Single stable check name: `CI Gate`
- Matrix can evolve freely without admin intervention
- Add Python 3.13? No admin action needed. Drop 3.11? No admin action needed.
- Clear failure reporting: CI Gate shows which upstream job failed

---

## 📚 Full Documentation

See `docs/operations/branch_protection_setup.md` for:
- Detailed configuration guide
- Maintenance checklist
- Historical context
- Troubleshooting

---

## 🆘 Troubleshooting

### "I don't see `CI Gate` in the status check list"
- Wait for a PR to run the workflow at least once
- The check populates from actual workflow runs
- You can trigger a workflow manually via Actions tab → Run workflow

### "Should I keep the old matrix checks too?"
- **No.** Remove all matrix-expanded checks (`test (3.11, cpu, core)`, `lint`, etc.)
- Requiring `CI Gate` alone is sufficient — it aggregates all upstream jobs
- Keeping old checks defeats the purpose (they'll break again when matrix changes)

### "What if I want to make `generate-manifest` optional?"
- Edit `.github/workflows/build.yml`
- Remove `generate-manifest` from the `ci_gate` job's `needs:` array
- CI Gate will no longer require manifest generation to pass

### "CI Gate shows as failed — how do I know what broke?"
- Click on the `CI Gate` check in the PR
- View the "Summarize upstream results" step output
- It shows each upstream job's result (success/failure/skipped)
- Navigate to the specific failed job for details

### "Still stuck?"
- Check `.github/workflows/build.yml` for the `ci_gate` job definition
- Verify workflow files have no syntax errors
- Contact repository maintainers with CI/CD label

---

**Questions?** See `docs/operations/branch_protection_setup.md` or create an issue with the `ci/cd` label.
