# Branch Cleanup Quick Reference

**Date:** 2026-02-16
**Status:** Post-PR #947 Merge
**Full assessment:** `docs/architecture/BRANCH_HYGIENE_ASSESSMENT_2026-02-16.md`

---

## Immediate Actions Required

### 🚨 Priority 1: Security (Within 7 Days)

**Issue:** `sentence-transformers` CVE-73169 (arbitrary code execution)

```bash
# Update requirements/ml.in
# Change: sentence-transformers>=2.2.0,<3
# To:     sentence-transformers>=3.1.0,<6

cd requirements
make compile
# Validate RAG system compatibility
# Create PR with security label
```

---

### ✅ Priority 2: Close Superseded PRs (This Week)

#### PR #845 — Close and Delete Branch
- **URL:** https://github.com/RC-88/Transformation_Portal/pull/845
- **Branch:** `RC219805-patch-4`
- **Reason:** Superseded by PRs #882, #909, #907
- **Action:** Close PR → Delete `origin/RC219805-patch-4`
- **Reference:** `docs/pr_archive/architecture/pr-review-845-883-superseded.md`

#### PR #883 — Close and Delete Branch
- **URL:** https://github.com/RC-88/Transformation_Portal/pull/883
- **Branch:** `automated/dependency-updates`
- **Reason:** Incomplete (only contains CI artifact), superseded by targeted PRs
- **Action:** Close PR → Delete `origin/automated/dependency-updates`
- **Reference:** `docs/pr_archive/architecture/pr-review-845-883-superseded.md`

---

### 🧹 Priority 3: Local Cleanup (Batch Commands)

#### Delete Branches Tracking Removed Remotes (10 branches)

```bash
cd /Users/rc/Projects/Transformation_Portal

# Prune remote-tracking references
git fetch --prune

# Delete [gone] branches
git branch -d docs/materials-v3-investigations
git branch -d docs/phase3-l1-cache-invariants
git branch -d feat/apex-research-ultra-phase1
git branch -d feat/spatial-ai-foundation
git branch -d feature/materials-v3-production-integration
git branch -d feature/materials-v3-water-sky-synthesis
git branch -d fix/apex-governance-blockers
git branch -d fix/ci-gate-preflight-classifier
git branch -d fix/depth-manifest-resolved-backend-v2
git branch -d test/da3-availability-guards

# Or use automated cleanup:
git branch -vv | grep ': gone]' | awk '{print $1}' | xargs git branch -d
```

#### Delete Local Review Branches (6 branches)

```bash
# Safe to delete (review artifacts)
git branch -d pr-785-review
git branch -d pr-803
git branch -d pr-841
git branch -d pr-851
git branch -d pr-871
git branch -d pr-906-review
```

#### Delete Superseded Local Branch

```bash
# PR #845 local branch (matches remote)
git branch -d RC219805-patch-4
```

---

### 📋 Priority 4: Audit Remaining Branches

#### Check for Unmerged Work

```bash
# For each local branch without remote tracking:
git log main..bugfix/materials-v3-critical-fixes
git log main..copilot/fix-coverage-artifact-upload
git log main..feat/apex-phase2-real-pipeline-integration
git log main..fix/depth-identity-resolved-backend
git log main..fix/depth-identity-resolved-backend-clean
git log main..precision-fix
git log main..temp/pr934-upscaling

# If output is empty → safe to delete
# If non-empty → assess whether to merge or abandon
```

#### Remote Branches to Audit

Check recent merge history and delete if merged:
- `origin/bugfix/materials-v3-critical-fixes`
- `origin/copilot/restore-ci-health-and-stability`
- `origin/copilot/review-prs-relevance`
- `origin/copilot/sub-pr-941`
- `origin/feature/phase4-ml-upscaling`
- `origin/feature/phase5-stagegraph-integration`
- `origin/feature/quick-wins-batch-1`
- `origin/feature/spatial-ai-linear-ingest-phase1`
- `origin/materials-v3/phase-c2-confidence-semantics` (likely PR #947)

---

## Summary

### Current State
- **GitHub remotes:** 12 branches (excluding main)
- **Local branches:** 24 total
- **PRs to close:** 2 (both superseded)
- **Security issue:** 1 (CVE-73169)

### Expected After Cleanup
- **GitHub remotes:** ~2-4 branches (active feature work only)
- **Local branches:** ~10-12 (active work + main)
- **PRs to close:** 0
- **Security issues:** 0

### Time Estimate
- **Security fix:** 2-3 hours (testing + PR)
- **PR closure:** 10 minutes
- **Local cleanup:** 5 minutes
- **Remote audit:** 30 minutes
- **Total:** ~4 hours

---

## Verification

After completion:

```bash
# Check clean state
git fetch --prune
git branch -vv | grep ': gone]'  # Should be empty
git branch -r | wc -l  # Should be ~5-8 (main + active features)

# Verify no unmerged work lost
git log --all --oneline --since="2 weeks ago" | grep -v main | wc -l
# Should align with expected active branches
```

---

## Decision Authority

**Architect verdict:** Close PRs #845 and #883 without merge (both superseded).

**Delegation:** Specialist may execute cleanup and security fix. Architect approval required for branch retention policy.

**Reference:** `BRANCH_HYGIENE_ASSESSMENT_2026-02-16.md` (full analysis)
