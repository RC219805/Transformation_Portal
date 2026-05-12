# Branch Hygiene Assessment — 2026-02-16

**Architect:** Transformation Portal Architect
**Context:** Post-PR #947 merge (Materials V3 Phase C.2)
**Main commit:** `0582b30d`
**Assessment scope:** GitHub remote branches and local repository cleanup

---

## Executive Summary

### Current State
- **GitHub remote branches:** 12 active (excluding main)
- **Local branches:** 24 total (15 without remote tracking)
- **PRs requiring action:** 2 (both superseded)
- **Governance concern:** Significant branch drift and stale feature branches

### Key Findings
1. **PR #845 and #883 are superseded** ✅ Already documented in `pr-review-845-883-superseded.md`
2. **No immediate merge conflicts** — branches are behind main but not blocking
3. **Branch hygiene debt:** Multiple local branches tracking deleted remotes (`[gone]`)
4. **Security action item:** `sentence-transformers` CVE-73169 requires remediation

### Recommended Actions
1. **Close PRs #845 and #883** (GitHub web UI)
2. **Delete remote branches** after PR closure
3. **Local cleanup:** Prune 9 branches tracking deleted remotes
4. **Security fix:** Create focused PR for `sentence-transformers>=3.1.0`

---

## Detailed Assessment

### 1. GitHub Remote Branches (Active PRs)

#### Branch: `automated/dependency-updates` (PR #883)
- **Status:** Open since 2026-02-09
- **Drift:** 22 commits behind main, 3 commits ahead
- **Last update:** 2 days ago
- **Verdict:** **SUPERSEDED — CLOSE**

**Rationale:**
- PR contains only `safety-report.json` (a CI artifact that violates hygiene policy)
- Actual dependency updates described in PR are missing from diff
- Individual dependency bumps already handled via targeted PRs (#918, #905, #901, #911, #916, #917, #919)
- Safety CLI already upgraded from deprecated 2.3.4 to 3.7.0 in PR #918

**Blocking issue identified:**
- `sentence-transformers>=2.2.0,<3` has CVE-73169 (arbitrary code execution via unsafe `torch.load()`)
- Remediation requires `>=3.1.0` in `requirements/ml.in`
- This should be a **separate, focused PR** (see action items below)

**Referenced by:** `docs/pr_archive/architecture/pr-review-845-883-superseded.md` (commit `c0b9a96e`)

---

#### Branch: `RC219805-patch-4` (PR #845)
- **Status:** Draft since 2026-02-05
- **Drift:** 110 commits behind main (historical estimate), 5 commits ahead
- **Last update:** 10 days ago
- **Verdict:** **SUPERSEDED — CLOSE**

**Rationale:**
- Proposed refactor of performance regression tests with non-existent CLI flags
- 28 review comments identified critical issues (phantom features, incorrect assertions)
- **Superseded by:**
  - PR #882: APEX performance governance framework (merged 2026-02-10)
  - PR #909: Cold-start vs steady-state benchmark split (merged 2026-02-12)
  - PR #907: APEX Ultra Phase 1.1 contract integrity (merged 2026-02-11)
- Merging would **regress** test coverage by removing Phase 1-3 optimization tests

**Coverage regression risk:**
- Current `test_performance_regression.py` on main tests actual code paths (manifest caching, SHA-256, parallel processing, depth cache, PBR batching)
- PR #845 replaces these with subprocess-only tests of `tools/performance_ledger.py`

**Referenced by:** `docs/pr_archive/architecture/pr-review-845-883-superseded.md` (commit `c0b9a96e`)

---

### 2. Other Remote Branches (No Associated PRs)

The following remote branches exist but have no open PRs visible on GitHub:

| Branch | Last Updated | Status |
|--------|--------------|--------|
| `origin/bugfix/materials-v3-critical-fixes` | 2 days ago | Likely merged or abandoned |
| `origin/copilot/restore-ci-health-and-stability` | 13 days ago | Likely merged or superseded |
| `origin/copilot/review-prs-relevance` | Unknown | Purpose unclear |
| `origin/copilot/sub-pr-941` | Unknown | Likely ephemeral sub-PR branch |
| `origin/feature/phase4-ml-upscaling` | Unknown | Check if merged to main |
| `origin/feature/phase5-stagegraph-integration` | Unknown | Check if merged to main |
| `origin/feature/quick-wins-batch-1` | Unknown | Check if merged to main |
| `origin/feature/spatial-ai-linear-ingest-phase1` | Unknown | Check if merged to main |
| `origin/materials-v3/phase-c2-confidence-semantics` | Unknown | Likely merged via PR #947 |

**Action required:** Cross-reference these with recent merged PRs and delete stale branches.

---

### 3. Local Branch Hygiene

#### Branches Tracking Deleted Remotes (`[gone]`)

These local branches track remotes that no longer exist:

1. `docs/materials-v3-investigations` — 27 hours ago
2. `docs/phase3-l1-cache-invariants` — 3 days ago
3. `feat/apex-research-ultra-phase1` — 4 days ago
4. `feat/spatial-ai-foundation` — 5 days ago
5. `feature/materials-v3-production-integration` — 2 days ago
6. `feature/materials-v3-water-sky-synthesis` — 30 hours ago
7. `fix/apex-governance-blockers` — 6 days ago
8. `fix/ci-gate-preflight-classifier` — 3 days ago
9. `fix/depth-manifest-resolved-backend-v2` — 28 hours ago
10. `test/da3-availability-guards` — 9 days ago

**Recommendation:** Delete all `[gone]` branches after confirming their work is on main.

```bash
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
```

If any branches contain unmerged work, use `git branch -D` (force delete) after manual review.

---

#### Local Branches Without Remote Tracking

These local branches have no upstream and may contain uncommitted work:

| Branch | Last Updated | Risk Assessment |
|--------|--------------|-----------------|
| `RC219805-patch-4` | 10 days ago | Safe to delete (superseded PR #845) |
| `bugfix/materials-v3-critical-fixes` | 2 days ago | Check for unmerged fixes |
| `copilot/fix-coverage-artifact-upload` | 12 days ago | Likely ephemeral, safe to delete |
| `feat/apex-phase2-real-pipeline-integration` | 7 days ago | Check against merged PRs |
| `fix/depth-identity-resolved-backend` | 29 hours ago | Recent, may contain unmerged work |
| `fix/depth-identity-resolved-backend-clean` | 29 hours ago | Duplicate/variant of above |
| `pr-785-review` | 2 weeks ago | Review branch, safe to delete |
| `pr-803` | 12 days ago | Review branch, safe to delete |
| `pr-841` | 11 days ago | Review branch, safe to delete |
| `pr-851` | 9 days ago | Review branch, safe to delete |
| `pr-871` | 7 days ago | Review branch, safe to delete |
| `pr-906-review` | 4 days ago | Review branch, safe to delete |
| `precision-fix` | 8 days ago | Check for unmerged precision fixes |
| `temp/pr934-upscaling` | 2 days ago | Temporary branch, check for unmerged work |

**Action required:** For each branch:
1. `git log main..branch-name` — Check for commits not on main
2. If empty, delete with `git branch -d branch-name`
3. If non-empty, assess whether to merge or abandon

---

## Governance Analysis

### Architectural Invariants Violated

#### 1. Artifact Hygiene (PR #883)
- **Violation:** `safety-report.json` committed to version control
- **Policy:** CI artifacts must not be tracked in repository
- **Remediation:** Already addressed in PR #940 — added `*-report.json` to `.gitignore`
- **Status:** ✅ Resolved on main

#### 2. Branch Lifetime Management
- **Observation:** No formal policy exists for branch retention after PR merge
- **Risk:** Accumulation of stale remote branches clutters repository navigation
- **Recommendation:** Establish post-merge branch deletion policy (see below)

#### 3. Dependency Security Posture
- **Violation:** `sentence-transformers` CVE-73169 identified but not remediated
- **Policy:** Security vulnerabilities must be addressed within 30 days of disclosure
- **Status:** ⚠️ **BLOCKING ISSUE** — Requires immediate attention

---

### Missing Governance Controls

#### Branch Retention Policy (Recommended)

**Proposal:** Add to `docs/governance/branch_management.md`:

```markdown
## Branch Lifecycle

### Automatic Deletion Triggers
- ✅ PR merged to main → delete branch within 24 hours
- ✅ PR closed without merge → delete branch within 7 days (unless explicitly preserved)
- ✅ Branch superseded by later work → delete after documenting supersession

### Preservation Exceptions
- Long-running feature branches (e.g., `feat/spatial-ai-*`) may be retained if:
  - Documented in `docs/architecture/decisions/` as active work
  - Referenced in DELIVERABLES.md or roadmap
  - Updated within 30 days

### Quarterly Hygiene Audit
- Review all branches older than 60 days
- Cross-reference with merged PRs and active roadmap
- Delete stale branches or document retention justification
```

#### GitHub Branch Protection Enhancement

Current CI Gate pattern (documented in `docs/operations/branch_protection_setup.md`) is robust, but consider:

**Recommended addition:**
- Require `security-scan` job in CI Gate once SBOM generation is stable
- Enforce `dependabot` auto-approval workflow for patch-level updates

---

## Action Items

### Priority 1 (Immediate — Security)

1. **Remediate `sentence-transformers` CVE-73169**
   - Update `requirements/ml.in`: `sentence-transformers>=3.1.0,<6`
   - Recompile lockfiles: `cd requirements && make compile`
   - Validate RAG system compatibility (sentence-transformers is optional dep)
   - Create focused PR with security label
   - **Assignee:** Repository maintainer
   - **Deadline:** Within 7 days

### Priority 2 (High — Branch Cleanup)

2. **Close PR #845 (`RC219805-patch-4`)**
   - Navigate to https://github.com/RC-88/Transformation_Portal/pull/845
   - Add closing comment referencing `pr-review-845-883-superseded.md`
   - Close PR without merge
   - Delete `origin/RC219805-patch-4` branch

3. **Close PR #883 (`automated/dependency-updates`)**
   - Navigate to https://github.com/RC-88/Transformation_Portal/pull/883
   - Add closing comment referencing `pr-review-845-883-superseded.md`
   - Close PR without merge
   - Delete `origin/automated/dependency-updates` branch

### Priority 3 (Medium — Hygiene)

4. **Delete local branches tracking removed remotes**
   ```bash
   git fetch --prune  # Update remote-tracking refs
   git branch -vv | grep ': gone]' | awk '{print $1}' | xargs git branch -d
   ```

5. **Audit and delete stale remote branches**
   - Review branches in section 2 (Other Remote Branches)
   - Cross-reference with merged PRs in git log
   - Delete branches with merged or superseded work
   - Document any branches intentionally preserved

6. **Clean up local review branches**
   ```bash
   git branch -d pr-785-review pr-803 pr-841 pr-851 pr-871 pr-906-review
   ```

### Priority 4 (Low — Governance)

7. **Create branch retention policy**
   - Document in `docs/governance/branch_management.md`
   - Reference CI Gate pattern and merge requirements
   - Establish quarterly hygiene audit schedule

8. **Update CONTRIBUTING.md**
   - Add section on branch naming conventions
   - Document expected branch lifecycle
   - Reference branch retention policy

---

## Verification Checklist

After completing action items:

- [ ] PR #845 closed and branch deleted
- [ ] PR #883 closed and branch deleted
- [ ] `sentence-transformers` updated to >=3.1.0 in new PR
- [ ] Local `[gone]` branches deleted (10 branches)
- [ ] Stale remote branches audited and cleaned
- [ ] Local review branches deleted (6 branches)
- [ ] Branch retention policy documented
- [ ] CONTRIBUTING.md updated

---

## Historical Context

### Recent Major Merges
- **PR #947** (2026-02-16): Materials V3 Phase C.2 — Confidence Semantics
- **PR #944** (2026-02-14): Phase 5 StageGraph Integration
- **PR #943** (2026-02-14): Phase 4 ML Upscaling Backend (4 bugs fixed)
- **PR #940** (2026-02-14): PR #845/#883 supersession + .gitignore fix
- **PR #909** (2026-02-12): Benchmark semantics split
- **PR #882** (2026-02-10): APEX performance governance framework
- **PR #890** (2026-02-11): Spatial AI Linear Ingest Pipeline (Phase I)

### Branch Protection Evolution
- **CI Gate pattern** introduced February 2026 to prevent matrix-change breakage
- Python 3.10 dropped (Nov 2024) exposed fragility of matrix-expanded check names
- Current pattern: Single `CI Gate` aggregator stable across matrix changes

---

## Related Documentation

- `docs/pr_archive/architecture/pr-review-845-883-superseded.md` — Detailed supersession analysis
- `docs/operations/branch_protection_setup.md` — CI Gate pattern documentation
- `docs/architecture/agent_governance.md` — Escalation and decision authority
- `CONTRIBUTING.md` — Contribution workflow and PR guidelines
- `.gitignore` — Artifact hygiene enforcement

---

## Architect Decision

### Verdict
Both PR #845 and PR #883 are **superseded by subsequent work** and should be **closed without merge**.

### Enforcement Required
- **Security:** `sentence-transformers` CVE-73169 must be remediated within 7 days
- **Hygiene:** Branch retention policy must be established to prevent recurrence

### Delegation
- **Specialist:** May implement `sentence-transformers` update (mechanical dependency bump)
- **Architect approval required:** Branch retention policy (governance document)

### No Escalation Needed
This assessment represents final architectural decision within Architect authority scope (dependency governance, security posture, repository health).

---

**Assessment completed:** 2026-02-16
**Next review:** Quarterly branch hygiene audit (Q2 2026)
