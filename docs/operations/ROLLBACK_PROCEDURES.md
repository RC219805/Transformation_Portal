# Rollback Procedures

**Version:** 1.0.0
**Last Updated:** 2025-01-28
**Authority:** Transformation Portal Architect
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Git Rollback Procedures](#git-rollback-procedures)
3. [PyPI Package Rollback](#pypi-package-rollback)
4. [Production Deployment Rollback](#production-deployment-rollback)
5. [Quality Firewall Rollback](#quality-firewall-rollback)
6. [Communication Templates](#communication-templates)
7. [Post-Rollback Checklist](#post-rollback-checklist)

---

## 1. Overview

### Purpose

This document provides **production-ready rollback procedures** for the Transformation Portal repository. Rollback is a critical operational capability that ensures system stability and user trust when issues are detected in production.

### When to Rollback vs Forward Fix

**Use Rollback When:**
- **Critical security vulnerability** discovered in recent release
- **Data corruption or loss** affecting users
- **Complete feature failure** blocking core functionality
- **Performance regression** > 50% degradation
- **Compatibility break** affecting majority of users
- **Compliance violation** requiring immediate remediation
- **Cannot forward-fix within 2 hours** of detection

**Use Forward Fix When:**
- Issue is **minor** (cosmetic, edge case, non-blocking)
- Fix is **simple and well-tested** (< 30 minutes to implement)
- Rollback would **introduce other issues** (data migration complexity)
- Issue affects **< 5% of users** or specific configuration
- **Testing rollback** would take longer than forward fix
- Forward fix **prevents regression** better than rollback

**Decision Authority:**
- **Critical Issues (Security, Data Loss):** Architect approval required for rollback
- **Major Issues (Feature Failure, Performance):** Lead Developer can approve rollback
- **Minor Issues:** Team consensus or forward fix preferred

### Rollback Principles

1. **Preserve History:** Never force-push to protected branches. Use `git revert`.
2. **Communicate Early:** Notify stakeholders before executing rollback.
3. **Test Rollback:** Verify rollback in staging/test environment when possible.
4. **Document Rationale:** Always document why rollback was chosen over forward fix.
5. **Post-Mortem:** Every rollback requires a post-mortem to prevent recurrence.

---

## 2. Git Rollback Procedures

### 2.1 Revert Single Commit

**Scenario:** A single commit introduced a bug and needs to be undone.

**Procedure:**

```bash
# 1. Identify the problematic commit
git log --oneline -20

# 2. Verify the commit contents
git show <commit-sha>

# 3. Create a revert commit
git revert <commit-sha>

# 4. Edit the commit message to explain the rollback
# Git will open an editor with a default message.
# Add context:
#   - What issue required the revert
#   - Link to issue/ticket
#   - Impact assessment

# 5. Push the revert
git push origin main

# 6. Verify CI passes on reverted state
gh run list --branch main --limit 1
```

**Example Commit Message:**

```
Revert "Add experimental depth fusion algorithm"

This reverts commit abc12345.

Reason: Memory leak detected in depth fusion causing OOM errors in
production for images > 8K resolution.

Impact: Affects ~15% of enterprise users processing ultra-high-res imagery.
Rollback restores stable depth processing using ADR-019 backends.

Issue: #1234
Post-Mortem: To be completed by 2025-01-30
```

### 2.2 Revert Multiple Commits

**Scenario:** A series of related commits need to be reverted (e.g., a feature branch merged via merge commit).

**Procedure:**

```bash
# 1. Identify the range of commits to revert
# Example: Revert commits between abc123 (oldest) and def456 (newest)
git log --oneline --graph --all

# 2. Revert in reverse chronological order (newest first)
# This preserves logical coherence and reduces conflicts
git revert --no-commit def456
git revert --no-commit cde345
git revert --no-commit abc123

# 3. Create a single revert commit
git commit -m "Revert experimental feature X (commits abc123..def456)

Reason: [Detailed explanation]
Impact: [User impact]
Issue: #XXXX
"

# 4. Push the revert
git push origin main
```

**Alternative: Revert a Range (Use with Caution)**

```bash
# Revert all commits from abc123 to def456 (inclusive)
# WARNING: This can create complex conflicts if range is large
git revert --no-commit abc123^..def456
git commit -m "Revert feature X (range abc123..def456)"
git push origin main
```

### 2.3 Revert Merged Pull Request

**Scenario:** A pull request was merged to `main` but needs to be reverted.

**Procedure:**

```bash
# 1. Find the merge commit SHA
git log --oneline --merges -10
# Look for: "Merge pull request #123 from feature-branch"

# 2. Revert the merge commit using -m 1 flag
# -m 1 specifies we want to keep the mainline (main branch) history
git revert -m 1 <merge-commit-sha>

# 3. Edit commit message
# Explain why PR is being reverted
git commit --amend

# 4. Push revert
git push origin main
```

**Important Notes:**
- Reverting a merge commit with `-m 1` keeps the `main` branch as the "mainline"
- If you later want to re-merge the original branch, you must **revert the revert** first
- Document in PR comments: "This PR was reverted in commit XYZ due to [reason]"

**Example:**

```bash
# Revert PR #906 merge
git revert -m 1 9af004ee
# Edit message to explain rollback
git push origin main

# Later, if you want to re-introduce the feature:
# 1. Revert the revert
git revert <revert-commit-sha>
# 2. Fix the original issue
# 3. Create new PR with fixes
```

### 2.4 Handle Conflicts During Revert

**Scenario:** `git revert` encounters merge conflicts.

**Procedure:**

```bash
# 1. Attempt revert
git revert <commit-sha>
# Git reports conflicts

# 2. View conflicted files
git status

# 3. Resolve conflicts manually
# Edit each conflicted file, remove conflict markers
# Choose whether to keep changes, discard them, or hybrid

# 4. Mark conflicts as resolved
git add <resolved-file>

# 5. Continue revert
git revert --continue

# 6. Verify state
git diff HEAD~1 HEAD
pytest tests/ -ra -m "not ml and not slow"

# 7. Push
git push origin main
```

**Conflict Resolution Principles:**
- **When in doubt, favor the pre-change state** (more conservative)
- **Test thoroughly** after resolving conflicts
- **Document resolution strategy** in commit message
- **Escalate to Architect** if conflicts are complex or risk is high

---

## 3. PyPI Package Rollback

### 3.1 Yank a Bad Release

**Scenario:** A PyPI release contains a critical bug and must be removed from default installation.

**Procedure:**

```bash
# 1. Identify the problematic version
# Example: transformation-portal==0.1.5 is broken

# 2. Yank the release using twine
pip install --upgrade twine
twine upload --repository pypi --yank \
    "Critical bug: see issue #XXXX" \
    dist/transformation-portal-0.1.5*

# Alternative: Use PyPI web UI
# - Log into PyPI
# - Navigate to release page
# - Click "Options" → "Yank release"
# - Provide reason (will be visible to users)
```

**What "Yank" Does:**
- Release **remains in PyPI** but is **not installed by default**
- Users with pinned versions (`transformation-portal==0.1.5`) can still install it
- `pip install transformation-portal` will **skip** yanked version
- Existing installations are **not affected**
- Yank reason is visible on PyPI page

**What "Yank" Does NOT Do:**
- Does **not delete** the release (deletion is rarely allowed on PyPI)
- Does **not force** users to upgrade
- Does **not break** existing environments with the yanked version

### 3.2 Communicate to Users

**Immediately After Yanking:**

1. **Update PyPI Release Page:**
   - Edit release description to add **prominent warning**
   - Link to issue tracker and fixed version

2. **Post Security Advisory (if security issue):**
   ```bash
   # Create GitHub Security Advisory
   gh api repos/OWNER/REPO/security-advisories \
       -F summary="Critical vulnerability in v0.1.5" \
       -F description="..." \
       -F severity="high" \
       -F cwe_ids[]=CWE-XXX
   ```

3. **Notify via Issue Tracker:**
   - Open public issue: "Critical Issue in v0.1.5 - Please Upgrade to v0.1.6"
   - Pin the issue to repository

4. **Email Notification (if enterprise users):**
   - Use [Communication Template 2](#template-2-user-facing-package-recall)

### 3.3 Re-Release Corrected Version

**Procedure:**

```bash
# 1. Fix the issue in code
git checkout main
git pull origin main

# 2. Create fix branch
git checkout -b fix/critical-issue-XXXX

# 3. Implement and test fix
# ... code changes ...
pytest tests/ --maxfail=1
pytest tests/integration/ --maxfail=1

# 4. Update version number (bump patch or minor)
# Edit pyproject.toml:
# OLD: version = "0.1.5"
# NEW: version = "0.1.6"

# 5. Update CHANGELOG.md
# Document the fix and reference yanked version

# 6. Commit and push
git add -A
git commit -m "Fix critical issue from v0.1.5

- [Detailed fix description]
- Yanked v0.1.5 from PyPI
- Bumped to v0.1.6

Issue: #XXXX
Yanked Version: 0.1.5
"
git push origin fix/critical-issue-XXXX

# 7. Create PR, get approval, merge

# 8. Tag and release new version
git checkout main
git pull origin main
git tag -a v0.1.6 -m "v0.1.6: Fix critical issue from v0.1.5"
git push origin v0.1.6

# 9. Build and publish to PyPI
python -m build
twine upload dist/transformation-portal-0.1.6*

# 10. Verify release
pip install --upgrade transformation-portal
python -c "import transformation_portal; print(transformation_portal.__version__)"
# Should output: 0.1.6

# 11. Update GitHub Release page with detailed notes
gh release create v0.1.6 \
    --title "v0.1.6: Critical Fix for v0.1.5" \
    --notes "This release fixes [issue] introduced in v0.1.5 (yanked).

All users on v0.1.5 should upgrade immediately.

Changes:
- [Fix detail]

See CHANGELOG.md for full details."
```

**Version Numbering Strategy:**
- **Patch bump (0.1.5 → 0.1.6):** Bug fix, no API changes
- **Minor bump (0.1.5 → 0.2.0):** Bug fix + new features or API changes
- **Major bump (0.1.5 → 1.0.0):** Breaking changes (avoid during rollback)

---

## 4. Production Deployment Rollback

### 4.1 Deployment Rollback Strategies

**Note:** This repository is primarily a library/CLI tool distributed via PyPI. If you have production deployments (e.g., containerized services, Lambda functions, cloud pipelines), apply these strategies.

**Strategy 1: Blue-Green Deployment Rollback**

```bash
# Scenario: You have two environments (blue=old, green=new)
# Green deployment fails, rollback to blue

# 1. Verify blue environment health
curl https://blue.example.com/health
# Expected: 200 OK

# 2. Switch traffic back to blue
# (Method depends on your infrastructure: load balancer, DNS, etc.)

# AWS ALB example:
aws elbv2 modify-listener \
    --listener-arn <listener-arn> \
    --default-actions Type=forward,TargetGroupArn=<blue-target-group-arn>

# Verify traffic switch
curl https://production.example.com/health
# Should route to blue environment

# 3. Keep green running for investigation
# Do not destroy green environment until root cause is understood
```

**Strategy 2: Container Image Rollback**

```bash
# Scenario: Docker/Kubernetes deployment with tagged images

# 1. Identify last known good image tag
docker images | grep transformation-portal
# Example: transformation-portal:v0.1.4 (good), v0.1.5 (bad)

# 2. Rollback Kubernetes deployment
kubectl set image deployment/transformation-portal \
    transformation-portal=transformation-portal:v0.1.4

# 3. Monitor rollout
kubectl rollout status deployment/transformation-portal

# 4. Verify health
kubectl get pods -l app=transformation-portal
kubectl logs -l app=transformation-portal --tail=50

# Alternative: Use kubectl rollout undo
kubectl rollout undo deployment/transformation-portal
# This rolls back to previous revision automatically
```

**Strategy 3: Lambda Function Rollback (AWS)**

```bash
# Scenario: AWS Lambda function using transformation-portal as layer

# 1. List function versions
aws lambda list-versions-by-function \
    --function-name transformation-portal-processor

# 2. Identify last known good version
# Example: Version 12 is current (broken), Version 11 is good

# 3. Update alias to point to good version
aws lambda update-alias \
    --function-name transformation-portal-processor \
    --name PROD \
    --function-version 11

# 4. Verify
aws lambda get-alias \
    --function-name transformation-portal-processor \
    --name PROD
# Should show FunctionVersion: 11
```

### 4.2 Data Migration Considerations

**Critical: Rollbacks with Schema Changes**

If the rolled-back version has **database schema changes** or **data format changes**:

1. **Backup Current State:**
   ```bash
   # Example: Backup S3 bucket before rollback
   aws s3 sync s3://prod-outputs s3://prod-outputs-backup-20250128
   ```

2. **Check Migration Reversibility:**
   - Can schema/data be downgraded without loss?
   - Are there backward-incompatible changes?

3. **Options:**
   - **Forward Fix:** If data migration is irreversible, prefer forward fix
   - **Restore from Backup:** Rollback code + restore pre-migration data
   - **Dual-Version Support:** Keep both versions running temporarily

**Example: Rollback with Data Restore**

```bash
# Scenario: v0.1.5 changed output metadata format (incompatible with v0.1.4)

# 1. Stop production workload
# (Kubernetes example)
kubectl scale deployment/transformation-portal --replicas=0

# 2. Restore data to pre-migration state
aws s3 sync s3://prod-outputs-backup-20250127 s3://prod-outputs --delete

# 3. Rollback application code
kubectl set image deployment/transformation-portal \
    transformation-portal=transformation-portal:v0.1.4

# 4. Restart workload
kubectl scale deployment/transformation-portal --replicas=3

# 5. Verify data compatibility
# Run smoke tests to confirm v0.1.4 can process restored data
```

### 4.3 State Restoration

**For Stateful Systems:**

1. **Identify State Components:**
   - Databases (PostgreSQL, DynamoDB, etc.)
   - File systems (S3, EFS, local volumes)
   - Caches (Redis, Memcached)
   - Message queues (SQS, RabbitMQ)

2. **Snapshot Before Rollback:**
   ```bash
   # Database snapshot
   aws rds create-db-snapshot \
       --db-instance-identifier prod-db \
       --db-snapshot-identifier prod-db-pre-rollback-20250128

   # S3 versioning (if enabled)
   aws s3api list-object-versions \
       --bucket prod-outputs \
       --prefix snapshots/
   ```

3. **Restore State:**
   ```bash
   # Restore from RDS snapshot
   aws rds restore-db-instance-from-db-snapshot \
       --db-instance-identifier prod-db-restored \
       --db-snapshot-identifier prod-db-pre-rollback-20250128

   # Point application to restored DB
   # Update connection strings, DNS, etc.
   ```

---

## 5. Quality Firewall Rollback

### 5.1 Temporarily Adjust Thresholds (Emergency)

**Scenario:** Quality firewall gates are failing CI due to environmental issues (not code quality), blocking urgent hotfix deployment.

**Authority Required:** Architect approval mandatory

**Procedure:**

```bash
# 1. Document emergency justification
# Create issue: "Emergency Quality Threshold Adjustment - Hotfix #XXXX"

# 2. Identify failing gate
# Example: Performance regression gate failing due to CI resource contention

# 3. Create emergency bypass branch
git checkout -b emergency/quality-bypass-XXXX

# 4. Adjust threshold in quality firewall config
# Example: .github/workflows/quality_firewall.yml
# OLD: PERF_THRESHOLD=5.0
# NEW: PERF_THRESHOLD=10.0  # Temporarily relaxed for emergency deploy

# 5. Add loud warning comment in config
# EMERGENCY BYPASS: Threshold relaxed for hotfix #XXXX
# MUST BE REVERTED by 2025-01-29 (24 hours)
# Approved by: [Architect Name]
# Issue: #XXXX

# 6. Commit with explicit emergency marker
git commit -m "EMERGENCY: Relax quality threshold for hotfix #XXXX

TEMPORARY BYPASS - MUST REVERT WITHIN 24 HOURS

Reason: [Detailed justification]
Approved By: [Architect Name]
Revert By: 2025-01-29 23:59 UTC
Issue: #XXXX
"

# 7. Push and merge with expedited review
git push origin emergency/quality-bypass-XXXX
gh pr create --title "EMERGENCY: Quality Bypass for Hotfix #XXXX" \
    --body "Emergency bypass - see commit message for justification"

# 8. Merge and deploy hotfix

# 9. IMMEDIATELY schedule revert
# Add calendar reminder, create revert issue, set GitHub Action reminder
```

### 5.2 Document Exception Rationale

**Required Documentation:**

1. **In Commit Message:**
   - Clear "EMERGENCY" or "TEMPORARY BYPASS" marker
   - Detailed justification (why bypass is necessary)
   - Architect approval name
   - Explicit revert deadline

2. **In Issue Tracker:**
   - Create issue: "Revert Quality Bypass #XXXX"
   - Assign to person responsible for revert
   - Set due date (max 24 hours)
   - Link to bypass commit

3. **In Post-Mortem:**
   - Why was bypass necessary?
   - Could it have been avoided?
   - Process improvements to prevent future bypasses

### 5.3 Restore Strict Enforcement

**Procedure:**

```bash
# 1. Create revert PR (within 24 hours of bypass)
git checkout main
git pull origin main
git checkout -b revert/quality-bypass-XXXX

# 2. Revert the bypass commit
git revert <bypass-commit-sha>

# 3. Verify strict enforcement is restored
# Check config file: thresholds back to original values
cat .github/workflows/quality_firewall.yml | grep THRESHOLD

# 4. Run full quality firewall locally
# Ensure all gates pass with strict thresholds
pytest tests/ --maxfail=1
./scripts/quality/run_quality_firewall.sh

# 5. Commit and push
git push origin revert/quality-bypass-XXXX

# 6. Create PR
gh pr create --title "Restore strict quality enforcement (revert bypass #XXXX)" \
    --body "Revert emergency bypass from commit <sha>.

All quality gates now passing with strict thresholds.
Issue #XXXX resolved."

# 7. Merge after review

# 8. Close related issue
gh issue close XXXX --comment "Quality bypass reverted, strict enforcement restored."
```

**Monitoring:**
- Set up alerts if quality bypasses are not reverted within 48 hours
- Review bypass frequency quarterly (should be extremely rare)

---

## 6. Communication Templates

### Template 1: Internal Team Notification

**Subject:** [URGENT] Rollback Initiated - [Component Name] [Version]

**Body:**

```
Team,

A rollback has been initiated for [Component/Feature] as of [Timestamp UTC].

ROLLBACK DETAILS:
- Component: [e.g., transformation-portal PyPI package]
- Rolled Back From: [e.g., v0.1.5]
- Rolled Back To: [e.g., v0.1.4]
- Rollback Method: [e.g., Git revert, PyPI yank, container rollback]

REASON:
[Detailed explanation of the issue that required rollback]

IMPACT:
- User Impact: [e.g., 15% of enterprise users processing 8K+ images]
- System Impact: [e.g., Memory usage reduced by 40%]
- Data Impact: [e.g., No data loss, all outputs remain valid]

TIMELINE:
- Issue Detected: [Timestamp]
- Rollback Decision: [Timestamp]
- Rollback Executed: [Timestamp]
- Rollback Verified: [Timestamp]
- Estimated Recovery: [Timestamp]

NEXT STEPS:
1. [e.g., Root cause analysis by 2025-01-30]
2. [e.g., Fix implementation and testing]
3. [e.g., Re-release as v0.1.6]
4. [e.g., Post-mortem scheduled for 2025-01-31]

POINT OF CONTACT:
[Name, Role, Contact Info]

TRACKING:
- Issue: #XXXX
- Rollback Commit: [Git SHA]
- Post-Mortem: [Link to doc/issue]

Thank you for your attention.

[Your Name]
[Your Role]
```

---

### Template 2: User-Facing Package Recall

**Subject:** Important: Security Update for transformation-portal v0.1.5

**Body:**

```
Dear transformation-portal users,

We have identified a critical issue in version 0.1.5 of the transformation-portal package and have yanked it from PyPI to prevent further installations.

WHAT HAPPENED:
[Brief, non-technical explanation of the issue]

AFFECTED VERSIONS:
- transformation-portal v0.1.5 (released [Date])

RECOMMENDED ACTION:
If you are using v0.1.5, please upgrade to v0.1.6 immediately:

pip install --upgrade transformation-portal

If you are on v0.1.4 or earlier, you are not affected but should upgrade for security best practices.

VERIFICATION:
Check your current version:
python -c "import transformation_portal; print(transformation_portal.__version__)"

DETAILS:
- Issue: [Link to GitHub issue]
- Security Advisory: [Link if applicable]
- CHANGELOG: [Link to CHANGELOG.md]

We apologize for any inconvenience. If you have questions or need assistance, please:
- Open an issue: [GitHub Issues URL]
- Email: [Support email if applicable]

Thank you for your understanding.

The Transformation Portal Team
```

---

### Template 3: Post-Mortem Template

**Title:** Post-Mortem: Rollback of [Component] [Version] on [Date]

**Document:**

```markdown
# Post-Mortem: Rollback of [Component] [Version]

**Date of Incident:** [YYYY-MM-DD]
**Date of Post-Mortem:** [YYYY-MM-DD]
**Participants:** [Names and Roles]
**Incident Owner:** [Name]

---

## Executive Summary

[2-3 sentence summary of what happened, impact, and resolution]

---

## Timeline (All times UTC)

| Time       | Event                                                      |
|------------|-------------------------------------------------------------|
| 10:00      | Issue first detected by [Person/System]                     |
| 10:15      | Root cause identified: [Brief description]                  |
| 10:30      | Rollback decision made by [Decision Maker]                  |
| 10:45      | Rollback executed [Method]                                  |
| 11:00      | Rollback verified, system stable                            |
| 11:30      | User notification sent                                      |

---

## Impact Assessment

### User Impact
- **Users Affected:** [Number or percentage]
- **Impact Severity:** [Critical / Major / Minor]
- **Impact Duration:** [Hours/minutes of degraded service]
- **Data Loss:** [Yes/No - details if yes]

### System Impact
- **Services Affected:** [List]
- **Performance Degradation:** [Metrics]
- **Availability:** [Uptime percentage during incident]

---

## Root Cause Analysis

### What Happened
[Detailed technical explanation of the issue]

### Why It Happened
[Underlying causes - code defect, process failure, environmental issue, etc.]

### Why It Wasn't Caught Earlier
[Analysis of testing/review gaps]

---

## What Went Well

- [e.g., Rollback procedure was well-documented and executed quickly]
- [e.g., Team responded promptly and coordinated effectively]
- [e.g., User communication was clear and timely]

---

## What Went Poorly

- [e.g., Issue was not detected in staging environment]
- [e.g., Rollback took longer than expected due to data migration complexity]
- [e.g., Monitoring did not alert on the issue immediately]

---

## Corrective Actions

| Action                                                                 | Owner        | Due Date   | Status      |
|------------------------------------------------------------------------|--------------|------------|-------------|
| Add integration test for [specific scenario]                          | [Name]       | 2025-02-05 | In Progress |
| Improve monitoring for [metric]                                        | [Name]       | 2025-02-10 | Not Started |
| Update deployment checklist to include [step]                         | [Name]       | 2025-02-01 | Complete    |
| Conduct training on [process/tool]                                     | [Name]       | 2025-02-15 | Not Started |

---

## Lessons Learned

1. **[Lesson 1]:** [Description and how to apply in future]
2. **[Lesson 2]:** [Description and how to apply in future]
3. **[Lesson 3]:** [Description and how to apply in future]

---

## References

- Incident Issue: #XXXX
- Rollback Commit: [Git SHA]
- User Notification: [Link]
- Related ADRs: [If applicable]

---

**Sign-off:**
- Incident Owner: [Name, Date]
- Architect Approval: [Name, Date]
```

---

## 7. Post-Rollback Checklist

### 7.1 Verify System Health

**Immediate Checks (within 15 minutes of rollback):**

- [ ] **CI/CD Pipeline:** Verify all workflows pass on rolled-back state
  ```bash
  gh run list --branch main --limit 5
  # All runs should show green checkmark
  ```

- [ ] **Test Suite:** Run full test suite (including integration tests)
  ```bash
  pytest tests/ --maxfail=1
  pytest tests/integration/ --maxfail=1
  ```

- [ ] **Linting and Quality Gates:** Verify code quality standards
  ```bash
  flake8 src/ tests/
  black --check src/ tests/
  mypy src/
  ```

- [ ] **Smoke Tests:** Run critical path smoke tests
  ```bash
  # Example: Test core functionality
  python -c "from transformation_portal.depth.backends import create_backend; print('OK')"
  ```

- [ ] **Performance Baselines:** Verify no performance regressions
  ```bash
  # Run performance ledger comparison
  python -m transformation_portal.performance_ledger.cli compare \
      --baseline-version v0.1.4 \
      --current-run performance_ledger.json \
      --strict
  ```

**Production Health Checks (if deployed):**

- [ ] **Service Availability:** Verify endpoints respond
  ```bash
  curl -f https://production.example.com/health || echo "FAIL"
  ```

- [ ] **Error Rates:** Check logs for increased error rates
  ```bash
  # Example: CloudWatch Logs
  aws logs filter-log-events \
      --log-group-name /aws/lambda/transformation-portal \
      --filter-pattern "ERROR" \
      --start-time $(date -u -d '15 minutes ago' +%s)000
  ```

- [ ] **Performance Metrics:** Verify latency/throughput returned to baseline
  ```bash
  # Check application metrics (method depends on monitoring stack)
  ```

### 7.2 Update Issue Tracking

**Required Updates:**

- [ ] **Mark Original Issue as Resolved by Rollback**
  ```bash
  gh issue comment XXXX --body "Resolved by rollback to v0.1.4. See rollback commit: <sha>"
  gh issue close XXXX
  ```

- [ ] **Create Follow-Up Issue for Proper Fix**
  ```bash
  gh issue create \
      --title "Proper fix for issue #XXXX (rolled back in v0.1.5)" \
      --body "The issue originally addressed in v0.1.5 was rolled back due to [reason].

      This issue tracks the proper fix with adequate testing.

      Context:
      - Original Issue: #XXXX
      - Rollback Commit: <sha>
      - Rollback Reason: [Brief description]

      Requirements:
      - [ ] Fix the original issue
      - [ ] Add regression test
      - [ ] Verify no performance/stability impact
      - [ ] Update documentation
      "
  ```

- [ ] **Create Post-Mortem Issue**
  ```bash
  gh issue create \
      --title "Post-Mortem: Rollback of v0.1.5 on 2025-01-28" \
      --label "post-mortem" \
      --body "Post-mortem required for rollback incident.

      Use template: docs/operations/ROLLBACK_PROCEDURES.md#template-3-post-mortem-template

      Due Date: 2025-01-31
      Owner: [Assign to incident owner]
      "
  ```

### 7.3 Document Root Cause

**Immediate Documentation (within 1 hour):**

- [ ] **Add entry to CHANGELOG.md**
  ```markdown
  ## [0.1.5] - 2025-01-27 [YANKED]

  **This version was yanked from PyPI on 2025-01-28 due to [issue].**

  Users should upgrade to v0.1.6 or downgrade to v0.1.4.

  ### Added
  - [Original feature description]

  ### Known Issues (Yanked)
  - Critical: [Description of issue that caused yank]
  - Impact: [User impact description]
  - Resolution: Fixed in v0.1.6
  ```

- [ ] **Update README.md (if applicable)**
  - Add note about yanked version if it affects installation instructions

- [ ] **Document in Git Commit**
  - Rollback commit message should contain full context (see templates above)

**Detailed Documentation (within 24 hours):**

- [ ] **Complete Post-Mortem** (see Template 3)

- [ ] **Update ADRs if applicable**
  - If rollback reveals flaws in architectural decision, create superseding ADR

- [ ] **Update Testing Strategy**
  - Document gaps that allowed issue to reach production
  - Add to test plan: "Test for [specific scenario] to prevent recurrence"

### 7.4 Prevent Recurrence

**Process Improvements:**

- [ ] **Add Regression Test**
  ```python
  def test_prevent_issue_XXXX_regression():
      """Regression test for issue #XXXX (rolled back in v0.1.5).

      This test ensures the specific failure mode does not recur.
      """
      # Test implementation
      pass
  ```

- [ ] **Enhance CI/CD Gates**
  - Add quality firewall check if issue could have been caught by automated gate
  - Example: Add performance benchmark if rollback was due to performance regression

- [ ] **Update Code Review Checklist**
  - Add item: "Verify [specific aspect] to prevent issue #XXXX recurrence"

- [ ] **Improve Monitoring/Alerting**
  - If issue was detected late, add monitoring for early warning signs

**Knowledge Sharing:**

- [ ] **Team Retrospective**
  - Schedule 30-minute team discussion within 48 hours
  - Review post-mortem findings
  - Discuss process improvements

- [ ] **Update Documentation**
  - If rollback revealed documentation gaps, update docs

- [ ] **Share Lessons Learned**
  - Add note to `docs/architecture/lessons_learned.md` (create if needed)

---

## Appendix A: Rollback Decision Tree

```
┌──────────────────────────┐
│  Issue Detected          │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Assess Severity         │
│  - Data loss?            │
│  - Security vuln?        │
│  - User impact?          │
└───────────┬──────────────┘
            │
            ▼
    ┌───────┴────────┐
    │                │
 Critical?        Minor?
    │                │
    ▼                ▼
┌────────┐      ┌────────────┐
│Rollback│      │Forward Fix │
└────────┘      └────────────┘
    │
    ▼
┌──────────────────────────┐
│ Can forward-fix in < 2h? │
└───────────┬──────────────┘
            │
        ┌───┴───┐
        │       │
       Yes     No
        │       │
        ▼       ▼
  ┌────────┐ ┌────────┐
  │Forward │ │Rollback│
  │  Fix   │ │        │
  └────────┘ └────────┘
```

---

## Appendix B: Emergency Contacts

**Escalation Path:**

1. **Lead Developer:** [Name, Email, Phone]
   - Authority: Approve rollback for major issues
   - Availability: Standard business hours

2. **Transformation Portal Architect:** [Name, Email, Phone]
   - Authority: Approve rollback for critical issues, security, quality bypasses
   - Availability: On-call 24/7 for critical issues

3. **Security Lead:** [Name, Email, Phone] (if applicable)
   - Authority: Approve security-related rollbacks
   - Availability: On-call 24/7 for security incidents

**Communication Channels:**

- **Emergency Slack Channel:** #transformation-portal-incidents
- **Email Distribution List:** transformation-portal-team@example.com
- **On-Call Rotation:** [Link to PagerDuty/OpsGenie schedule]

---

## Version History

| Version | Date       | Changes                                      | Author              |
|---------|------------|----------------------------------------------|---------------------|
| 1.0.0   | 2025-01-28 | Initial rollback procedures documentation    | Architect           |

---

**For Questions or Updates:** Open an issue with label `documentation` or contact the Transformation Portal Architect.
