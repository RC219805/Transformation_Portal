# GitHub Actions Workflow Inventory

**Purpose**: Track all workflows, their purposes, health, and optimization opportunities
**Last Updated**: 2026-01-02
**Total Workflows**: 24 active

---

## 🟢 Core CI/CD Workflows (Critical Path)

### 1. `ci-consolidated.yml` - Primary CI/CD Pipeline
- **Purpose**: Main build, test, and validation pipeline
- **Triggers**: Push to main/develop, PRs to main, workflow_dispatch
- **Status**: ✅ Healthy (66.67% success, 458s avg duration)
- **Optimization**: Cache already implemented, consider parallelization
- **Priority**: HIGH - This is the critical path

### 2. `security-gates.yml` - Security Validation
- **Purpose**: Security checks, CVE validation, dependency scanning
- **Triggers**: Push, PR
- **Status**: ✅ Healthy (100% success, 53s avg)
- **Optimization**: Excellent - no action needed
- **Priority**: HIGH - Security critical

### 3. `codeql.yml` - Code Security Analysis
- **Purpose**: GitHub Advanced Security scanning
- **Triggers**: Push, PR, schedule (weekly)
- **Status**: ✅ Healthy (100% success, 166s avg)
- **Optimization**: Scheduled runs prevent PR bottlenecks
- **Priority**: HIGH - Security critical

---

## 🟡 Quality Assurance Workflows

### 4. `quality-gate.yml` - Pre-commit Checks
- **Purpose**: Run pre-commit hooks on changed files
- **Triggers**: PR to main, push to main
- **Status**: ✅ Healthy (100% success, 59s avg)
- **Optimization**: Pre-commit cache implemented
- **Priority**: MEDIUM

### 5. `quality-gate-golden.yml` - Golden Image Validation
- **Purpose**: Weekly regression testing against golden images
- **Triggers**: workflow_dispatch, schedule (Mon 07:00 UTC)
- **Status**: ✅ Manual/scheduled only
- **Optimization**: N/A - by design
- **Priority**: LOW - Quality assurance only

### 6. `architecture-hardening.yml` - Architecture Guardrails
- **Purpose**: Validate architectural patterns, module boundaries
- **Triggers**: Push, PR
- **Status**: ✅ Healthy (100% success, 74s avg)
- **Optimization**: Fast, no action needed
- **Priority**: MEDIUM

---

## 🔵 Specialized Testing Workflows

### 7. `materialsv3_tests.yml` - Materials V3 Testing
- **Purpose**: Specialized tests for materials detection system
- **Triggers**: Push, PR affecting materials_v3 code
- **Status**: ✅ Healthy (100% success, 167s avg)
- **Optimization**: Consider merging into ci-consolidated.yml
- **Priority**: MEDIUM

### 8. `depth_quality.yml` - Depth Pipeline Smoke Tests
- **Purpose**: Validate depth processing pipeline
- **Triggers**: Push, PR affecting depth code
- **Status**: ✅ Healthy (100% success, 61s avg)
- **Optimization**: Consider merging into ci-consolidated.yml
- **Priority**: MEDIUM

### 9. `observability-smoke.yml` - Observability Checks
- **Purpose**: Validate monitoring, logging, metrics
- **Triggers**: Push, PR
- **Status**: ✅ Healthy (100% success, 46s avg)
- **Optimization**: Fast, no action needed
- **Priority**: LOW

### 10. `performance-monitor.yml` - Performance Regression
- **Purpose**: Track performance metrics over time
- **Triggers**: Push, PR
- **Status**: ✅ Healthy (100% success, 65s avg)
- **Optimization**: Consider sampling (not every commit)
- **Priority**: MEDIUM

---

## 🟠 Experimental/Optional Workflows

### 11. `experimental-boundary.yml` - Experimental Feature Guard
- **Purpose**: Prevent experimental code from leaking to production
- **Triggers**: Push, PR
- **Status**: ✅ Healthy (100% success, 45s avg)
- **Optimization**: Fast, no action needed
- **Priority**: LOW

### 12. `feature-freeze-check.yml` - Feature Freeze Enforcement
- **Purpose**: Block changes during feature freeze periods
- **Triggers**: PR
- **Status**: ✅ Healthy (context-dependent)
- **Optimization**: N/A - policy enforcement
- **Priority**: LOW

---

## 🤖 AI-Powered Workflows (Require OPENAI_API_KEY)

### 13. `ai-code-review.yml` - AI Code Review (GPT-4o)
- **Purpose**: Automated code review with AI analysis
- **Triggers**: PR to main/develop
- **Status**: ⚠️ **FIXED** (was 0%, now skips gracefully)
- **Optimization**: Implemented check-api-key job
- **Priority**: LOW - Optional enhancement
- **Note**: Requires OPENAI_API_KEY secret

### 14. `smart-issue-management.yml` - AI Issue Triage
- **Purpose**: Automated issue classification and labeling
- **Triggers**: Issue/PR opened, labeled
- **Status**: ⚠️ **FIXED** (was 0%, now skips gracefully)
- **Optimization**: Implemented check-api-key job
- **Priority**: LOW - Optional enhancement
- **Note**: Requires OPENAI_API_KEY secret

### 15. `summary.yml` - AI Issue Summarization
- **Purpose**: Generate AI summaries of issues/PRs
- **Triggers**: Issue comment, PR, PR review
- **Status**: ⚠️ **FIXED** (was 0%, now skips gracefully)
- **Optimization**: Implemented check-api-key job
- **Priority**: LOW - Optional enhancement
- **Note**: Requires OPENAI_API_KEY secret

---

## 📦 Deployment & Publishing Workflows

### 16. `submit-pypi.yml` - PyPI Publishing
- **Purpose**: Publish releases to PyPI via Trusted Publishing
- **Triggers**: Push tags matching v*.*.*
- **Status**: ⚠️ 25% success (3 failures, 1 success)
- **Optimization**: Review failure logs, improve error handling
- **Priority**: HIGH - Production deployment
- **Note**: Uses OIDC Trusted Publishing (no API key needed)

### 17. `dependency-submission.yml` - Dependency Graph
- **Purpose**: Submit dependency graph to GitHub
- **Triggers**: Push to main
- **Status**: ✅ Healthy (100% success, 119s avg)
- **Optimization**: No action needed
- **Priority**: LOW - GitHub feature support

---

## 📊 Monitoring & Reporting Workflows

### 18. `trend-dashboard.yml` - Performance Trends
- **Purpose**: Generate performance trend visualizations
- **Triggers**: Schedule, workflow_dispatch
- **Status**: ✅ Healthy (limited runs by design)
- **Optimization**: N/A - scheduled reporting
- **Priority**: LOW

### 19. `issue_printer.yml` - Issue Reporting
- **Purpose**: Generate issue summary reports
- **Triggers**: Issues workflow
- **Status**: ✅ Used (27 runs)
- **Optimization**: No action needed
- **Priority**: LOW

### 20. `pr-context.yml` - PR Context Analysis
- **Purpose**: Add context metadata to PRs
- **Triggers**: PR events
- **Status**: ✅ Healthy (100 runs)
- **Optimization**: No action needed
- **Priority**: LOW

---

## 🔧 Maintenance Workflows

### 21. `dependency-update.yml` - Automated Dependency Updates
- **Purpose**: Create PRs for dependency updates
- **Triggers**: Schedule, workflow_dispatch
- **Status**: ✅ Used (9 runs)
- **Optimization**: No action needed
- **Priority**: LOW - Maintenance only

### 22. `security-auto-remediation.yml` - Security Auto-Fix
- **Purpose**: Automatically fix security vulnerabilities
- **Triggers**: Security events, workflow_dispatch
- **Status**: ✅ Available for emergencies
- **Optimization**: N/A - emergency use only
- **Priority**: LOW

### 23. `security-scan.yml` - Periodic Security Scan
- **Purpose**: Deep security scanning (bandit, safety)
- **Triggers**: Schedule, workflow_dispatch
- **Status**: ✅ Healthy (100% success, 177s)
- **Optimization**: No action needed
- **Priority**: MEDIUM

---

## 📚 Documentation Workflows

### 24. `pages-docs.yml` - Documentation Deployment
- **Purpose**: Build and deploy docs to GitHub Pages
- **Triggers**: Push to main
- **Status**: ✅ (if exists)
- **Optimization**: Check if actually needed
- **Priority**: LOW

---

## 📋 Optimization Opportunities

### High Priority (Do This Week)
1. ✅ **Fix AI workflows** - COMPLETE (3 workflows fixed)
2. ❌ **Investigate PyPI failures** - 25% success rate needs fixing
3. ❌ **Consolidate test workflows** - Merge materialsv3_tests and depth_quality into ci-consolidated

### Medium Priority (Do This Month)
4. ❌ **Test parallelization** - Use pytest-xdist in ci-consolidated
5. ❌ **Sampling for performance-monitor** - Don't run on every commit
6. ❌ **Job dependency optimization** - Maximize parallel execution

### Low Priority (Future)
7. ❌ **Workflow cleanup** - Archive truly unused workflows
8. ❌ **Dashboard creation** - Unified CI health dashboard
9. ❌ **Alert thresholds** - Automated alerts for workflow failures

---

## 🎯 Success Metrics

| Category | Current | Target | Status |
|----------|---------|--------|--------|
| **Overall Success Rate** | 68% | 95%+ | 🟡 Improving |
| **Critical Path Duration** | 458s | <300s | 🟡 Needs work |
| **Zero-Failure Workflows** | 11/17 | 15/17 | 🟢 Good |
| **Cache Hit Rate** | 60%+ | 80%+ | 🟡 Needs measurement |

---

## 🔄 Review Schedule

- **Daily**: Monitor workflow health via `scripts/workflow_health_check.py`
- **Weekly**: Review optimization progress
- **Monthly**: Update this inventory with new workflows/changes

---

## 📖 Related Documentation

- [CI/CD Monitoring Guide](CI_CD_MONITORING.md)
- [CI Optimization Plan](CI_OPTIMIZATION_PLAN.md)
- [Workflow Health Check Script](../scripts/workflow_health_check.py)

---

**Maintainer Notes**:
- All workflows use latest action versions (@v6, @v5, @v4)
- Security workflows are critical - do not disable
- AI workflows are optional and fail gracefully
- Cache is implemented but could be more aggressive
