# Phase 2 RAG System - Operational Deployment Guide

## Executive Summary

This guide provides step-by-step instructions for deploying the four operational components that transform the Phase 2 RAG system from static infrastructure into a living intelligence layer. Each component builds upon the previous, creating a self-reinforcing feedback loop that compounds value with every CI execution.

**The Transformation:**
```
Before: CI runs → Reports → Forgets
After:  CI runs → Ingests → Analyzes → Remembers → Predicts → Informs
```

---

## Prerequisites

Before deploying, ensure:

1. **Phase 2 Infrastructure Deployed**
   ```bash
   # Verify Phase 2 files exist
   ls -la .github/agents/rag_system/git_hooks.py
   ls -la .github/agents/rag_system/knowledge_feedback.py
   ls -la .github/agents/rag_system/dependency_analysis.py
   ```

2. **Python 3.10+ Available**
   ```bash
   python3 --version  # Should be 3.10 or higher
   ```

3. **Repository Write Access**
   - Push access to main/develop branches
   - Workflow write permissions

---

## Action 1: Git Hook Installation

### Purpose
Enable real-time RAG index synchronization on every git operation.

### Deployment Steps

```bash
# Navigate to repository root
cd /path/to/Transformation_Portal

# Make installation script executable
chmod +x phase2_operational_package/1_install_git_hooks.sh

# Preview installation (dry run)
./phase2_operational_package/1_install_git_hooks.sh --dry-run

# Install hooks
./phase2_operational_package/1_install_git_hooks.sh

# Verify installation
python3 .github/agents/rag_system/git_hooks.py status
```

### Verification

```bash
# Make a test commit
echo "# Test" >> /tmp/test.txt
git add /tmp/test.txt
git commit -m "Test commit for hook verification"

# Check that hook executed (look for RAG index update messages)
# Then remove test commit
git reset --hard HEAD~1
```

### Expected Behavior

| Git Operation | Hook Triggered | Action |
|---------------|----------------|--------|
| `git commit` | post-commit | Index updated with committed files |
| `git merge` | post-merge | Index updated with merged changes |
| `git checkout` | post-checkout | Cache validated for new branch |
| `git push` | pre-push | Consistency check before push |

### Rollback

```bash
python3 .github/agents/rag_system/git_hooks.py uninstall
```

---

## Action 2: JUnit XML Export Integration

### Purpose
Enable granular test intelligence through structured test result reporting.

### Deployment Steps

1. **Open the consolidated CI workflow**
   ```bash
   vi .github/workflows/ci-consolidated.yml
   ```

2. **Apply the patches from `2_ci_workflow_patch.yml`**:

   **Patch 1: Add environment variables** (after line ~57)
   ```yaml
   env:
     # ... existing variables ...
     JUNIT_XML_OUTPUT: 'test-results/junit.xml'
     COVERAGE_XML_OUTPUT: 'test-results/coverage.xml'
     KNOWLEDGE_ENGINE_ENABLED: '1'
     RAG_CACHE_DIR: '.rag_cache'
   ```

   **Patch 2: Modify pytest command** (in core-tests job)
   ```yaml
   - name: Run Core Tests with JUnit XML Export
     run: |
       python -m pytest tests/ \
         --ignore=tests/foundation \
         --ignore=tests/perceptual \
         -v --tb=short \
         --junitxml=test-results/junit-${{ matrix.python-version }}.xml \
         --cov=src \
         --cov-report=xml:test-results/coverage-${{ matrix.python-version }}.xml
   ```

   **Patch 3: Add knowledge-ingestion job** (after summary job)
   - Copy the entire `knowledge-ingestion` job from `2_ci_workflow_patch.yml`

3. **Commit and push**
   ```bash
   git add .github/workflows/ci-consolidated.yml
   git commit -m "Enable JUnit XML export for Knowledge Engine ingestion"
   git push
   ```

### Verification

After CI runs:
1. Check workflow logs for "Knowledge Engine Ingestion" job
2. Verify artifact `knowledge-base-state` is created
3. Check `.rag_cache/knowledge/` for updated files

### Expected Outcomes

- Per-test duration tracking (enables performance regression detection)
- Failure stack traces captured (enables pattern recognition)
- Skip reasons recorded (validates ML-gating strategy)
- Coverage data integrated (tracks code quality trends)

---

## Action 3: PR Context Integration

### Purpose
Enrich every pull request with historical test intelligence and failure pattern warnings.

### Deployment Steps

1. **Copy workflow file**
   ```bash
   cp phase2_operational_package/3_pr_context_workflow.yml \
      .github/workflows/pr-context.yml
   ```

2. **Commit and push**
   ```bash
   git add .github/workflows/pr-context.yml
   git commit -m "Add PR context generation workflow"
   git push
   ```

### Verification

1. Open a new pull request
2. Wait for "PR Context Generation" workflow to complete
3. Check PR comments for Knowledge Engine analysis

### Expected PR Comment Structure

```markdown
## 🔍 PR Context (Knowledge Engine)

### 🧪 Test Impact Analysis
**X tests** are historically associated with the changed files:
- `test_module::test_function`
- ...

### ⚠️ Historical Failure Patterns
The following tests have failed previously when similar files were changed:
- **`test_id`**: N historical failure(s)

### 📊 Quality Trends
Test pass rate is improving/declining: X% → Y%

---
*Generated by Phase 2 RAG System Knowledge Engine*
```

---

## Action 4: Trend Dashboard Scheduling

### Purpose
Enable predictive quality monitoring through automated weekly trend analysis.

### Deployment Steps

1. **Copy workflow file**
   ```bash
   cp phase2_operational_package/4_trend_dashboard_cron.yml \
      .github/workflows/trend-dashboard.yml
   ```

2. **Commit and push**
   ```bash
   git add .github/workflows/trend-dashboard.yml
   git commit -m "Add scheduled quality trend dashboard"
   git push
   ```

3. **Trigger initial run** (optional)
   - Go to Actions → Quality Trend Dashboard
   - Click "Run workflow"
   - Set analysis period (default: 30 days)

### Verification

1. Check workflow runs on schedule (Mondays 9:00 UTC)
2. Verify `trend-dashboard-*` artifact is created
3. Check for auto-created issues if regressions detected

### Expected Dashboard Output

```markdown
# 📊 Quality Trend Dashboard

## 📈 Quality Trends

### Test Pass Rate ✅
- **Current:** 86.2%
- **Average:** 84.1%
- **Trend:** improving (+2.1%)

### Execution Time ⚡
- **Current:** 11.2s
- **Average:** 12.4s
- **Change:** -9.7%

## 🎲 Flaky Tests
| Test | Pass | Fail | Flakiness |
|------|------|------|-----------|
| `test_async_timeout` | 8 | 4 | 33% |

## 💡 Insights
- ✅ Test pass rate is trending upward
- ⚠️ Identified 3 flaky tests requiring attention

## 📋 Recommendations
1. Prioritize stabilizing flaky tests
```

---

## Post-Deployment Validation

### Complete System Check

```bash
# 1. Verify git hooks
python3 .github/agents/rag_system/git_hooks.py status

# 2. Check knowledge base state
cat .rag_cache/knowledge/knowledge_state.json

# 3. List active workflows
gh workflow list

# 4. Check recent workflow runs
gh run list --limit 5
```

### Health Indicators

| Component | Health Check | Expected Result |
|-----------|--------------|-----------------|
| Git Hooks | `git_hooks.py status` | 4 hooks installed |
| JUnit Export | Check CI artifacts | `junit-*.xml` files present |
| PR Context | Open test PR | Comment generated |
| Trend Dashboard | Manual workflow run | Report artifact created |

---

## Troubleshooting

### Git Hooks Not Triggering

```bash
# Check hook permissions
ls -la .git/hooks/post-commit

# Reinstall hooks
python3 .github/agents/rag_system/git_hooks.py uninstall
python3 .github/agents/rag_system/git_hooks.py install --verbose
```

### Knowledge Engine Empty

```bash
# Manually trigger ingestion
python3 phase2_operational_package/scripts/ingest_ci_results.py \
  --junit test-results/junit.xml

# Check cache directory
ls -la .rag_cache/knowledge/
```

### PR Context Not Appearing

1. Check workflow permissions (needs `pull-requests: write`)
2. Verify RAG cache is being shared between workflows
3. Check workflow logs for errors

### Trend Dashboard Issues

```bash
# Run manually with verbose output
python3 phase2_operational_package/scripts/run_trend_analysis.py \
  --days 30 --verbose
```

---

## Operational Metrics

After full deployment, monitor these metrics:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Index Freshness | <500ms post-commit | Hook execution time |
| Test Intelligence | 100% JUnit coverage | Artifact presence |
| PR Context Coverage | 100% of PRs | Comment presence |
| Trend Accuracy | <5% variance | Prediction vs. actual |

---

## Using the Helper Scripts

### Generate PR Context Locally

```bash
# Auto-detect changed files from git
python3 phase2_operational_package/scripts/generate_pr_context.py --auto

# Specify files explicitly
python3 phase2_operational_package/scripts/generate_pr_context.py \
  --changed-files src/module.py tests/test_module.py

# Output to file
python3 phase2_operational_package/scripts/generate_pr_context.py \
  --auto --output pr_context.md
```

### Run Trend Analysis Locally

```bash
# Run with default 30-day window
python3 phase2_operational_package/scripts/run_trend_analysis.py

# Custom analysis period
python3 phase2_operational_package/scripts/run_trend_analysis.py --days 14

# Generate markdown report
python3 phase2_operational_package/scripts/run_trend_analysis.py \
  --output trend_report.json --markdown
```

### Ingest CI Results Manually

```bash
# Ingest single JUnit file
python3 phase2_operational_package/scripts/ingest_ci_results.py \
  --junit test-results/junit.xml

# Ingest from directory
python3 phase2_operational_package/scripts/ingest_ci_results.py \
  --dir test-results/

# Include coverage
python3 phase2_operational_package/scripts/ingest_ci_results.py \
  --junit results.xml --coverage coverage.xml
```

---

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review `.rag_cache/knowledge/` for data integrity
3. Consult Phase 2 Implementation Guide

---

*Phase 2 RAG System v2.1.0 - Transformation Portal*
*Operational Deployment Guide v1.0*
