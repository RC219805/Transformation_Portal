# CI/CD Monitoring Guide

## Overview

This guide explains how to monitor and maintain the health of the Transformation Portal's CI/CD infrastructure.

## Workflow Health Monitoring

### Quick Check

```bash
# Check all workflows (last 50 runs)
make workflow-health

# Check specific workflow
python scripts/workflow_health_check.py --workflow ci-consolidated.yml

# JSON output for automation
python scripts/workflow_health_check.py --json
```

### Health Metrics

The workflow health check provides:

- **Success Rate**: Percentage of successful runs (target: >90%)
- **Failure Rate**: Number and percentage of failed runs
- **Average Duration**: Mean execution time in seconds
- **Flaky Detection**: Workflows with success rate <90% flagged as flaky
- **Recommendations**: Actionable insights for improvements

### Interpreting Results

| Success Rate | Status | Action Required |
|-------------|--------|-----------------|
| ≥90% | ✅ Healthy | Monitor periodically |
| 70-89% | ⚠️ Warning | Investigate root cause |
| <70% | ❌ Critical | Immediate attention required |

### Common Issues

#### 1. AI Code Review Failures

**Symptoms**: `.github/workflows/ai-code-review.yml` showing 0% success rate

**Causes**:
- Missing API keys or authentication
- Rate limiting on AI service
- Network connectivity issues

**Resolution**:
```bash
# Check workflow logs
gh run view --log --workflow=ai-code-review.yml

# Disable if not critical
# (Workflows can be disabled in GitHub UI)
```

#### 2. Summary Workflow Failures

**Symptoms**: `.github/workflows/summary.yml` failing consistently

**Causes**:
- Permission issues with GitHub token
- Missing or invalid workflow artifacts
- Dependency on other failed workflows

**Resolution**:
```bash
# Verify token permissions
gh auth status

# Check workflow dependencies
grep -r "needs:" .github/workflows/summary.yml
```

#### 3. Smart Issue Management Failures

**Symptoms**: `.github/workflows/smart-issue-management.yml` failing

**Causes**:
- GitHub API rate limiting
- Invalid issue query syntax
- Permission restrictions

**Resolution**:
```bash
# Check rate limit status
gh api rate_limit

# Test issue queries manually
gh issue list --state all --limit 5
```

#### 4. PyPI Upload Failures

**Symptoms**: Submit to PyPI workflow showing low success rate

**Recent Fix**: ✅ Migrated to Trusted Publishing (OIDC) on 2026-01-02

**Verification**:
```bash
# Check latest run
gh run list --workflow=submit-pypi.yml --limit 1

# Verify OIDC configuration
cat .github/workflows/submit-pypi.yml | grep -A 5 "id-token:"
```

## Workflow Optimization

### CI/CD Pipeline Duration

**Current Status**: Average ~10m54s (654 seconds)

**Optimization Opportunities**:

1. **Parallel Test Execution**
   ```yaml
   # Already implemented in ci-consolidated.yml
   strategy:
     matrix:
       python-version: ['3.10', '3.11', '3.12']
   ```

2. **Dependency Caching**
   ```yaml
   # Verify cache hits
   grep "cache-hit" .github/workflows/ci-consolidated.yml
   ```

3. **Selective Test Execution**
   ```yaml
   # Run only tests for changed files
   # Implemented via changed-files detection
   ```

### Reducing Flakiness

**Best Practices**:

1. **Add Retries for Network Operations**
   ```yaml
   - uses: nick-fields/retry@v2
     with:
       timeout_minutes: 10
       max_attempts: 3
   ```

2. **Use Stable Test Data**
   - Avoid external API dependencies
   - Mock network requests
   - Use fixtures for test data

3. **Increase Timeouts for Heavy Tests**
   ```yaml
   timeout-minutes: 30  # For ML model tests
   ```

## Monitoring Dashboard

### GitHub Actions Dashboard

Access the official dashboard:
```
https://github.com/RC219805/Transformation_Portal/actions
```

### Custom Monitoring

Create automated alerts using workflow health check:

```bash
#!/bin/bash
# Daily health check (add to cron)

cd /path/to/Transformation_Portal
python scripts/workflow_health_check.py --json > /tmp/workflow-health.json

# Alert if overall success rate < 80%
SUCCESS_RATE=$(jq -r '.[] | .success_rate' /tmp/workflow-health.json | awk '{s+=$1; c++} END {print s/c}')

if (( $(echo "$SUCCESS_RATE < 80" | bc -l) )); then
    echo "⚠️ CI/CD health degraded: ${SUCCESS_RATE}%" | mail -s "CI Alert" team@example.com
fi
```

## Workflow Maintenance

### Regular Tasks

**Weekly**:
- Review workflow health report
- Investigate any flaky workflows
- Check for deprecated GitHub Actions

**Monthly**:
- Update GitHub Actions to latest versions
- Review and optimize slow workflows
- Audit workflow permissions

**Quarterly**:
- Security audit of workflow configurations
- Review and update CI/CD best practices
- Evaluate new GitHub Actions features

### Workflow Updates

```bash
# Update all actions to latest versions
python scripts/update_workflow_actions.py  # TODO: Create this script

# Validate workflow syntax
gh workflow view ci-consolidated.yml

# Test workflow changes on branch
git checkout -b test/workflow-update
# Make changes
git push origin test/workflow-update
# Monitor run on GitHub
```

## Troubleshooting

### Debug Mode

Enable debug logging in workflows:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### Manual Workflow Runs

Trigger workflows manually for testing:

```bash
# Trigger with default inputs
gh workflow run ci-consolidated.yml

# Trigger with custom inputs
gh workflow run ci-consolidated.yml \
  -f force_full_test=true \
  -f enable_phase2_features=true
```

### Viewing Logs

```bash
# View latest run logs
gh run view --log

# View specific workflow run
gh run view 20653165274 --log

# Download logs for analysis
gh run download 20653165274
```

## Metrics and KPIs

### Key Performance Indicators

1. **Overall Success Rate**: Target >90%
2. **Average Duration**: Target <15 minutes for CI
3. **Flaky Workflow Count**: Target 0
4. **Time to Fix**: Target <24 hours for critical failures
5. **Build Frequency**: Monitor daily commits triggering CI

### Tracking Trends

```bash
# Generate 30-day health report
python scripts/workflow_health_check.py --limit 200 --json > health-report.json

# Analyze trends
python scripts/analyze_ci_trends.py health-report.json  # TODO: Create script
```

## Best Practices

### Workflow Design

1. **Keep workflows focused**: One workflow = one purpose
2. **Use matrix builds**: Test across Python versions/platforms
3. **Cache dependencies**: Speed up builds with pip cache
4. **Fail fast**: Stop on first critical error
5. **Provide context**: Clear job names and helpful error messages

### Security

1. **Use OIDC**: Prefer Trusted Publishing over API tokens
2. **Minimize permissions**: Use least-privilege principle
3. **Pin action versions**: Use SHA instead of tags for security
4. **Secret scanning**: Enable GitHub secret scanning
5. **Audit regularly**: Review workflow permissions quarterly

### Performance

1. **Parallel execution**: Use matrix strategy
2. **Smart caching**: Cache pip, npm, and build artifacts
3. **Selective testing**: Run only affected tests
4. **Optimize dependencies**: Install only what's needed
5. **Monitor duration**: Alert on >20% increase

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Health Check Script](../scripts/workflow_health_check.py)
- [CI Consolidated Workflow](../.github/workflows/ci-consolidated.yml)
- [PyPI Trusted Publishing Setup](../.github/workflows/PYPI_TRUSTED_PUBLISHING_SETUP.md)

## Support

For CI/CD issues:
1. Check workflow logs: `gh run view --log`
2. Review health report: `make workflow-health`
3. Consult this guide for common issues
4. Open issue with `ci-cd` label if unresolved
