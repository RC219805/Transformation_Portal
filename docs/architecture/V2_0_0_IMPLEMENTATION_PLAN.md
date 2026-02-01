# v2.0.0 Pre-Release Implementation Plan

**Status**: ACTION REQUIRED
**Deadline**: 2 days before production deployment
**Owner**: DevOps + Architect

---

## P0: Critical Blockers (MUST COMPLETE)

### 1. Add Code Coverage to CI ⏱️ 3 hours

**File**: `.github/workflows/build.yml`

**Location**: After line 178 (in the test job)

**Changes Required**:

```yaml
      - name: Install core dependencies
        run: |
          set -euo pipefail
          python -m pip install --upgrade pip wheel
          python -m pip install --no-cache-dir pytest pytest-cov  # ✅ Already there
          python -m pip install --no-cache-dir -r requirements-ci.txt
          pip cache purge || true

      # ... existing steps ...

      - name: Run tests with coverage  # ⬅️ REPLACE existing "Run tests" step
        run: |
          set -euo pipefail

          # Run tests with coverage measurement
          python -m pytest tests/ \
            -m "${{ matrix.markexpr }}" \
            -v \
            --tb=short \
            --cov=src/transformation_portal/lux_depth_v3 \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --cov-report=html:htmlcov \
            --cov-fail-under=70

          echo "✅ Tests passed with ≥70% coverage"

      - name: Upload coverage reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.python-version }}-${{ matrix.test-type }}
          path: |
            coverage.xml
            htmlcov/
          retention-days: 30

      - name: Upload to Codecov (optional)
        if: matrix.python-version == '3.11' && matrix.test-type == 'core'
        uses: codecov/codecov-action@v5
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false  # Don't block on Codecov failures
```

**Validation**:
```bash
# Test locally first
cd /Users/rc/Projects/Transformation_Portal
python3 -m pytest tests/test_pbr_processor.py tests/test_pbr_presets.py \
  --cov=src/transformation_portal/lux_depth_v3 \
  --cov-report=term-missing \
  --cov-fail-under=70

# Expected: ✅ Tests pass, coverage ≥70%
```

**Effort**: 3 hours (implementation + testing + validation)

---

### 2. Add Security Scanning to PR Workflow ⏱️ 2 hours

**File**: `.github/workflows/build.yml`

**Location**: Add new job after `lint` job (before `test` job)

**Changes Required**:

```yaml
  security:
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    permissions:
      contents: read
      security-events: write

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install security tools
        run: |
          set -euo pipefail
          python -m pip install --upgrade pip
          python -m pip install pip-audit

      - name: Dependency vulnerability scan (pip-audit)
        run: |
          set -euo pipefail
          echo "Scanning requirements.txt for vulnerabilities..."
          pip-audit --requirement requirements.txt --format json --output pip-audit-report.json || true

          # Check if critical vulnerabilities exist
          CRITICAL_COUNT=$(jq '.dependencies | length' pip-audit-report.json 2>/dev/null || echo "0")

          if [ "$CRITICAL_COUNT" -gt 0 ]; then
            echo "⚠️ Found $CRITICAL_COUNT vulnerable dependencies"
            jq '.dependencies' pip-audit-report.json
            exit 1
          fi

          echo "✅ No critical vulnerabilities found"

      - name: Banned dependency check
        run: |
          set -euo pipefail
          echo "Checking for banned dependencies..."
          python scripts/security/verify_banned_dependencies.py
          echo "✅ No banned dependencies detected"

      - name: Upload security reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: security-reports
          path: pip-audit-report.json
          retention-days: 30

  test:
    needs: [lint, security]  # ⬅️ ADD security dependency
    runs-on: ubuntu-24.04
    # ... rest of test job unchanged
```

**Update**: Modify `test` job dependency:
```yaml
  test:
    needs: [lint, security]  # ⬅️ Was: needs: lint
```

**Validation**:
```bash
# Test security scan locally
cd /Users/rc/Projects/Transformation_Portal
pip install pip-audit
pip-audit --requirement requirements.txt

# Test banned dependency check
python3 scripts/security/verify_banned_dependencies.py
```

**Effort**: 2 hours

---

### 3. Configure Branch Protection ⏱️ 1 hour

**Action**: Manual configuration in GitHub repository settings

**Steps**:

1. Navigate to: `https://github.com/RC219805/Transformation_Portal/settings/branches`

2. Edit protection rule for `main` branch:

   **Required Status Checks**:
   ```
   ☑ Require status checks to pass before merging
     ☑ lint
     ☑ security
     ☑ test (3.10, cpu, core)
     ☑ test (3.11, cpu, ml)
     ☑ test (3.12, cpu, core)
     ☑ Analyze (python)      # CodeQL
     ☑ Analyze (actions)     # CodeQL
   ```

   **Additional Settings**:
   ```
   ☑ Require branches to be up to date before merging
   ☑ Require conversation resolution before merging
   ☑ Do not allow bypassing the above settings
   ```

3. **Verify**: Create test PR and ensure all checks run

**Effort**: 1 hour (setup + validation)

---

### 4. Document Rollback Procedures ⏱️ 2 hours

**File**: `docs/deployment/ROLLBACK_PROCEDURES.md` (new)

**Content**:

```markdown
# Emergency Rollback Procedures: v2.0.0

**Purpose**: Fast recovery from production issues
**Audience**: On-call engineers, DevOps, Architect
**Last Updated**: 2026-02-01

---

## Rollback Triggers

Execute rollback immediately if ANY of these conditions occur:

| Condition | Threshold | Detection Method |
|-----------|-----------|------------------|
| Error rate spike | >10% for 15 minutes | Application logs |
| Memory leak | +2GB growth/hour | System metrics |
| Crash rate | >1% of requests | Error tracking |
| Data corruption | Any instance | User reports + validation |
| Critical bug | Severity P0 | GitHub issue + assessment |

---

## Rollback Decision Matrix

| Scenario | Action | ETA |
|----------|--------|-----|
| Error rate 5-10% | Monitor closely, prepare rollback | 0 min |
| Error rate >10% | **Execute rollback immediately** | 5 min |
| Memory leak detected | Restart + monitor → rollback if persists | 10 min |
| Critical bug confirmed | **Execute rollback** | 5 min |

---

## Rollback Procedure

### Option A: Feature Flag Disable (Fastest - 2 minutes)

**Use When**: PBR feature causing issues, depth pipeline still works

```bash
# 1. SSH to production server
ssh production-server

# 2. Disable PBR generation via environment variable
export TRANSFORMATION_PORTAL_DISABLE_PBR=true

# 3. Restart application
sudo systemctl restart transformation-portal

# 4. Verify
curl https://api.transformation-portal/health | jq '.features.pbr'
# Expected: {"enabled": false, "status": "disabled"}
```

**Validation**:
- Depth pipeline continues to work
- PBR generation returns 404 or graceful fallback
- Error rate returns to baseline

---

### Option B: Version Revert (Full - 15 minutes)

**Use When**: Systemic issues, need complete rollback to v1.9.5

```bash
#!/bin/bash
set -euo pipefail

echo "🚨 EMERGENCY ROLLBACK: v2.0.0 → v1.9.5"
echo "Initiated by: $(whoami) at $(date)"

# 1. Stop production services (2 min)
echo "Step 1/5: Stopping services..."
sudo systemctl stop transformation-portal
sudo systemctl stop transformation-portal-worker

# 2. Backup current state (1 min)
echo "Step 2/5: Backing up v2.0.0 state..."
mkdir -p /var/backups/transformation-portal/v2.0.0-rollback-$(date +%Y%m%d-%H%M%S)
cp -r /opt/transformation-portal /var/backups/transformation-portal/v2.0.0-rollback-$(date +%Y%m%d-%H%M%S)/

# 3. Checkout previous version (3 min)
echo "Step 3/5: Reverting to v1.9.5..."
cd /opt/transformation-portal
git fetch origin --tags
git checkout v1.9.5
git status

# 4. Reinstall dependencies (5 min)
echo "Step 4/5: Reinstalling dependencies..."
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt

# 5. Restart services (2 min)
echo "Step 5/5: Restarting services..."
sudo systemctl start transformation-portal
sudo systemctl start transformation-portal-worker

# 6. Health check (2 min)
echo "Step 6/6: Validating rollback..."
sleep 10  # Allow services to initialize

curl -f https://api.transformation-portal/health || {
  echo "❌ Health check failed! Investigate immediately."
  exit 1
}

echo "✅ Rollback complete. Services running v1.9.5."
echo "Next steps:"
echo "  1. Monitor error rates for 15 minutes"
echo "  2. Create incident post-mortem: docs/incidents/v2.0.0-rollback-$(date +%Y%m%d).md"
echo "  3. Schedule hotfix planning meeting"
```

**Save as**: `scripts/deployment/emergency_rollback.sh`

**Permissions**:
```bash
chmod +x scripts/deployment/emergency_rollback.sh
```

---

### Option C: Database Rollback (If Needed)

**Use When**: Schema changes in v2.0.0 need reverting

⚠️ **NOTE**: v2.0.0 does NOT include database changes. This section is for future reference.

```bash
# If database migrations exist:
# 1. Identify migration version
psql -U transformation_portal -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# 2. Rollback migration
alembic downgrade -1  # Or specific version

# 3. Verify
psql -U transformation_portal -c "\dt"
```

---

## Post-Rollback Checklist

After rollback execution:

- [ ] **Verify services healthy** (error rate <1%, latency normal)
- [ ] **Notify stakeholders** (Slack #engineering, email)
- [ ] **Create incident report** (`docs/incidents/YYYY-MM-DD-v2.0.0-rollback.md`)
- [ ] **Update status page** (if public-facing)
- [ ] **Schedule post-mortem** (within 48 hours)
- [ ] **Plan hotfix** (v2.0.1 with fixes)

---

## Rollback Validation

After rollback, confirm:

```bash
# 1. Version check
curl https://api.transformation-portal/version
# Expected: {"version": "1.9.5"}

# 2. Core functionality
curl -X POST https://api.transformation-portal/depth/estimate \
  -H "Content-Type: application/json" \
  -d '{"image": "base64..."}'
# Expected: 200 OK with depth map

# 3. Error rate
tail -100 /var/log/transformation-portal/app.log | grep ERROR | wc -l
# Expected: <5 errors in last 100 lines

# 4. Memory usage
ps aux | grep transformation-portal | awk '{print $6}'
# Expected: <6GB RSS
```

---

## Hotfix Process (v2.0.1)

After successful rollback, plan hotfix:

1. **Root Cause Analysis** (4-8 hours):
   - Reproduce issue locally
   - Identify exact failure point
   - Write failing test

2. **Develop Fix** (4-12 hours):
   - Implement minimal fix
   - Add regression test
   - Validate locally

3. **Fast-Track Review** (2-4 hours):
   - Create PR with `[HOTFIX]` prefix
   - Architect + Specialist review
   - Skip non-critical checks

4. **Deploy v2.0.1** (1-2 hours):
   - Tag release
   - Deploy to staging
   - 6-hour canary
   - Full production

**Timeline**: Target 24-48 hours from rollback to hotfix

---

## Communication Templates

### Slack Announcement (Rollback Initiated)

```
🚨 **PRODUCTION ROLLBACK IN PROGRESS** 🚨

**Component**: Transformation Portal
**From**: v2.0.0
**To**: v1.9.5
**Reason**: [Error rate spike / Memory leak / Critical bug]
**ETA**: 15 minutes
**Impact**: [Brief description]
**Initiated by**: @architect

Monitoring: #incident-response
Updates: Every 5 minutes
```

### Slack Announcement (Rollback Complete)

```
✅ **ROLLBACK COMPLETE**

**Status**: Services restored to v1.9.5
**Error Rate**: [Current rate, should be <1%]
**Latency**: [p95 latency, should be <2s]
**Impact Duration**: [Total downtime/degradation]

**Next Steps**:
1. Post-mortem scheduled: [Date/Time]
2. Hotfix planning: [Date/Time]
3. User communication: [If needed]

Details: docs/incidents/v2.0.0-rollback-YYYY-MM-DD.md
```

---

## On-Call Contacts

| Role | Primary | Secondary |
|------|---------|-----------|
| **Incident Commander** | @architect | @product-owner |
| **Technical Lead** | @specialist | @devops-lead |
| **DevOps** | @devops-primary | @devops-secondary |
| **Communications** | @product-manager | @architect |

**Escalation Path**:
1. On-call engineer (0-15 min)
2. Technical lead (15-30 min)
3. Architect (30-60 min)
4. CTO (>60 min or business-critical)

---

## Testing Rollback Procedure

**Quarterly drill** (recommended):

1. **Staging Rollback Test**:
   ```bash
   # Deploy v2.0.0 to staging
   # Execute rollback procedure
   # Validate services restored
   # Document lessons learned
   ```

2. **Metrics**:
   - Rollback execution time (target: <15 min)
   - Time to detection (target: <5 min)
   - False positive rate (target: <10%)

---

## Lessons from Past Rollbacks

*This section will be populated after first production rollback.*

---

**Document Owner**: Architect
**Review Frequency**: After each rollback
**Next Review**: Post v2.0.0 deployment
```

**Create directory**:
```bash
mkdir -p docs/deployment
```

**Validation**: Have peer review procedure for clarity

**Effort**: 2 hours (writing + review)

---

### 5. Create Staging Validation Checklist ⏱️ 2 hours

**File**: `docs/deployment/STAGING_VALIDATION.md` (new)

**Content**:

```markdown
# Staging Environment Validation: v2.0.0

**Purpose**: Pre-production verification
**Owner**: QA + Architect
**Duration**: 24-48 hours

---

## Staging Environment Details

**URL**: `staging.transformation-portal.internal` (VPN required)
**Infrastructure**: Mirrors production (same Python version, dependencies, hardware)
**Data**: Anonymized production sample (50 images, 5 videos)
**Access**: DevOps, QA, Architect only

---

## Pre-Deployment Checklist

Before deploying v2.0.0 to staging:

- [ ] All P0 items completed (coverage, security, branch protection)
- [ ] CI tests passing: 177/177 tests ✅
- [ ] Security scan clean (no critical vulnerabilities)
- [ ] Documentation reviewed and approved
- [ ] Rollback procedure tested (dry run)

---

## Deployment to Staging

```bash
#!/bin/bash
# Deploy v2.0.0 to staging

set -euo pipefail

echo "Deploying v2.0.0 to staging..."

# 1. SSH to staging server
ssh staging-server

# 2. Backup current version
sudo systemctl stop transformation-portal
cp -r /opt/transformation-portal /var/backups/transformation-portal-pre-v2.0.0

# 3. Deploy v2.0.0
cd /opt/transformation-portal
git fetch origin --tags
git checkout v2.0.0
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt

# 4. Restart services
sudo systemctl start transformation-portal

# 5. Health check
sleep 10
curl -f https://staging.transformation-portal/health || exit 1

echo "✅ v2.0.0 deployed to staging"
```

---

## Smoke Tests (30 minutes)

### Test 1: Basic PBR Generation

```bash
# Test all 8 presets with sample image
cd /opt/transformation-portal

for preset in standard premium draft wood metal glass stone fabric; do
  echo "Testing preset: $preset"

  python3 -m transformation_portal.lux_depth_v3.pbr_cli \
    --depth data/staging/sample_depth.npy \
    --preset "$preset" \
    --output /tmp/staging-test-$preset/

  # Verify outputs exist
  test -f /tmp/staging-test-$preset/sample_normal.png || exit 1
  test -f /tmp/staging-test-$preset/sample_roughness.png || exit 1
  test -f /tmp/staging-test-$preset/sample_ao.png || exit 1

  echo "✅ Preset $preset: PASS"
done
```

**Expected**: All 8 presets generate valid outputs

### Test 2: Batch Processing

```bash
# Test batch processing (50 images)
python3 -m transformation_portal.lux_depth_v3.pbr_cli \
  --depth-dir data/staging/depth_batch/ \
  --preset premium \
  --output /tmp/staging-batch-test/

# Verify all 50 outputs
OUTPUT_COUNT=$(ls -1 /tmp/staging-batch-test/*.png | wc -l)
EXPECTED_COUNT=150  # 50 images × 3 maps

if [ "$OUTPUT_COUNT" -eq "$EXPECTED_COUNT" ]; then
  echo "✅ Batch test: PASS ($OUTPUT_COUNT/$EXPECTED_COUNT files)"
else
  echo "❌ Batch test: FAIL ($OUTPUT_COUNT/$EXPECTED_COUNT files)"
  exit 1
fi
```

### Test 3: Orchestrator Integration

```bash
# Test full enhancement pipeline with PBR
python3 << 'EOF'
from pathlib import Path
from transformation_portal.lux_depth_v3.pbr_presets import PREMIUM_QUALITY
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Configure
output = Path("/tmp/staging-orchestrator-test")
orchestrator = EnhanceOrchestrator(PREMIUM_QUALITY, output)

# Process single image
result = orchestrator.enhance_single(
    "data/staging/sample_interior.jpg",
    base_name="interior_test"
)

# Verify PBR maps generated
assert (output / "interior_test_normal.png").exists()
assert (output / "interior_test_roughness.png").exists()
assert (output / "interior_test_ao.png").exists()

print("✅ Orchestrator integration: PASS")
EOF
```

### Test 4: Error Handling

```bash
# Test graceful error handling
python3 << 'EOF'
from transformation_portal.lux_depth_v3.pbr_processor import PBRProcessor
from transformation_portal.lux_depth_v3.pbr import PBRConfig
import numpy as np

config = PBRConfig()

# Test 1: Missing file
try:
    PBRProcessor.from_cached_depth("/nonexistent.npy", config)
    print("❌ Should have raised FileNotFoundError")
    exit(1)
except FileNotFoundError as e:
    print(f"✅ Missing file error: {e}")

# Test 2: NaN values
try:
    depth = np.full((100, 100), np.nan)
    processor = PBRProcessor(config)
    processor.from_depth(depth, save=False)
    print("❌ Should have rejected NaN values")
    exit(1)
except ValueError as e:
    print(f"✅ NaN rejection: {e}")

print("✅ Error handling: PASS")
EOF
```

---

## Performance Validation (1 hour)

### Benchmark 1: Single Image Latency

```bash
# Measure p50, p95, p99 latency
python3 << 'EOF'
import time
import numpy as np
from transformation_portal.lux_depth_v3.pbr_processor import PBRProcessor
from transformation_portal.lux_depth_v3.pbr_presets import get_preset

config = get_preset("premium").to_pbr_config()
processor = PBRProcessor(config)

# Load sample depth
depth = np.load("data/staging/sample_depth.npy")

# Warm-up
for _ in range(3):
    processor.from_depth(depth, save=False)

# Benchmark 100 iterations
latencies = []
for _ in range(100):
    start = time.perf_counter()
    processor.from_depth(depth, save=False)
    latencies.append(time.perf_counter() - start)

latencies.sort()
p50 = latencies[50]
p95 = latencies[95]
p99 = latencies[99]

print(f"Latency p50: {p50:.3f}s")
print(f"Latency p95: {p95:.3f}s")
print(f"Latency p99: {p99:.3f}s")

# Validation
assert p95 < 2.0, f"p95 latency too high: {p95:.3f}s"
assert p99 < 3.0, f"p99 latency too high: {p99:.3f}s"

print("✅ Performance benchmark: PASS")
EOF
```

**Expected**:
- p50 < 1.0s
- p95 < 2.0s
- p99 < 3.0s

### Benchmark 2: Batch Throughput

```bash
# Test batch processing performance (50 images)
time python3 -m transformation_portal.lux_depth_v3.pbr_cli \
  --depth-dir data/staging/depth_batch/ \
  --preset standard \
  --output /tmp/staging-throughput-test/

# Expected: <60 seconds for 50 images (>50 images/min)
```

### Benchmark 3: Memory Usage

```bash
# Monitor memory during batch processing
/usr/bin/time -v python3 -m transformation_portal.lux_depth_v3.pbr_cli \
  --depth-dir data/staging/depth_batch/ \
  --preset premium \
  --output /tmp/staging-memory-test/ \
  2>&1 | grep "Maximum resident set size"

# Expected: <8GB peak memory
```

---

## 24-Hour Monitoring

After smoke tests pass, monitor for 24 hours:

### Metrics to Track

| Metric | Check Frequency | Alert Threshold |
|--------|-----------------|-----------------|
| Error rate | Every 2 hours | >1% |
| Memory usage | Every 2 hours | >8GB sustained |
| Disk usage | Every 4 hours | >80% |
| Log anomalies | Every 4 hours | >10 errors/hour |

### Automated Monitoring Script

```bash
#!/bin/bash
# Save as: scripts/staging_monitor.sh

while true; do
  echo "=== Staging Health Check: $(date) ==="

  # 1. Error rate
  ERROR_COUNT=$(tail -1000 /var/log/transformation-portal/app.log | grep ERROR | wc -l)
  echo "Errors (last 1000 lines): $ERROR_COUNT"

  # 2. Memory usage
  MEM_USAGE=$(ps aux | grep transformation-portal | awk '{sum+=$6} END {print sum/1024/1024}')
  echo "Memory usage: ${MEM_USAGE}GB"

  # 3. Disk usage
  DISK_USAGE=$(df -h /opt/transformation-portal | tail -1 | awk '{print $5}')
  echo "Disk usage: $DISK_USAGE"

  # 4. Health endpoint
  curl -sf https://staging.transformation-portal/health > /dev/null && echo "✅ Health: OK" || echo "❌ Health: FAIL"

  echo "---"
  sleep 7200  # 2 hours
done
```

---

## Sign-Off Criteria

Staging validation PASSES if:

- ✅ All smoke tests pass (8/8 presets, batch, integration, errors)
- ✅ Performance benchmarks meet targets (p95 < 2s, throughput > 50 img/min)
- ✅ 24-hour monitoring: Error rate <1%, no memory leaks, no crashes
- ✅ No P0/P1 bugs discovered
- ✅ Logs clean (no unexpected warnings)

**Sign-Off Required**:
- [ ] QA Engineer: _______________
- [ ] Architect: _______________
- [ ] DevOps Lead: _______________

**Date**: _______________

---

## If Validation Fails

1. **Document issues** in GitHub (label: `staging-blocker`)
2. **Rollback staging** to previous version
3. **Fix issues** and retest locally
4. **Re-deploy to staging** and repeat validation
5. **Do NOT proceed to production** until staging passes

---

**Document Owner**: Architect + QA
**Last Updated**: 2026-02-01
**Next Review**: After v2.0.0 staging deployment
```

**Create directory** (if needed):
```bash
mkdir -p docs/deployment
```

**Effort**: 2 hours (documentation + script creation)

---

## P0 Summary

**Total Effort**: 12 hours (can parallelize to 2 days)

| Task | Owner | Duration | Dependencies |
|------|-------|----------|--------------|
| Add coverage to CI | DevOps | 3 hours | None |
| Add security to PR | DevOps | 2 hours | None |
| Branch protection | Admin | 1 hour | Security job created |
| Rollback docs | Architect | 2 hours | None |
| Staging validation | QA + Architect | 2 hours | None |
| **Integration testing** | All | 2 hours | All above |

**Timeline**:
- **Day 1** (8 hours): Items 1-3 (coverage, security, branch protection)
- **Day 2** (4 hours): Items 4-5 (documentation), integration testing

---

## P1: High Priority (v2.0.1) - Can Defer

### 6. Add Type Hints to pbr.py ⏱️ 30 minutes

**File**: `src/transformation_portal/lux_depth_v3/pbr.py`

**Change**:
```python
from typing import Tuple

def generate_pbr_maps(
    depth: np.ndarray,
    config: PBRConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate PBR maps from depth array.

    Returns:
        Tuple of (normal, roughness, ao) as uint8 arrays (H, W, C)
    """
```

---

### 7. Add mypy to CI ⏱️ 1 hour

**File**: `.github/workflows/build.yml`

Add after lint step:

```yaml
- name: Type check with mypy
  run: |
    python -m pip install mypy types-PyYAML
    mypy src/transformation_portal/lux_depth_v3/pbr*.py \
      --config-file mypy.ini \
      --no-error-summary
```

---

### 8. CLI Test Suite ⏱️ 6-8 hours

**File**: `tests/test_pbr_cli.py` (new)

**Minimal implementation**:

```python
"""Tests for PBR CLI interface."""
import pytest
from pathlib import Path
from typer.testing import CliRunner
from transformation_portal.lux_depth_v3.pbr_cli import app

runner = CliRunner()

def test_cli_help():
    """Test CLI help displays."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Generate PBR maps" in result.stdout

def test_cli_single_file(tmp_path, sample_depth_npy):
    """Test CLI with single depth file."""
    result = runner.invoke(app, [
        "--depth", str(sample_depth_npy),
        "--output", str(tmp_path),
        "--preset", "premium"
    ])
    assert result.exit_code == 0
    assert (tmp_path / "sample_normal.png").exists()

# ... 18 more tests for argument combinations
```

**Target**: 20+ tests, 80% CLI coverage

---

## Validation Checklist

Before marking P0 items complete:

### CI Coverage Integration
- [ ] `build.yml` updated with coverage steps
- [ ] Coverage threshold set to 70%
- [ ] Artifact upload working
- [ ] PR shows coverage report in checks

### Security Scanning
- [ ] New `security` job added to `build.yml`
- [ ] `pip-audit` step working
- [ ] Banned dependency check running
- [ ] `test` job depends on `security`

### Branch Protection
- [ ] All required checks configured
- [ ] Cannot merge failing PRs
- [ ] Cannot bypass checks (admin override disabled)
- [ ] Test PR validates enforcement

### Documentation
- [ ] Rollback procedures complete and reviewed
- [ ] Staging validation checklist complete
- [ ] Scripts tested (dry run)
- [ ] Peer review completed

### Integration Testing
- [ ] All P0 changes tested together
- [ ] PR workflow validates security + coverage
- [ ] False positive rate acceptable (<5%)

---

## Success Criteria

**P0 Complete When**:
✅ Create test PR → all checks pass (coverage, security, tests)
✅ Rollback procedure dry-run successful
✅ Staging validation documented and approved
✅ Architect sign-off

**Ready for Production Deployment**: YES

---

**Plan Owner**: Architect
**Last Updated**: 2026-02-01
**Execution Timeline**: 2 days before production release
