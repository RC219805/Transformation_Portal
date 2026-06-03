# Rollback Procedures - Transformation Portal

**Version**: 2.1.0
**Last Updated**: 2026-02-10
**Status**: Production Ready
**Owner**: DevOps + Architect

---

## Overview

This document provides step-by-step procedures for rolling back deployments of the Transformation Portal in case of critical issues during or after release.

### Rollback Decision Criteria

**Initiate rollback immediately if**:
- Error rate exceeds 5% during canary deployment
- Critical functionality completely broken (depth estimation, PBR generation)
- Data corruption or loss detected
- Security vulnerability discovered in deployed code
- Performance degradation >50% from baseline
- Memory leak causing OOM crashes

**Do NOT rollback for**:
- Minor UI issues or cosmetic bugs
- Non-critical feature bugs (can be hotfixed)
- Error rate <2% (monitor and hotfix)
- Performance degradation <20% (investigate first)

---

## Rollback Strategies

### Strategy 1: Git Tag Rollback (Recommended for Releases)

**Use When**: Rolling back a tagged release (e.g., v2.0.0 → v1.9.0)
**Estimated Time**: 15 minutes
**Risk**: Low (well-tested previous version)

#### Prerequisites
```bash
# Ensure you have access to production environment
git remote -v  # Verify origin points to production repo
git fetch --tags  # Ensure all tags are up to date
```

#### Step-by-Step Procedure

**Step 1: Identify target version**
```bash
# List recent tags
git tag --sort=-version:refname | head -10

# Example output:
# v2.0.0  (current, broken)
# v1.9.0  (last known good)
# v1.8.0
```

**Step 2: Create rollback branch**
```bash
# Create rollback branch from last known good version
git checkout -b rollback-to-v1.9.0 v1.9.0

# Verify you're on correct version
git log --oneline -5
git describe --tags
```

**Step 3: Deploy rollback**
```bash
# If using pip install from git:
pip install --force-reinstall git+https://github.com/RC219805/Transformation_Portal.git@v1.9.0

# If using local editable install:
cd /path/to/Transformation_Portal
git checkout v1.9.0
pip install --force-reinstall -e .
```

**Step 4: Verify rollback**
```bash
# Check installed version
python -c "import transformation_portal; print(transformation_portal.__version__)"

# Run smoke tests
pytest tests/test_core_functionality.py -v --maxfail=1

# Verify PBR processing (if applicable)
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli --help
```

**Step 5: Restart services** (if applicable)
```bash
# If running as a service:
sudo systemctl restart transformation-portal

# If using Docker:
docker-compose down
docker-compose up -d

# Verify service is healthy
curl http://localhost:8000/health || .venv/bin/python scripts/verification/verify_core.py
```

**Step 6: Monitor and communicate**
```bash
# Monitor logs for 30 minutes
tail -f /var/log/transformation-portal/app.log

# Or if using Docker:
docker logs -f transformation-portal --tail=100
```

**Communication Template**:
```
[INCIDENT] Transformation Portal rollback completed

Timeline:
- [HH:MM] Issue detected: [describe problem]
- [HH:MM] Rollback initiated
- [HH:MM] Rollback completed and verified
- [HH:MM] Service restored

Current Status: v1.9.0 (stable)
Next Steps: Root cause analysis, hotfix plan for v2.0.1

Affected users: [estimate]
Mitigation: [describe any required user action]
```

---

### Strategy 2: Feature Flag Rollback (For Incremental Features)

**Use When**: New feature causing issues but rest of system is stable
**Estimated Time**: 5 minutes
**Risk**: Very Low (instant toggle)

#### Prerequisites
- Feature must be behind a feature flag (e.g., `ENABLE_PBR_V2`)
- Access to configuration management system

#### Procedure

**Step 1: Identify feature flag**
```python
# Example: Disable PBR v2 processing
# In config/feature_flags.yml or environment variables
ENABLE_PBR_V2: false
ENABLE_DEPTH_V2: true  # Keep other features enabled
```

**Step 2: Update configuration**
```bash
# Environment variable approach:
export ENABLE_PBR_V2=false

# Or update config file:
vim config/production.yml
# Set: enable_pbr_v2: false

# Restart service to pick up changes
sudo systemctl restart transformation-portal
```

**Step 3: Verify feature disabled**
```python
# Test that PBR v2 is not used
python -c "
from transformation_portal.config import get_config
config = get_config('production')
assert not config.enable_pbr_v2, 'PBR v2 still enabled!'
print('✅ PBR v2 successfully disabled')
"
```

**Step 4: Fallback verification**
```bash
# Verify system falls back to v1 PBR or skips PBR
python examples/test_pbr_fallback.py

# Should output: "Using PBR v1 (fallback)" or "PBR disabled"
```

---

### Strategy 3: Container Rollback (For Dockerized Deployments)

**Use When**: Using Docker/container-based deployment
**Estimated Time**: 10 minutes
**Risk**: Low (immutable containers)

#### Prerequisites
```bash
# List available container images
docker images | grep transformation-portal

# Example output:
# transformation-portal   v2.0.0    abc123   2 hours ago    1.2GB
# transformation-portal   v1.9.0    def456   2 weeks ago    1.1GB
```

#### Procedure

**Step 1: Stop current container**
```bash
# If using docker-compose:
docker-compose down

# If using standalone Docker:
docker stop transformation-portal
docker rm transformation-portal
```

**Step 2: Start previous container version**
```bash
# Update docker-compose.yml:
services:
  transformation-portal:
    image: ghcr.io/rc219805/transformation-portal:v1.9.0  # Changed from v2.0.0
    # ... rest of config

# Or pull and run directly:
docker pull ghcr.io/rc219805/transformation-portal:v1.9.0
docker run -d --name transformation-portal \
  -p 8000:8000 \
  -v /data:/data \
  ghcr.io/rc219805/transformation-portal:v1.9.0
```

**Step 3: Verify container health**
```bash
# Check container is running
docker ps | grep transformation-portal

# Check health endpoint (if configured)
docker exec transformation-portal curl localhost:8000/health

# Check logs
docker logs transformation-portal --tail=50
```

**Step 4: Run smoke tests**
```bash
# Execute tests inside container
docker exec transformation-portal pytest tests/test_core_functionality.py -v

# Or from host:
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"image": "test.jpg", "preset": "default"}'
```

---

### Strategy 4: Hot Patch Rollback (For Critical Bugs in Current Version)

**Use When**: Single file/function causing issue in otherwise stable release
**Estimated Time**: 20 minutes
**Risk**: Medium (manual patching)

#### Prerequisites
- Root cause identified to specific file/function
- Previous known-good version of that file available

#### Procedure

**Step 1: Identify problematic code**
```bash
# Example: PBR processor has a bug
git log --oneline -- src/transformation_portal/lux_depth_v3/pbr_processor.py

# Identify last known good commit for that file
git blame src/transformation_portal/lux_depth_v3/pbr_processor.py
```

**Step 2: Extract good version of file**
```bash
# Get file from previous tag
git show v1.9.0:src/transformation_portal/lux_depth_v3/pbr_processor.py > pbr_processor_v1.9.0.py

# Compare with current version
diff pbr_processor_v1.9.0.py src/transformation_portal/lux_depth_v3/pbr_processor.py
```

**Step 3: Apply patch**
```bash
# Backup current version
cp src/transformation_portal/lux_depth_v3/pbr_processor.py pbr_processor_broken_backup.py

# Replace with good version
cp pbr_processor_v1.9.0.py src/transformation_portal/lux_depth_v3/pbr_processor.py

# Re-install package (if needed)
pip install --force-reinstall -e .
```

**Step 4: Test patched version**
```bash
# Run targeted tests for patched module
pytest tests/test_pbr_processor.py -v

# Run integration tests
pytest tests/test_integration/ -v -m "not slow"
```

**Step 5: Document patch**
```bash
# Create emergency patch commit
git add src/transformation_portal/lux_depth_v3/pbr_processor.py
git commit -m "HOTFIX: Rollback pbr_processor.py to v1.9.0

Reason: [describe bug]
Affects: v2.0.0
Rollback to: commit abc123 from v1.9.0
Temporary fix until v2.0.1
"

git tag v2.0.0-hotfix1
git push origin v2.0.0-hotfix1
```

---

## PyPI Package Rollback Procedures

**Context**: If the Transformation Portal is published to PyPI (public or private), rollback requires special handling since PyPI does not allow version deletion or replacement.

### Current Status (as of v2.2.0)
- **PyPI Publication**: Not currently published to public PyPI
- **Distribution Method**: Git-based installation (`pip install git+...`) or local editable install

### If Published to PyPI (Future)

#### Option 1: Publish New Rollback Version (Recommended)

```bash
# Starting from rollback branch/tag
git checkout v2.1.0

# Bump to new patch version indicating rollback
# Edit pyproject.toml: version = "2.2.1"
sed -i 's/version = "2.2.0"/version = "2.2.1"/' pyproject.toml

# Add notice in CHANGELOG.md
echo "## [2.2.1] - $(date +%Y-%m-%d)
### Changed
- Emergency rollback to v2.1.0 codebase due to [incident reference]
- All functionality equivalent to v2.1.0
" >> CHANGELOG.md

# Build distribution packages
.venv/bin/python -m build
ls -lh dist/  # Verify wheel and sdist created

# Verify package contents
unzip -l dist/transformation_portal-2.2.1-py3-none-any.whl | head -20

# Test installation in isolated environment
python -m venv /tmp/test-rollback
source /tmp/test-rollback/bin/activate
pip install dist/transformation_portal-2.2.1-py3-none-any.whl
python -c "import transformation_portal; print(transformation_portal.__version__)"
deactivate

# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# Verify from Test PyPI
pip install --index-url https://test.pypi.org/simple/ transformation-portal==2.2.1

# If verified, upload to production PyPI
python -m twine upload dist/*
```

#### Option 2: Yank Broken Version

**Important**: Yanking prevents new installs but does not affect existing installations.

```bash
# Using PyPI web interface:
# 1. Navigate to: https://pypi.org/manage/project/transformation-portal/releases/
# 2. Select broken version (e.g., 2.2.0)
# 3. Click "Options" → "Yank release"
# 4. Provide reason: "Critical bug in [component], use 2.2.1 or 2.1.0 instead"

# Using PyPI API (requires token):
# Note: PyPI does not provide CLI yanking; use web UI
```

**Update Documentation**:
```bash
# Update README.md installation section
echo "**Note**: Version 2.2.0 has been yanked due to [issue]. Use 2.2.1 or 2.1.0." >> README.md

# Update GitHub release notes
gh release edit v2.2.0 --notes "⚠️ This release has been yanked. Use v2.2.1 or v2.1.0 instead.

Reason: [describe critical issue]
Recommended: Install v2.2.1 (rollback to v2.1.0 codebase) or v2.1.0 directly."
```

### Dependency Pinning Rollback

The repository uses tiered dependency management (as of v2.2.0):
- `requirements/core.txt` - Minimal runtime dependencies
- `requirements/ml.txt` - ML model dependencies
- `requirements/dev.txt` - Development and testing tools
- `requirements-ci.txt` - CI-specific pinned versions
- `requirements.txt` - Consolidated requirements

#### Procedure: Revert Dependency Changes

**Step 1: Identify dependency differences**
```bash
# Compare requirements between versions
git diff v2.1.0..v2.2.0 -- requirements/

# Show only changed package versions
git diff v2.1.0..v2.2.0 -- requirements/ | grep -E '^[+-][^+-]' | grep -v '^[+-]#'

# Example output:
# -torch==2.0.1
# +torch==2.1.0
# -transformers==4.38.0
# +transformers==4.40.0
```

**Step 2: Restore previous requirements files**
```bash
# Checkout requirements from target version
git checkout v2.1.0 -- requirements/

# Verify changes
git status
git diff requirements/
```

**Step 3: Reinstall with reverted dependencies**
```bash
# Remove current environment (recommended for clean state)
deactivate
rm -rf venv/

# Create fresh environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install -r requirements/core.txt

# Install ML dependencies (if needed)
pip install -r requirements/ml.txt

# Install package in editable mode
pip install -e .

# Verify critical dependencies
pip show torch transformers pillow numpy opencv-python
```

**Step 4: Lock exact versions**
```bash
# Freeze exact versions after successful rollback
pip freeze > requirements-rollback-$(date +%Y%m%d).txt

# Document in incident report
echo "Dependency state after rollback:" > rollback-report.md
pip freeze >> rollback-report.md
```

#### Dependency Conflict Resolution

If dependency conflicts occur during rollback:

```bash
# Option 1: Rebuild the governed core environment
make repair-core-venv

# Option 2: Use a rollback constraints file for the editable package install
.venv/bin/python -m pip install --constraint requirements-rollback-YYYYMMDD.txt -e .

# Option 3: Regenerate and reinstall the governed core lane
make install-core
```

**Known Compatibility Issues**:
- PyTorch versions must match CUDA version (if GPU enabled)
- Transformers version must be compatible with PyTorch version
- Pillow-SIMD conflicts with standard Pillow (only one can be installed)

### Data Migration Rollback

**Current State (as of v2.2.0)**: The Transformation Portal is **stateless** with respect to persistent data:
- No database (SQLite, PostgreSQL, etc.)
- No schema migrations
- Outputs are self-contained artifacts (images, JSON metadata, provenance logs)

**Therefore**: Data migration rollback is **not currently required**.

#### Future Considerations

If future versions introduce stateful components, this section must be updated with:

**Database Rollback Procedure** (template for future):
```bash
# Example: If SQLite database is added in future versions

# Backup current database
cp data/transformation_portal.db data/transformation_portal.db.backup-$(date +%Y%m%d)

# Apply down-migration to rollback schema
alembic downgrade -1  # Roll back one migration
# or
alembic downgrade <target-revision>  # Roll back to specific revision

# Verify schema version
alembic current
```

#### Output Artifact Compatibility

**Scenario**: Rolling back from v2.2.0 to v2.1.0 when metadata schema changed.

**Example**: v2.2.0 added new provenance fields (`hardware_acceleration`, `checkpoint_hash`).

**Impact Assessment**:
- **v2.1.0 reading v2.2.0 outputs**: Older code ignores unknown fields (graceful degradation)
- **v2.2.0 reading v2.1.0 outputs**: Newer code handles missing fields with defaults

**Action Required**: None, schemas are forward/backward compatible.

**If incompatibility detected**:
```bash
# Archive outputs from newer version
mkdir -p archive/output-v2.2.0-$(date +%Y%m%d)/
mv output_*/ archive/output-v2.2.0-$(date +%Y%m%d)/

# Verify older version can process fresh inputs
.venv/bin/python -m transformation_portal.cli depth-estimation \
  --input tests/fixtures/test_image.jpg \
  --output /tmp/verify-rollback/
```

#### Checkpoint and Model Weights Rollback

**Scenario**: Model weights changed between versions (e.g., Depth model updated).

**Procedure**:
```bash
# Check current checkpoint version
ls -lh weights/
# depth_anything_v2_vitl.pth
# depth_pro.pt

# Restore archived weights from target version
# (Assumes weights are version-controlled or archived externally)
cp archive/weights-v2.1.0/depth_pro.pt weights/
cp archive/weights-v2.1.0/depth_anything_v2_vitl.pth weights/

# Verify checkpoint compatibility
python scripts/verify_checkpoint.py --model depth_pro --version v2.1.0

# Update configuration if needed
# Edit config/depth_config.yaml:
# checkpoint_path: weights/depth_pro.pt
# checkpoint_version: v2.1.0
```

**Critical**: Model weights are typically NOT version-controlled (too large). Establish external archival:
- Cloud storage (S3, GCS) with versioning enabled
- DVC (Data Version Control) for large files
- Model registry (MLflow, Weights & Biases)

---

## Incident Response Integration

### Incident Lifecycle Integration

Rollback procedures integrate with the incident response lifecycle:

```
Detection → Assessment → Containment → Rollback → Recovery → Post-Mortem
    ↓           ↓            ↓            ↓          ↓           ↓
Alerting   Severity    Communication  Execute    Verify     Root Cause
Monitoring  Analysis   Stakeholder    This       Testing    Analysis
                       Notice         Document              + Prevention
```

### Escalation Matrix

| Severity | Criteria | Decision Authority | Response Time | Notification |
|----------|----------|-------------------|---------------|--------------|
| **P0 (Critical)** | • Data corruption<br>• Security breach<br>• System unusable | On-Call Engineer | Immediate (<5 min) | Slack + PagerDuty + Email (all) |
| **P1 (High)** | • Major feature broken<br>• Error rate >10%<br>• Perf degradation >50% | Platform Lead | <15 minutes | Slack + Email (team) |
| **P2 (Medium)** | • Minor feature broken<br>• Error rate 5-10%<br>• Perf degradation 20-50% | Engineering Director | <1 hour | Slack (team) |
| **P3 (Low)** | • Cosmetic issues<br>• Error rate <5%<br>• Non-critical bugs | Product Owner | Next business day | Issue tracker |

### Communication Templates

#### Rollback Initiation Notice

```
Subject: [INCIDENT-XXX] [P0] Emergency Rollback - Transformation Portal

Team,

We are executing an emergency rollback of the Transformation Portal due to [brief description of issue].

**Rollback Details:**
- Current Version: v2.2.0 (failing)
- Target Version: v2.1.0 (last known good)
- Incident ID: INC-2026-02-10-001
- Severity: P0 (Critical)

**Timeline:**
- Issue Detected: 2026-02-10 14:35 UTC
- Rollback Initiated: 2026-02-10 14:40 UTC
- Expected Completion: 2026-02-10 15:10 UTC (+30min)
- Verification Complete: 2026-02-10 15:25 UTC (+45min)

**Impact:**
- Users: All production users affected
- Functionality: [Describe what's broken/unavailable]
- Workaround: [If any temporary workaround exists]

**Actions Required:**
- Users: Please pause new processing runs until rollback is verified
- Team: Monitor #transformation-portal-incidents for updates

Status updates will be posted every 15 minutes.

[On-Call Engineer Name]
Incident Commander
```

#### Rollback Completion Notice

```
Subject: [INCIDENT-XXX] [RESOLVED] Rollback Complete - v2.1.0 Restored

Team,

Rollback to v2.1.0 is complete and verified. The system is operational.

**Summary:**
- Rollback Completed: 2026-02-10 15:08 UTC
- Verification Status: ✅ PASS (all smoke tests passed)
- Current Version: v2.1.0 (stable)
- Total Downtime: 33 minutes

**Verification Results:**
- ✅ Smoke tests: 12/12 passed
- ✅ Integration tests: 47/47 passed
- ✅ Performance: Within baseline (p95: 1.8s)
- ✅ Error rate: <0.1% (normal)

**Next Steps:**
1. Root cause analysis: [Link to incident doc]
2. Fix development: Targeting v2.2.1 by [date]
3. Post-mortem: Scheduled for [date/time]

**Affected Users:**
- Total impact: ~50 processing runs queued during rollback
- Retry instructions: [Link to docs or instructions]

Thank you for your patience during this incident.

[On-Call Engineer Name]
Incident Commander
```

---

## Post-Rollback Checklist

After completing rollback, verify the following:

### Immediate Verification (15 minutes)

- [ ] **Version Check**: Confirm correct version deployed
  ```bash
  python -c "import transformation_portal; print(transformation_portal.__version__)"
  ```

- [ ] **Smoke Tests Pass**: Core functionality works
  ```bash
  pytest tests/test_core_functionality.py -v --maxfail=1
  ```

- [ ] **Service Health**: Endpoints/APIs respond correctly
  ```bash
  curl http://localhost:8000/health  # Or equivalent
  ```

- [ ] **Error Rate**: Monitor logs for 15 minutes
  ```bash
  tail -f /var/log/transformation-portal/app.log | grep ERROR
  ```

- [ ] **Resource Usage**: Memory/CPU within normal range
  ```bash
  top -p $(pgrep -f transformation-portal)
  # Or: docker stats transformation-portal
  ```

### Extended Monitoring (2 hours)

- [ ] **Error Rate <1%**: Monitor application logs
- [ ] **Performance Baseline**: Latency back to normal (p95 <2s)
- [ ] **No Memory Leaks**: Memory usage stable over time
- [ ] **User Reports**: No increase in support tickets
- [ ] **Batch Jobs**: Scheduled tasks complete successfully

### Communication & Documentation

- [ ] **Incident Report**: Document timeline and root cause
- [ ] **Stakeholder Notification**: Inform product, QA, users (if external)
- [ ] **Postmortem Scheduled**: Plan root cause analysis meeting
- [ ] **Hotfix Plan**: Create plan for v2.0.1 with fix
- [ ] **Rollback Log**: Update this document with lessons learned

---

## Rollback Testing (Staging)

**Perform rollback dry-run in staging quarterly** to ensure procedures work.

### Dry-Run Checklist

```bash
# 1. Deploy v2.0.0 to staging
git checkout v2.0.0
pip install -e .

# 2. Simulate issue (e.g., inject error)
# ... (manual simulation)

# 3. Execute rollback procedure
git checkout v1.9.0
pip install --force-reinstall -e .

# 4. Verify rollback successful
pytest tests/ -v -m "not slow"

# 5. Document dry-run results
echo "Dry-run completed on $(date)" >> docs/deployment/ROLLBACK_HISTORY.md
```

---

## Contact Information

### Escalation Path

| Role | Contact | Availability |
|------|---------|--------------|
| **DevOps Lead** | devops@example.com | 24/7 (on-call) |
| **Architect** | architect@example.com | Business hours + urgent |
| **Product Owner** | product@example.com | Business hours |
| **QA Lead** | qa@example.com | Business hours |

### Emergency Contacts

- **Slack Channel**: `#transformation-portal-incidents`
- **PagerDuty**: (if configured)
- **On-Call Rotation**: (link to rotation schedule)

---

## Appendix A: Rollback Decision Matrix

| Symptom | Severity | Recommended Action | Response Time |
|---------|----------|-------------------|---------------|
| Error rate >10% | Critical | Full rollback (Strategy 1) | Immediate (<5 min) |
| Error rate 5-10% | High | Feature flag disable (Strategy 2) | <10 minutes |
| Error rate 2-5% | Medium | Monitor + hotfix plan | <30 minutes |
| Single feature broken | Medium | Feature flag or hot patch | <20 minutes |
| Performance degradation >50% | High | Full rollback | <15 minutes |
| Memory leak causing OOMs | Critical | Full rollback or container restart | Immediate |
| Security vulnerability | Critical | Full rollback + incident response | Immediate |
| Minor UI bug | Low | Document + plan hotfix | Next business day |

---

## Appendix B: Rollback Script Template

```bash
#!/bin/bash
# rollback.sh - Automated rollback script

set -euo pipefail

ROLLBACK_VERSION="${1:-v1.9.0}"
REPO_PATH="${2:-/opt/transformation-portal}"

echo "🔄 Initiating rollback to $ROLLBACK_VERSION"

# 1. Checkout target version
cd "$REPO_PATH"
git fetch --tags
git checkout "$ROLLBACK_VERSION"

# 2. Reinstall package
pip install --force-reinstall -e .

# 3. Restart service
sudo systemctl restart transformation-portal

# 4. Wait for service to be ready
sleep 10

# 5. Smoke tests
pytest tests/test_core_functionality.py -v --maxfail=1

# 6. Verify version
DEPLOYED_VERSION=$(python -c "import transformation_portal; print(transformation_portal.__version__)")
echo "✅ Rollback complete. Current version: $DEPLOYED_VERSION"

# 7. Monitor for 5 minutes
echo "📊 Monitoring logs for 5 minutes..."
timeout 300 tail -f /var/log/transformation-portal/app.log || true

echo "🎉 Rollback verified. Please continue manual monitoring."
```

**Usage**:
```bash
chmod +x rollback.sh
./rollback.sh v1.9.0 /opt/transformation-portal
```

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.1.0 | 2026-02-10 | Added PyPI rollback, dependency pinning, data migration, incident response integration | Architect |
| 2.0.0 | 2026-02-01 | Enhanced with multiple strategies, verification procedures | Architect |
| 1.0.0 | 2026-02-01 | Initial rollback procedures | Architect |

---

**Next Review Date**: 2026-03-01
**Document Status**: Active
**Approval**: Pending (DevOps Lead, Architect, QA Lead)
