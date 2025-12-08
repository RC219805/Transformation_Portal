# Phase 2: Hardening Expansion & Automation Pack

**Status**: 🚀 **IN PROGRESS**  
**Version**: 1.0  
**Date**: 2025-12-08  
**Author**: Transformation Portal Architect  
**Phase 1 Dependency**: PR #519 (Architecture Hardening Pack), PR #525 (Observability Pack)

---

## Executive Summary

Phase 2 builds on the security, reproducibility, and observability foundation established in Phase 1 by:

1. **Expanding hardening coverage** beyond lux_depth_v2 to legacy pipelines and batch processors
2. **Automating security remediation** with CI/CD workflows for dependency updates
3. **Production deployment readiness** with runbooks, Docker hardening, and health checks
4. **Advanced monitoring** with Grafana dashboards and alert rules
5. **Adoption tooling** to migrate legacy code to hardened patterns

**Key Deliverables**:
- Universal hardening wrapper for any pipeline
- Automated dependency security workflow
- Production deployment guides and runbooks
- Grafana dashboards for observability metrics
- Legacy pipeline migration utilities
- Comprehensive integration tests

---

## Phase 1 Recap

### What Phase 1 Delivered (December 8, 2025)

**PR #519: Architecture Hardening Pack**
- ✅ Security: Input validation, CVE enforcement, path protection
- ✅ Reproducibility: Report stamping, config hashing, git tracking
- ✅ Performance: Stage profiling (<5% overhead)
- ✅ CI/CD: Dependency ban guard, security scanning

**PR #525: Observability Pack**
- ✅ Prometheus /metrics endpoint
- ✅ JSON structured logging
- ✅ Request correlation (X-Request-ID)
- ✅ HTTP and pipeline metrics

**Coverage**: lux_depth_v2 only (hardened, observable, production-ready)

**Gaps Identified**:
- ❌ Legacy pipelines still unprotected
- ❌ No automated security remediation
- ❌ Missing production deployment guides
- ❌ Observability metrics not visualized
- ❌ No migration path for legacy code

---

## Phase 2 Architecture

### Goal 1: Universal Hardening Framework

**Problem**: Phase 1 hardening is tightly coupled to lux_depth_v2.

**Solution**: Extract generic hardening layer that works with any pipeline.

#### Generic Pipeline Wrapper

```python
# transformation_portal/hardening/universal.py

from typing import Any, Callable, Dict, Protocol
from pathlib import Path
import time
import hashlib
import json

class Pipeline(Protocol):
    """Protocol for any pipeline."""
    def process(self, input_path: Path, **kwargs) -> Any:
        ...

class UniversalHardenedWrapper:
    """Universal hardening wrapper for any pipeline."""
    
    def __init__(
        self,
        pipeline: Pipeline,
        policy: 'HardeningPolicy',
        enable_profiling: bool = True,
        enable_stamping: bool = True
    ):
        self.pipeline = pipeline
        self.policy = policy
        self.enable_profiling = enable_profiling
        self.enable_stamping = enable_stamping
    
    def process(self, input_path: Path, **kwargs) -> Dict[str, Any]:
        """Process with hardening applied."""
        # 1. Input validation
        validated_path = self.policy.validate_input(input_path)
        
        # 2. Profiling start
        if self.enable_profiling:
            start = time.perf_counter()
        
        # 3. Execute wrapped pipeline
        try:
            result = self.pipeline.process(validated_path, **kwargs)
        except Exception as e:
            # Log and re-raise with context
            self._log_failure(input_path, e)
            raise
        
        # 4. Profiling end
        if self.enable_profiling:
            duration_ms = (time.perf_counter() - start) * 1000
        else:
            duration_ms = None
        
        # 5. Report stamping
        if self.enable_stamping:
            report = self._create_report(
                input_path=input_path,
                result=result,
                duration_ms=duration_ms,
                kwargs=kwargs
            )
            return {"result": result, "report": report}
        
        return {"result": result}
    
    def _create_report(self, input_path, result, duration_ms, kwargs):
        """Create reproducibility report."""
        from lux_depth_v2.hardening.stamping import stamp_report
        from lux_depth_v2.hardening.runtime import gather_runtime_info
        
        config_hash = hashlib.sha256(
            json.dumps(kwargs, sort_keys=True).encode()
        ).hexdigest()
        
        return stamp_report(
            report={},
            config_hash=config_hash,
            runtime_info=gather_runtime_info(),
            include_input_hash=False
        )
    
    def _log_failure(self, input_path, exception):
        """Log processing failure."""
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"Pipeline failed: {input_path}",
            extra={
                "input_path": str(input_path),
                "exception_type": type(exception).__name__,
                "exception_msg": str(exception)
            }
        )
```

#### Retrofit Utility for Legacy Pipelines

```python
# transformation_portal/hardening/retrofit.py

from pathlib import Path
from typing import Optional
import inspect

def retrofit_pipeline(
    pipeline_module: str,
    policy_path: Optional[Path] = None
):
    """Retrofit hardening onto legacy pipeline."""
    import importlib
    
    # Load legacy pipeline
    module = importlib.import_module(pipeline_module)
    
    # Find process function
    process_fn = None
    for name, obj in inspect.getmembers(module):
        if callable(obj) and name.startswith("process"):
            process_fn = obj
            break
    
    if not process_fn:
        raise ValueError(f"No process function found in {pipeline_module}")
    
    # Load policy
    from lux_depth_v2.hardening.policy import HardeningPolicy
    policy = HardeningPolicy.load(policy_path)
    
    # Create wrapper adapter
    class LegacyAdapter:
        def process(self, input_path: Path, **kwargs):
            return process_fn(input_path, **kwargs)
    
    # Wrap with hardening
    from transformation_portal.hardening.universal import UniversalHardenedWrapper
    return UniversalHardenedWrapper(
        pipeline=LegacyAdapter(),
        policy=policy
    )
```

### Goal 2: Automated Security Remediation

**Problem**: Manual dependency updates are slow and error-prone.

**Solution**: Automated workflow to detect, patch, and test security issues.

#### Security Auto-Remediation Workflow

```yaml
# .github/workflows/security-auto-remediation.yml

name: Security Auto-Remediation

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 02:00 UTC
  workflow_dispatch:     # Manual trigger

permissions:
  contents: write
  pull-requests: write
  security-events: write

jobs:
  detect-vulnerabilities:
    name: Detect Security Issues
    runs-on: ubuntu-latest
    outputs:
      has_vulns: ${{ steps.audit.outputs.has_vulns }}
      vuln_report: ${{ steps.audit.outputs.report }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Run pip-audit
        id: audit
        continue-on-error: true
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.txt \
            --requirement requirements-dev.txt \
            --format json \
            --output audit-report.json \
            --vulnerability-service osv
          
          if [ $? -eq 0 ]; then
            echo "has_vulns=false" >> $GITHUB_OUTPUT
          else
            echo "has_vulns=true" >> $GITHUB_OUTPUT
            cat audit-report.json | jq -r '.vulnerabilities[].id' > vuln-ids.txt
            echo "report<<EOF" >> $GITHUB_OUTPUT
            cat audit-report.json >> $GITHUB_OUTPUT
            echo "EOF" >> $GITHUB_OUTPUT
          fi
      
      - name: Upload audit report
        uses: actions/upload-artifact@v4
        if: steps.audit.outputs.has_vulns == 'true'
        with:
          name: audit-report
          path: audit-report.json
  
  auto-patch:
    name: Attempt Auto-Patch
    needs: detect-vulnerabilities
    if: needs.detect-vulnerabilities.outputs.has_vulns == 'true'
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Download audit report
        uses: actions/download-artifact@v4
        with:
          name: audit-report
      
      - name: Attempt automated patching
        id: patch
        run: |
          # Install pip-audit and pur (pip update requirements)
          pip install pip-audit pur
          
          # Try to upgrade vulnerable packages
          pur -r requirements.txt --patch --skip basicsr,realesrgan,gfpgan
          pur -r requirements-dev.txt --patch
          
          # Verify fixes
          pip install -r requirements.txt -r requirements-dev.txt
          pip-audit --requirement requirements.txt \
            --requirement requirements-dev.txt || true
          
          # Check if git diff exists
          if git diff --quiet; then
            echo "patched=false" >> $GITHUB_OUTPUT
          else
            echo "patched=true" >> $GITHUB_OUTPUT
          fi
      
      - name: Run tests
        if: steps.patch.outputs.patched == 'true'
        run: |
          pip install pytest pytest-xdist
          pytest tests/ -n auto --tb=short -v
      
      - name: Create PR
        if: steps.patch.outputs.patched == 'true'
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'security: Auto-patch vulnerabilities'
          title: '[Security] Automated Dependency Patch'
          body: |
            ## Security Auto-Remediation
            
            This PR was automatically created to patch security vulnerabilities detected by pip-audit.
            
            ### Vulnerabilities Addressed
            ${{ needs.detect-vulnerabilities.outputs.vuln_report }}
            
            ### Changes Made
            - Updated dependencies to patched versions
            - Verified with pip-audit
            - Ran test suite
            
            **Action Required**: Manual review recommended before merge.
          branch: security/auto-patch-${{ github.run_number }}
          labels: security,automated
```

### Goal 3: Production Deployment Readiness

**Deliverable**: Comprehensive production deployment guide and tooling.

#### Production Deployment Runbook

```markdown
# docs/operations/PRODUCTION_DEPLOYMENT_RUNBOOK.md

# Production Deployment Runbook

## Pre-Deployment Checklist

### Security
- [ ] All Dependabot alerts resolved
- [ ] Security scan passing (Bandit, pip-audit)
- [ ] Secrets properly configured (env vars, not hardcoded)
- [ ] CVE-2024-27763 enforcement verified
- [ ] TLS/SSL certificates valid

### Configuration
- [ ] Environment variables documented
- [ ] HardeningPolicy configuration reviewed
- [ ] Output directories have correct permissions
- [ ] Log rotation configured
- [ ] Metrics endpoint accessible

### Infrastructure
- [ ] Docker image built and scanned
- [ ] Resource limits configured (memory, CPU)
- [ ] Health check endpoint responding
- [ ] Prometheus scraping configured
- [ ] Log aggregation configured

### Testing
- [ ] All CI/CD tests passing
- [ ] Smoke tests run against production config
- [ ] Load testing completed
- [ ] Rollback procedure tested

## Deployment Steps

### 1. Build Docker Image

```bash
docker build -t transformation-portal:latest \
  --build-arg PYTHON_VERSION=3.11 \
  --target production \
  .

# Scan for vulnerabilities
docker scan transformation-portal:latest
```

### 2. Configure Environment

```bash
# Create production .env file
cat > .env.production <<EOF
# Hardening Policy
LUX_HARDEN_MAX_INPUT_BYTES=100000000
LUX_HARDEN_ENABLE_RATE_LIMIT=true
LUX_HARDEN_STAMP_INCLUDE_INPUT_HASH=false

# Observability
LUX_METRICS_ENABLED=1
LUX_METRICS_TOKEN=<SECURE_TOKEN>
LUX_LOG_FORMAT=json
LUX_LOG_LEVEL=INFO

# Service
LUX_OUTPUT_DIR=/data/output
LUX_PORT=8088
LUX_WORKERS=4
EOF
```

### 3. Deploy with Docker Compose

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  lux-depth-service:
    image: transformation-portal:latest
    container_name: lux-depth-service
    restart: unless-stopped
    
    env_file:
      - .env.production
    
    ports:
      - "8088:8088"
    
    volumes:
      - ./data/input:/data/input:ro
      - ./data/output:/data/output
      - ./logs:/app/logs
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8088/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    resources:
      limits:
        memory: 8G
        cpus: '4.0'
      reservations:
        memory: 4G
        cpus: '2.0'
    
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
  
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    
    ports:
      - "9090:9090"
    
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
  
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    
    ports:
      - "3000:3000"
    
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=<SECURE_PASSWORD>
    
    volumes:
      - ./config/grafana-dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./config/grafana-datasources:/etc/grafana/provisioning/datasources:ro
      - grafana-data:/var/lib/grafana

volumes:
  prometheus-data:
  grafana-data:
```

### 4. Start Services

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Start services
docker-compose -f docker-compose.production.yml up -d

# Verify health
curl http://localhost:8088/health
curl http://localhost:8088/metrics
```

### 5. Smoke Test

```bash
# Run smoke test
python scripts/smoke_test_production.py \
  --endpoint http://localhost:8088 \
  --input tests/fixtures/sample.tif
```

## Monitoring & Alerts

### Prometheus Metrics

Key metrics to monitor:
- `lux_http_request_duration_seconds{quantile="0.95"}` - p95 latency
- `lux_http_requests_total` - Request rate
- `lux_http_inflight_requests` - Current load
- `lux_exceptions_total` - Error rate

### Alert Rules

```yaml
# config/prometheus-alerts.yml
groups:
  - name: lux_depth_alerts
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, lux_http_request_duration_seconds_bucket) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p95 latency: {{ $value }}s"
      
      - alert: HighErrorRate
        expr: rate(lux_exceptions_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate {{ $value }}/s"
      
      - alert: ServiceDown
        expr: up{job="lux-depth-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
```

## Rollback Procedure

### Emergency Rollback

```bash
# Stop current version
docker-compose -f docker-compose.production.yml down

# Restore previous version
docker tag transformation-portal:previous transformation-portal:latest

# Restart
docker-compose -f docker-compose.production.yml up -d

# Verify
curl http://localhost:8088/health
```

## Troubleshooting

### Service Won't Start
1. Check logs: `docker-compose logs lux-depth-service`
2. Verify environment variables: `docker-compose config`
3. Check file permissions: `/data/output` must be writable

### High Memory Usage
1. Check metrics: `curl http://localhost:8088/metrics | grep memory`
2. Reduce batch size in configuration
3. Restart service to clear cache

### Slow Response Times
1. Check p95 latency in Grafana
2. Review stage timings in metrics
3. Enable profiling to identify bottlenecks
```

#### Docker Hardening

```dockerfile
# Dockerfile.production

FROM python:3.11-slim AS base

# Security: Non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Security: Minimal system packages
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-repo.txt ./
RUN pip install --no-cache-dir -r requirements-repo.txt

# Copy application
COPY --chown=appuser:appuser lux_depth_v2/ ./lux_depth_v2/
COPY --chown=appuser:appuser transformation_portal/ ./transformation_portal/

# Security: Switch to non-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8088/health || exit 1

# Expose ports
EXPOSE 8088

# Run service
CMD ["python", "-m", "lux_depth_v2.service", "--output-dir", "/data/output", "--service", "--port", "8088"]
```

### Goal 4: Grafana Dashboards

**Deliverable**: Pre-built Grafana dashboards for observability metrics.

#### Main Dashboard JSON

```json
{
  "title": "Transformation Portal - Lux Depth V2",
  "panels": [
    {
      "title": "Request Rate",
      "targets": [
        {
          "expr": "rate(lux_http_requests_total[5m])"
        }
      ]
    },
    {
      "title": "P95 Latency",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, lux_http_request_duration_seconds_bucket)"
        }
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {
          "expr": "rate(lux_exceptions_total[5m])"
        }
      ]
    },
    {
      "title": "Pipeline Stage Breakdown",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, lux_pipeline_stage_duration_seconds_bucket)"
        }
      ]
    }
  ]
}
```

### Goal 5: Integration Tests

**Deliverable**: End-to-end integration tests for hardened pipelines.

#### Integration Test Suite

```python
# tests/integration/test_hardened_pipeline_e2e.py

import pytest
from pathlib import Path
import requests
import time

@pytest.mark.integration
def test_hardened_service_e2e():
    """Test full hardening + observability stack."""
    base_url = "http://localhost:8088"
    
    # 1. Health check
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    
    # 2. Metrics endpoint
    response = requests.get(f"{base_url}/metrics")
    assert response.status_code == 200
    assert "lux_http_requests_total" in response.text
    
    # 3. Process request with correlation
    test_image = Path("tests/fixtures/sample.tif")
    request_id = "test-e2e-12345"
    
    with open(test_image, "rb") as f:
        response = requests.post(
            f"{base_url}/v2/process",
            files={"file": f},
            headers={"X-Request-ID": request_id},
            params={"preset": "INTERIOR_LUXURY"}
        )
    
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert "X-Elapsed-MS" in response.headers
    
    # 4. Verify report stamping
    result = response.json()
    assert "meta" in result
    assert "run_id" in result["meta"]
    assert "config_hash" in result["meta"]
    assert "git_commit" in result["meta"]
    
    # 5. Verify metrics updated
    time.sleep(1)
    response = requests.get(f"{base_url}/metrics")
    metrics_text = response.text
    assert f'route="/v2/process"' in metrics_text

@pytest.mark.integration
def test_universal_hardening_wrapper():
    """Test universal hardening works with legacy pipeline."""
    from transformation_portal.hardening.universal import UniversalHardenedWrapper
    from lux_depth_v2.hardening.policy import HardeningPolicy
    
    # Create mock legacy pipeline
    class LegacyPipeline:
        def process(self, input_path: Path, **kwargs):
            return {"processed": True}
    
    # Wrap with hardening
    policy = HardeningPolicy.load()
    wrapped = UniversalHardenedWrapper(
        pipeline=LegacyPipeline(),
        policy=policy
    )
    
    # Process
    result = wrapped.process(Path("tests/fixtures/sample.tif"))
    
    # Verify hardening applied
    assert "report" in result
    assert "meta" in result["report"]
    assert "run_id" in result["report"]["meta"]
```

---

## Implementation Timeline

### Week 1: Core Infrastructure (Dec 8-15)
- [ ] Universal hardening wrapper
- [ ] Retrofit utility for legacy pipelines
- [ ] Security auto-remediation workflow
- [ ] Unit tests for new components

### Week 2: Production Readiness (Dec 16-22)
- [ ] Production deployment runbook
- [ ] Docker hardening (Dockerfile.production)
- [ ] docker-compose.production.yml
- [ ] Prometheus alert rules
- [ ] Grafana dashboards

### Week 3: Integration & Testing (Dec 23-29)
- [ ] End-to-end integration tests
- [ ] Smoke test production deployment
- [ ] Performance benchmarking
- [ ] Documentation updates

### Week 4: Migration & Adoption (Dec 30 - Jan 5)
- [ ] Legacy pipeline migration guide
- [ ] Example migrations
- [ ] Training materials
- [ ] PR submission

---

## Success Criteria

### Technical
- ✅ Universal hardening works with any pipeline
- ✅ Auto-remediation workflow creates PRs successfully
- ✅ Docker production deployment < 5min
- ✅ Grafana dashboards display all metrics
- ✅ Integration tests pass 100%

### Operational
- ✅ Production deployment runbook complete
- ✅ Alert rules configured
- ✅ Rollback procedure tested
- ✅ Zero downtime deployment verified

### Adoption
- ✅ At least 1 legacy pipeline migrated
- ✅ Migration guide with examples
- ✅ Developer training completed

---

## Risk Mitigation

### Medium Risks
- **Docker security**: Mitigated with non-root user, minimal packages
- **Alert fatigue**: Mitigated with proper thresholds
- **Migration complexity**: Mitigated with retrofit utility

### Rollback Strategy
- Universal hardening is opt-in (feature flag)
- Docker image tagged with version
- Quick rollback via docker tag swap

---

## Deliverables Summary

1. **Code**:
   - `transformation_portal/hardening/universal.py` - Universal wrapper
   - `transformation_portal/hardening/retrofit.py` - Legacy retrofit
   - `.github/workflows/security-auto-remediation.yml` - Auto-patching
   - `Dockerfile.production` - Hardened Docker image
   - `docker-compose.production.yml` - Full stack

2. **Documentation**:
   - `docs/operations/PRODUCTION_DEPLOYMENT_RUNBOOK.md`
   - `docs/operations/GRAFANA_DASHBOARDS.md`
   - `docs/operations/MIGRATION_GUIDE.md`

3. **Configuration**:
   - `config/prometheus-alerts.yml` - Alert rules
   - `config/grafana-dashboards/*.json` - Pre-built dashboards
   - `.env.production.example` - Production env template

4. **Tests**:
   - `tests/integration/test_hardened_pipeline_e2e.py`
   - `tests/integration/test_production_deployment.py`
   - `scripts/smoke_test_production.py`

---

**Status**: Ready for Implementation  
**Next Review**: 2025-12-22 (2-week milestone)
