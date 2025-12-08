# Phase 2 Production Deployment Guide

**Transformation Portal - Lux Depth V2 Production Stack**  
**Last Updated**: December 8, 2025  
**Status**: Production Ready

---

## Overview

Phase 2 introduces the production-ready Lux Depth V2 pipeline with:

- **GPU-accelerated depth processing** - Depth Anything V2 with Apple Neural Engine optimization
- **Advanced material segmentation** - ONNX, SegFormer, and heuristic backends
- **FastAPI service mode** - RESTful API with health checks and metrics
- **Security hardening** - CVE-2024-27763 mitigation, input validation, rate limiting
- **Production monitoring** - Prometheus metrics, Grafana dashboards
- **Docker containerization** - Multi-stage builds with non-root users

---

## Quick Start

### 1. Standard Deployment (docker-compose)

```bash
# Clone repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Copy environment template
cp deployment/.env.production.example .env.production

# Edit environment variables
nano .env.production  # Set secure passwords and tokens

# Start Lux Depth V2 service
docker-compose up -d lux-depth-v2-service

# Check service health
curl http://localhost:8088/health

# Process an image via API
curl -X POST http://localhost:8088/v2/process \
  -F "image=@input.jpg" \
  -F "preset=interior_luxury"
```

### 2. Production Stack (with monitoring)

```bash
# Start full production stack
docker-compose -f deployment/docker-compose.production.yml up -d

# Services:
# - lux-depth-service: Main processing service (port 8088)
# - prometheus: Metrics collection (port 9090)
# - grafana: Visualization dashboards (port 3000)

# Access Grafana
# URL: http://localhost:3000
# Default credentials: admin / <see .env.production>
```

### 3. GPU-Enabled Deployment

```bash
# Ensure NVIDIA Docker runtime is installed
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Start GPU service
docker-compose up -d lux-depth-v2-gpu

# Verify GPU access
docker exec lux-depth-v2-gpu python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Architecture

### Service Topology

```
┌─────────────────────────────────────────────────────┐
│                 Load Balancer (nginx)               │
│                  (HTTPS termination)                │
└───────────────────┬─────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Lux V2  │   │ Lux V2  │   │ Lux V2  │
│ Service │   │ Service │   │ Service │
│ (CPU)   │   │ (GPU)   │   │ Worker  │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     └──────┬──────┴──────┬──────┘
            │             │
            ▼             ▼
      ┌──────────┐  ┌──────────┐
      │Prometheus│  │  Grafana │
      └──────────┘  └──────────┘
```

### Component Responsibilities

| Component | Purpose | Port | Health Check |
|-----------|---------|------|--------------|
| **lux-depth-v2-service** | Main API service | 8088 | `/health` |
| **lux-depth-v2-gpu** | GPU-accelerated service | 8089 | `/health` |
| **lux-depth-v2-worker** | Batch processing | - | N/A |
| **prometheus** | Metrics collection | 9090 | `/-/healthy` |
| **grafana** | Visualization | 3000 | `/api/health` |

---

## Security Configuration

### 1. CVE-2024-27763 Mitigation

**Issue**: Command injection vulnerability in basicsr ≤1.4.x  
**CVSS**: 9.8 (Critical)

**Mitigation Status**: ✅ **RESOLVED**

All Docker images use `lux_depth_v2/requirements-repo.txt` which:
- Excludes `basicsr`, `realesrgan`, and `gfpgan`
- Uses safe alternative upscaling backends (torch, ONNX)
- Includes security verification in build process

**Verification**:
```bash
# Check that basicsr is NOT present in container
docker exec lux-depth-v2-service python -c "import basicsr"
# Expected output: ImportError (this is correct)

# Run automated security check
docker exec lux-depth-v2-service python scripts/utilities/verify_no_basicsr_imports.py --check-pkg
# Expected output: OK: basicsr is not importable
```

### 2. Input Validation

All file paths and user inputs are validated before processing:

```python
# Implemented in lux_depth_v2/service.py
- Path traversal prevention
- Symlink attack protection
- File size limits (100MB default)
- MIME type validation
```

### 3. Rate Limiting

API endpoints are rate-limited to prevent abuse:

```bash
# Configure via environment variables
LUX_HARDEN_ENABLE_RATE_LIMIT=true  # Enable rate limiting
LUX_HARDEN_MAX_REQUESTS_PER_MINUTE=10  # 10 requests/minute per IP
```

### 4. Authentication (Optional - Recommended for Production)

```bash
# Set API key in environment
export LUX_DEPTH_API_KEY="your-secure-key-here"

# Restart service
docker-compose restart lux-depth-v2-service

# Use API key in requests
curl -X POST http://localhost:8088/v2/process \
  -H "X-API-Key: your-secure-key-here" \
  -F "image=@input.jpg"
```

### 5. HTTPS/TLS

**Production deployments MUST use HTTPS.**

**Option A: Reverse Proxy (Recommended)**
```nginx
# /etc/nginx/sites-available/lux-depth-v2
server {
    listen 443 ssl http2;
    server_name lux-depth-api.example.com;

    ssl_certificate /etc/ssl/certs/lux-depth.crt;
    ssl_certificate_key /etc/ssl/private/lux-depth.key;

    location / {
        proxy_pass http://localhost:8088;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Option B: Direct TLS (Development Only)**
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run uvicorn with TLS
docker run -p 8088:8088 \
  -v $(pwd)/key.pem:/app/key.pem \
  -v $(pwd)/cert.pem:/app/cert.pem \
  transformation-portal:lux-depth-v2 \
  uvicorn lux_depth_v2.service:app \
    --host 0.0.0.0 --port 8088 \
    --ssl-keyfile /app/key.pem \
    --ssl-certfile /app/cert.pem
```

---

## Environment Configuration

### Required Variables

```bash
# Service
LUX_OUTPUT_DIR=/data/output  # Output directory path

# Security
LUX_HARDEN_ENABLE_RATE_LIMIT=true
LUX_HARDEN_MAX_INPUT_BYTES=100000000  # 100MB max file size

# Observability
LUX_METRICS_ENABLED=1
LUX_LOG_LEVEL=INFO
LUX_LOG_FORMAT=json
```

### Optional Variables

```bash
# Authentication
LUX_DEPTH_API_KEY=your-secret-key  # Enable API key auth

# Performance
LUX_DEVICE=cuda  # cpu, cuda, or mps
LUX_BATCH_SIZE=4  # Batch processing size
LUX_NUM_WORKERS=2  # Data loader workers

# Features
LUX_ENABLE_UPSCALING=true
LUX_UPSCALER_BACKEND=torch  # torch, onnx
LUX_MATERIAL_SEGMENTATION_BACKEND=onnx  # onnx, segformer, heuristic

# Monitoring
LUX_METRICS_TOKEN=your-prometheus-token
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=secure-password
```

---

## Monitoring & Observability

### Prometheus Metrics

Metrics endpoint: `http://localhost:8088/metrics`

**Key Metrics**:
- `lux_depth_requests_total` - Total requests processed
- `lux_depth_request_duration_seconds` - Processing latency
- `lux_depth_errors_total` - Error count by type
- `lux_depth_queue_size` - Processing queue depth
- `lux_depth_gpu_memory_bytes` - GPU memory usage (if applicable)

### Grafana Dashboards

Access: `http://localhost:3000` (default credentials: admin / see .env.production)

**Pre-configured Dashboards**:
1. **Service Overview** - Request rate, latency, error rate
2. **Resource Usage** - CPU, memory, GPU utilization
3. **Quality Metrics** - Output quality scores, validation results
4. **Security Events** - Rate limit violations, auth failures

**Import Dashboard**:
```bash
# Copy dashboard JSON to Grafana provisioning directory
cp config/production/grafana-dashboards/lux-depth-v2.json \
   deployment/config/production/grafana-dashboards/
```

### Log Aggregation

Logs are written to `/app/logs` inside containers and mounted to host `./logs`.

**View logs**:
```bash
# Service logs
docker logs lux-depth-v2-service

# Structured JSON logs (for parsing)
cat logs/lux-depth-v2.log | jq '.level, .message'

# Error logs only
docker logs lux-depth-v2-service 2>&1 | grep ERROR
```

---

## Performance Tuning

### Resource Allocation

**Recommended Specs**:
- **CPU Service**: 2-4 cores, 4-8GB RAM
- **GPU Service**: 4-8 cores, 8-16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Worker**: 2 cores, 2-4GB RAM per worker

**Configure Limits**:
```yaml
# docker-compose.yml
services:
  lux-depth-v2-service:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

### Throughput Optimization

**Single Image Processing**:
- CPU: 5-15 seconds per image (4K)
- GPU (CUDA): 2-5 seconds per image (4K)
- Apple Silicon (MPS): 3-8 seconds per image (4K)

**Batch Processing**:
- CPU: 127 images/hour (without upscaling)
- GPU: 400+ images/hour (with upscaling)

**Optimization Tips**:
1. Use GPU service for high-throughput workloads
2. Enable batch processing for 10+ images
3. Disable AI upscaling for faster processing (--no-upscale)
4. Use torch backend (fastest, secure) instead of ONNX
5. Tune batch size based on available memory

---

## Health Checks & Readiness

### Service Health Endpoint

```bash
# Check service health
curl http://localhost:8088/health

# Expected response:
{
  "status": "healthy",
  "version": "2.0.0",
  "device": "cuda",
  "uptime_seconds": 12345
}
```

### Docker Health Checks

Health checks run automatically every 30 seconds:

```bash
# Check health status
docker inspect lux-depth-v2-service | jq '.[0].State.Health.Status'

# View health check logs
docker inspect lux-depth-v2-service | jq '.[0].State.Health.Log'
```

### Readiness Probes (Kubernetes)

```yaml
# kubernetes/deployment.yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8088
  initialDelaySeconds: 40
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8088
  initialDelaySeconds: 20
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

---

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptom**: Container exits immediately

**Diagnosis**:
```bash
docker logs lux-depth-v2-service
```

**Common Causes**:
- Missing environment variables → Check `.env.production`
- Port conflict → Change `LUX_PORT` in environment
- Permission issues → Ensure volumes are writable

#### 2. GPU Not Detected

**Symptom**: Service falls back to CPU

**Diagnosis**:
```bash
docker exec lux-depth-v2-gpu python -c "import torch; print(torch.cuda.is_available())"
```

**Solution**:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 3. High Memory Usage

**Symptom**: Container OOM (out of memory) killed

**Solution**:
```bash
# Reduce batch size
docker-compose down
# Edit .env.production:
# LUX_BATCH_SIZE=1

# Increase memory limit
# Edit docker-compose.yml:
# memory: 16G

docker-compose up -d
```

#### 4. Security Scan Failing

**Symptom**: CI/CD security workflow fails

**Diagnosis**:
```bash
# Run local security check
python scripts/utilities/verify_no_basicsr_imports.py --check-pkg

# Check if vulnerable package is present
pip show basicsr
```

**Solution**:
```bash
# Ensure using repo dependencies
pip uninstall basicsr realesrgan gfpgan -y
pip install -r lux_depth_v2/requirements-repo.txt
```

---

## CI/CD Integration

### GitHub Actions Workflow

Security scanning runs automatically on:
- Push to `main` branch
- Pull requests
- Dependency file changes
- Daily scheduled scan

**Workflow**: `.github/workflows/security-scan.yml`

**Key Jobs**:
1. **quick-check** - Verifies basicsr is not present
2. **dependency-scan** - Scans for vulnerabilities with Safety
3. **docker-build** - Validates Docker images pass security checks
4. **update-knowledge-base** - Updates RAG security knowledge

### Manual Security Validation

```bash
# Run full security audit
make security-audit

# Or run individual checks:
python scripts/utilities/verify_no_basicsr_imports.py --check-pkg
safety check -r lux_depth_v2/requirements-repo.txt
bandit -r lux_depth_v2/ -ll
```

---

## Backup & Disaster Recovery

### Data Backup

```bash
# Backup output directory
tar -czf output-backup-$(date +%Y%m%d).tar.gz output/

# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/

# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/
```

### Container Recovery

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (CAUTION: deletes data)
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build
```

### Database Backup (if using persistent storage)

```bash
# Backup Prometheus data
docker exec prometheus tar -czf - /prometheus > prometheus-backup.tar.gz

# Backup Grafana dashboards
docker exec grafana tar -czf - /var/lib/grafana > grafana-backup.tar.gz
```

---

## Production Checklist

### Pre-Deployment

- [ ] Review and customize `.env.production`
- [ ] Set secure passwords for Grafana admin
- [ ] Set secure API key (`LUX_DEPTH_API_KEY`)
- [ ] Configure resource limits (CPU, memory)
- [ ] Set up HTTPS/TLS (via reverse proxy or direct)
- [ ] Configure firewall rules (restrict ports 8088, 8089)
- [ ] Set up log rotation
- [ ] Configure backup schedule

### Security

- [ ] Verify CVE-2024-27763 mitigation (no basicsr)
- [ ] Enable rate limiting
- [ ] Set file upload size limits
- [ ] Enable input validation
- [ ] Configure API key authentication
- [ ] Set up security monitoring alerts
- [ ] Review security scan results

### Monitoring

- [ ] Verify Prometheus scraping service metrics
- [ ] Import Grafana dashboards
- [ ] Configure alert rules
- [ ] Set up log aggregation
- [ ] Test health check endpoints
- [ ] Configure uptime monitoring

### Testing

- [ ] Process test image via API
- [ ] Verify output quality
- [ ] Load test with concurrent requests
- [ ] Test failover and recovery
- [ ] Verify GPU acceleration (if applicable)
- [ ] Test with different presets

---

## Support & Resources

- **Security Issues**: See `/SECURITY.md` for vulnerability reporting
- **Module Documentation**: `lux_depth_v2/README.md`
- **Security Guide**: `lux_depth_v2/SECURITY.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

---

**Version**: 2.0.0  
**Status**: Production Ready  
**Last Updated**: December 8, 2025
