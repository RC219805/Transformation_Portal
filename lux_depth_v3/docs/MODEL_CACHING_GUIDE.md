# DA3 Model Caching Guide

Comprehensive guide to downloading, managing, and deploying Depth Anything 3 (DA3) model variants.

## Table of Contents

1. [Overview](#overview)
2. [Why Pre-Cache Models](#why-pre-cache-models)
3. [Model Variants](#model-variants)
4. [Cache Location](#cache-location)
5. [Usage](#usage)
6. [Offline Operation](#offline-operation)
7. [Deployment Strategies](#deployment-strategies)
8. [Storage Requirements](#storage-requirements)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The DA3 model caching system provides comprehensive management for all 10 official model variants:

- **Pre-cache models** before production/benchmarking runs
- **Manage cache location** and storage
- **Support offline operation** after initial download
- **Validate cached models** before use
- **Provide deployment-ready snapshots**

---

## Why Pre-Cache Models

### Production Benefits

1. **Eliminate Download Latency**: Models are ready immediately when needed
2. **Offline Operation**: Work without internet connectivity
3. **Consistent Performance**: No download overhead during processing
4. **Deployment Ready**: Package cached models for production environments
5. **Bandwidth Efficiency**: Download once, use many times

### Common Scenarios

- **CI/CD Pipelines**: Pre-cache in Docker images for consistent builds
- **Edge Deployment**: Download on development machines, deploy to edge devices
- **Benchmarking**: Ensure all models are cached before running benchmarks
- **Production Servers**: Avoid download delays during production workloads

---

## Model Variants

All 10 official DA3 models are supported:

| Model Key | HuggingFace ID | Params | Purpose | Recommended |
|-----------|----------------|--------|---------|-------------|
| `nested-giant-large-v1.1` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | ~2B | Full any-view + metric | ✅ Yes |
| `giant-v1.1` | `depth-anything/DA3-GIANT-1.1` | ~1.15B | Any-view + pose + 3DGS | ✅ Yes |
| `large-v1.1` | `depth-anything/DA3-LARGE-1.1` | ~0.35B | Balanced multi-view | ✅ Yes |
| `nested-giant-large` | `depth-anything/DA3NESTED-GIANT-LARGE` | ~2B | Legacy v1.0 | - |
| `giant` | `depth-anything/DA3-GIANT` | ~1.15B | Legacy v1.0 | - |
| `large` | `depth-anything/DA3-LARGE` | ~0.35B | Legacy v1.0 | - |
| `base` | `depth-anything/DA3-BASE` | ~0.12B | Compact foundation | - |
| `small` | `depth-anything/DA3-SMALL` | ~0.08B | Lightweight | - |
| `metric-large` | `depth-anything/DA3METRIC-LARGE` | ~0.35B | Metric depth only | ✅ Yes |
| `mono-large` | `depth-anything/DA3MONO-LARGE` | ~0.35B | Monocular only | - |

### Model Sets

Pre-defined sets for common use cases:

- **`essential`**: Minimal production set (nested-giant-large-v1.1, metric-large)
- **`production`**: Recommended for production (all v1.1 models + metric-large)
- **`benchmark`**: Full benchmark suite (all variants except legacy v1.0)
- **`all`**: Complete collection (all 10 variants)

---

## Cache Location

### Default Location

```
~/.cache/huggingface/hub
```

### Environment Variables

Control cache location via environment variables:

```bash
# Base cache directory
export HF_HOME=/data/models

# Hub cache location
export HF_HUB_CACHE=/data/models/hub

# Force offline mode (no downloads)
export HF_HUB_OFFLINE=1
```

### Custom Cache Directory

Specify custom location when downloading:

```bash
lux-depth-v3 cache-download --cache-dir /data/models
```

---

## Usage

### CLI Commands

#### Download Models

```bash
# Download essential models (recommended)
lux-depth-v3 cache-download --set essential

# Download production models
lux-depth-v3 cache-download --set production

# Download all models for benchmarking
lux-depth-v3 cache-download --set benchmark

# Download specific models
lux-depth-v3 cache-download --models nested-giant-large-v1.1,metric-large

# Custom cache directory
lux-depth-v3 cache-download --set production --cache-dir /data/models

# Force re-download
lux-depth-v3 cache-download --set production --force

# Skip verification (faster)
lux-depth-v3 cache-download --set essential --no-verify
```

#### List Cached Models

```bash
lux-depth-v3 cache-list
```

**Example Output:**

```
📦 Cached Models (4)
======================================================================
✓ depth-anything/DA3NESTED-GIANT-LARGE-1.1
   Size: 7.84 GB | Cached: 2025-12-19T10:30:00
✓ depth-anything/DA3-GIANT-1.1
   Size: 4.52 GB | Cached: 2025-12-19T10:35:00
✓ depth-anything/DA3-LARGE-1.1
   Size: 1.38 GB | Cached: 2025-12-19T10:37:00
✓ depth-anything/DA3METRIC-LARGE
   Size: 1.38 GB | Cached: 2025-12-19T10:38:00

======================================================================
Total: 15.12 GB in 4 models
Cache: /home/user/.cache/huggingface/hub
```

#### Show Cache Statistics

```bash
lux-depth-v3 cache-stats
```

**Example Output:**

```
📊 Cache Statistics
======================================================================
Location: /home/user/.cache/huggingface/hub
Models: 4
Total Size: 15.12 GB
Last Updated: 2025-12-19T10:38:00
```

### Python API

```python
from pathlib import Path
from lux_depth_v3.model_cache import ModelCacheManager, precache_models

# Quick pre-cache
results = precache_models("essential")

# Advanced usage
manager = ModelCacheManager(cache_dir=Path("/data/models"))

# Download specific models
info = manager.download_model("nested-giant-large-v1.1")
print(f"Downloaded: {info.size_bytes / (1024**3):.2f} GB")

# Download model set
results = manager.download_models(
    model_set="production",
    force=False,
    verify=True
)

# List cached models
cached = manager.list_cached_models()
for model in cached:
    print(f"{model.model_id}: {model.size_bytes / (1024**3):.2f} GB")

# Get statistics
stats = manager.get_cache_stats()
print(f"Total: {stats['total_size_gb']:.2f} GB")
```

### Automation Scripts

#### Bash Script

```bash
# Use environment variables for configuration
export DA3_MODEL_SET=production
export DA3_CACHE_DIR=/data/models
export DA3_VERIFY=true

./scripts/precache_models.sh
```

#### Python Script

```bash
# Essential models (default)
./scripts/precache_models.py

# Production models
./scripts/precache_models.py --set production

# Custom cache directory
./scripts/precache_models.py --set benchmark --cache-dir /data/models

# Force re-download
./scripts/precache_models.py --set all --force
```

---

## Offline Operation

### Initial Download

Download models once with internet connectivity:

```bash
lux-depth-v3 cache-download --set production
```

### Enable Offline Mode

After caching, force offline operation:

```bash
export HF_HUB_OFFLINE=1
```

### Verification

Test offline operation:

```bash
# With internet disabled or HF_HUB_OFFLINE=1
lux-depth-v3 process --input-dir renders/ --output-dir output/
```

Models will load from cache without attempting downloads.

---

## Deployment Strategies

### Docker Containers

**Dockerfile Example:**

```dockerfile
FROM python:3.10

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Pre-cache models during build
ENV HF_HOME=/models
RUN lux-depth-v3 cache-download --set production --cache-dir /models

# Runtime
CMD ["lux-depth-v3", "process", "--input-dir", "/data/input", "--output-dir", "/data/output"]
```

### Edge Deployment

1. **Development Machine**: Download models with fast internet
2. **Package**: Create tarball of cache directory
3. **Edge Device**: Extract tarball to target cache location

```bash
# On development machine
lux-depth-v3 cache-download --set production --cache-dir /tmp/models
tar -czf da3-models-production.tar.gz -C /tmp/models .

# On edge device
mkdir -p ~/.cache/huggingface/hub
tar -xzf da3-models-production.tar.gz -C ~/.cache/huggingface/hub
```

### CI/CD Pipelines

**GitHub Actions Example:**

```yaml
- name: Cache DA3 Models
  uses: actions/cache@v3
  with:
    path: ~/.cache/huggingface/hub
    key: da3-models-production-${{ hashFiles('requirements.txt') }}

- name: Download Models
  run: |
    if [ ! -d ~/.cache/huggingface/hub ]; then
      lux-depth-v3 cache-download --set production
    fi
```

---

## Storage Requirements

### Model Sizes (Approximate)

| Model | Size | Precision |
|-------|------|-----------|
| nested-giant-large-v1.1 | ~8 GB | FP32 |
| giant-v1.1 | ~4.5 GB | FP32 |
| large-v1.1 | ~1.4 GB | FP32 |
| base | ~0.5 GB | FP32 |
| small | ~0.3 GB | FP32 |
| metric-large | ~1.4 GB | FP32 |
| mono-large | ~1.4 GB | FP32 |

### Set Sizes

- **Essential**: ~10 GB (2 models)
- **Production**: ~15 GB (4 models)
- **Benchmark**: ~18 GB (7 models)
- **All**: ~20 GB (10 models)

### Recommendations

- **Development**: Essential set (10 GB)
- **Production**: Production set (15 GB)
- **Research/Benchmarking**: Benchmark or All set (18-20 GB)

---

## Troubleshooting

### Common Issues

#### Download Failures

**Problem**: Model download times out or fails

**Solutions:**
```bash
# Retry with force flag
lux-depth-v3 cache-download --set production --force

# Use snapshot strategy for unreliable connections
# (Set in Python API: strategy=CacheStrategy.SNAPSHOT)
```

#### Insufficient Disk Space

**Problem**: Not enough space for models

**Solutions:**
```bash
# Check available space
df -h ~/.cache

# Download essential set only
lux-depth-v3 cache-download --set essential

# Use custom location with more space
lux-depth-v3 cache-download --set production --cache-dir /data/models
```

#### Cache Corruption

**Problem**: Cached models are corrupted or incomplete

**Solutions:**
```bash
# Re-download with verification
lux-depth-v3 cache-download --set production --force --verify

# Check cache statistics
lux-depth-v3 cache-stats

# Manually delete cache and re-download
rm -rf ~/.cache/huggingface/hub/lux_depth_v3_cache.json
lux-depth-v3 cache-download --set production
```

#### Offline Mode Fails

**Problem**: Models don't load in offline mode

**Solutions:**
```bash
# Verify models are cached
lux-depth-v3 cache-list

# Re-download if missing
lux-depth-v3 cache-download --set production

# Check environment variables
echo $HF_HUB_OFFLINE
echo $HF_HOME
```

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from lux_depth_v3.model_cache import ModelCacheManager
manager = ModelCacheManager()
```

---

## Best Practices

1. **Pre-cache before production**: Always cache models before deploying to production
2. **Use production set**: Start with production set unless specific needs require others
3. **Verify downloads**: Use `--verify` flag to ensure model integrity
4. **Monitor disk space**: Ensure sufficient space before downloading large sets
5. **Version control**: Track which models are cached in deployment documentation
6. **Test offline mode**: Verify offline operation before deploying to restricted environments
7. **Automate caching**: Include model caching in CI/CD pipelines
8. **Document cache location**: Make cache directory location explicit in deployment docs

---

## Quick Reference

### Essential Commands

```bash
# Download essential models
lux-depth-v3 cache-download --set essential

# List cached models
lux-depth-v3 cache-list

# Show cache stats
lux-depth-v3 cache-stats

# Download specific model
lux-depth-v3 cache-download --models nested-giant-large-v1.1

# Enable offline mode
export HF_HUB_OFFLINE=1
```

### Environment Variables

```bash
HF_HOME              # Base cache directory
HF_HUB_CACHE         # Hub cache location
HF_HUB_OFFLINE       # Force offline mode (1 = offline)
DA3_CACHE_DIR        # Cache directory (for scripts)
DA3_MODEL_SET        # Model set to download (for scripts)
DA3_VERIFY           # Verify downloads (for scripts)
```

---

## Support

For issues or questions:

1. Check cache statistics: `lux-depth-v3 cache-stats`
2. Review logs with `--verbose` flag
3. Consult troubleshooting section above
4. Open issue on GitHub with cache stats and error logs
