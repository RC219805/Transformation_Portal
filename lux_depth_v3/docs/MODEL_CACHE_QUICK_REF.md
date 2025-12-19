# DA3 Model Cache - Quick Reference

## Installation

```bash
pip install depth-anything-3
cd lux_depth_v3
pip install -r requirements.txt
```

## Quick Start

```bash
# Cache essential models (recommended)
lux-depth-v3 cache-download --set essential

# List cached models
lux-depth-v3 cache-list

# Show statistics
lux-depth-v3 cache-stats
```

## CLI Commands

### Download Models

```bash
# By set
lux-depth-v3 cache-download --set essential      # 2 models, ~10 GB
lux-depth-v3 cache-download --set production     # 4 models, ~15 GB
lux-depth-v3 cache-download --set benchmark      # 7 models, ~18 GB
lux-depth-v3 cache-download --set all            # 10 models, ~20 GB

# Specific models
lux-depth-v3 cache-download --models nested-giant-large-v1.1,metric-large

# Custom cache directory
lux-depth-v3 cache-download --set production --cache-dir /data/models

# Force re-download
lux-depth-v3 cache-download --set production --force

# Skip verification (faster)
lux-depth-v3 cache-download --set essential --no-verify
```

### List Cached

```bash
lux-depth-v3 cache-list
```

### Statistics

```bash
lux-depth-v3 cache-stats
```

## Python API

```python
from lux_depth_v3.model_cache import ModelCacheManager, precache_models

# Quick pre-cache
precache_models("production")

# Advanced usage
manager = ModelCacheManager(cache_dir="/data/models")

# Download single model
info = manager.download_model("nested-giant-large-v1.1")
print(f"Size: {info.size_bytes / (1024**3):.2f} GB")

# Download set
results = manager.download_models(
    model_set="production",
    force=False,
    verify=True
)

# List cached
cached = manager.list_cached_models()
for model in cached:
    print(f"{model.model_id}: {model.size_bytes / (1024**3):.2f} GB")

# Get statistics
stats = manager.get_cache_stats()
print(f"Total: {stats['total_size_gb']:.2f} GB in {stats['num_models']} models")
```

## Automation Scripts

### Bash

```bash
# Configure with environment variables
export DA3_MODEL_SET=production
export DA3_CACHE_DIR=/data/models
export DA3_VERIFY=true

./scripts/precache_models.sh
```

### Python

```bash
./scripts/precache_models.py --set production
./scripts/precache_models.py --set benchmark --cache-dir /data/models
./scripts/precache_models.py --set all --force
```

## Environment Variables

```bash
# HuggingFace cache configuration
export HF_HOME=/data/models             # Base cache directory
export HF_HUB_CACHE=/data/models/hub    # Hub cache location
export HF_HUB_OFFLINE=1                 # Force offline mode

# Script configuration
export DA3_CACHE_DIR=/data/models       # Cache directory (scripts)
export DA3_MODEL_SET=production         # Model set (scripts)
export DA3_VERIFY=true                  # Verify downloads (scripts)
```

## Model Sets

| Set | Models | Size | Use Case |
|-----|--------|------|----------|
| **essential** | 2 | ~10 GB | Development, minimal deployment |
| **production** | 4 | ~15 GB | Production deployment |
| **benchmark** | 7 | ~18 GB | Benchmarking, research |
| **all** | 10 | ~20 GB | Complete collection |

## Model Variants

| Key | HuggingFace ID | Params | Priority |
|-----|----------------|--------|----------|
| `nested-giant-large-v1.1` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | ~2B | ⭐ Recommended |
| `giant-v1.1` | `depth-anything/DA3-GIANT-1.1` | ~1.15B | ⭐ Recommended |
| `large-v1.1` | `depth-anything/DA3-LARGE-1.1` | ~0.35B | ⭐ Recommended |
| `metric-large` | `depth-anything/DA3METRIC-LARGE` | ~0.35B | ⭐ Recommended |
| `base` | `depth-anything/DA3-BASE` | ~0.12B | - |
| `small` | `depth-anything/DA3-SMALL` | ~0.08B | - |
| `mono-large` | `depth-anything/DA3MONO-LARGE` | ~0.35B | - |
| `nested-giant-large` | `depth-anything/DA3NESTED-GIANT-LARGE` | ~2B | Legacy v1.0 |
| `giant` | `depth-anything/DA3-GIANT` | ~1.15B | Legacy v1.0 |
| `large` | `depth-anything/DA3-LARGE` | ~0.35B | Legacy v1.0 |

## Offline Operation

```bash
# 1. Download models (with internet)
lux-depth-v3 cache-download --set production

# 2. Enable offline mode
export HF_HUB_OFFLINE=1

# 3. Use cached models (no internet required)
lux-depth-v3 process --input-dir renders/ --output-dir output/
```

## Deployment

### Docker

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

CMD ["lux-depth-v3", "process", "--input-dir", "/data/input", "--output-dir", "/data/output"]
```

### Edge Deployment

```bash
# On dev machine
lux-depth-v3 cache-download --set production --cache-dir /tmp/models
tar -czf da3-models-production.tar.gz -C /tmp/models .

# On edge device
mkdir -p ~/.cache/huggingface/hub
tar -xzf da3-models-production.tar.gz -C ~/.cache/huggingface/hub
```

### CI/CD

```yaml
- name: Cache DA3 Models
  uses: actions/cache@v3
  with:
    path: ~/.cache/huggingface/hub
    key: da3-models-production-${{ hashFiles('requirements.txt') }}

- name: Download Models
  run: |
    lux-depth-v3 cache-download --set production
```

## Troubleshooting

### Download Failures

```bash
# Retry with force
lux-depth-v3 cache-download --set production --force
```

### Disk Space

```bash
# Check available space
df -h ~/.cache

# Download smaller set
lux-depth-v3 cache-download --set essential

# Use custom location
lux-depth-v3 cache-download --set production --cache-dir /data/models
```

### Cache Corruption

```bash
# Re-download with verification
lux-depth-v3 cache-download --set production --force --verify

# Clear and re-download
rm -rf ~/.cache/huggingface/hub/lux_depth_v3_cache.json
lux-depth-v3 cache-download --set production
```

## Documentation

- **Comprehensive Guide**: `lux_depth_v3/docs/MODEL_CACHING_GUIDE.md`
- **Implementation Report**: `DA3_MODEL_CACHE_COMPLETE.md`
- **Main README**: `lux_depth_v3/README.md`

## Support

```bash
# Show help
lux-depth-v3 cache-download --help
lux-depth-v3 cache-list --help
lux-depth-v3 cache-stats --help

# Check statistics
lux-depth-v3 cache-stats
```
