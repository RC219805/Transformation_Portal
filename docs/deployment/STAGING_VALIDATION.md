# Staging Validation Checklist - Transformation Portal

**Version**: 2.0.0
**Last Updated**: 2026-02-01
**Status**: Production Ready
**Owner**: QA + DevOps

---

## Overview

This document provides a comprehensive validation checklist for staging environment deployments before promoting to production.

**Purpose**: Ensure all critical functionality works correctly in a production-like environment before exposing to users.

**Validation Timeline**: Minimum 24 hours for minor releases, 48 hours for major releases.

---

## Pre-Validation Setup

### 1. Staging Environment Readiness

**Environment Checklist**:
- [ ] Staging environment matches production specs (CPU, RAM, storage)
- [ ] Same OS version and Python version as production
- [ ] All required dependencies installed (from `requirements-ci.txt`)
- [ ] Environment variables configured (`config/staging.yml` or `.env.staging`)
- [ ] Test data available (sample images, depth maps, presets)
- [ ] Logging configured and accessible
- [ ] Monitoring/alerting disabled (or set to staging channels)

**Deployment Commands**:
```bash
# Deploy to staging
cd /opt/transformation-portal-staging
git fetch --tags
git checkout v2.0.0  # Or target version
make install-core

# Verify version
.venv/bin/python -c "import transformation_portal; print(transformation_portal.__version__)"
# Expected output: 2.0.0

# Verify environment
.venv/bin/python scripts/verification/verify_core.py
```

---

## Validation Phases

### Phase 1: Installation & Configuration (30 minutes)

**Goal**: Verify package installs correctly and configuration is valid.

#### 1.1 Installation Verification
```bash
# Fresh repo-managed install
make venv
make install-core

# Verify imports
.venv/bin/python -c "
from transformation_portal.lux_depth_v3 import PBRProcessor, EnhanceConfig
from transformation_portal.lux_depth_v3.pbr_presets import get_preset
print('✅ All imports successful')
"
```

**Expected Result**: No import errors, all modules load cleanly.

**Failure Action**: Check for missing dependencies or circular imports.

#### 1.2 Configuration Validation
```bash
# Verify config files parse correctly
python -c "
from transformation_portal.config import load_config
config = load_config('config/staging.yml')
assert config.output_root, 'output_root not set'
assert config.cache_dir, 'cache_dir not set'
print('✅ Configuration valid')
"
```

**Expected Result**: Configuration loads without errors.

**Failure Action**: Review config syntax, check for missing required fields.

#### 1.3 CLI Availability
```bash
# Verify console scripts registered
which transform-process || echo "Warning: transform-process not in PATH"

# Test CLI help
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli --help

# Expected output: Usage information, no errors
```

**Expected Result**: CLI displays help without errors.

**Failure Action**: Check `pyproject.toml` entry points, reinstall package.

**Sign-off**: _____________ (QA Lead) Date: _______

---

### Phase 2: Core Functionality (1 hour)

**Goal**: Verify all core processing pipelines work end-to-end.

#### 2.1 Depth Estimation
```bash
# Test depth estimation on sample image
python -c "
from pathlib import Path
from transformation_portal.depth_tools import estimate_depth

image_path = Path('data/sample_images/interior_sample.jpg')
depth = estimate_depth(image_path, model='depth_anything_v2_vitl')

assert depth is not None, 'Depth estimation failed'
assert depth.shape[0] > 0 and depth.shape[1] > 0, 'Invalid depth shape'
print(f'✅ Depth estimated: {depth.shape}')
"
```

**Expected Result**: Depth map generated with valid dimensions.

**Failure Action**: Check model availability, GPU/CPU configuration.

#### 2.2 PBR Processing (New in v2.0.0)
```bash
# Test PBR processor with default preset
python examples/pbr_preset_example.py

# Expected output:
# ✅ Normal map generated
# ✅ Roughness map generated
# ✅ Ambient occlusion generated
# Processing time: ~2-3 seconds for 4K image
```

**Expected Result**: All PBR maps generated, timing within expected range.

**Failure Action**: Check depth quality, PBR algorithm parameters.

#### 2.3 Preset System
```bash
# Test all 8 PBR presets
python -c "
from transformation_portal.lux_depth_v3.pbr_presets import (
    get_all_presets, get_preset
)

presets = get_all_presets()
assert len(presets) == 8, f'Expected 8 presets, got {len(presets)}'

for name in ['default', 'premium', 'balanced', 'performance']:
    preset = get_preset(name)
    assert preset is not None, f'Preset {name} not found'
    print(f'✅ Preset {name} loaded')
"
```

**Expected Result**: All 8 presets load successfully.

**Failure Action**: Check `pbr_presets.py` for syntax errors or missing entries.

#### 2.4 Batch Processing
```bash
# Test batch processing with 5 images
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli \
  --input-dir data/sample_images/ \
  --output-dir /tmp/staging_batch_test \
  --preset balanced \
  --batch-size 2

# Verify output
ls -lh /tmp/staging_batch_test/
# Expected: 5 sets of PBR maps (normal, roughness, AO)
```

**Expected Result**: All images processed, outputs exist, no crashes.

**Failure Action**: Check batch logic, memory limits, error handling.

**Sign-off**: _____________ (QA Lead) Date: _______

---

### Phase 3: Integration Tests (2 hours)

**Goal**: Run full test suite in staging environment.

#### 3.1 Unit Tests
```bash
# Run core unit tests (fast)
pytest tests/ -v -m "not ml and not slow" --maxfail=5

# Expected: >90% pass rate (allow for environment differences)
```

**Expected Result**: All core tests pass or only known failures.

**Failure Action**: Investigate failures, compare with CI results.

#### 3.2 Integration Tests
```bash
# Run PBR integration tests
pytest tests/test_pbr_processor.py -v --maxfail=1

# Expected: All 53 tests pass
```

**Expected Result**: All integration tests pass.

**Failure Action**: Check test data, depth model availability.

#### 3.3 Preset Validation Tests
```bash
# Run preset validation suite
pytest tests/test_pbr_presets.py -v --maxfail=1

# Expected: All 85 tests pass
```

**Expected Result**: All preset tests pass.

**Failure Action**: Check preset definitions, parameter ranges.

#### 3.4 Code Coverage Check (if applicable)
```bash
# Run with coverage
pytest tests/ --cov=src/transformation_portal --cov-report=term-missing

# Expected: >70% coverage for core modules
```

**Expected Result**: Coverage meets or exceeds 70% threshold.

**Failure Action**: Review untested code paths, add tests if critical.

**Sign-off**: _____________ (QA Lead) Date: _______

---

### Phase 4: Performance & Scalability (2 hours)

**Goal**: Verify performance meets baseline requirements.

#### 4.1 Single Image Performance
```bash
# Test with 4K image (typical luxury real estate size)
time python -c "
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor
from transformation_portal.lux_depth_v3.pbr_presets import get_preset

processor = PBRProcessor(get_preset('premium'))
result = processor.process_from_path(Path('data/4k_sample.jpg'))
print(f'✅ Processing complete: {result}')
"

# Expected time: <3 seconds for depth + PBR on typical hardware
```

**Expected Result**: Processing completes in <3 seconds.

**Failure Action**: Profile bottlenecks, check for inefficiencies.

#### 4.2 Batch Throughput
```bash
# Test batch processing throughput
python -c "
import time
from pathlib import Path
from transformation_portal.lux_depth_v3.pbr_cli import batch_process

start = time.time()
batch_process(
    input_dir=Path('data/sample_images'),
    output_dir=Path('/tmp/staging_throughput_test'),
    preset='balanced',
    batch_size=4
)
elapsed = time.time() - start

images_processed = len(list(Path('data/sample_images').glob('*.jpg')))
throughput = images_processed / elapsed * 3600  # images/hour
print(f'Throughput: {throughput:.0f} images/hour')
print(f'Expected: >1000 images/hour')
assert throughput > 1000, f'Throughput too low: {throughput}'
"
```

**Expected Result**: Throughput >1,000 images/hour for balanced preset.

**Failure Action**: Check CPU/GPU utilization, memory constraints.

#### 4.3 Memory Usage
```bash
# Monitor memory during batch processing
python -c "
import psutil
import time
from pathlib import Path
from transformation_portal.lux_depth_v3.pbr_cli import batch_process

process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024  # MB

batch_process(
    input_dir=Path('data/sample_images'),
    output_dir=Path('/tmp/staging_memory_test'),
    preset='premium',
    batch_size=1
)

mem_after = process.memory_info().rss / 1024 / 1024  # MB
mem_increase = mem_after - mem_before

print(f'Memory before: {mem_before:.1f} MB')
print(f'Memory after: {mem_after:.1f} MB')
print(f'Memory increase: {mem_increase:.1f} MB')
print(f'Expected: <1000 MB increase')
assert mem_increase < 1000, f'Memory leak suspected: {mem_increase} MB'
"
```

**Expected Result**: Memory increase <1 GB for batch processing.

**Failure Action**: Profile memory usage, check for leaks.

#### 4.4 Concurrent Processing
```bash
# Test with 4 concurrent processes (if supported)
# (Skip if single-process only)
for i in {1..4}; do
  .venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli \
    --input data/sample_$i.jpg \
    --output /tmp/staging_concurrent_$i \
    --preset balanced &
done

wait
echo "✅ All concurrent processes completed"
```

**Expected Result**: All processes complete successfully, no conflicts.

**Failure Action**: Check for shared resource conflicts (file locks, cache).

**Sign-off**: _____________ (QA Lead) Date: _______

---

### Phase 5: Error Handling & Edge Cases (1 hour)

**Goal**: Verify system handles errors gracefully.

#### 5.1 Invalid Input Handling
```bash
# Test with missing file
python -c "
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor

processor = PBRProcessor()
try:
    processor.process_from_path(Path('nonexistent.jpg'))
    print('❌ Should have raised FileNotFoundError')
    exit(1)
except FileNotFoundError:
    print('✅ Correctly raised FileNotFoundError')
"
```

**Expected Result**: Appropriate exception raised with clear message.

**Failure Action**: Improve error messages, add input validation.

#### 5.2 Corrupt Image Handling
```bash
# Test with corrupt image file
echo "not an image" > /tmp/corrupt.jpg
python -c "
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor

processor = PBRProcessor()
try:
    processor.process_from_path(Path('/tmp/corrupt.jpg'))
    print('❌ Should have raised exception for corrupt image')
    exit(1)
except Exception as e:
    print(f'✅ Correctly raised exception: {type(e).__name__}')
"
```

**Expected Result**: Graceful error handling, no crashes.

**Failure Action**: Add image validation, improve error messages.

#### 5.3 Out-of-Memory Simulation
```bash
# Test with very large image (if available)
# Or set memory limit: ulimit -v 512000  # 500 MB virtual memory

python -c "
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor

# Attempt to process large image with limited memory
# Should either succeed efficiently or fail gracefully
"
```

**Expected Result**: Either completes successfully or raises MemoryError.

**Failure Action**: Add memory checks, improve documentation on limits.

#### 5.4 Invalid Preset Handling
```bash
# Test with nonexistent preset
python -c "
from transformation_portal.lux_depth_v3.pbr_presets import get_preset

try:
    preset = get_preset('nonexistent_preset')
    print('❌ Should have raised KeyError')
    exit(1)
except KeyError:
    print('✅ Correctly raised KeyError for invalid preset')
"
```

**Expected Result**: Clear error message indicating invalid preset.

**Failure Action**: Improve preset validation, list available presets in error.

**Sign-off**: _____________ (QA Lead) Date: _______

---

### Phase 6: Documentation & Examples (30 minutes)

**Goal**: Verify all documented examples work correctly.

#### 6.1 README Examples
```bash
# Test all code snippets from README.md
# (Manual or automated with doctest)

# Example: Quick Start
make install-core
.venv/bin/python scripts/verification/verify_core.py

# Expected: All commands execute successfully
```

**Expected Result**: All README examples work without modification.

**Failure Action**: Update README with corrections, test again.

#### 6.2 Example Scripts
```bash
# Test example scripts
python examples/pbr_preset_example.py
python examples/process_750_picacho_pbr.py  # (if sample data available)

# Expected: Scripts complete successfully with expected output
```

**Expected Result**: All example scripts execute successfully.

**Failure Action**: Fix examples or update to match current API.

#### 6.3 API Documentation
```bash
# Generate API docs (if using Sphinx)
cd docs/
make html

# Verify no warnings or errors
# Open in browser: docs/_build/html/index.html
```

**Expected Result**: Documentation builds cleanly, all APIs documented.

**Failure Action**: Fix docstrings, update API docs.

**Sign-off**: _____________ (QA Lead) Date: _______

---

### Phase 7: Extended Monitoring (24-48 hours)

**Goal**: Monitor staging environment for stability over extended period.

#### 7.1 Continuous Smoke Tests
```bash
# Run smoke tests every hour for 24-48 hours
# (Use cron or systemd timer)

# Example crontab entry:
# 0 * * * * /opt/staging/run_smoke_tests.sh >> /var/log/staging_smoke.log 2>&1

# run_smoke_tests.sh:
#!/bin/bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli \
  --input data/smoke_test_image.jpg \
  --output /tmp/smoke_test_$(date +%s) \
  --preset balanced

pytest tests/test_core_functionality.py -v --maxfail=1
```

**Expected Result**: All hourly smoke tests pass for 24-48 hours.

**Failure Action**: Investigate intermittent failures, check for resource leaks.

#### 7.2 Log Monitoring
```bash
# Monitor logs for warnings/errors
tail -f /var/log/transformation-portal/app.log | grep -i "error\|warning"

# Expected: No unexpected errors or warnings
```

**Expected Result**: No errors, only informational logs.

**Failure Action**: Investigate and fix any warnings or errors.

#### 7.3 Resource Monitoring
```bash
# Monitor CPU, memory, disk usage
watch -n 60 'top -b -n 1 | head -20; df -h; free -h'

# Expected: Stable resource usage over 24-48 hours
```

**Expected Result**: No memory leaks, CPU usage normal, disk usage stable.

**Failure Action**: Profile and fix resource leaks.

**Sign-off**: _____________ (QA Lead) Date: _______

---

## Final Sign-Off

### Validation Summary

**Date Completed**: __________
**Duration**: __________ hours
**Tested By**: __________

**Test Results**:
- [ ] Phase 1: Installation & Configuration - PASS
- [ ] Phase 2: Core Functionality - PASS
- [ ] Phase 3: Integration Tests - PASS
- [ ] Phase 4: Performance & Scalability - PASS
- [ ] Phase 5: Error Handling & Edge Cases - PASS
- [ ] Phase 6: Documentation & Examples - PASS
- [ ] Phase 7: Extended Monitoring (24-48h) - PASS

**Issues Found**: (List any issues discovered, severity, and resolution)

| Issue ID | Description | Severity | Status | Notes |
|----------|-------------|----------|--------|-------|
| STG-001 | Example | Low | Resolved | Fixed in v2.0.1 |
|          |             |          |        |       |

**Production Readiness**: (Check one)
- [ ] ✅ **APPROVED** - Ready for production deployment
- [ ] ⚠️ **APPROVED WITH NOTES** - Ready with minor known issues (documented above)
- [ ] ❌ **REJECTED** - Not ready, critical issues must be resolved

**Approvals**:

- QA Lead: _____________ Date: _______
- Architect: _____________ Date: _______
- DevOps Lead: _____________ Date: _______
- Product Owner: _____________ Date: _______ (optional)

---

## Appendix A: Staging Environment Specifications

**Hardware**:
- CPU: (e.g., 4 cores, Intel Xeon)
- RAM: (e.g., 16 GB)
- Storage: (e.g., 100 GB SSD)
- GPU: (e.g., None / NVIDIA Tesla T4)

**Software**:
- OS: (e.g., Ubuntu 24.04 LTS)
- Python: (e.g., 3.11.5)
- Key Dependencies: (list major dependencies and versions)

**Configuration**:
- Config file: `config/staging.yml`
- Environment variables: `.env.staging`
- Data directory: `/opt/transformation-portal-staging/data`
- Log directory: `/var/log/transformation-portal`

---

## Appendix B: Quick Validation Script

```bash
#!/bin/bash
# quick_staging_validation.sh - Run essential checks quickly

set -euo pipefail

echo "🔍 Quick Staging Validation for Transformation Portal v2.0.0"
echo "================================================================"

# 1. Version check
echo "1. Checking version..."
VERSION=$(python -c "import transformation_portal; print(transformation_portal.__version__)")
if [ "$VERSION" != "2.0.0" ]; then
  echo "❌ Version mismatch: expected 2.0.0, got $VERSION"
  exit 1
fi
echo "✅ Version: $VERSION"

# 2. Import check
echo "2. Checking imports..."
python -c "
from transformation_portal.lux_depth_v3 import PBRProcessor
from transformation_portal.lux_depth_v3.pbr_presets import get_preset
print('✅ Imports successful')
"

# 3. Core tests
echo "3. Running core tests..."
pytest tests/test_pbr_processor.py -v -k "test_processor_initialization" --maxfail=1

# 4. Example run
echo "4. Testing example script..."
python examples/pbr_preset_example.py > /dev/null

# 5. Performance check
echo "5. Quick performance check..."
time python -c "
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor
from transformation_portal.lux_depth_v3.pbr_presets import get_preset
processor = PBRProcessor(get_preset('balanced'))
# (assumes test image available)
"

echo ""
echo "================================================================"
echo "✅ Quick validation complete!"
echo "Run full validation checklist for production deployment."
```

**Usage**:
```bash
chmod +x quick_staging_validation.sh
./quick_staging_validation.sh
```

---

**Next Review Date**: 2026-03-01
**Document Status**: Active
**Version**: 1.0.0
