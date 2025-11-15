# Dependency Submission Workflow - Fix Documentation

## Problem

The automatic GitHub dependency submission workflow was failing with:
```
OSError: [Errno 28] No space left on device
```

This occurred when `pip-compile` tried to resolve and download all dependencies from the heavy `requirements.txt` file containing large ML packages (PyTorch, Diffusers, etc.).

## Root Cause

1. **Heavy requirements.txt**: The root requirements file included all ML dependencies (~10GB)
2. **Limited disk space**: GitHub Actions runners have ~14GB free space
3. **pip-compile behavior**: Downloads packages during dependency resolution
4. **No cleanup**: The automatic workflow didn't free disk space before processing

## Comprehensive Fix Applied

### 1. Custom Dependency Submission Workflow (`.github/workflows/dependency-submission.yml`)

**Features:**
- ✅ **Aggressive disk cleanup** before processing (~30GB freed)
- ✅ **Proper environment variables** (`PIP_NO_CACHE_DIR=1`, etc.)
- ✅ **Minimal pip installation** (only pip-tools, not full dependencies)
- ✅ **Optimized directory exclusions** (tests, docs, cache directories)
- ✅ **Shallow checkout** (fetch-depth: 1) to save space
- ✅ **Comprehensive cleanup** after submission
- ✅ **Better error reporting** with disk usage diagnostics
- ✅ **Manual trigger support** for testing (`workflow_dispatch`)

### 2. Optimized Requirements Structure

**Before:**
```
requirements.txt (50 lines, all ML deps included)
├── torch, torchvision (~6GB)
├── diffusers, transformers (~2GB)
├── realesrgan, basicsr (~1GB)
└── Many other heavy packages
```

**After:**
```
requirements.txt → references requirements/base.txt (lightweight)
├── Core dependencies only (~500MB)
├── ML deps commented out by default
└── Clear documentation on installation options

requirements/
├── base.txt       # Core runtime (500MB)
├── ml.txt         # ML features (10GB) - optional
├── dev.txt        # Dev tools
├── ci.txt         # CI/CD tools
└── all.txt        # Everything combined
```

### 3. Updated Documentation

**requirements.txt** now includes:
- Installation options for different use cases
- Size estimates for each option
- Clear comments on enabling ML features
- Reference to pyproject.toml extras

**pyproject.toml** already has:
- `[ml]` extras for ML dependencies
- `[dev]` extras for development
- `[ci]` extras for CI/CD
- `[all]` for complete installation

## Installation Options

### Option 1: Lightweight Install (Core Features Only)
```bash
pip install -r requirements.txt
# or
pip install -e .
```
**Size:** ~500MB | **Features:** Image processing, batch operations, basic color grading

### Option 2: Full Install with ML Features
```bash
pip install -r requirements/all.txt
# or
pip install -e ".[ml]"
```
**Size:** ~10GB | **Features:** All features including AI upscaling, depth estimation, ControlNet

### Option 3: CI Environment
```bash
pip install -r requirements-ci.txt
```
**Size:** ~500MB + test tools | **Purpose:** Fast CI builds without ML dependencies

### Option 4: Development
```bash
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt
```
**Size:** ~500MB + dev tools | **Features:** Linting, testing, type checking

## Testing the Fix

### Test the workflow manually:
```bash
# From GitHub UI:
1. Go to Actions → Dependency Submission
2. Click "Run workflow"
3. Select branch
4. Click "Run workflow"
```

### Expected behavior:
1. ✅ Disk cleanup frees ~30GB
2. ✅ Checkout completes successfully
3. ✅ pip-tools installs without issues
4. ✅ Dependency detection processes requirements/base.txt
5. ✅ Submission completes within disk limits
6. ✅ Cleanup runs successfully

## Monitoring

The workflow includes comprehensive logging:
- Initial disk usage
- Disk usage after cleanup
- Disk usage after pip setup
- Final disk usage

If the workflow fails, check the logs for:
- Available disk space at each step
- Which step consumed the most space
- Error messages from pip-compile

## Benefits

1. **Faster CI/CD**: Core tests run faster without ML dependencies
2. **Flexible installation**: Users choose lightweight or full install
3. **Better dependency management**: Layered system is more maintainable
4. **Reduced disk issues**: Comprehensive cleanup prevents space problems
5. **Clear documentation**: Users understand installation options

## Migration Guide

For existing installations, no changes needed. The new structure is backward compatible:
- `requirements.txt` still works (installs core only)
- Add `[ml]` extra to get ML features: `pip install -e ".[ml]"`
- Or use `requirements/all.txt` for full install

## Related Files

- `.github/workflows/dependency-submission.yml` - Custom workflow (NEW)
- `requirements.txt` - Optimized root requirements (UPDATED)
- `requirements-ci.txt` - CI requirements (UPDATED)
- `requirements/` - Layered requirements system (EXISTING)
- `pyproject.toml` - Package metadata with extras (EXISTING)

## References

- GitHub Dependency Graph: https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-the-dependency-graph
- Component Detection Action: https://github.com/actions/component-detection-dependency-submission-action
- Disk Space Management: https://github.com/actions/runner-images/issues/2840
