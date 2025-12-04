# Setup Scripts

This directory contains installation and setup scripts for the Transformation Portal repository.

## Quick Start

For new developers, run the unified setup script:

```bash
# Full development setup with RAG hooks
python scripts/setup/dev_setup.py --all

# Minimal setup (core deps only)
python scripts/setup/dev_setup.py --minimal

# Setup with ML extras
python scripts/setup/dev_setup.py --with-ml --with-rag
```

## Available Scripts

### `dev_setup.py` ⭐ (Recommended)

Unified developer setup script with RAG hooks integration.

**Usage:**
```bash
python scripts/setup/dev_setup.py              # Standard setup
python scripts/setup/dev_setup.py --minimal    # Core deps only
python scripts/setup/dev_setup.py --with-ml    # Include ML extras
python scripts/setup/dev_setup.py --with-rag   # Include RAG git hooks
python scripts/setup/dev_setup.py --all        # Everything
```

**What it does:**
- Creates/uses virtual environment
- Installs dependencies (core + optional extras)
- Installs RAG system git hooks for incremental indexing
- Sets up pre-commit hooks
- Configures development environment

**RAG Hooks installed:**
- `post-commit`: Automatic RAG index updates on commits
- `post-merge`: Index sync after pulls/merges
- `pre-push`: Cache consistency validation

### `auto-organize-install.sh`

Installs the automated repository organization system, including the pre-commit hook.

**Usage:**
```bash
./scripts/setup/auto-organize-install.sh
```

**What it does:**
- Installs the pre-commit hook to prevent misplaced files
- Makes the `.auto-organize.sh` script executable
- Validates the repository structure

**Requirements:**
- Git repository
- Bash shell (Linux, macOS, or Windows with WSL/Git Bash)

### `pre-commit-check.sh`

Pre-commit hook that validates file organization before allowing commits.

**Usage:**
This script is automatically run by git when you commit. You can also run it manually:

```bash
./scripts/setup/pre-commit-check.sh
```

**What it does:**
- Checks for files in the repository root that should be elsewhere
- Provides suggestions for where files should go
- Prevents commits with misplaced files (unless `--no-verify` is used)

**Bypass (Emergency Only):**
```bash
git commit --no-verify -m "Emergency commit"
```

### `install_models.py`

Downloads and installs AI/ML models required by the Transformation Portal pipelines.

**Usage:**
```bash
python scripts/setup/install_models.py [--model MODEL_NAME] [--all]
```

**Options:**
- `--model MODEL_NAME`: Install a specific model
- `--all`: Install all required models

**Models:**
- Depth Anything V2 (depth estimation)
- Real-ESRGAN (upscaling)
- Stable Diffusion XL (AI enhancement)
- ControlNet models (edge-preserving processing)

### `download_depth_models.py`

Downloads depth estimation models for the Depth Pipeline.

**Usage:**
```bash
python scripts/setup/download_depth_models.py
```

**What it does:**
- Downloads Depth Anything V2 models
- Installs CoreML variants for Apple Silicon optimization
- Validates model integrity

## Installation Guide

### First-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install organization system
./scripts/setup/auto-organize-install.sh

# 4. (Optional) Install ML models
python scripts/setup/install_models.py --all
```

### Organization System Only

If you only want to install the organization system:

```bash
./scripts/setup/auto-organize-install.sh
```

### Verify Installation

```bash
# Check that pre-commit hook is installed
ls -la .git/hooks/pre-commit

# Test organization script
./.auto-organize.sh --dry-run

# Test pre-commit hook (should show help if no misplaced files)
./scripts/setup/pre-commit-check.sh
```

## Troubleshooting

### Script Won't Execute

**Problem:** Permission denied when running scripts

**Solution:**
```bash
chmod +x scripts/setup/*.sh
chmod +x .auto-organize.sh
```

### Pre-Commit Hook Not Working

**Problem:** Hook doesn't run on commit

**Solution:**
```bash
# Reinstall the hook
./scripts/setup/auto-organize-install.sh

# Verify installation
ls -la .git/hooks/pre-commit

# Make sure it's executable
chmod +x .git/hooks/pre-commit
```

### Line Ending Issues (Windows)

**Problem:** Scripts fail with `\r` errors

**Solution:**
```bash
# Install dos2unix
# Ubuntu/Debian: sudo apt-get install dos2unix
# macOS: brew install dos2unix
# Windows: Available in Git Bash

# Convert line endings
dos2unix scripts/setup/*.sh
dos2unix .auto-organize.sh

# Or use sed
sed -i 's/\r$//' scripts/setup/*.sh
sed -i 's/\r$//' .auto-organize.sh
```

### Model Download Fails

**Problem:** Network errors or timeout during model download

**Solution:**
```bash
# Try downloading specific models one at a time
python scripts/setup/install_models.py --model depth_anything_v2

# Check disk space
df -h

# Check network connection
curl -I https://huggingface.co
```

## Development

### Adding a New Setup Script

When adding a new setup script:

1. Place it in `scripts/setup/`
2. Make it executable: `chmod +x scripts/setup/your_script.sh`
3. Add documentation to this README
4. Follow the existing script patterns for error handling
5. Test on multiple platforms (Linux, macOS, Windows)

### Script Template

```bash
#!/usr/bin/env bash
#
# your_script.sh
# Brief description of what this script does
#

set -euo pipefail  # Exit on error, undefined vars, pipe failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Your script logic here

echo "✓ Script completed successfully"
```

## Related Documentation

- [Repository Organization System](../../REPO_ORGANIZATION.md)
- [Main README](../../README.md)
- [Contributing Guide](../../CONTRIBUTING.md) (if exists)

## Support

For issues with setup scripts:

- Check this README first
- Review error messages carefully
- Check [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)
- Create a new issue if your problem isn't covered

## Version History

- **v1.1.0** (December 2025): Added unified developer setup
  - Added `dev_setup.py` with RAG hooks integration
  - Streamlined developer onboarding process
  - Integrated RAG incremental indexing hooks
- **v1.0.0** (November 2025): Initial setup script collection
  - Added `auto-organize-install.sh`
  - Added `pre-commit-check.sh`
  - Documented existing model installation scripts
