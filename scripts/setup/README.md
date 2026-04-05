# Setup Scripts

This directory contains installation and setup scripts for the Transformation Portal repository.

## Available Scripts

### `auto-organize-install.sh`

Installs the automated repository organization system, including the standard repository pre-commit hook.

**Usage:**
```bash
./scripts/setup/auto-organize-install.sh
```

**What it does:**
- Installs the standard repository pre-commit hook at `.git/hooks/pre-commit`
- Makes the `.auto-organize.sh` script executable
- Validates the repository structure

### `install_da3_runtime.sh`

Bootstraps the repo-local Depth Anything 3 subprocess runtime used by
the auto-discovered `./.venv-da3/bin/python` contract and by explicit
`--da3-python` overrides.

**Usage:**
```bash
./scripts/setup/install_da3_runtime.sh
```

**What it does:**
- Clones Depth Anything 3 into `.runtime/Depth-Anything-3` if it is missing
- Synchronizes that checkout to the validated default ref unless `--ref` overrides it
- Creates the isolated DA3 venv at `./.venv-da3`
- Installs a pinned DA3-compatible dependency set without upstream `xformers`
- Captures a runtime package snapshot at `.runtime/da3-pip-freeze.txt`
- Runs the DA3 worker readiness check used by the subprocess adapter

**Stable contract:**
```bash
lux-depth-v3 --input-dir ./input --output-dir ./output
```

If you need a non-default interpreter, override it explicitly:
```bash
lux-depth-v3 --input-dir ./input --output-dir ./output --da3-python ~/venvs/da3/bin/python
```

**Requirements:**
- Git repository
- Bash shell (Linux, macOS, or Windows with WSL/Git Bash)

### `pre-commit-check.sh`

Canonical root-placement validator used by the repository hook set and by the
compatibility quality-gate wrapper.

**Usage:**
The primary git hook is installed with `make install-hooks` or
`./scripts/setup/auto-organize-install.sh`, both of which install the
repository's `pre-commit` hook. `scripts/pre_commit_hook.sh` remains available
as a compatibility wrapper around
`scripts/utilities/pre-commit-quality-check.py`; that Python quality gate
invokes `scripts/setup/pre-commit-check.sh` so the root-placement policy stays
single-sourced. This script also remains available for manual organization-only
checks:

```bash
./scripts/setup/pre-commit-check.sh
```

**What it does:**
- Checks for files in the repository root that should be elsewhere
- Provides suggestions for where files should go
- Prevents commits with misplaced files (unless `--no-verify` is used)

### `run_lint_tool.sh`

Bootstraps the CI-pinned lint toolchain locally and runs formatter/import checks from that environment.

**Usage:**
```bash
./scripts/setup/run_lint_tool.sh black path/to/file.py
./scripts/setup/run_lint_tool.sh isort path/to/file.py
./scripts/setup/run_lint_tool.sh parity
```

**What it does:**
- Creates `.venv-lint/` with `python3.12` if needed
- Installs `requirements-lint.txt` when the lockfile changes
- Runs Black and isort with the same versions used by GitHub Actions
- Powers `make lint-parity` and the local Black/isort pre-commit hooks

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

# Run the same hook set manually
make pre-commit

# Or run only the organization sub-check
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
make install-hooks

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

- [Repository Organization System](../../docs/governance/REPO_ORGANIZATION.md)
- [Main README](../../README.md)
- [Contributing Guide](../../CONTRIBUTING.md) (if exists)

## Support

For issues with setup scripts:

- Check this README first
- Review error messages carefully
- Check [GitHub Issues](https://github.com/RC219805/Transformation_Portal/issues)
- Create a new issue if your problem isn't covered

## Version History

- **v1.0.0** (November 2025): Initial setup script collection
  - Added `auto-organize-install.sh`
  - Added `pre-commit-check.sh`
  - Documented existing model installation scripts
