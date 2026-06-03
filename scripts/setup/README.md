# Setup Scripts

This directory contains installation and setup scripts for the Transformation Portal repository.

## Available Scripts

### `auto-organize-install.sh`

Installs the automated repository organization system, including the
repo-managed pre-commit and pre-push hook set.

**Usage:**
```bash
./scripts/setup/auto-organize-install.sh
```

**What it does:**
- Delegates to `make install-hooks` so `.git/hooks/pre-commit` and
  `.git/hooks/pre-push` match `.pre-commit-config.yaml`
- Makes the `.auto-organize.sh` script executable

### `install_da3_runtime.sh`

Bootstraps the repo-local Depth Anything 3 subprocess runtime used by
the auto-discovered `./.runtime/Depth-Anything-3/.venv-da3/bin/python`
contract and by explicit `--da3-python` overrides.

**Usage:**
```bash
./scripts/setup/install_da3_runtime.sh
```

**What it does:**
- Clones Depth Anything 3 into `.runtime/Depth-Anything-3` if it is missing
- Fetches the validated PR #110 runtime-contract ref by default, then synchronizes
  the checkout to the pinned commit unless `--ref` / `DA3_RUNTIME_REF` and
  `--fetch-ref` / `DA3_RUNTIME_FETCH_REF` override it
- Resolves a Python 3.11+ bootstrap interpreter (preferring the repo `.venv` when available)
- Creates the isolated DA3 venv at `.runtime/Depth-Anything-3/.venv-da3`
- Installs the pinned DA3-compatible baseline dependency profile without
  `pycolmap` or `xformers`
- Supports explicit `colmap` and `xformers` profiles when those optional
  feature lanes are requested. `pycolmap` is pinned by the script. `xformers`
  is intentionally operator-managed by default because compatible wheels vary
  by torch/platform; set `DA3_XFORMERS_SPEC` to a pinned pip spec when your
  environment has a known-good wheel.
- Uses the PR #110-style dependency contract for the default ref: NumPy 2,
  optional `pycolmap`, optional `xformers`, and baseline `open3d`
- Captures a runtime package snapshot at `.runtime/da3-pip-freeze.txt`
- Runs the DA3 worker readiness check used by the subprocess adapter

**Stable contract:**
```bash
lux-depth-v3 --input-dir ./input_images --output-dir ./output
```

If you need a non-default interpreter, override it explicitly:
```bash
lux-depth-v3 --input-dir ./input_images --output-dir ./output --da3-python ~/venvs/da3/bin/python
```

**Requirements:**
- Git repository
- Bash shell (Linux, macOS, or Windows with WSL/Git Bash)

### `install_depth_pro_runtime.sh`

Bootstraps the repo-local Depth Pro subprocess runtime used by the
auto-discovered `./.venv-depth-pro/bin/python` contract and by explicit
`--depth-pro-python` overrides.

**Usage:**
```bash
./scripts/setup/install_depth_pro_runtime.sh --skip-verify
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -o checkpoints/depth_pro.pt
./scripts/setup/install_depth_pro_runtime.sh
```

**What it does:**
- Resolves a Python 3.11+ bootstrap interpreter (preferring the repo `.venv` when available)
- Creates the isolated Depth Pro venv at `./.venv-depth-pro`
- Installs the repo-owned pinned Depth Pro surface (`torch==2.7.1`, `torchvision==0.22.1`, `numpy==1.26.4`) plus the Apple `ml-depth-pro` package from a pinned git ref
- Captures a runtime package snapshot at `.runtime/depth-pro-pip-freeze.txt`
- Runs `pip check`
- Runs the Depth Pro worker readiness check used by the subprocess adapter

**Stable contract:**
```bash
lux-depth-v3 --input-dir ./input_images --output-dir ./output --depth-pro-python ./.venv-depth-pro/bin/python
```

Use `--verify-device cpu` when you only need a CPU-safe contract. The default
`--verify-device auto` validates `mps` on Apple Silicon and `cpu` elsewhere.

### `install_raw_runtime.sh`

Bootstraps the repo-local RAW subprocess runtime used by the
auto-discovered `./.venv-raw/bin/python` contract and by explicit
`--raw-python` overrides.

**Usage:**
```bash
./scripts/setup/install_raw_runtime.sh
```

**What it does:**
- Resolves a Python 3.11+ bootstrap interpreter (preferring the repo `.venv` when available)
- Creates the isolated RAW venv at `./.venv-raw`
- Installs the local project with the `raw` extra into that venv
- Captures a runtime package snapshot at `.runtime/raw-pip-freeze.txt`
- Runs the RAW worker readiness check used by canonical ingest

**Stable contract:**
```bash
lux-depth-v3 --input-dir ./input_images --output-dir ./output --raw-python ./.venv-raw/bin/python
```

If you want the repo-local runtime to be used automatically for RAW batches,
just create `./.venv-raw/bin/python` with this script and omit `--raw-python`.

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
.venv/bin/python scripts/setup/install_models.py [--all] [--dry-run] [--force]
```

**Options:**
- `--all`: Check/install optional model families in addition to required models
- `--dry-run`: Show what would be checked or downloaded without downloading artifacts
- `--force`: Re-download verified local weights when supported

**Models:**
- Depth Anything V2 (depth estimation)
- Governed upscaling model assets where configured; the external `realesrgan`
  package remains unsupported
- Stable Diffusion XL (AI enhancement)
- ControlNet models (edge-preserving processing)

### `download_depth_models.py`

Prints Depth Anything CoreML setup instructions and verifies local Lux Depth V3
artifact status.

**Usage:**
```bash
.venv/bin/python scripts/setup/download_depth_models.py
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

# 2. Install pinned core runtime and dev tooling
make venv
make install-core

# 3. Install organization system
./scripts/setup/auto-organize-install.sh

# 4. (Optional) Install ML models
.venv/bin/python scripts/setup/install_models.py --all
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
ls -la .git/hooks/pre-push

# Run the root-placement check directly
./scripts/setup/pre-commit-check.sh --all
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
.venv/bin/python scripts/setup/install_models.py --model depth_anything_v2

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
