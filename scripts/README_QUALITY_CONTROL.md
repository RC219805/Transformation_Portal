# Quality Control System

Comprehensive pre-commit quality control system for the Transformation Portal repository.

## Overview

This system prevents CI/CD failures by catching issues locally before pushing:

- **Flake8 errors** (undefined variables, import issues, syntax errors)
- **Test failures** (documentation organization, Python syntax)
- **Code quality issues** (trailing whitespace, debugging statements)
- **CI configuration problems** (invalid YAML, incorrect matrix setup)

## Quick Start

### 1. Install Git Pre-commit Hook

```bash
make install-hooks
```

This installs a pre-commit hook that automatically runs quality checks before each commit.

### 2. Run Quality Checks Before Pushing

```bash
# Fast check (recommended during development)
make ci

# Comprehensive CI simulation (before pushing)
make ci-full

# Manual pre-commit check
make pre-commit
```

## Tools

### 1. Pre-commit Hooks

The default git hook is installed with `make install-hooks`, which runs
`pre-commit install -f` against this repository's `.pre-commit-config.yaml`.
The legacy `scripts/pre_commit_hook.sh` entrypoint remains available as a
compatibility wrapper for running `scripts/utilities/pre-commit-quality-check.py`
directly.

**Installation:**
```bash
make install-hooks
```

**What the default installed hook checks:**
- Trailing whitespace / EOF hygiene
- Large added files, merge-conflict markers, line-ending issues, YAML syntax
- Repository-root file placement
- CI-parity `black` / `isort` checks for `src/` and `tests/`
- ML test isolation guardrails
- Dependency-constraint validation for requirement files

**Additional checks available through the compatibility wrapper / manual gate:**
- Untracked core files
- Trailing whitespace / EOF hygiene with auto-fix and re-stage
- Large added files, merge-conflict markers, line-ending issues, YAML syntax
- Repository-root file placement
- Lint tool parity from `requirements-lint.txt`
- Staged Python `flake8` critical/F821 checks
- Staged Python `black` / `isort` checks
- Repo-specific import heuristics (`iio`, `cv2`)

**Manual run:**
```bash
make pre-commit

# Optional compatibility path
./scripts/pre_commit_hook.sh

# Run the canonical implementation directly
python3 scripts/utilities/pre-commit-quality-check.py
```

### 2. Local CI Simulation (`scripts/local_ci_check.sh`)

Replicates the exact CI environment locally to catch issues before pushing.

**Usage:**
```bash
# Full check (all Python versions, all tests)
./scripts/local_ci_check.sh

# Quick mode (fast tests only)
./scripts/local_ci_check.sh --quick

# Specific Python version
./scripts/local_ci_check.sh --python 3.10

# Via Makefile
make ci-full       # Full simulation
make ci-quick      # Quick mode
```

**What it checks:**
1. Environment setup (required tools)
2. Flake8 linting (critical errors)
3. Pylint (changed files, non-blocking warnings)
4. Documentation structure (markdown count)
5. Test suite (pytest with optional parallel execution)
6. Additional checks (debugging statements, TODO/FIXME)

**Output:**
```
╔════════════════════════════════════════════╗
║  Transformation Portal - Local CI Check   ║
╚════════════════════════════════════════════╝

[1/6] Environment Setup
[2/6] Flake8 (Critical Errors)
[3/6] Pylint (Changed Files)
[4/6] Documentation Structure
[5/6] Test Suite
[6/6] Additional Quality Checks

╔════════════════════════════════════════════╗
║  ✓ ALL CHECKS PASSED - READY TO PUSH!     ║
╚════════════════════════════════════════════╝
```

### 3. Auto-fix Utility (`scripts/auto_fix_quality.py`)

Automatically fixes common quality issues.

**Usage:**
```bash
# Check what would be fixed (dry-run)
python scripts/auto_fix_quality.py --dry-run

# Fix all issues automatically
python scripts/auto_fix_quality.py --fix-all

# Fix specific issues
python scripts/auto_fix_quality.py --whitespace  # Trailing whitespace only
python scripts/auto_fix_quality.py --imports     # Import issues only
python scripts/auto_fix_quality.py --format      # Format with autopep8

# Fix specific files/directories
python scripts/auto_fix_quality.py path/to/file.py
python scripts/auto_fix_quality.py src/

# Via Makefile
make fix-quality     # Auto-fix all
make check-quality   # Dry-run mode
```

**What it fixes:**
- Trailing whitespace
- Missing common imports (`numpy`, `PIL.Image`, `pathlib.Path`, etc.)
- Code formatting (with autopep8)
- Line length issues (max 127 characters)

### 4. Documentation Organizer (`scripts/organize_docs.sh`)

Organizes excessive markdown files from root to `docs/` subdirectories.

**Usage:**
```bash
# Preview what would be moved
./scripts/organize_docs.sh --dry-run

# Move files automatically
./scripts/organize_docs.sh --auto

# Interactive mode (asks for confirmation)
./scripts/organize_docs.sh

# Via Makefile
make organize-docs    # Interactive
make check-docs       # Dry-run
```

**Directory structure:**
- `docs/migration/` - Migration guides
- `docs/deprecation/` - Deprecation notices
- `docs/guides/` - User guides and tutorials
- `docs/reference/` - Technical documentation
- `docs/archive/` - Historical documents

**Files kept in root:**
- README.md
- START_HERE.md
- LICENSE, LICENSE.md
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- SECURITY.md
- CHANGELOG.md

**Output:**
```
Categorization Plan:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Keep in root: README.md
  ✓ Keep in root: START_HERE.md
  → Move to docs/migration: MIGRATION_GUIDE.md
  → Move to docs/deprecation: DEPRECATION_POLICY.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keep in root: 4
Move to docs/: 2
✓ Final count (4) will be within limit (10)
```

### 5. CI Config Validator (`scripts/validate_ci_config.py`)

Validates GitHub Actions workflow configurations.

**Usage:**
```bash
# Validate all workflows
python scripts/validate_ci_config.py

# Validate specific workflow
python scripts/validate_ci_config.py .github/workflows/build.yml

# Auto-fix common issues
python scripts/validate_ci_config.py --fix

# Via Makefile
make validate-ci
```

**What it validates:**
- YAML syntax
- Required workflow fields (`name`, `on`, `jobs`)
- Python version matrix (3.11, 3.12)
- Job dependencies (`needs` clause)
- Checkout action versions
- Flake8 configuration consistency
- Common misconfigurations

## Makefile Integration

All tools are integrated into the Makefile for easy access:

```bash
# Quality checks
make lint              # Run flake8 + pylint
make ci                # Quick CI check (lint + fast tests)
make ci-full           # Full CI simulation
make ci-quick          # Fast CI simulation
make pre-commit        # Run pre-commit checks manually
make quality-check     # All quality validations

# Auto-fix
make fix-quality       # Auto-fix all issues
make check-quality     # Dry-run (show what would be fixed)

# Tools
make install-hooks     # Install git pre-commit hook
make validate-ci       # Validate GitHub Actions configs
make organize-docs     # Organize markdown files
make check-docs        # Preview documentation organization

# Testing
make test-fast         # Fast tests (development)
make test-full         # Full test suite
make test-structure    # Structure validation tests

# Utility
make clean             # Clean build artifacts
make setup             # Install package
```

## Workflow Integration

### Recommended Development Workflow

```bash
# 1. Start development
git checkout -b feature/my-feature

# 2. Make changes
vim my_file.py

# 3. Check quality before staging
make check-quality     # See what would be fixed
make fix-quality       # Auto-fix issues

# 4. Stage changes
git add my_file.py

# 5. Commit (pre-commit hook runs automatically)
git commit -m "feat: add new feature"

# 6. Run full CI simulation before pushing
make ci-full

# 7. Push to remote
git push origin feature/my-feature
```

### First-Time Setup

```bash
# 1. Install pre-commit hook
make install-hooks

# 2. Run initial quality check
make quality-check

# 3. Fix any issues
make fix-quality

# 4. Organize docs if needed
make check-docs
make organize-docs    # If count > 10

# 5. Validate CI configuration
make validate-ci

# 6. Run full test suite
make test-full
```

## CI/CD Configuration

### GitHub Actions Integration

The local CI simulation replicates the exact checks from `.github/workflows/build.yml`:

**Flake8:**
```yaml
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

**Pylint:**
```yaml
# Changed files only, non-blocking warnings
pylint $(git diff --name-only origin/main...HEAD | grep '\.py$')
# Exit codes: 1=fatal, 2=error, 4=warning, 8=refactor, 16=convention, 32=usage
# Fail only on: 1 (fatal), 2 (error), 32 (usage error)
```

**Test Matrix:**
- Python versions: 3.11, 3.12
- Devices: CPU, GPU
- Tasks: lint, test

## Troubleshooting

### Pre-commit hook not running

```bash
# Check if hook is installed
ls -la .git/hooks/pre-commit

# Reinstall
make install-hooks

# Check the generated hook
sed -n '1,40p' .git/hooks/pre-commit
```

### Flake8 errors not caught locally

```bash
# Ensure flake8 is installed
pip install flake8

# Run with exact CI config
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check specific file
flake8 path/to/file.py --select=E9,F63,F7,F82
```

### Test failures in CI but not locally

```bash
# Run with CI environment
./scripts/local_ci_check.sh

# Check Python version
python --version  # Should be 3.11 or 3.12

# Run tests with same flags as CI
pytest -v tests/ --maxfail=5

# Check for pytest-xdist (parallel execution)
pip install pytest-xdist
pytest -n auto tests/
```

### Too many markdown files

```bash
# Check current count
find . -maxdepth 1 -name "*.md" -type f | wc -l

# Preview organization
make check-docs

# Organize automatically
make organize-docs
```

### Import errors (e.g., `iio` undefined)

```bash
# Auto-fix imports
python scripts/auto_fix_quality.py --imports path/to/file.py

# Or manually add missing import
# For imageio v3:
import imageio.v3 as iio
```

## Configuration

### Flake8 Error Codes

The system checks for critical errors only:
- **E9**: Syntax errors
- **F63**: Invalid comparison, invalid print syntax
- **F7**: Syntax errors in docstrings
- **F82**: Undefined names (most common)

### Pylint Exit Codes

Pylint uses bitwise flags:
- **1**: Fatal
- **2**: Error
- **4**: Warning
- **8**: Refactor
- **16**: Convention
- **32**: Usage error

The system fails only on: 1 (fatal), 2 (error), 32 (usage error)

### Python Version Support

- **Minimum**: Python 3.11
- **Tested**: 3.11, 3.12
- **Recommended**: 3.12 (latest)

## Best Practices

1. **Run `make install-hooks` once** after cloning the repository
2. **Run `make ci` frequently** during development
3. **Run `make ci-full` before pushing** to catch all issues
4. **Use `make fix-quality`** to auto-fix common problems
5. **Check `make quality-check`** periodically for overall health
6. **Run `make validate-ci`** after modifying GitHub Actions workflows
7. **Organize docs** when markdown count exceeds 10: `make organize-docs`

## Performance

- **Pre-commit hook**: ~2-5 seconds (fast checks only)
- **Local CI (quick)**: ~10-30 seconds (fast tests)
- **Local CI (full)**: ~2-5 minutes (all tests, parallel)
- **Auto-fix**: ~5-10 seconds (all tracked files)

## Contributing

When adding new quality checks:

1. Keep `.pre-commit-config.yaml`, `make install-hooks`, and any compatibility wrapper behavior consistent
2. Update Makefile with new target if needed
3. Document in this README
4. Test with `--dry-run` mode first
5. Ensure checks match CI configuration exactly

## See Also

- [Main README](../README.md) - Project overview
- [CI Configuration](../.github/workflows/build.yml) - GitHub Actions setup
- [Testing Guide](../tests/TEST_STATUS.md) - Test suite documentation
- [Repository Architecture](../docs/architecture/ARCHITECTURE.md) - System architecture overview
