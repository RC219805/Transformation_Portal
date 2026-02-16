# Contributing to Transformation Portal

Thank you for considering contributing to the Transformation Portal! This document outlines the development workflow, quality standards, and CI requirements.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the coding standards below
4. Run local tests: `pytest -v tests/ -m "not ml and not slow"`
5. Commit with clear messages
6. Push and open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Transformation_Portal.git
cd Transformation_Portal

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Verify installation
pytest --version
python -c "import transformation_portal; print('OK')"
```

## Quality Standards

### Code Style
- **Line length**: Maximum 127 characters
- **Python version**: 3.11+ (minimum supported version)
- **Formatting**: Use `black` with line length 127
- **Import sorting**: Use `isort` with black profile
- **Type hints**: Required for public APIs

### Testing Requirements
All contributions must include tests:

- **Unit tests** for new functions/classes
- **Integration tests** for cross-module functionality
- **CLI contract tests** for command-line interfaces
- **Regression tests** for bug fixes

### Test Dependency Isolation (ADR-031)

**CRITICAL:** Tests are strictly isolated by dependency requirements to ensure fast, offline CI.

#### Test Classification

| Marker | Dependencies | Python Versions | Command | Runtime Target |
|--------|--------------|-----------------|---------|----------------|
| *(default)* | Core only (no ML) | 3.11, 3.12 | `pytest -m "not ml and not slow"` | ~30s |
| `@pytest.mark.ml` | transformers, torch, diffusers | 3.11 | `pytest -m "ml and not slow"` | ~5min |
| `@pytest.mark.slow` | Any | 3.11 | Manual/nightly | No limit |
| `@pytest.mark.benchmark` | Any | 3.11 | Manual/nightly | No limit |

#### Writing ML Tests: Required Patterns

ML dependencies (transformers, torch, diffusers) are **NOT installed** in core CI environments.

**Pattern A: Module-level import guard** ✅ (RECOMMENDED)

```python
# tests/spatial_ai/segmentation/test_material_classifier.py

try:
    import transformers
    import torch
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

@pytest.mark.ml
@pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies required")
class TestMaterialClassifier:
    def test_inference(self):
        # Safe to import here - skipif prevents collection in offline CI
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(...)
        ...
```

**Pattern B: Inline import skip** ✅

```python
@pytest.mark.ml
def test_inference(self):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    # Use imports...
```

#### Import-Before-Mock Anti-Pattern

**WRONG** ❌ - This will fail in offline CI:

```python
from unittest.mock import patch

@patch("transformers.CLIPModel")  # ❌ Imports transformers BEFORE patching!
def test_foo(mock_clip):
    pass
```

**Why it fails:** `@patch("transformers.CLIPModel")` imports the `transformers` module during test collection (before patching), causing `ModuleNotFoundError` when transformers is not installed.

**CORRECT** ✅ - Add import guard:

```python
try:
    import transformers
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

@pytest.mark.ml
@pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies required")
@patch("transformers.CLIPModel")  # ✅ Safe: skip decorator prevents collection
def test_foo(mock_clip):
    pass
```

#### Enforcement (3 Layers)

1. **Pre-commit hook**: Blocks commits with `@patch("transformers|torch")` without import guards
   - Script: `scripts/check_ml_test_isolation.sh`
   - Auto-runs on `git commit`

2. **CI validation**: Verifies core tests don't import ML dependencies
   - Job: `test-isolation` in Quality Firewall workflow
   - Fails fast with clear diagnostics

3. **Documentation**: Full specification and rationale
   - See: [`docs/architecture/ADR-031-test-dependency-isolation.md`](docs/architecture/ADR-031-test-dependency-isolation.md)

#### Quick Reference

```bash
# Run core tests (what CI runs for fast feedback)
pytest tests/ -m "not ml and not slow" -v

# Run ML tests (requires ML dependencies installed)
pytest tests/ -m "ml and not slow" -v

# Run all tests except slow/benchmark
pytest tests/ -m "not slow and not benchmark" -v

# Check if your test will violate isolation
bash scripts/check_ml_test_isolation.sh
```

### Test Stability & Flake Management

Tests must be **deterministic** and **reliable**. Flaky tests (intermittent failures) are not acceptable.

**Flake Rate Monitoring:**
- All tests tracked in `tests/flake_ledger.json`
- Repository target: **<1% flake rate**
- Tests with >3% flake rate are quarantined

**If you encounter a flaky test:**
1. Check ledger: `python scripts/analyze_flakes.py`
2. Reproduce locally (run 20+ times)
3. Fix root cause (see ADR-033 for common patterns)
4. **Do not** just re-run CI hoping it passes

**Quarantine mechanism:**
```python
import pytest

@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sometimes_flaky():
    # Last resort - prefer fixing root cause
    pass
```

**Common flake sources:**
- Race conditions / timing assumptions
- Environmental dependencies (network, filesystem)
- Test order dependencies
- Non-deterministic inputs (unseeded random)

See [`docs/architecture/ADR-033-test-flake-management.md`](docs/architecture/ADR-033-test-flake-management.md) for full guide.

### Documentation
- Docstrings for all public functions/classes
- Update relevant docs in `docs/` if behavior changes
- Update README if user-facing features change

## CI Quality Firewall

All pull requests must pass these automated gates before merge:

### 1. Linting (BLOCKING)
- **flake8**: No critical errors (E9, F63, F7, F82)
- **black**: Code must be formatted
- **isort**: Imports must be sorted
- **Exit code**: Must be 0

### 2. Type Checking (NON-BLOCKING)
- **mypy**: Run on critical modules
- **Warnings logged** but don't block merge

### 3. Security (BLOCKING)
- **bandit**: No high-severity security issues
- **gitleaks**: No secrets in commits
- **pip-audit**: No critical vulnerabilities
- **Exit code**: Must be 0

### 4. Tests (BLOCKING)
- **Core tests**: Python 3.11 and 3.12
- **ML tests**: Python 3.11
- **All tests must pass**
- **No skipped tests without justification**

### 5. Coverage Gates (ENFORCED)

#### Global Minimum
- Coverage must **not decrease** vs `main` branch
- Current baseline: **33%** (will increase over time)

#### Diff Coverage (KEY METRIC)
- **New/changed lines must be 80%+ covered**
- This is the primary quality ratchet mechanism
- Enforced via `diff-cover` tool

#### Critical Module Floors
Future enforcement (not yet active):
- `lux_depth_v3/`: 80% minimum
- `orchestrator.py`: 80% minimum
- `pbr_cli.py`: 80% minimum
- `preprocessing.py`: 70% minimum

### 6. Build Check (BLOCKING)
- Package must build successfully
- Wheel install must work
- `twine check` must pass

### 7. Repository Hygiene (BLOCKING)
- No workflow marker files in root
- No coverage artifacts committed
- Max 15 markdown files in root
- No directories with spaces in names

## Running CI Checks Locally

### Quick Check (before committing)
```bash
# Lint and format
black --check --line-length=127 src/ tests/
isort --check-only --profile=black --line-length=127 src/ tests/
flake8 src/ tests/ --max-line-length=127

# Core tests
pytest -v tests/ -m "not ml and not slow" --maxfail=3
```

### Full Pre-PR Check
```bash
# Format code
black --line-length=127 src/ tests/
isort --profile=black --line-length=127 src/ tests/

# Run security scans
./scripts/security_scan.sh  # Uses CI-aligned flags: -ll -ii

# Run all tests with coverage
pytest -v tests/ -m "not slow" \
  --cov=src/transformation_portal \
  --cov-report=term \
  --cov-report=html

# Check coverage threshold
coverage report --fail-under=33

# Build package
python -m build
twine check dist/*
```

## Branch Protection and Merge Requirements

The `main` branch is protected to ensure code quality and stability. All changes must:

1. **Go through Pull Request review**
   - Minimum 1 approval recommended (not currently enforced, but best practice)
   - 2 approvals required for architectural changes (ADRs, security, dependencies)

2. **Pass all required CI checks:**
   - CI Gate (build + core tests)
   - Lint (critical errors)
   - Core test suite

3. **Resolve all review conversations** (recommended)
   - Address all reviewer comments before merge

4. **Maintain linear history**
   - Use "Squash and merge" or "Rebase and merge"
   - Merge commits are allowed but squash is preferred

5. **Keep branch up to date**
   - Strict status checks enabled: must be up-to-date with main before merge

**For detailed branch protection verification procedures, troubleshooting, and governance:** See [Branch Protection Verification](docs/governance/BRANCH_PROTECTION_VERIFICATION.md)

---

### Current Branch Protection Rules (Verified 2026-02-10)

### Required Status Checks (✅ ENFORCED)
- **CI Gate** (GitHub App ID: 15368) must pass
- **Strict status checks**: Branches must be up-to-date before merging
- **Cannot bypass**: Status checks are mandatory

### Pull Request Reviews (⚠️ PARTIAL)
- **Approving reviews required**: 0 (not enforced, but recommended)
- **Dismiss stale reviews**: ✅ Enabled (stale approvals dismissed on new commits)
- **Code owner reviews**: Not required
- **Last push approval**: Not required

### History and Force Push Protection (✅ ENFORCED)
- **Force pushes**: ❌ Disabled (history is immutable)
- **Branch deletions**: ❌ Disabled (main cannot be deleted)
- **Linear history**: ✅ Required (merge commits or squash merges only)

### Additional Protections (NOT ENABLED)
- **Enforce for admins**: ❌ Not enabled (admins can bypass)
- **Require signed commits**: ❌ Not enabled
- **Require conversation resolution**: ❌ Not enabled
- **Lock branch**: ❌ Not enabled

### Recommended Improvements
Based on governance best practices, consider enabling:

1. **Required approving reviews**: Set to 1+ reviewer minimum
   ```bash
   gh api -X PATCH repos/RC219805/Transformation_Portal/branches/main/protection/required_pull_request_reviews \
     -f required_approving_review_count=1
   ```

2. **Enforce for admins**: Prevent accidental bypasses
   ```bash
   gh api -X POST repos/RC219805/Transformation_Portal/branches/main/protection/enforce_admins
   ```

3. **Require signed commits**: For supply chain security (optional)
   ```bash
   gh api -X POST repos/RC219805/Transformation_Portal/branches/main/protection/required_signatures
   ```

### Current Configuration Summary
```json
{
  "required_status_checks": {
    "strict": true,
    "checks": ["CI Gate"]
  },
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": true
  },
  "enforce_admins": false,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_signatures": false
}
```

**Verification**: Run `gh api repos/RC219805/Transformation_Portal/branches/main/protection` to verify current settings.

## Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer explanation if needed.

Fixes #issue-number
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`

**Examples**:
```
feat(pbr): add --dry-run flag to pbr_cli
fix(orchestrator): handle missing depth files gracefully
test(cli): add contract tests for pbr_cli exit codes
docs(readme): update installation instructions
```

## PR Guidelines

### PR Title
Use conventional commit format:
```
feat(module): Add new feature
fix(module): Fix specific bug
```

### PR Description
Include:
1. **What** changed (high-level summary)
2. **Why** it changed (problem being solved)
3. **How** it works (approach taken)
4. **Testing** done (manual + automated)
5. **Breaking changes** (if any)

### PR Size
- **Small PRs** (< 400 lines) preferred
- **Large PRs** (> 400 lines) require justification
- **Refactors** should be separate from features

### Draft PRs
Use draft PRs for:
- Work in progress
- Seeking early feedback
- Demonstrating approach before full implementation

## Coverage Strategy

We use a **ratcheting coverage strategy**:

1. **Never decrease global coverage** (enforced in CI)
2. **New code must be well-tested** (80% diff coverage)
3. **Critical modules** have floor thresholds (coming soon)
4. **Incremental improvement** over time

### Writing Testable Code

**Good practices**:
- Small, focused functions
- Dependency injection for external resources
- Avoid global state
- Mock external dependencies (FFmpeg, file I/O, models)

**Bad practices**:
- Large functions with many responsibilities
- Direct filesystem access without abstraction
- Hardcoded paths or configurations
- Side effects in pure functions

## Debugging CI Failures

### Lint Failures
```bash
# Auto-fix most issues
black --line-length=127 src/ tests/
isort --profile=black --line-length=127 src/ tests/

# Check remaining issues
flake8 src/ tests/ --max-line-length=127
```

### Test Failures
```bash
# Run failing test in verbose mode
pytest -vvs tests/test_module.py::TestClass::test_method

# Run with debugger
pytest --pdb tests/test_module.py::TestClass::test_method

# Check test output
pytest -v --tb=long tests/
```

### Coverage Failures
```bash
# Generate HTML coverage report
pytest --cov=src/transformation_portal --cov-report=html
# Open htmlcov/index.html in browser

# Check diff coverage locally
diff-cover coverage.xml --compare-branch=main --fail-under=80
```

### Build Failures
```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# Build fresh
python -m build

# Test wheel
pip install dist/*.whl --force-reinstall
```

## Release Process

Releases follow semantic versioning:
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist
1. Update `CHANGELOG.md`
2. Bump version in `pyproject.toml`
3. Create release PR
4. After merge, tag: `git tag v0.2.0 && git push --tags`
5. GitHub Actions will build and publish to PyPI

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Architecture**: See `docs/architecture/` for design decisions

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Help others learn and grow

---

Thank you for contributing to making Transformation Portal better!
