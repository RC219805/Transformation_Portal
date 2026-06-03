# Contributing to Transformation Portal

Thank you for considering contributing to the Transformation Portal! This document outlines the development workflow, quality standards, and CI requirements.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the coding standards below
4. Run local tests: `pytest -v tests/ -m "(unit or security or regression or golden or integration) and not slow"`
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
| *(core)* | Core only (no ML) | 3.11, 3.12 | `pytest -m "(unit or security or regression or golden or integration) and not slow"` | ~30s |
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
pytest tests/ -m "(unit or security or regression or golden or integration) and not slow" -v

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

### 2. Security (BLOCKING)
- **bandit**: No high-severity security issues
- **gitleaks**: No secrets in commits
- **pip-audit**: No critical vulnerabilities
- **Exit code**: Must be 0

### 3. Tests (BLOCKING)
- **Core tests**: Python 3.11 and 3.12
- **ML tests**: Python 3.11
- **All tests must pass**
- **No skipped tests without justification**

### 4. Coverage Gates (ENFORCED)

#### Global Minimum
- Combined coverage must stay **≥25%** (enforced via `coverage report --fail-under=25`)
- Current baseline: **25.44%** (Q2 2026 target: 28%)

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

---

### 5. Type Checking (BLOCKING)

- **mypy**: Hard-fail type checking on critical modules (`lux_depth_v3/`)
- **Exit code**: Must be 0

> **Note:** `build.yml` (the canonical PR gate) now includes a dedicated typecheck gate.
> This aligns with `ci.yml` post-merge enforcement for parity across quality workflows.

---

## Post-Merge Quality Signals (Non-Blocking)

The following checks run after merge and do not block PRs:

### Extended Type Checking (POST-MERGE)
- **ci.yml**: Hard-fail mypy on critical modules (same as PR gate)
- **ci-quality-firewall.yml**: Soft-fail mypy for advisory checks

---

## Additional Pre-Merge Gates

### 6. Build Check (BLOCKING)
- Package must build successfully
- Wheel install must work
- `twine check` must pass

### 7. Repository Hygiene (BLOCKING)
- No workflow marker files in root
- No coverage artifacts committed
- Max 15 markdown files in root
- No directories with spaces in names

### Local Ignore Rules
- Treat `.git/info/exclude` as machine-local scratch only.
- Do not rely on `.git/info/exclude` for team policy; shared ignore rules must live in `.gitignore`.
- If you add a local exclude that others will need, promote it to `.gitignore` in the same PR.

## Running CI Checks Locally

### Quick Check (before committing)
```bash
# Lint and format
black --check --line-length=127 src/ tests/
isort --check-only --profile=black --line-length=127 src/ tests/
flake8 src/ tests/ --max-line-length=127

# Core tests
pytest -v tests/ -m "(unit or security or regression or golden or integration) and not slow" --maxfail=3
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
coverage report --fail-under=25

# Build package
python -m build
twine check dist/*
```

## CI/CD Control Plane

The repository uses a layered CI/CD architecture with distinct workflow roles.

### Quality Control Plane (Q2 2026)

| Workflow | Trigger | Role | Branch Protection | Action Refs |
|----------|---------|------|-------------------|-------------|
| `build.yml` | PR, push, dispatch | **Canonical PR Gate** | ✅ Required | SHA-pinned |
| `ci.yml` | push | Post-merge validation | No | SHA-pinned |
| `ci-quality-firewall.yml` | workflow_run, dispatch | Post-CI verification | No | SHA-pinned |
| `quality-gate.yml` | PR, push | Legacy helper | No | SHA-pinned |

**Canonical Workflow:** `build.yml` is the only workflow required for branch protection.
All quality-control workflows use SHA-pinned action refs (normalized Q2 2026).

### Test Marker Selection

> **Current State (2026-03-23):** CI uses **positive marker selection** for core tests.
> This explicitly selects test categories (unit, security, regression, golden, integration)
> rather than excluding unwanted tiers.

```bash
# PR gating expression (positive marker selection)
pytest -v tests/ -m "(unit or security or regression or golden or integration) and not ml and not slow and not benchmark" --maxfail=1

# ML tier expression
pytest -v tests/ -m "ml and not slow and not integration and not benchmark" --maxfail=1
```

## Branch Protection and Merge Requirements

The `main` branch is protected to ensure code quality and stability. All changes must:

1. **Go through Pull Request review**
   - Minimum 1 approval recommended (not currently enforced, but best practice)
   - 2 approvals required for architectural changes (ADRs, security, dependencies)

2. **Pass all required CI checks (merge blockers):**
   - **CI Gate** (`.github/workflows/build.yml`) is the only required status check in branch protection
   - CI Gate explicitly aggregates and enforces:
     - `lightweight`
     - `lint` (when `run_full=true`)
     - `test` matrix (when `run_full=true`)
     - `generate-manifest` (when `run_full=true`)

   **Post-merge quality signals (non-blocking):**
   - `CI Quality Firewall (post-CI) / Quality Gate Summary`
   - `CI Quality Firewall (post-CI) / Flake Rate Analysis`
   - `Nightly Deep Checks / Nightly Summary`

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

# Check diff coverage locally (matches `make coverage-diff`; 85% is the
# Phase 0 target in docs/testing/test_coverage_improvement_plan.md)
diff-cover coverage.xml --compare-branch=origin/main --fail-under=85
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

## Dependency Management

Transformation Portal uses `pip-compile` for dependency management. All dependencies are defined in abstract `.in` files and compiled to pinned `.txt` files for deterministic builds.

### Constraint Style Guidelines (ADR-032)

**When adding dependencies, use the correct constraint style:**

| Style              | Format          | Use Case                                          | Example                                  |
|--------------------|-----------------|---------------------------------------------------|------------------------------------------|
| **Range Pin**      | `>=X.Y,<Z`      | Production dependencies (base.in, ml.in)          | `numpy>=1.24,<2.5.0`                     |
| **Strict Pin**     | `==X.Y.Z`       | Deterministic builds, known incompatibilities     | `rawpy==0.26.0  # RAW demosaic`          |
| **Lower-bound**    | `>=X.Y`         | Dev tools with stable CLI (dev.in, ci.in only)    | `black>=24.8  # Formatter`               |
| **Unpinned**       | (none)          | **NEVER ALLOWED** (causes non-deterministic builds) | ❌                                       |

**Decision tree:**

1. Is this a production dependency (`base.in` or `ml.in`)?
   - **Yes**: Use **range pin** (`>=X.Y,<Z`) unless determinism is critical
   - **No**: Continue to step 2

2. Is this a development tool with a stable CLI (`dev.in` or `ci.in`)?
   - **Yes**: Use **lower-bound** (`>=X.Y`) if benefits from latest features
   - **No**: Use **range pin** for safety

3. Does it need deterministic behavior (ML models, RAW processing)?
   - **Yes**: Use **strict pin** (`==X.Y.Z`) with inline comment

### Adding/Updating Dependencies

**Workflow:**

```bash
# 1. Edit the appropriate .in file
vim requirements/base.in       # Production runtime deps
vim requirements/ml.in         # Optional ML/AI deps
vim requirements/dev.in        # Testing and linting tools
vim requirements/ci.in         # CI-only tools

# 2. Add dependency with correct constraint style
echo "new-package>=1.0.0,<2    # Brief description" >> requirements/base.in

# 3. Recompile all .txt files
cd requirements && make compile

# 4. Validate constraints
cd .. && ./scripts/validate_dependency_constraints.sh

# 5. Commit both .in and .txt files
git add requirements/*.in requirements/*.txt
git commit -m "deps: add new-package for feature XYZ"
```

### Platform-Specific ML Lockfile Generation

**IMPORTANT:** ML lockfiles contain platform-specific packages (e.g., PyTorch wheels for different OS/architectures). These lockfiles **must be regenerated on their authoritative host platform** to avoid cross-platform contamination.

#### Trust Model

| Target Platform | Lockfile | Authoritative Host | Status |
|-----------------|----------|-------------------|--------|
| macOS Apple Silicon | `ml-core-darwin-arm64.txt` | Native Darwin arm64 | **Active** |
| Linux x86_64 | *(none)* | *(none)* | **Retired unsupported; fails closed** |
| macOS Intel | *(none)* | *(none)* | **Retired unsupported; fails closed** |

**Never generate target-owned ML locks from a non-authoritative host** — pip-compile resolves host-specific wheels that will fail on the target platform. New Linux or macOS Intel ML support requires a separate governed lane before any installable lockfile is checked in.

#### When to Regenerate ML Lockfiles

Regenerate when:
- Adding/updating packages in `ml-core-darwin-arm64.in` or shared ML `.in` files
- Updating `base.txt` (ML locks constrain against it)
- Security patches require ML package updates

#### Regeneration Workflow

**For macOS Apple Silicon ML lock** (on native Darwin arm64 host):
```bash
cd requirements
make compile-ml-darwin-arm64   # Compile from native M1/M2/M3 Mac
make check-ml-darwin-arm64     # Verify lock is current
```

**For Linux x86_64 ML lock**:
```bash
cd requirements
# RETIRED - unsupported lane; do not regenerate without a new governed lane
make compile-ml-linux-x86_64   # Fails closed
make check-ml-linux-x86_64     # Fails closed
```

**For macOS Intel ML lock**:
```bash
cd requirements
# RETIRED - unsupported lane; do not regenerate without a new governed lane
make compile-ml-darwin-x86_64  # Fails closed
make check-ml-darwin-x86_64    # Fails closed
```

#### Retired Lane Handling

Linux x86_64 and macOS Intel ML lanes are historical governance records only.
Their top-level Make targets remain as fail-closed stubs so stale automation and
operator commands cannot silently recreate unsupported lockfiles.

```bash
cd requirements
make compile-ml-linux-x86_64   # Fails closed
make compile-ml-darwin-x86_64  # Fails closed
```

#### Automated CI Validation

CI automatically validates:
- **No Linux/CUDA markers in Darwin locks** (rejects unsupported target contamination)
- **Lock ownership authority** (prevents off-lane modifications)
- **Retired lane absence** (Linux/macOS Intel ML manifests must not reappear as installable checked-in locks)

See `scripts/validation/check_requirements_lock_contract.py` for enforcement details.

#### Common Issues

**Error: "Darwin arm64 ML lock generation is authoritative only on native Darwin arm64"**
→ You're trying to compile macOS locks from Linux. Run on a Mac.

**Error: "Linux x86_64 ML lock lane is retired unsupported"**
→ The Linux ML lock lane is not part of the current installable checked-in contract. Open a governed lane decision before regenerating it.

**Error: "Darwin x86_64 ML lock lane is retired unsupported"**
→ The Intel Mac ML lock lane is not part of the current installable checked-in contract. Do not regenerate it without a governed lane decision.

### Banned Packages

The following packages are **banned** and must not be added:

| Package      | Reason                                      | Alternative                                      |
|--------------|---------------------------------------------|--------------------------------------------------|
| `realesrgan` | Unmaintained (no updates since 2022)        | Use local implementation in `src/transformation_portal/spatial_ai/reconstruction/` |

### Security Minimums

Certain packages require minimum versions due to CVEs:

| Package                 | Minimum Version | Reason                              |
|-------------------------|-----------------|-------------------------------------|
| `sentence-transformers` | >=3.1.0         | CVE-73169 (arbitrary code execution) |
| `Pillow`                | >=10.0.0        | Multiple CVEs in 9.x series         |

### Approved Exceptions

Lower-bound-only constraints are approved for these development tools:

- `mypy`, `black`, `flake8`, `pylint` (linters/formatters with stable CLIs)
- `pypdf` (CI utilities with backward compat)
- `PyYAML`, `coremltools`, `psutil` (optional ML deps with stable APIs)

See [`docs/architecture/ADR-032-dependency-pinning-strategy.md`](docs/architecture/ADR-032-dependency-pinning-strategy.md) for full rationale.

### Enforcement

Dependencies are validated automatically:

**Pre-commit hook** (local):
```bash
# Runs on git commit for .in or .txt changes
# Blocks unpinned deps, banned packages, stale .txt files
```

**CI job** (automated):
```bash
# Runs in CI Quality Firewall workflow
# Blocks PR merge on violations
```

**Manual validation:**
```bash
./scripts/validate_dependency_constraints.sh --verbose
```

### Exception Process

If you need to violate a constraint rule:

1. Document rationale (why is the exception necessary?)
2. Assess risk (what could break?)
3. Add to ADR-032 approved exceptions table, or
4. Add inline comment in `.in` file with reviewer name and date

**Example:**
```
# Exception: new-lib has no stable release yet (approved by @reviewer, 2026-02-16)
new-lib>=0.5.0
```

### Quarterly Dependency Audit

Transformation Portal conducts quarterly dependency audits:

- **Security patches**: CVEs reviewed and updated
- **Staleness check**: Packages >6 months old evaluated
- **Banned packages**: New unmaintained packages identified
- **Exception review**: Approved exceptions re-validated

Next audit: **2026-08-16 (Q3 2026)**

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
