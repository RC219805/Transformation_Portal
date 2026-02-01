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
- **Python version**: 3.10+ (test on 3.10, 3.11, 3.12)
- **Formatting**: Use `black` with line length 127
- **Import sorting**: Use `isort` with black profile
- **Type hints**: Required for public APIs

### Testing Requirements
All contributions must include tests:

- **Unit tests** for new functions/classes
- **Integration tests** for cross-module functionality
- **CLI contract tests** for command-line interfaces
- **Regression tests** for bug fixes

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
- **Core tests**: Python 3.10 and 3.12
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
bandit -r src/ -ll

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

## Branch Protection Rules

The `main` branch is protected with these rules:

1. **Require PR reviews**: 1+ approving review required
2. **Require status checks**: All CI jobs must pass
3. **No force push**: History is immutable
4. **Linear history**: Prefer rebase/squash merges

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
