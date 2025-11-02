# Hypothesis Dependency Fix

## Issue
Tests were failing with `ModuleNotFoundError: No module named 'hypothesis'`.

## Root Cause
The `hypothesis` package is a required dependency for property-based testing but was not installed in the environment.

## Solution
Install the required testing dependencies:

```bash
# Option 1: Install just hypothesis
pip install hypothesis

# Option 2: Install all CI dependencies (recommended)
pip install -r requirements-ci.txt

# Option 3: Install development dependencies
pip install -r requirements-dev.txt

# Option 4: Install via pyproject.toml extras
pip install -e ".[dev]"
```

## Verification
After installation, verify that hypothesis is available:

```bash
python -c "import hypothesis; print(hypothesis.__version__)"
```

Run tests to confirm:
```bash
# Run fast test suite (recommended)
make test-fast

# Run all tests
pytest tests/ -v

# Run specific test file that uses hypothesis
pytest tests/__init__.py -v
```

## Dependencies Configuration
The `hypothesis` package is configured in:
- `pyproject.toml` - Under `[project.optional-dependencies.dev]` (use `pip install -e ".[dev]"` to install)
- `requirements-dev.txt` - For development environments (version: `hypothesis>=6,<7`)
- `requirements-ci.txt` - For CI/CD pipelines (version: `hypothesis>=6,<7`)

## Additional Notes
- The fast test suite requires additional dependencies like `numpy`, `Pillow`, `scipy`, etc.
- For complete functionality, install: `pip install -r requirements-ci.txt`
- Version specified in pyproject.toml: `hypothesis>=6.115.0`
- Version specified in requirements files: `hypothesis>=6,<7`
- Currently installed: hypothesis 6.143.1
