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
# Run fast test suite
make test-fast

# Run specific test with hypothesis
pytest tests/__init__.py -v
```

## Dependencies Configuration
The `hypothesis` package is configured in:
- `pyproject.toml` - Under `[project.optional-dependencies.dev]`
- `requirements-dev.txt` - For development environments
- `requirements-ci.txt` - For CI/CD pipelines

## Additional Notes
- The fast test suite requires additional dependencies like `numpy`, `Pillow`, `scipy`, etc.
- For complete functionality, install: `pip install -r requirements-ci.txt`
- Version required: `hypothesis>=6,<7` (currently 6.143.1)
