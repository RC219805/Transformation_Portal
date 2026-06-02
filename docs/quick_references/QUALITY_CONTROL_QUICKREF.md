# Quality Control Quick Reference

## Pre-Commit Checklist

### 1. Code Quality
```bash
# Run the shared lint policy locally
./scripts/lint_runner.sh local

# Or run advisory lint via Make
make lint
```

### 2. Tests
```bash
# Run fast tests
make test-fast

# Run the local CI gate
make ci
```

### 3. Documentation
```bash
# Validate touched docs placement
python3 scripts/governance/check_docs_structure.py --changed-only

# Validate the full docs baseline
python3 scripts/governance/check_docs_structure.py --all
```

## Common Issues & Fixes

### Undefined Name Errors (F821)
**Problem**: Import inside try/except or conditional
**Solution**: Use top-level conditional imports

```python
# Bad
try:
    import optional_module
except ImportError:
    pass

def function():
    optional_module.do_something()

# Good
try:
    import optional_module
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False

def function():
    if HAS_OPTIONAL:
        optional_module.do_something()
```

### Trailing Whitespace (C0303)
**Quick Fix**:
```bash
# Run the configured repo hooks
make pre-commit
```

### Docs Topology Violations
**Problem**: A file was added or edited directly under `docs/`

**Current rule**:
- `docs/README.md` is the only allowed file directly under `docs/`
- Every other docs file must live in an approved top-level directory

**Quick Fix**:
```bash
# Example: move a quick reference into an approved docs subtree
mv docs/MY_NOTE.md docs/quick_references/MY_NOTE.md
```

## Quality Gates

### Must Pass Before Merge
1. `flake8` critical errors: 0
2. `pylint` fatal/error/usage: 0
3. Local CI: `make ci`
4. Documentation topology: `check_docs_structure.py --changed-only` and `--all`

### Best Practices
1. Type hints for function parameters
2. Docstrings for public functions
3. Meaningful variable names
4. DRY (Don't Repeat Yourself)
5. Single responsibility per function

## File Organization

### Python Modules
```text
transformation_portal/
|-- pipelines/       # Processing pipelines
|-- utils/           # Utility functions
|-- models/          # ML model handling
`-- cli/             # Command-line interfaces
```

### Documentation
```text
docs/
|-- README.md                  # Only allowed docs root file
|-- governance/                # Policy and repo organization docs
|-- guides/                    # How-to guides
|-- projects/                  # Historical project docs
`-- quick_references/          # Quick reference material
```

### Tests
```text
tests/
|-- test_*.py       # Unit tests
|-- conftest.py     # Pytest fixtures
`-- data/           # Test data
```

## Automation

### Git Hooks
Install the configured hooks:
```bash
make install-hooks

# Or directly:
pre-commit install
```

### Make Targets
```bash
make lint          # Run advisory lint via the shared helper
make test-fast     # Quick tests
make ci            # Local CI gate
make clean         # Clean build artifacts
```

## Emergency Fixes

### Quick Lint Fix
```bash
# Apply the repo-supported quality autofixes
make fix-quality
```

### Quick Test
```bash
# Test specific file
pytest tests/test_FILE.py -v
```

### Quick Doc Cleanup
```bash
# Move a root-level docs file into an approved subtree
mv docs/DOCUMENTATION.md docs/quick_references/DOCUMENTATION.md
```

## Resources

- Flake8 Docs: https://flake8.pycqa.org/
- Pylint Docs: https://pylint.pycqa.org/
- Pytest Docs: https://docs.pytest.org/
- Documentation policy: docs/governance/DOCUMENTATION_POLICY.md

## Help

### Getting Help
```bash
# Flake8 help
flake8 --help

# Pylint help
pylint --help

# Pytest help
pytest --help
```

### Understanding Exit Codes
- `flake8`: `0` = success, `1` = errors found
- `pylint`: bitwise flags (`1` fatal, `2` error, `4` warning, etc.)
- `pytest`: `0` = all pass, `1` = tests failed, `5` = no tests collected

---

Keep this handy for daily development.
