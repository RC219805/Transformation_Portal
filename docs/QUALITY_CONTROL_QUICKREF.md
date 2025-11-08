# Quality Control Quick Reference

## Pre-Commit Checklist

### 1. Code Quality
```bash
# Run flake8 critical errors check
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run pylint on changed files
pylint $(git diff --name-only origin/main...HEAD | grep '\.py$')
```

### 2. Tests
```bash
# Run fast tests
make test-fast

# Run full test suite
make test-full
```

### 3. Documentation
```bash
# Count root markdown files (must be ≤ 10)
ls -1 *.md 2>/dev/null | wc -l
```

## Common Issues & Fixes

### Undefined Name Errors (F821)
**Problem**: Import inside try/except or conditional
**Solution**: Use top-level conditional imports

```python
# ❌ Bad
try:
    import optional_module
except ImportError:
    pass

def function():
    optional_module.do_something()  # F821 error

# ✅ Good
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
# Remove trailing whitespace from file
python3 -c "
with open('file.py', 'r') as f:
    lines = [line.rstrip() for line in f]
with open('file.py', 'w') as f:
    f.write('\n'.join(lines) + '\n')
"
```

### Too Many Root Markdown Files
**Guidelines**:
- Keep in root: README, START_HERE, policies, migration guides
- Move to `docs/sessions/YYYY-MM-DD/`: Session-specific documentation
- Move to `docs/projects/PROJECT_NAME/`: Project-specific documentation  
- Move to `docs/`: Technical documentation

## Quality Gates

### Must Pass Before Merge
1. ✅ Flake8 critical errors: 0
2. ✅ Pylint fatal/error/usage: 0
3. ✅ Test suite: 100% passing
4. ✅ Root markdown files: ≤ 10

### Best Practices
1. ✅ Type hints for function parameters
2. ✅ Docstrings for public functions
3. ✅ Meaningful variable names
4. ✅ DRY (Don't Repeat Yourself)
5. ✅ Single responsibility per function

## File Organization

### Python Modules
```
transformation_portal/
├── pipelines/       # Processing pipelines
├── utils/          # Utility functions
├── models/         # ML model handling
└── cli/            # Command-line interfaces
```

### Documentation
```
docs/
├── sessions/       # Time-based session docs
├── projects/       # Project-specific docs
├── guides/         # How-to guides
└── *.md           # Technical documentation
```

### Tests
```
tests/
├── test_*.py       # Unit tests
├── conftest.py     # Pytest fixtures
└── data/          # Test data
```

## Automation

### Git Hooks (Recommended)
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Run quality checks before commit
flake8 . --count --select=E9,F63,F7,F82 --quiet || exit 1
python3 -m pytest tests/test_codebase_structure.py -q || exit 1
```

### Make Targets
```bash
make lint          # Run linting
make test-fast     # Quick tests
make test-full     # Complete test suite
make clean         # Clean build artifacts
```

## Emergency Fixes

### Quick Lint Fix
```bash
# Auto-fix simple issues
autopep8 --in-place --aggressive --aggressive FILE.py
```

### Quick Test
```bash
# Test specific file
pytest tests/test_FILE.py -v
```

### Quick Doc Cleanup
```bash
# List root markdown files
ls -1 *.md

# Move to appropriate location
mv DOCUMENTATION.md docs/sessions/$(date +%Y-%m-%d)/
```

## Resources

- Flake8 Docs: https://flake8.pycqa.org/
- Pylint Docs: https://pylint.pycqa.org/
- Pytest Docs: https://docs.pytest.org/
- PEP 8 Style Guide: https://pep8.org/

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
- **Flake8**: 0 = success, 1 = errors found
- **Pylint**: Bitwise flags (1=fatal, 2=error, 4=warning, etc.)
- **Pytest**: 0 = all pass, 1 = tests failed, 5 = no tests collected

---

**Keep this handy for daily development!**
