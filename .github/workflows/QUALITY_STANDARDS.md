# Code Quality Standards & Best Practices

## Overview
This document establishes baseline quality control standards to prevent recurring code issues and CI/CD failures.

## Current Issues Fixed (Nov 9, 2025)

### Critical Errors
1. **Missing @dataclass decorator** - `pdf_spec_parser.py:FinishSpec` was missing decorator
2. **Pylint directive syntax** - Updated to `pylint: disable-next=` for pylint 4.0 compatibility
3. **Too many root markdown files** - Reduced from 24 to 6 (limit: 10)

### Recurring Patterns
- Trailing whitespace (C0303)
- Unnecessary f-strings without interpolation (W1309)
- Import placement issues (C0413)
- Redefined outer scope variables (W0621)
- subprocess.run without check parameter (W1510)

## Automated Quality Gates

### Pre-commit Checks (Local)
```bash
# Runs automatically on commit via .pre-commit-quality-check.py
1. Trailing whitespace removal (autopep8)
2. Flake8 critical errors (E9, F63, F7, F82)
3. Python syntax validation
4. Import statement compilation
5. Markdown file count validation (≤10 in root)
```

### CI/CD Pipeline (.github/workflows/build.yml)
```yaml
Jobs:
  pre-commit-checks:
    - Trailing whitespace auto-fix
    - Flake8 critical errors only
    - Pylint (warnings allowed, only fail on fatal/error)
    - Markdown file count validation

  lint-and-test:
    Matrix: [Python 3.10, 3.11, 3.12] × [CPU, GPU]
    - Install dependencies
    - Run pytest (548 tests)
    - Code coverage reporting
```

## Best Practices Moving Forward

### 1. Import Organization
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from PIL import Image

# Local application
from transformation_portal.utils import load_image
```

### 2. Dataclass Usage
```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    """Always use @dataclass for structured data."""
    field1: str
    field2: int = 0
```

### 3. F-String Best Practices
```python
# Good - has interpolation
print(f"Processing {filename}")

# Bad - no interpolation, use regular string
print(f"Starting process")  # Should be: print("Starting process")
```

### 4. Subprocess Best Practices
```python
import subprocess

# Always specify check=True or handle errors
result = subprocess.run(
    ["command", "arg"],
    check=True,  # Raises CalledProcessError on failure
    capture_output=True,
    text=True
)
```

### 5. Pylint Suppression
```python
# For single-line suppression (pylint 4.0+)
# pylint: disable-next=undefined-all-variable
__all__ = ["DynamicImport"]

# For block suppression
# pylint: disable=too-many-arguments
def complex_function(a, b, c, d, e, f):
    pass
# pylint: enable=too-many-arguments
```

### 6. Trailing Whitespace
- **Automatic fix**: Handled by autopep8 in pre-commit
- **Prevention**: Configure editor to remove on save

### 7. Root Directory Organization
```
Root Directory Limits:
- Maximum 10 markdown files
- Move documentation to docs/
- Move session summaries to docs/sessions/
- Keep only: README.md, LICENSE, CONTRIBUTING.md, etc.
```

## Quality Metrics

### Target Standards
- **Pylint Score**: ≥9.5/10 (currently 9.35/10)
- **Test Coverage**: ≥80% (currently tracking)
- **Flake8**: Zero critical errors (E9, F63, F7, F82)
- **Test Pass Rate**: 100% (currently 548/549 passing)

### Current Status
```
✓ Flake8 critical errors: 0
✓ Pylint score: 9.35/10
✓ Test results: 548/549 passing (99.8%)
✗ Test failures: 1 (test_no_excessive_root_markdown_files - fixed)
```

## Proactive Safeguards

### 1. Pre-commit Hook Auto-fixes
- Trailing whitespace removal
- Line length enforcement (127 chars)
- Import sorting
- Basic formatting

### 2. CI/CD Failure Prevention
```bash
# Run before pushing
make test-fast          # Quick test subset
make lint               # Full linting
git diff origin/main    # Review changes
```

### 3. Documentation Standards
```markdown
# All markdown files must have:
1. Title (# heading)
2. Purpose/Overview
3. Date or version info
4. Clear structure

# Session summaries: docs/sessions/YYYY-MM-DD_topic.md
# Technical docs: docs/technical/topic.md
# User guides: docs/guides/topic.md
```

### 4. Code Review Checklist
- [ ] No trailing whitespace
- [ ] Imports at top of file
- [ ] @dataclass for data structures
- [ ] subprocess.run has check=True
- [ ] F-strings only when interpolating
- [ ] Pylint score ≥9.0
- [ ] All tests pass
- [ ] Documentation updated

## Enforcement Mechanisms

### Automatic (Enforced)
1. **Pre-commit hook**: Auto-fixes formatting
2. **CI/CD**: Blocks merge on critical errors
3. **Flake8**: Zero tolerance for E9, F63, F7, F82

### Warning (Non-blocking)
1. **Pylint warnings**: Logged but don't fail CI
2. **Test warnings**: Tracked but don't block
3. **Print statements**: Warned but allowed

### Manual Review Required
1. **Architecture changes**: Requires discussion
2. **New dependencies**: Must justify
3. **Breaking changes**: Document migration path

## Tools & Configuration

### Installed Quality Tools
```bash
pip install flake8 pylint autopep8 pytest pytest-cov
```

### Configuration Files
- `.flake8` - Flake8 configuration
- `pyproject.toml` - Pylint, pytest configuration
- `.github/workflows/build.yml` - CI/CD pipeline
- `.pre-commit-quality-check.py` - Local validation

### IDE Integration
```json
// VSCode settings.json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "files.trimTrailingWhitespace": true,
  "python.formatting.provider": "autopep8",
  "python.formatting.autopep8Args": ["--max-line-length=127"]
}
```

## Continuous Improvement

### Weekly Review
- Review CI/CD failures
- Update quality standards
- Refactor problem areas

### Monthly Audit
- Dependency updates
- Test coverage review
- Documentation updates

### Quarterly Goals
- Improve Pylint score to 9.8+
- Achieve 90%+ test coverage
- Reduce technical debt

## References

- [PEP 8 Style Guide](https://pep8.org/)
- [Pylint Documentation](https://pylint.pycqa.org/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- Repository CI/CD: `.github/workflows/build.yml`
- Quality Gate: `.pre-commit-quality-check.py`

---

**Last Updated**: November 9, 2025
**Maintainer**: Transformation Portal Team
**Review Cycle**: Monthly
