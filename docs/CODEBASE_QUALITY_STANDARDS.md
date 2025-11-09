# Codebase Quality Standards

## Overview
This document defines the quality standards and automated safeguards for the Transformation Portal codebase to prevent recurring issues identified in CI/CD workflows.

## Identified Recurring Issues (Nov 8, 2025)

### 1. **Undefined Names (F821 Errors)**
**Problem**: Variables like `iio` used without proper imports
**Solution**:
- Always use try/except blocks for optional imports
- Define module-level flags (e.g., `HAS_IMAGEIO = True/False`)
- Use import guards before using optional modules

```python
# ✅ CORRECT
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Later in code
if HAS_IMAGEIO:
    import imageio.v3 as iio  # Re-import in scope
    iio.imwrite(...)
```

### 2. **Trailing Whitespace (C0303)**
**Problem**: Excessive trailing whitespace violations (1000+ instances)
**Solution**:
- **Automated**: Pre-commit hook strips trailing whitespace
- **IDE**: Configure editor to remove trailing whitespace on save
- **Manual**: Run `autopep8 --in-place --max-line-length=127 *.py`

### 3. **Excessive Root Markdown Files**
**Problem**: Test failure when >10 markdown files in repository root
**Solution**:
- **Session summaries** → `docs/sessions/`
- **Project docs** → `docs/projects/{project_name}/`
- **Integration guides** → `docs/guides/`
- **Root only**: README.md, START_HERE.md, MIGRATION_GUIDE.md, DEPRECATION_POLICY.md

### 4. **Import Organization**
**Problem**: Imports at wrong position, circular dependencies
**Solution**:
- Standard library imports first
- Third-party imports second
- Local imports last
- Use `# noqa: E402` only when absolutely necessary with documentation

```python
# ✅ CORRECT ORDER
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from local_module import function
```

### 5. **Missing Type Hints**
**Problem**: Functions without type annotations
**Solution**: Add type hints to all new functions

```python
def process_image(
    input_path: Path,
    output_path: Path,
    quality: int = 95
) -> Optional[np.ndarray]:
    """Process image with quality control."""
    ...
```

## Automated Quality Gates

### Pre-Commit Checks
Location: `.github/pre-commit-hook.sh`

Automatically runs:
1. **Undefined name detection** (F821)
2. **Trailing whitespace removal**
3. **Markdown file count check**
4. **Import validation**

Install:
```bash
cp .github/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### CI/CD Workflow Quality Gate
Location: `.github/workflows/quality-gate.yml`

Runs on every PR and push to main:
1. Auto-fixes formatting issues
2. Flake8 critical errors (E9, F63, F7, F82)
3. Pylint non-blocking warnings
4. Markdown file count enforcement
5. Auto-commits formatting fixes

### Real-Time Monitoring
Script: `.codebase_health_monitor.py`

Monitors:
- Code quality metrics
- Test coverage
- Documentation completeness
- Performance regression

Run: `python3 .codebase_health_monitor.py --watch`

## Quality Standards by File Type

### Python Files (.py)

#### Required
- ✅ No flake8 F821 (undefined names)
- ✅ No trailing whitespace
- ✅ Max line length: 127 characters
- ✅ Type hints on public functions
- ✅ Docstrings on public functions/classes

#### Recommended
- 📋 Pylint score > 8.0/10
- 📋 Test coverage > 80%
- 📋 Complexity score < 10 (McCabe)

### Markdown Files (.md)

#### Required
- ✅ Root directory: Max 10 files
- ✅ Organized in appropriate subdirectories
- ✅ No trailing whitespace

#### Recommended
- 📋 Links verified (no 404s)
- 📋 Code blocks have language specifiers
- 📋 Tables are properly formatted

### Configuration Files

#### Required
- ✅ Valid YAML/JSON syntax
- ✅ No secrets in version control
- ✅ Documented configuration options

## Common Pitfalls & Solutions

### ❌ Pitfall: Importing inside try/except then using outside
```python
# DON'T DO THIS
try:
    import imageio.v3 as iio
except ImportError:
    pass

# Much later...
iio.imwrite(...)  # ❌ F821: undefined name 'iio'
```

### ✅ Solution: Re-import in scope or use guard
```python
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

if HAS_IMAGEIO:
    import imageio.v3 as iio  # ✅ Re-import in scope
    iio.imwrite(...)
```

### ❌ Pitfall: Creating markdown files in root for notes
```python
# DON'T DO THIS
with open("NOTES.md", "w") as f:
    f.write("Some notes...")
```

### ✅ Solution: Use appropriate subdirectory
```python
# DO THIS
notes_path = Path("docs/sessions/NOTES_2025_11_08.md")
notes_path.parent.mkdir(parents=True, exist_ok=True)
with open(notes_path, "w") as f:
    f.write("Some notes...")
```

### ❌ Pitfall: Not handling optional dependencies
```python
# DON'T DO THIS
import tifffile  # Crashes if not installed
```

### ✅ Solution: Graceful fallback
```python
# DO THIS
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

if HAS_TIFFFILE:
    # Use tifffile
else:
    # Fallback to PIL or raise informative error
    raise ImportError(
        "tifffile required for 16-bit TIFF support. "
        "Install with: pip install tifffile"
    )
```

## Enforcement Levels

### 🛑 **BLOCKING** (CI fails, must fix)
- F821: Undefined names
- E9: Syntax errors
- F63, F7, F82: Import/logic errors
- >10 markdown files in root
- Test failures

### ⚠️ **WARNING** (CI passes, should fix)
- Trailing whitespace (auto-fixed)
- Pylint warnings
- Line length >127
- Missing type hints
- Missing docstrings

### 💡 **SUGGESTION** (CI passes, nice to have)
- Code complexity reduction
- Performance optimizations
- Additional test coverage
- Documentation improvements

## Quality Metrics Dashboard

Track these metrics over time:

| Metric | Target | Current |
|--------|--------|---------|
| Flake8 Errors | 0 | Monitor |
| Pylint Score | >9.0 | 9.27 |
| Test Coverage | >85% | Monitor |
| Root .md Files | ≤10 | 4 ✅ |
| Undefined Names | 0 | 0 ✅ |

## Quick Reference

### Before Committing
```bash
# 1. Run auto-formatter
autopep8 --in-place --max-line-length=127 *.py

# 2. Check for undefined names
flake8 --select=F821 *.py

# 3. Check markdown count
find . -maxdepth 1 -name "*.md" | wc -l  # Must be ≤10

# 4. Run tests
pytest tests/ -v
```

### Installation
```bash
# Install pre-commit hook
cp .github/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install quality tools
pip install flake8 pylint autopep8 pytest
```

## Continuous Improvement

This document is living documentation. Update when:
- New quality issues are identified
- Standards change
- Tools are added/updated
- Best practices evolve

**Last Updated**: November 8, 2025
**Version**: 1.0
**Maintainer**: RC
