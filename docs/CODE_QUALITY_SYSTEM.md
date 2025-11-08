# Code Quality Control System

**Version:** 2.0
**Last Updated:** November 8, 2025
**Status:** Active

---

## Overview

This document establishes baseline quality standards and automated safeguards to prevent common code quality issues from reaching CI/CD and production.

---

## Recent Issues Resolved

### 1. **Undefined Variable Reference (Critical)**
- **Issue:** `iio` used before import in `process_750_picacho.py:147`
- **Fix:** Wrapped import in try-except block
- **Prevention:** Pre-commit hook validates all imports

### 2. **Excessive Root Markdown Files**
- **Issue:** 24 markdown files in root (limit: 10)
- **Fix:** Organized into `docs/session_summaries/` and `docs/archive/`
- **Prevention:** Pre-commit hook enforces limit

### 3. **Trailing Whitespace**
- **Issue:** Pylint warnings for trailing whitespace
- **Fix:** Automated removal with sed
- **Prevention:** Pre-commit hook auto-fixes

---

## Automated Quality Gates

### Level 1: Pre-Commit Hook (Local)

**Location:** `.git/hooks/pre-commit`
**Status:** ✅ Active

**Checks:**
1. ✅ Trailing whitespace (auto-fix)
2. ✅ Flake8 critical errors (E9, F63, F7, F82)
3. ✅ Python syntax validation
4. ✅ Import validation
5. ✅ Markdown file count (<= 10 in root)
6. ⚠️  Debugging statements detection
7. ⚠️  Print statement warnings

**To enable:**
```bash
chmod +x .git/hooks/pre-commit
```

### Level 2: GitHub Actions CI (Remote)

**Configuration:** `.github/workflows/build.yml`
**Status:** ✅ Active

**Matrix Testing:**
- Python: 3.10, 3.11, 3.12
- Mode: test, lint
- Device: cpu, gpu

**Quality Checks:**
1. Flake8 (critical errors only)
2. Pylint (warnings non-blocking)
3. Pytest (548 tests)
4. Test coverage reporting

---

## Code Quality Standards

### Python Code

#### Imports
```python
# ✅ GOOD: Import at module level
import numpy as np
from pathlib import Path

def process_image(img_path):
    arr = np.array(...)

# ✗ BAD: Import inside function without try-except
def process_image(img_path):
    import imageio.v3 as iio  # May fail at runtime
    iio.imwrite(...)

# ✅ GOOD: Conditional import with fallback
def process_image(img_path):
    try:
        import tifffile
        tifffile.imwrite(...)
    except ImportError:
        import imageio.v3 as iio
        iio.imwrite(...)
```

#### Whitespace
```python
# ✅ GOOD: No trailing whitespace
def function():
    return value

# ✗ BAD: Trailing whitespace (will be auto-fixed)
def function():
    return value
```

#### Documentation
```python
# ✅ GOOD: Docstrings for public functions
def enhance_image(img, strength=0.7):
    """Apply AI enhancement to image.

    Args:
        img: Input image array
        strength: Enhancement strength (0-1)

    Returns:
        Enhanced image array
    """
    pass

# ✗ BAD: No documentation
def enhance_image(img, strength=0.7):
    pass
```

### File Organization

#### Root Directory (Max 10 Markdown Files)
```
✅ ALLOWED:
  - README.md (project overview)
  - START_HERE.md (quickstart)
  - MIGRATION_GUIDE.md (migration docs)
  - DEPRECATION_POLICY.md (policy docs)

✗ NOT ALLOWED:
  - Session summaries → docs/session_summaries/
  - Implementation reports → docs/
  - Archive material → docs/archive/
  - Project-specific docs → docs/projects/
```

#### Documentation Structure
```
docs/
├── session_summaries/     # Temporary session notes
├── archive/               # Old summaries, reports
├── projects/              # Project-specific docs
│   └── 750_PICACHO_*/     # Current project docs
├── Version_History/       # Change logs
└── depth_pipeline/        # Technical docs
```

---

## Quality Workflow

### Before Committing

1. **Run Tests Locally**
   ```bash
   make test-fast  # Quick validation
   ```

2. **Check Linting**
   ```bash
   flake8 . --select=E9,F63,F7,F82  # Critical only
   ```

3. **Organize Documentation**
   ```bash
   # If adding new markdown files
   mv NEW_DOC.md docs/session_summaries/
   ```

4. **Commit**
   ```bash
   git add <files>
   git commit -m "message"
   # Pre-commit hook runs automatically
   ```

### After Push

1. **Monitor CI** https://github.com/RC219805/Transformation_Portal/actions

2. **Fix Failures Immediately**
   - Don't stack commits on failing CI
   - Fix root cause, not symptoms

3. **Update Documentation**
   - If API changes, update docs
   - If new features, add examples

---

## Common Issues & Solutions

### Issue: `undefined name 'xyz'`

**Cause:** Variable used before import or definition

**Solution:**
```python
# Before
def func():
    xyz.method()  # ✗ xyz not imported

# After
import xyz  # ✅ Import at top

def func():
    xyz.method()
```

### Issue: `Too many markdown files in root`

**Cause:** Documentation not organized

**Solution:**
```bash
# Organize docs
mv SESSION_*.md docs/session_summaries/
mv *_SUMMARY.txt docs/archive/
git add docs/
```

### Issue: `Trailing whitespace`

**Cause:** Editor not configured to remove whitespace

**Solution:**
```bash
# Auto-fix all Python files
find . -name "*.py" -exec sed -i '' 's/[[:space:]]*$//' {} \;

# Or configure editor to remove on save
# VSCode: "files.trimTrailingWhitespace": true
```

### Issue: `Import cv2` warnings

**Cause:** cv2 imported but not always used

**Solution:**
```python
# Use lazy import
def process_with_opencv(img):
    import cv2  # Import only when needed
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

---

## Quality Metrics

### Current Status

| Metric                  | Current | Target | Status |
|-------------------------|---------|--------|--------|
| Test Coverage           | 85%     | 90%    | 🟨     |
| Passing Tests           | 548/549 | 100%   | 🟩     |
| Flake8 Critical Errors  | 0       | 0      | 🟩     |
| Pylint Score            | 9.27/10 | 9.0+   | 🟩     |
| Root Markdown Files     | 4       | ≤10    | 🟩     |

### Quality Trends

```
November 2025:
  Tests: 511 → 548 (+37)
  Coverage: 78% → 85% (+7%)
  Flake8 Errors: 2 → 0 (-2) ✅
  Root MD Files: 24 → 4 (-20) ✅
```

---

## Continuous Improvement

### Monthly Quality Review

**Schedule:** First Friday of each month
**Participants:** All contributors
**Agenda:**
1. Review quality metrics
2. Identify recurring issues
3. Update standards if needed
4. Improve automation

### Quality Improvement Ideas

**Short Term:**
- [ ] Add type hints to all public APIs
- [ ] Increase test coverage to 90%
- [ ] Add property-based testing for math functions
- [ ] Document all public classes

**Long Term:**
- [ ] Integrate mutation testing
- [ ] Add performance benchmarks
- [ ] Create visual regression tests
- [ ] Automated dependency updates

---

## Tools & Resources

### Required Tools
```bash
pip install flake8 pylint pytest pytest-cov hypothesis
```

### Recommended IDE Settings

**VSCode** (`.vscode/settings.json`):
```json
{
  "editor.rulers": [127],
  "files.trimTrailingWhitespace": true,
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": true,
  "python.testing.pytestEnabled": true
}
```

**PyCharm**:
- Settings → Editor → Code Style → Python → Wrap at: 127
- Settings → Editor → General → Strip trailing spaces on save
- Settings → Tools → External Tools → Add flake8, pylint

---

## Enforcement

### Required Checks (Must Pass)
1. ✅ Flake8 critical errors (E9, F63, F7, F82)
2. ✅ Python syntax validation
3. ✅ Pytest all tests passing
4. ✅ Markdown file count ≤ 10 in root

### Warning Checks (Should Pass)
1. ⚠️  Pylint score ≥ 9.0/10
2. ⚠️  Test coverage ≥ 85%
3. ⚠️  No debugging statements
4. ⚠️  Minimal print statements

### Emergency Override

**When:** Critical production fix needed immediately
**How:** Add `[skip ci]` to commit message
**Follow-up:** Create issue to fix quality violations within 24 hours

```bash
git commit -m "HOTFIX: Critical production issue [skip ci]"
# Then immediately:
gh issue create --title "Fix quality issues in hotfix XYZ"
```

---

## Questions & Support

**Documentation:** `docs/`
**GitHub Issues:** https://github.com/RC219805/Transformation_Portal/issues
**Pre-commit Hook:** `.git/hooks/pre-commit`
**CI Configuration:** `.github/workflows/build.yml`

---

**Version History:**
- 2.0 (Nov 8, 2025): Comprehensive overhaul, automated safeguards
- 1.0 (Oct 2025): Initial quality guidelines
