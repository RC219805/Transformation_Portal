# Dependency Update Process

This document describes the automated dependency update workflow and how to review dependency update PRs.

## 📋 Overview

The Transformation Portal uses an automated dependency update system that:
- Runs weekly on Mondays at 9 AM UTC (via GitHub Actions)
- Updates all Python dependencies to their latest allowed versions
- Creates a PR with the changes
- Performs automated validation and security scanning
- Can be triggered manually via workflow dispatch

## 🔄 Automated Workflow

### What Happens Automatically

The `dependency-update.yml` workflow performs these steps:

1. **Dependency Update**
   - Uses `pip-compile` to update `requirements/*.txt` files
   - Respects version constraints in `requirements/*.in` files
   - Follows the layered requirements system (base, ml, dev, ci, all)

2. **Security Scanning**
   - Runs Safety 3.x vulnerability scanner on `requirements/all.txt`
   - Generates `safety-report.json` (uploaded as workflow artifact)
   - Does not block PR creation (report-only mode)

3. **Validation Gates** ✅ NEW
   - Uploads security report as workflow artifact (30-day retention)
   - Runs pre-commit checks on all changed files
   - Tests installation on Python 3.10 (minimum supported version)
   - Tests installation on Python 3.12 (latest supported version)
   - YAML parsing smoke test (validates ruamel-yaml upgrade)
   - Import checks (validates Python version markers)

4. **PR Creation**
   - Creates PR on branch `automated/dependency-updates`
   - Includes summary of changes and validation results
   - Labels: `dependencies`, `automated`

## 🔍 Reviewing Dependency Update PRs

### Quick Checklist

When reviewing an automated dependency PR (e.g., PR #656):

- [ ] **Check CI Status** - Ensure all automated validation checks passed
- [ ] **Review Security Report** - Download `safety-report` artifact from workflow run
- [ ] **Check Breaking Changes** - Review the diff for major version bumps
- [ ] **Verify Python Compatibility** - Ensure tests passed on Python 3.10, 3.11, 3.12
- [ ] **Test YAML Workflows** - If ruamel-yaml was updated, test config parsing
- [ ] **Run Local Smoke Tests** - See "Local Testing" section below

### Understanding the Diff

Dependency update PRs will typically change these files:

```
requirements/base.txt     # Core runtime dependencies
requirements/ml.txt       # ML and deep learning packages
requirements/dev.txt      # Development tools
requirements/ci.txt       # CI/CD tools
requirements/all.txt      # Combined requirements
```

**What to look for:**

1. **Version Bumps** - Major changes (e.g., `certifi 2023.7.22 → 2024.8.30`)
2. **New Dependencies** - Transitive dependencies added/removed
3. **Removed Packages** - Dependencies that are no longer needed
4. **Version Markers** - Environment markers like `; python_version < "3.11"`

### Security Report Location

⚠️ **Important:** The security report is **NOT committed** to the PR. It's available as a **workflow artifact**.

**How to access:**
1. Go to the dependency update workflow run (Actions tab)
2. Scroll to "Artifacts" section at the bottom
3. Download `safety-report.json`
4. Review for any critical or high-severity vulnerabilities

### CI Validation Steps

The automated workflow validates:

| Validation Step | What It Tests | Why It Matters |
|----------------|---------------|----------------|
| **Python 3.10 Install** | `base.txt` + `dev.txt` installation | Ensures minimum Python version works |
| **Python 3.12 Install** | `base.txt` + `dev.txt` installation | Ensures latest Python version works |
| **Pre-commit Checks** | Code quality, security, formatting | Catches common issues early |
| **YAML Smoke Test** | ruamel-yaml round-trip parsing | Detects ruamel-yaml breaking changes |
| **Import Validation** | Python version marker gating | Ensures backports are correctly gated |

## 🧪 Local Testing

### Manual Validation Steps

Before merging a dependency update PR:

#### 1. Test Installation (Python 3.12)

```bash
# Create clean virtual environment
python3.12 -m venv .venv-test
source .venv-test/bin/activate

# Install updated dependencies
pip install -c requirements/constraints.txt -r requirements/base.txt
pip install -c requirements/constraints.txt -r requirements/dev.txt

# Verify imports
python -c "import numpy, PIL, yaml; print('✅ Core imports successful')"
```

#### 2. YAML Config Smoke Test (if ruamel-yaml updated)

```bash
# Test YAML parsing with a real config file
python -c "
from ruamel.yaml import YAML
yaml = YAML()

# Test with actual pipeline config
with open('config/interior_preset.yaml', 'r') as f:
    data = yaml.load(f)
    print(f'✅ Loaded config: {data.get(\"name\", \"unknown\")}')
"
```

#### 3. Pipeline Hello World Test

```bash
# Quick smoke test of core pipeline
python -c "
from pathlib import Path
import sys

# Test that basic pipeline imports work
try:
    from lux_depth_v2.config import PipelineConfig
    cfg = PipelineConfig()
    print(f'✅ Pipeline config loads: preset={cfg.preset}')
except ImportError as e:
    print(f'⚠️  lux_depth_v2 not available: {e}')
"
```

#### 4. Check for Vulnerable Packages

```bash
# Verify basicsr is not installed (CVE-2024-27763)
if pip show basicsr > /dev/null 2>&1; then
  echo "❌ SECURITY VIOLATION: basicsr is installed!"
  exit 1
else
  echo "✅ basicsr not installed (as expected)"
fi
```

## 🚨 Known Issues and Gotchas

### ruamel-yaml 0.18 → 0.19 Upgrade

**Impact:** ruamel-yaml 0.19 dropped the `ruamel-yaml-clib` C extension dependency.

**What to watch for:**
- Performance changes in YAML parsing (though unlikely to be noticeable)
- Edge case behavior changes in YAML round-trip preservation
- Platform-specific issues (especially on older systems)

**Mitigation:**
- The YAML smoke test in the workflow catches basic issues
- Test with actual pipeline configs before merging
- Monitor for issues in production after deployment

### backports-asyncio-runner Python Version Gating

**Issue:** `backports-asyncio-runner` should only install on Python < 3.11.

**Validation:**
- The workflow automatically checks this with import validation
- On Python 3.12, `importlib.util.find_spec('asyncio_runner')` should return `None`

**If the check fails:**
- Verify `requirements/dev.txt` and `requirements/all.txt` have the marker:
  ```
  backports-asyncio-runner==1.2.0 ; python_version < "3.11"
  ```

### Large Dependency Changes

If the PR shows many dependencies changing:

1. Check if `constraints.txt` was modified (security exclusions)
2. Look for transitive dependency cascades (one change affects many)
3. Verify no CVE-2024-27763 related packages snuck in (basicsr, realesrgan, gfpgan)

## 🛠️ Troubleshooting

### Workflow Fails on Validation Steps

**Symptom:** Python 3.10 or 3.12 installation test fails

**Diagnosis:**
1. Check the workflow logs for the specific error
2. Look for dependency conflicts or missing packages
3. Verify `constraints.txt` isn't blocking required packages

**Fix:**
- Update version constraints in `requirements/*.in` files
- Re-run `make update` in `requirements/` directory
- Test locally before pushing

### Security Report Shows Critical Vulnerabilities

**Symptom:** `safety-report.json` contains critical/high-severity issues

**Diagnosis:**
1. Review the vulnerability details in the JSON report
2. Check if patches are available for the vulnerable packages
3. Determine if the vulnerability affects our usage

**Fix:**
- If patch available: Update version constraint in `.in` file and re-compile
- If no patch: Add to `constraints.txt` to exclude the package
- Document the decision in `requirements/constraints.txt` comments

### Pre-commit Checks Fail

**Symptom:** Pre-commit hook failures in the workflow

**Common causes:**
- Trailing whitespace in requirements files
- YAML syntax errors
- Large files accidentally included

**Fix:**
- Run `pre-commit run --all-files` locally
- Fix issues and commit
- Re-run the workflow

## 📚 References

- [Layered Dependency Management](../requirements/README.md)
- [pip-tools Documentation](https://github.com/jazzband/pip-tools)
- [Safety 3.x Documentation](https://docs.safetycli.com/)
- [CVE-2024-27763 Mitigation](../requirements/constraints.txt)

## 🔗 Related Workflows

- `.github/workflows/dependency-update.yml` - Main automation workflow
- `.github/workflows/ci-consolidated.yml` - CI validation on PRs
- `.github/workflows/security-unified.yml` - Security scanning
- `.github/workflows/dependency-submission.yml` - GitHub dependency graph

## 📝 Change Log

- **2026-01-05**: Enhanced workflow with validation gates (Python 3.10/3.12, pre-commit, smoke tests)
- **2025**: Initial automated dependency update workflow implemented
