# Disk Space Issues in CI - Solutions Applied

> Current status note (2026-05-12): This is a historical incident note, not the
> current CI contract. For current workflow ownership and blocking status, use
> `docs/ci/WORKFLOW_MATRIX.md`. Current relevant surfaces are
> `.github/workflows/build.yml` for the primary `CI Gate` and lint/test
> installation posture, and `.github/workflows/dependency-submission.yml` for
> GitHub dependency graph submission.

## Problem
GitHub Actions workflows were failing with `OSError: [Errno 28] No space left on device` errors due to large dependency installations.

## Failures Fixed

### 1. CI Lint Job (`.github/workflows/build.yml`)
**Issue**: Installing 5-8GB of ML dependencies (torch, transformers, diffusers, etc.) just to lint a few Python files.

**Current status**:
- `build.yml` remains the primary required PR gate through the stable `CI Gate`
  aggregator.
- Lint tooling is still isolated through `requirements-lint.txt`.
- The workflow now uses current cache and install policy from `build.yml`; do
  not treat this historical note as the source of truth for exact cache flags.

### 2. Dependency Submission Workflow
**Issue**: Dependency graph submission previously risked large dependency
resolution and runner disk pressure in this ML-heavy repository.

**Current status**:
- `.github/workflows/dependency-submission.yml` is the custom dependency graph
  submission workflow.
- It runs `advanced-security/component-detection-dependency-submission-action`
  with disk cleanup, pip cache controls, and directory exclusions.
- `.github/dependency-submission-config.yml` documents the current local
  configuration intent.

**To disable GitHub's automatic dependency submission**:
1. Go to repository **Settings**
2. Navigate to **Code security and analysis**
3. Under **Dependency graph** → **Dependency submission**
4. Disable **"Automatic dependency submission"**

### 3. AI Code Review Syntax Error (`.github/workflows/ai-code-review.yml`)
**Issue**: Invalid Python syntax with emoji characters outside strings.

**Solution**:
- Fixed system_prompt definition
- Removed emoji characters from Python code
- Fixed changes_text building logic
- Simplified comment posting

## Files Modified

- `.github/workflows/build.yml` - Optimized lint and test jobs
- `requirements-lint.txt` (new) - Minimal dependencies for linting
- `.github/workflows/dependency-submission.yml` (new) - Custom dependency graph submission
- `.github/workflows/ai-code-review.yml` - Fixed Python syntax errors
- `.github/dependency-submission-config.yml` (new) - Configuration documentation

## Disk Space Savings Summary

| Workflow | Before | After | Savings |
|----------|--------|-------|---------|
| Lint Job | ~8GB | ~500MB | ~7.5GB |
| Dependency Submission | ~8GB | ~200MB | ~7.8GB |
| Test Jobs (each) | ~10GB | ~8GB | ~2GB |

**Total disk space saved per CI run**: ~15-20GB
