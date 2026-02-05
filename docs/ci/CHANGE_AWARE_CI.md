# Change-Aware CI Quick Reference

## Overview

Change-aware CI uses path-based workflow triggers to skip irrelevant jobs on PRs, reducing CI runtime and cost while maintaining safety.

## How It Works

### Path Filters

PRs trigger workflows based on which files changed:

**Code/Test Changes** → Full CI Suite
- `src/**`, `tests/**`, `scripts/**`
- `requirements*.txt`, `pyproject.toml`
- `.github/workflows/**`

**Documentation-Only Changes** → Docs Build Only
- `docs/**`, `*.md`, `README*`
- Skips: test jobs, lint (except doc validation)

### Safety Guarantees

✅ **Always run on main**: No filters on protected branches
✅ **Manual override**: `workflow_dispatch` runs full suite
✅ **Conservative filters**: False positives OK, false negatives NOT OK
✅ **Explicit skip messages**: Clear logs when jobs skipped

## Expected Time Savings

| PR Type | Time Savings | Example |
|---------|--------------|---------|
| Doc-only | 70%+ | 10min → 3min |
| Test-only | 30%+ | 12min → 8min |
| Workflow-only | 0% | Full suite |
| Code changes | 0% | Full suite |

## Manual Override

Run full suite on any PR:

```bash
# Via GitHub UI
Actions → CI → Run workflow → Select branch

# Or via gh CLI
gh workflow run "CI (Lint, Tests & Manifest)" --ref your-branch-name
```

## Validation

Check path filter configuration:

```bash
python scripts/validate_path_filters.py
```

## Rollback

If filters cause issues:

```bash
git revert <commit-sha>
git push origin main
```

## Monitoring

Track effectiveness:
- Compare PR completion times (before/after)
- Review workflow logs for skip messages
- Monitor for false negatives (required checks missed)

## Related

- **ADR-0016**: Design decision and rationale
- **build.yml**: Primary CI workflow with filters
- **docs.yml**: Documentation workflow (already filtered)

## Last Updated

2026-02-04
