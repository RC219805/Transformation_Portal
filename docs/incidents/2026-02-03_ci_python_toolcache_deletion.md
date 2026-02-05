# Incident Report: CI Python Toolcache Deletion

**Date**: 2026-02-03
**Severity**: High (CI completely broken on main)
**Status**: ✅ Resolved
**Duration**: ~3 hours (identification → stabilization → root cause fix)

## Summary

The CI Quality Firewall workflow was failing consistently because the "Free disk space" cleanup step was deleting `$AGENT_TOOLSDIRECTORY` **after** Python was installed, removing the interpreter that subsequent steps needed to run tests.

## Timeline

### Discovery
- Multiple CI runs on main failing with "No such file or directory" when trying to execute Python
- Issue #806 opened to track investigation

### Incident Response
1. **Stabilization** (PR #812): Merged fallback logic + diagnostic telemetry to stop the bleeding
2. **Root Cause Analysis**: Examined CI logs from stabilized runs
3. **Real Fix** (PR #813): Reordered workflow steps to prevent toolcache deletion

### Resolution
- PR #813 merged with step reorder fix
- CI fully green without fallback logic
- Follow-up cleanup to remove temporary workarounds

## Root Cause

### The Problem

The `Free disk space` step was positioned **after** `actions/setup-python@v5`:

```yaml
steps:
  - uses: actions/checkout@v4

  - name: Set up Python 3.11                    # ← Installs to /opt/hostedtoolcache
    id: setup-python-ml
    uses: actions/setup-python@v5
    with:
      python-version: "3.11"

  - name: Free disk space                       # ← Deletes the toolcache!
    run: |
      sudo rm -rf "$AGENT_TOOLSDIRECTORY"
```

### Evidence from CI Logs

Run [21651099423](https://github.com/RC219805/Transformation_Portal/actions/runs/21651099423):

```
22:57:59.869 - Set up Python 3.11 completes
               pythonLocation: /opt/hostedtoolcache/Python/3.11.14/x64

22:57:59.882 - Free disk space begins
               Executes: sudo rm -rf "$AGENT_TOOLSDIRECTORY"
               → Deletes /opt/hostedtoolcache

22:58:32.092 - Install dependencies tries:
               ${{ steps.setup-python.outputs.python-path }}
               → /opt/hostedtoolcache/Python/3.11.14/x64/bin/python
               → ERROR: No such file or directory
```

### Why It Happened

- GitHub Actions stores installed tools in `$AGENT_TOOLSDIRECTORY` (typically `/opt/hostedtoolcache`)
- The cleanup step was intended to free disk space for ML tests
- Moving the cleanup after Python setup was a logical change (to maximize free space)
- But it inadvertently deleted the Python interpreter itself

## The Fix

**Simple step reorder** - cleanup before installation:

```yaml
steps:
  - uses: actions/checkout@v4

  - name: Free disk space                       # ← Now runs FIRST
    run: |
      sudo rm -rf "$AGENT_TOOLSDIRECTORY"
      # ... other cleanup ...

  - name: Set up Python 3.11                    # ← Installs after cleanup
    id: setup-python-ml
    uses: actions/setup-python@v5
    with:
      python-version: "3.11"
```

Python gets installed into a clean toolcache, no conflicts.

## PRs Involved

### #812 - Incident Stabilization (Merged)
- **Purpose**: Stop the bleeding with fallback logic
- **Strategy**: If `python-path` doesn't exist, try multiple fallback locations
- **Outcome**: Stabilized main, captured diagnostic telemetry
- **Status**: ✅ Merged, served its purpose

### #813 - Root Cause Fix (Merged)
- **Purpose**: Eliminate the root cause
- **Strategy**: Reorder "Free disk space" before "Set up Python"
- **Changes**: 12 lines moved (no logic changes)
- **Outcome**: CI fully green without workarounds
- **Status**: ✅ Merged

### Follow-up - Cleanup
- **Purpose**: Remove #812's fallback logic
- **Timing**: After #813 proven stable (multiple green runs on main)
- **Scope**: Remove ~100 lines of diagnostic/fallback code

## Lessons Learned

### What Went Well
1. **Fast stabilization**: #812 merged quickly to stop CI bleeding
2. **Good telemetry**: Diagnostic output made root cause obvious
3. **Minimal fix**: Step reorder is clean, low-risk, easy to review
4. **Credit to Copilot**: AI code review identified the theory correctly

### What Could Improve
1. **Workflow validation**: Could add a pre-commit check to validate step ordering
2. **CI testing**: Could test workflow changes in a branch before main merge
3. **Documentation**: Disk cleanup strategy should be documented with rationale

### Process Improvements
1. **ADR for workflow structure**: Document why steps are ordered as they are
2. **CI workflow test suite**: Validate critical step dependencies
3. **Incident response template**: This report serves as a template for future incidents

## Impact Assessment

### Blast Radius
- **Affected**: CI Quality Firewall workflow (main and develop branches)
- **Not affected**: Production code, deployed artifacts, user workflows
- **Duration**: ~3 hours from first failure to stabilization

### Risk During Incident
- ❌ Could not verify PRs with required CI checks
- ❌ Could not merge changes safely
- ✅ No production impact (CI only)
- ✅ No security implications

## Prevention

### Immediate
- ✅ Step reorder applied to both `test-core` and `test-ml` jobs
- ✅ CI passing consistently without workarounds

### Long-term
- Consider pinning `actions/setup-python` to specific SHA
- Document critical step dependencies in workflow comments
- Add workflow validation to pre-commit checks

## References

- **Issue**: #806
- **Stabilization PR**: #812
- **Fix PR**: #813
- **CI Run (evidence)**: [21651099423](https://github.com/RC219805/Transformation_Portal/actions/runs/21651099423)
- **Copilot Analysis**: PR #812 code review comments

---

**Report Author**: Transformation Portal Architect
**Report Date**: 2026-02-03
**Last Updated**: 2026-02-03
