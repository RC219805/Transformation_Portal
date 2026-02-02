# ADR-020: Drop Python 3.10 Support

**Status:** Adopted
**Date:** 2026-02-02
**Authority:** Transformation Portal Architect
**Supersedes:** None
**Related:** PR #793a (Python 3.10 drop), PR #793b (ML stack upgrades)

---

## Context

Python 3.10 reached end-of-life for mainstream support in October 2026. The repository currently declares `requires-python = ">=3.10"` but CI workflows already test only Python 3.11.

### Ecosystem Signal

**scikit-learn 1.8.0** (released January 2026) dropped Python 3.10 support. This is a significant ecosystem signal:
- scikit-learn is a mature, widely-deployed core ML dependency
- The scikit-learn team has a conservative compatibility policy
- Their decision reflects broad Python 3.10 obsolescence in the scientific Python ecosystem

### Current State

- **pyproject.toml:** Declares `requires-python = ">=3.10"`
- **CI Actual:** Only tests Python 3.11 (de facto unsupported)
- **Production Use:** No known Python 3.10 deployments ("unused to my knowledge" per maintainer)
- **Dependency Pressure:** scikit-learn 1.8.0 requires Python >=3.11

---

## Decision

**Drop Python 3.10 support. Set minimum required Python version to 3.11.**

### Rationale

1. **CI Reality:** Python 3.10 is already untested in CI (no matrix entry)
2. **Ecosystem Alignment:** Follow mature dependencies (scikit-learn) that have made this transition
3. **No Production Impact:** No identified production deployments using Python 3.10
4. **Modern Baseline:** Python 3.11 (released October 2022) is reasonable minimum for 2026
5. **Cleaner Messaging:** Align declared support (`requires-python`) with tested support (CI matrix)

### Non-Decision

This ADR **intentionally does not** upgrade any dependencies. Dependency version bumps are deferred to a separate PR (#793b) to minimize risk and improve reviewability.

---

## Consequences

### Breaking Changes

- **Users on Python 3.10 must upgrade to Python 3.11+** before installing future versions
- This is a **breaking change** requiring a changelog entry and clear communication

### Benefits

1. **Honest Declaration:** `requires-python` now matches CI coverage
2. **Dependency Freedom:** Unblocks future upgrades to packages dropping 3.10 support
3. **Simplified Testing:** No need to maintain/test 3.10-specific compatibility
4. **Modern Features:** Can leverage Python 3.11+ features without compatibility concerns

### Risks (Mitigated)

- **Risk:** Users unknowingly on Python 3.10 will face install failures
- **Mitigation:** Clear changelog, README update, and pip will fail fast with clear error message

### Migration Path

For users on Python 3.10:
```bash
# Check current Python version
python --version

# Upgrade Python (example using pyenv)
pyenv install 3.11
pyenv global 3.11

# Or use system package manager
# Ubuntu/Debian: sudo apt install python3.11
# macOS: brew install python@3.11
```

---

## Implementation

### Changes Required (Minimal)

1. **pyproject.toml:**
   - `requires-python = ">=3.11"` (was `">=3.10"`)
   - `target-version = ["py311", "py312"]` in black config (was `["py310", "py311", "py312"]`)

2. **README.md:**
   - Update badge: `python-3.11%2B` (was `python-3.10%2B`)

3. **CHANGELOG.md:**
   - Document breaking change in `[Unreleased]` section

4. **CI Workflows:**
   - Update `.github/workflows/build.yml` matrix to test 3.11 and 3.12 (remove 3.10)
   - Update `.github/workflows/ci.yml` matrix to test 3.11 and 3.12 (remove 3.10)
   - Update `.github/workflows/python-app.yml` comment to reflect 3.11+ support

### What Does NOT Change

- No dependency version bumps (kept for PR #793b)
- No code changes (purely policy/declaration)

---

## Validation

### Pre-Merge Checklist

- [x] `pyproject.toml` updated
- [x] README badge updated
- [x] README system requirements updated
- [x] CHANGELOG.md entry added
- [x] ADR-020 created
- [x] CI workflows updated to remove Python 3.10
- [x] No dependency version changes (verified)
- [x] No test changes required (except structure test for CHANGELOG.md)

### Post-Merge Validation

- [ ] Pip install fails cleanly on Python 3.10 with clear error
- [ ] Pip install succeeds on Python 3.11+
- [ ] CI remains green (no behavior change)

---

## Rollout Strategy

This is **PR #793a** in a two-part rollout:
- **PR #793a (this):** Policy change only (Python 3.10 drop)
- **PR #793b (follows):** ML stack upgrades (torch, scikit-learn, etc.)

**Merge Order:** #793a must merge first. #793b depends on #793a.

---

## Alternatives Considered

### Alternative 1: Keep Python 3.10, Pin scikit-learn <1.8

**Rejected:** Holding back ecosystem dependencies is not sustainable. Would accumulate technical debt.

### Alternative 2: Add Python 3.10 to CI Matrix

**Rejected:** No production usage identified. Adding CI coverage for unused version wastes resources.

### Alternative 3: Combine with Dependency Upgrades (Original PR #793)

**Rejected:** Too much risk in single PR. Split strategy improves reviewability and rollback capability.

---

## References

- [Python 3.10 Release Schedule (PEP 619)](https://peps.python.org/pep-0619/)
- [scikit-learn 1.8.0 Release Notes](https://scikit-learn.org/stable/whats_new/v1.8.html)
- PR #793 (original combined PR, superseded)
- PR #793a (this decision)
- PR #793b (ML stack upgrades, depends on this)
