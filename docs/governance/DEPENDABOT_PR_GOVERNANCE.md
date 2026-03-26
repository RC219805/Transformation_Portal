# Dependabot PR Governance

**Purpose**: Define triage policy and merge criteria for Dependabot-generated pull requests
**Owner**: Transformation Portal Architect
**Created**: 2026-03-26
**Last Updated**: 2026-03-26

---

## Overview

This document establishes governance for reviewing and merging Dependabot PRs in accordance with the repository's enforced policy controls. Dependabot updates fall into two categories:

1. **GitHub Actions updates** (`package-ecosystem: "github-actions"`)
2. **Python dependency updates** (`package-ecosystem: "pip"`)

Each category has different risk profiles and merge criteria.

---

## Repository Policy Enforcement Points

The following are the enforced controls that constrain Dependabot PR acceptance:

| Control | Enforcement Location | Implication |
|---------|---------------------|-------------|
| Python `>=3.11` | `pyproject.toml` | Dependencies must support Python 3.11+ |
| Web stack exact pins | `requirements/base.in` | FastAPI/Starlette/uvicorn are intentionally pinned for API/UI parity |
| Action SHA pinning | `enforcement.yml` → `action-pins` job | All workflow actions must be SHA-pinned |
| Banned dependencies | `enforcement.yml` → `banned-dependencies` job | Blocked packages cannot be introduced |
| Dependency constraints | `build.yml` → `validate_constraints` job | ADR-032 constraint validation |

---

## Risk Classification

### GitHub Actions Updates

| Risk Level | Criteria | Examples | Merge Policy |
|------------|----------|----------|--------------|
| **Low** | Patch bump, narrow scope, non-blocking workflow | `download-artifact` 8.0.0 → 8.0.1 | Merge now (CI green) |
| **Low** | SHA bump, non-blocking workflow | `component-detection-dependency-submission-action` | Merge now (CI green) |
| **Low-Medium** | Minor/major bump, narrow scope, simple usage | `github-script` v7 → v8 (single issue workflow) | Merge after quick smoke |
| **Medium** | Major bump, post-merge validation required | `deploy-pages` v4 → v5 (Pages deployment) | Merge after targeted smoke |
| **Medium** | Major bump, non-blocking but sensitive workflow | `codecov-action` v4 → v5 (coverage upload) | Smoke first, do not blind-merge |

### Python Dependency Updates

| Risk Level | Criteria | Examples | Merge Policy |
|------------|----------|----------|--------------|
| **Low** | Patch bump, non-pinned dependency | General patches outside web stack | Merge after CI green |
| **Medium** | Minor bump, non-pinned dependency | Feature updates to utilities | Review changelog, test coverage |
| **High** | Any change to exact-pinned dependencies | `starlette`, `fastapi`, `uvicorn` | **HOLD** - requires curated PR |
| **Critical** | Major version bump to exact-pinned dependencies | `starlette` 0.x → 1.0 | **DO NOT MERGE** as routine bump |

---

## Exact-Pinned Web Stack Policy

The following dependencies are **exact-pinned** in `requirements/base.in` for API/UI parity and security baseline:

```
fastapi==0.135.1
starlette==0.52.1
uvicorn==0.42.0
aiofiles==25.1.0
```

### Why Exact Pins?

1. **API/UI Parity**: Ensures consistent behavior between orchestrator API and any UI consumers
2. **Security Baseline**: Pins are updated deliberately after CVE review (see ADR-032)
3. **Contract Stability**: Prevents subtle behavioral changes from affecting pipeline outputs

### Updating Exact-Pinned Dependencies

Dependabot PRs that bump exact-pinned dependencies **must not be merged as routine bumps**. Instead:

1. **Close** the Dependabot PR (or leave it open for reference)
2. **Create a curated PR** that:
   - Updates all related dependencies together (e.g., FastAPI + Starlette pair)
   - Validates direct imports in the codebase
   - Tests API/UI parity behavior
   - References changelog and migration notes
   - Documents CVE status if applicable

---

## Current Open Dependabot PRs (2026-03-26)

### Merge Order Recommendations

| Order | PR | Change | Risk | Recommendation |
|-------|-----|--------|------|----------------|
| 1 | #1273 | `actions/download-artifact` 8.0.0 → 8.0.1 | Low | ✅ **Merge now** |
| 2 | #1274 | `component-detection-dependency-submission-action` SHA bump | Low | ✅ **Merge now** |
| 3 | #1270 | `actions/github-script` 7.0.1 → 8.0.0 | Low-Medium | ✅ **Merge after quick smoke** |
| 4 | #1272 | `actions/deploy-pages` 4.0.5 → 5.0.0 | Medium | ⚠️ **Merge after targeted Pages smoke** |
| 5 | #1271 | `codecov/codecov-action` 4.6.0 → 5.5.3 | Medium | ⚠️ **Do not blind-merge; smoke first** |
| 6 | #1275 | `starlette` 0.52.1 → 1.0.0 | High | 🛑 **HOLD - do not merge as-is** |

### PR Details

#### #1273 - actions/download-artifact (MERGE NOW)
- **Scope**: `submit-pypi.yml` only (release upload jobs)
- **Impact**: Patch bump with minimal blast radius
- **CI Status**: Green
- **Validation**: PR CI is sufficient

#### #1274 - component-detection-dependency-submission-action (MERGE NOW)
- **Scope**: `dependency-submission.yml` (dependency graph workflow)
- **Impact**: SHA bump, intentionally non-blocking workflow
- **CI Status**: Green
- **Validation**: PR CI is sufficient

#### #1270 - actions/github-script (MERGE AFTER QUICK SMOKE)
- **Scope**: `issue_printer.yml` only (issues-triggered workflow)
- **Impact**: Major version (v7 → v8), Node 24 runtime
- **Runner Requirement**: v2.327.1+ (GitHub-hosted runners satisfy this)
- **Validation**: Single step workflow, low risk if on GitHub-hosted runners

#### #1272 - actions/deploy-pages (MERGE AFTER TARGETED SMOKE)
- **Scope**: `apex_performance.yml` → `dashboard_deploy` job
- **Impact**: Major version (v4 → v5), Pages/OIDC deployment
- **Limitation**: PR CI does NOT validate real deployment (runs only on `main`)
- **Validation**: Accept that first real validation is post-merge, or test manually

#### #1271 - codecov/codecov-action (SMOKE FIRST)
- **Scope**: `ci.yml` and `ci-quality-firewall.yml` coverage uploads
- **Impact**: Major version (v4 → v5), `fail_ci_if_error: false` mitigates risk
- **Limitation**: Coverage upload behavior may change; PR CI doesn't fully validate
- **Validation**: Test coverage upload on a non-critical branch before merge

#### #1275 - starlette (HOLD - DO NOT MERGE)
- **Scope**: Core web stack dependency
- **Impact**: Framework major (0.x → 1.0), exact-pinned dependency
- **Conflict**: Violates `requirements/base.in` exact-pin policy
- **mergeable_state**: `behind` (not mergeable as-is)
- **Required Action**: Close this PR; create a curated compatibility PR that validates FastAPI/Starlette pair

---

## Batching Policy

**Do NOT batch all Dependabot PRs.**

### Acceptable Batching

- PRs #1273 + #1274 (both low-risk, workflow-only, CI green)

### Must Be Separate

- #1270 (major action runtime change)
- #1272 (post-merge validation required)
- #1271 (coverage behavior validation required)
- #1275 (runtime dependency, exact-pin violation)

---

## Enforcement Checklist

Before merging any Dependabot PR:

- [ ] CI is green on the PR
- [ ] Change does not modify exact-pinned dependencies in `requirements/base.in`
- [ ] For major action bumps: runner compatibility verified
- [ ] For post-merge-validation workflows: risk accepted or manual smoke completed
- [ ] No batching of unrelated risk levels

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-26 | Initial creation with PR #1270-#1275 assessment | Architect |

---

**Maintained by**: Transformation Portal Architect
**Review Frequency**: As needed (when new policy patterns emerge)
