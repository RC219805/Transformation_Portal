# Dependabot PR Governance

**Purpose**: Define triage policy and merge criteria for Dependabot-generated pull requests
**Owner**: Transformation Portal Architect
**Created**: 2026-03-26
**Last Updated**: 2026-08-20

---

## Overview

This document establishes governance for reviewing and merging Dependabot PRs in accordance with the repository's enforced policy controls. Dependabot updates fall into three categories:

1. **GitHub Actions updates** (`package-ecosystem: "github-actions"`)
2. **Python dependency updates** (`package-ecosystem: "pip"`)
3. **Node dependency updates** (`package-ecosystem: "npm"`) for the paired root/Worker manifests and `/web/secure-landing`

Each category has different risk profiles and merge criteria.
All configured Dependabot entries carry the `dependencies` and `automated`
labels and use staggered Tuesday UTC schedules so update PRs remain
classifiable without arriving in one burst.

Current grouping and compatibility controls keep coupled changes atomic:

- `github/codeql-action/init` and `github/codeql-action/analyze` are grouped as
  one action-family update and must retain the same immutable release SHA. The
  Dependabot contract fails if that group remains configured after the last
  live `github/codeql-action/*` workflow reference is removed.
- Root and Cloudflare Worker `wrangler` updates are grouped across both npm
  directories with the Worker's `@cloudflare/workers-types` peer. The coupled
  group is validated together before merge by
  `python3 scripts/validation/check_worker_dependency_parity.py`, including
  stable numeric manifest pins, the complete shared lock graph, and the
  Wrangler peer range.
- Frontdoor security updates are grouped separately from routine version
  updates. Its Dependabot entry intentionally omits `target-branch` so GitHub
  applies security grouping while targeting the repository default branch,
  `main`.
- Redis major updates and core Transformers minor updates are ignored because
  the governed contracts remain Redis `<7` and Transformers `>=5.5,<5.6`.
  Compatible patch updates remain eligible.

---

## Repository Policy Enforcement Points

The following are the enforced controls that constrain Dependabot PR acceptance:

| Control | Enforcement Location | Implication |
|---------|---------------------|-------------|
| Python `>=3.11` | `pyproject.toml` | Dependencies must support Python 3.11+ |
| Web stack exact pins | `requirements/base.in` | FastAPI/Starlette/uvicorn/websockets are intentionally pinned for API/UI parity |
| Action SHA pinning | `enforcement.yml` → `action-pins` job | Third-party actions must be SHA-pinned; official `actions/*@v...` tags are currently allowed, and enforcement is strictest in critical workflows |
| Banned dependencies | `enforcement.yml` → `banned-dependencies` job | Blocked packages cannot be introduced |
| Dependency constraints | `build.yml` → `dependency-constraints` job; `ci-quality-firewall.yml` → `validate-dependency-constraints` job | ADR-032 constraint validation |
| Node runtime boundaries | `web/secure-landing/package.json`; `cloudflare/transformationportal-worker/package.json`; root `package.json` | Frontdoor updates must preserve Node 22, exact Next pin posture, and Cloudflare Worker tooling separation |

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
| **High** | Any change to exact-pinned dependencies | `starlette`, `fastapi`, `uvicorn`, `websockets` | **HOLD** - requires curated PR |
| **Critical** | Major version bump to exact-pinned dependencies | `starlette` 0.x → 1.0 | **DO NOT MERGE** as routine bump |

### Node Dependency Updates

Dependabot tracks three checked-in npm lockfile roots:

- `/` for root worker-build tooling.
- `/web/secure-landing` for the managed Next frontdoor.
- `/cloudflare/transformationportal-worker` for the frontdoor-only Cloudflare Worker package.

Root and Worker updates share one multi-directory entry so Wrangler stays in
parity. Other minor and patch version updates are grouped by runtime surface to
reduce PR noise. Major updates remain separate PRs and must be reviewed as
compatibility changes.

Run `python3 scripts/validation/check_worker_dependency_parity.py` after any
root/Worker manifest or lockfile change. `make validate-ci` and the scheduled
dependency updater run the same check before accepting or generating updates.

| Risk Level | Criteria | Examples | Merge Policy |
|------------|----------|----------|--------------|
| **Low** | Patch bump in isolated tooling | `wrangler` patch, `@cloudflare/workers-types` patch | Merge after package-lock diff and relevant dry-run/test pass |
| **Medium** | Minor bump in frontdoor runtime or test tooling | `@playwright/test`, `stylelint`, `argon2` minor | Run `make test-frontdoor-contract` or narrower frontdoor lane before merge |
| **High** | Exact-pinned frontdoor framework or lock metadata change | `next`, npm package-manager metadata | Require frontdoor contract/build validation and package-lock review |
| **Critical** | Major framework/runtime boundary change | `next` major, React major, Node engine range change | Curated PR only; do not routine-merge |

---

## Exact-Pinned Web Stack Policy

The following dependencies are **exact-pinned** in `requirements/base.in` for API/UI parity and security baseline:

```
fastapi==0.141.1
starlette==1.3.1
uvicorn==0.52.1
websockets==17.0.1
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

## Resolved Dependabot Wave (2026-03-26)

### Merge Order Recommendations

| Order | PR | Change | Risk | Outcome |
|-------|-----|--------|------|---------|
| 1 | #1273 | `actions/download-artifact` 8.0.0 → 8.0.1 | Low | ✅ Merged |
| 2 | #1274 | `component-detection-dependency-submission-action` SHA bump | Low | ✅ Merged |
| 3 | #1270 | `actions/github-script` 7.0.1 → 8.0.0 | Low-Medium | ✅ Merged after validation |
| 4 | #1272 | `actions/deploy-pages` 4.0.5 → 5.0.0 | Medium | ✅ Merged after targeted review |
| 5 | #1271 | `codecov/codecov-action` 4.6.0 → 5.5.3 | Medium | ✅ Merged after targeted review |
| 6 | #1275 | `starlette` 0.52.1 → 1.0.0 | Critical | ✅ Closed in favor of curated issue/PR flow |
| 7 | #1278 | Curated Starlette 1.0 compatibility PR | High | ✅ Merged |

### PR Details

#### #1273 - actions/download-artifact (MERGE NOW)
- **Scope in this PR**: `submit-pypi.yml` only (release upload jobs)
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

#### #1275 - starlette (CLOSED, REPLACED BY CURATED PR)
- **Scope**: Core web stack dependency
- **Impact**: Framework major (0.x → 1.0), exact-pinned dependency
- **Conflict**: Violates `requirements/base.in` exact-pin policy
- **Disposition**: Closed and replaced with issue `#1277` + curated PR `#1278`
- **Result**: Starlette 1.0 validated and merged without bundling the invalid cross-platform ML lock regeneration

#### #1278 - curated Starlette compatibility (MERGED)
- **Scope**: `pyproject.toml`, `requirements/base.in`, `requirements/base.txt`, `requirements/all.txt`
- **Validated Set**: FastAPI `0.136.0`, Starlette `1.0.0`, Uvicorn `0.42.0`
- **Validation**: `make test-orchestrator-contract`, `make ci`, curated live orchestrator smoke
- **Governance Lesson**: Exact-pinned web stack updates require issue-first / PR-second handling when Dependabot crosses a compatibility boundary

#### #2042 - curated FastAPI compatibility
- **Scope**: FastAPI source bounds, generic locks, FastVLM subprocess manifest, contract tests, and current baseline documentation
- **Validated Set**: FastAPI `0.141.1`, Starlette `1.3.1`, Uvicorn `0.52.1`
- **Validation**: generic-lock freshness, orchestrator/API/dashboard/frontdoor contracts, FastVLM isolated install, and full CI
- **Governance Lesson**: Dependabot source-bound changes are incomplete until exact locks and every governed baseline consumer move in the same reviewed change

---

## "Dep Pin Changed" Checklist

Use this checklist whenever a governed dependency pin changes, especially under
`requirements/base.in` or the compiled lock contract:

- [ ] Update the manifest input first (`requirements/*.in`, and `pyproject.toml`
      only when the compatibility bound must move with it).
- [ ] Regenerate only the affected governed lockfiles through the existing lock
      workflow; do not hand-edit compiled lock output.
- [ ] Re-run `make check-requirements-lock-contract` after regeneration.
- [ ] Update workflow or contract tests that encode action pins, lock paths, or
      compatibility expectations.
- [ ] Update the relevant dependency-governance docs so the recorded baseline,
      validation path, and runtime/toolchain requirements stay current.
- [ ] Re-run the validation path for the touched surface (`make ci` at minimum;
      add frontdoor or browser validation when the web/runtime boundary moved).

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
- [ ] For Node major bumps: frontdoor/worker compatibility and package-lock platform metadata reviewed
- [ ] For post-merge-validation workflows: risk accepted or manual smoke completed
- [ ] No batching of unrelated risk levels

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-26 | Initial creation with PR #1270-#1275 assessment | Architect |
| 2026-03-26 | Updated after #1275 closure and curated Starlette merge via #1278 | Architect |
| 2026-04-16 | Recorded the FastAPI 0.136.0 curated baseline and the "dep pin changed" checklist | Architect |
| 2026-04-16 | Added ML alert dismissal/remediation governance for supported vs frozen target-owned lanes | Architect |
| 2026-04-23 | Synced the exact-pinned web stack block with the curated Uvicorn 0.45.0 baseline | Architect |
| 2026-05-12 | Added npm Dependabot coverage and per-directory grouped minor/patch policy | Architect |
| 2026-05-26 | Recorded the curated Starlette 1.0.1 patch for PYSEC-2026-161 and synced the current exact-pinned web stack baseline | Architect |
| 2026-05-26 | Recorded the curated Python/runtime refresh for Uvicorn 0.48.0, SQLAlchemy 2.0.50, diff-cover 10.2.1, and FastVLM idna/fsspec pins | Architect |
| 2026-06-10 | Added PyTorch alert wave triage for the `torch==2.12.0` supported baseline rotation and no-patch `torch.jit.script` dismissals | Architect |
| 2026-08-04 | Grouped CodeQL, root/Worker Wrangler, and frontdoor security updates; guarded Redis and Transformers compatibility bounds | Architect |
| 2026-08-06 | Rotated supported ML and subprocess runtimes to Pillow 12.3.0, torch 2.13.0 / torchvision 0.28.0, and Transformers 5.5.x after patched releases became available | Architect |
| 2026-08-19 | Curated FastAPI 0.141.1 via PR #2042 and synchronized its locks, runtime manifest, contract tests, and current web-stack baseline | Architect |

---

## ML Alert Governance (2026-04-16)

Dependabot ML alert waves must be triaged by advisory reachability and lane support status, not by raw alert count.

### Default ML dispositions

| Scenario | Disposition | Notes |
|----------|-------------|-------|
| Supported target-owned lock (`ml-core-darwin-arm64.txt`) with a reachable vulnerable runtime dependency | Curated remediation PR | Prefer controlled lock/input rotation with focused validation |
| Retired unsupported Linux/macOS Intel ML lane | Remove scan-visible manifest or dismiss `not_used` when no manifest exists | Document that the lane is retired, unsupported, and not part of supported ML posture |
| Vulnerability only affects an unused dependency code path (for example `transformers.Trainer`) | Dismiss `not_used` | Include repo search evidence showing the vulnerable path is unreachable |

### ML review rules

1. Treat repeated alerts across target-owned ML lockfiles as a single advisory wave.
2. Do not take a broad or pre-release dependency upgrade just to clear the dashboard.
3. Keep retired Darwin x86_64 and Linux historical lanes out of supported baseline decisions.
4. Keep managed checkpoint and model-load trust boundaries tight while rotating versions.
5. Record dismissals and supported-lane remediation evidence in a dedicated triage artifact.

### Most recent detailed advisory mapping

See [DEPENDABOT_ML_ALERT_TRIAGE_2026-06-10.md](DEPENDABOT_ML_ALERT_TRIAGE_2026-06-10.md) for the most recent detailed advisory mapping: the June 10, 2026 PyTorch dismissal evidence and supported-lane rotation. Current supported-runtime baselines remain governed by the target-owned manifests and locks.
The April 16, 2026 record remains available at
[DEPENDABOT_ML_ALERT_TRIAGE_2026-04-16.md](DEPENDABOT_ML_ALERT_TRIAGE_2026-04-16.md).

---

**Maintained by**: Transformation Portal Architect
**Review Frequency**: As needed (when new policy patterns emerge)
