# Transformation Portal Custom Agent Guide

## Overview

The repository now has three live custom agent profiles:

- **Transformation Portal Architect** for governance, contracts, security, dependency, and CI/CD decisions
- **Portal App Steward** for the managed browser boundary and portal shell
- **Transformation Portal Specialist** for backend/orchestrator, Lux Depth, archive, ingest, and machine-mode execution work

These roles are complementary. The Architect defines system invariants. The Steward and Specialist execute inside them.

Current baseline: `main` through PR #1721. Current documentation navigation
lives in `README.md`, `docs/README.md`,
`docs/governance/DOCUMENTATION_MAP.md`, and
`docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md`, with historical
classification context retained in
`docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md`.

## Live Agent Roles

### Transformation Portal Architect

Use `@transformation-portal-architect` when the task involves:

- dependency or runtime policy
- CI/CD or workflow enforcement
- documentation topology, Copilot instructions, or live custom-agent role boundaries
- security posture or trust-boundary changes
- public HTTP, CLI, schema, or packaging contracts
- ADR interpretation or architectural trade-offs

Example:

```text
@transformation-portal-architect review this proposal to change the managed /portal bootstrap contract and list the required enforcement updates
```

### Portal App Steward

Use `@portal-app-steward` when the task involves the managed browser boundary:

- `web/secure-landing/`
- `/`, `/login`, `/portal`, `/portal/bootstrap`
- `/portal/assets/*`, `/portal/video/*`, managed `/healthz`
- `portal.html`
- `public/portal-assets/*`
- `config/portal_asset_manifest.json`
- browser-smoke selectors, bootstrap states, Node 22 frontdoor checks, and frontdoor/portal validation

The Steward is the first-choice execution agent for browser-surface work. It should preserve route, auth, asset, and browser-validation contracts and treat direct-debug as secondary to managed mode.

Examples:

```text
@portal-app-steward update the managed /login shell copy without changing auth, cache, or CSP contracts
```

```text
@portal-app-steward trace why the portal review shell is not reflecting degraded bootstrap state and list the smallest safe validation set
```

```text
@portal-app-steward fix the manifest-backed portal asset wiring and keep data-ui anchors stable
```

### Transformation Portal Specialist

Use `@transformation-portal-specialist` when the task involves governed non-browser execution surfaces:

- `app.py`
- backend `/healthz`, `/ready`, and `/v1/readiness`
- typed `/v1/*` behavior
- Lux Depth V3
- archive-gate
- ingest and machine-mode
- docs/tests/tooling that move with those contracts

Examples:

```text
@transformation-portal-specialist debug why /v1/jobs/{id}/events is missing the terminal done event in contract tests
```

```text
@transformation-portal-specialist update app.py validation without changing the /ready or typed /v1/* response contracts
```

```text
@transformation-portal-specialist fix this tp.meta.machine.v1 payload drift and list the docs and tests that must move together
```

## Steward vs Specialist

Use the Steward when the primary change is browser-facing:

- homepage, login, portal shell, bootstrap, recovery, browser assets, selector stability, or browser validation

Use the Specialist when the primary change is backend or workflow-facing:

- typed API behavior, queue/runtime logic, Lux Depth execution, archive behavior, machine-mode, ingest, or backend validation

If browser work requires backend contract changes:

1. Architect direction stays authoritative for contract decisions.
2. Steward owns the browser-side plan and validation.
3. Specialist owns the backend implementation that changes typed behavior or backend hardening.

Current backend contract anchors:

- `/healthz`, `/ready`, and `/v1/readiness` expose typed OpenAPI response
  models while keeping existing wire contracts stable.
- Job lifecycle routes use typed response models while preserving their
  existing `/v1/*` wire contracts.
- `/v1/readiness` keeps transport success separate from per-pipeline
  `ready` / `degraded` / `blocked` execution truth.
- Archive Gates A/B/C readiness evidence is captured in
  `docs/governance/audit/archive-gates-2026-04-27.md`.

## Working With The Agents

Best prompt pattern:

1. name the active surface
2. identify the failing route, test, selector, or command
3. ask for the smallest safe change
4. ask for tests and docs to move with the behavior

Good examples:

```text
@portal-app-steward preserve the ?view=overview|build|operate|review route contract while tightening narrow-width layout in the review shell
```

```text
@transformation-portal-specialist explain which contract tests cover this /v1 payload change and patch the smallest safe backend fix
```

```text
@transformation-portal-architect assess whether this proposal reopens ADR-050 and list the rollback and enforcement requirements
```

## Canonical Docs

The live custom agent surface is defined by:

- `.github/agents/transformation-portal-architect.md`
- `.github/agents/portal-app-steward.md`
- `.github/agents/transformation-portal-specialist.md`
- `.github/agents/README.md`
- `.github/copilot-instructions.md`
- `docs/architecture/agent_governance.md`
- `docs/governance/DOCUMENTATION_MAP.md`
- `docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md`

Supporting quick references:

- `.github/agents/QUICK_START_v2.md`
- `docs/reference/AGENT_QUICK_REFERENCE.md`

Historical or milestone-style documents are not authoritative for live agent behavior.

## Updating The Live Agent Surface

When role boundaries or live guidance change, update the source of truth together:

1. the relevant profile under `.github/agents/`
2. `.github/agents/README.md`
3. `.github/agents/QUICK_START_v2.md`
4. `.github/copilot-instructions.md` when repo-wide Copilot behavior changes
5. this guide
6. `docs/reference/AGENT_QUICK_REFERENCE.md`
7. `docs/governance/DOCUMENTATION_MAP.md` when canonical navigation changes
8. `tests/test_custom_agent_config.py`

Do not leave the README, guide, quick reference, and contract test out of sync.

## Support Material

Specialist-specific RAG and support materials may remain Specialist-owned unless they are promoted into the live canonical custom-agent surface.
