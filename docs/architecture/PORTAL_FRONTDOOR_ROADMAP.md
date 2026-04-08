# Portal Frontdoor Roadmap

Date: 2026-04-07
Scope: `web/secure-landing/` plus the managed browser boundary that fronts the FastAPI portal/orchestrator surfaces

## Objective

Track the managed frontdoor as its own roadmap lane now that the FastAPI
portal/orchestrator baseline is already re-baselined in
`docs/architecture/PORTAL_ORCHESTRATOR_ROADMAP.md`.

This roadmap is intentionally narrower than a full shell rewrite. The next
delivery horizon is production hardening over the next 3-5 PRs, with FastAPI
and `portal.html` remaining the operator-shell system of record.

## Completed Baseline

The following slices are already shipped and should remain closed:

- PR `#1328` established the thin managed frontdoor, server-side backend secret
  injection, SQLite-backed sessions, CSRF protection, and browser-facing `/v1/*`
  proxying.
- PR `#1330` made managed access fail closed behind verified Cloudflare Access
  JWT validation and introduced `managed_unavailable` behavior.
- PR `#1333` and PR `#1337` refined the branded login/media shell and wired the
  approved DNA brand assets into the managed entry path.
- PR `#1335` split the browser entry into `/`, `/login`, and `/portal` while
  keeping the FastAPI console contract intact.
- PR `#1368`, merged on April 7, 2026, replaced the inline homepage with the
  server-rendered verifier-backed Dynamic Neural Access landing page.
- PR `#1369`, merged on April 7, 2026, added the same-origin `/portal/assets/*`
  proxy as an interim availability bridge for the operator shell assets.

## Active Risks

- SQLite-backed sessions remain single-instance by design for this roadmap
  horizon. Shared or externalized session state is still intentionally deferred
  until a real deployment target requires multi-instance or ephemeral-runtime
  support.
- Managed browser validation still depends on live operator/browser smoke in
  addition to the now-normalized contract suite.

## Current Repo Status

### PR 1: First-class frontdoor CI

- Shipped on `main`.
- Public Make targets now cover frontdoor contract and browser validation.
- GitHub Actions change classification treats `web/secure-landing/**`,
  frontdoor smoke scripts, and secure-frontdoor docs as runtime-affecting.
- Frontdoor CI runs `npm ci`, `npm test`, and `npm run build` when the managed
  frontdoor changes.

### PR 2: Readiness and deployment guardrails

- Shipped on `main`.
- `GET /healthz` now reports structured checks for backend connectivity, Access
  configuration, user-source availability, and session-store readiness.
- The route returns `503` when required production checks fail while preserving
  the top-level `ok` contract.
- Local bypass and `direct_debug` behavior remain confined to explicit
  development flows.

### PR 3: Managed observability and recovery

- Shipped on `main`.
- Managed frontdoor failures are normalized across `/portal`,
  `/portal/bootstrap`, `/portal/assets/*`, `/portal/video/*`, and `/v1/*`
  through the shared `managed_surface_failure` audit taxonomy.
- Operator-visible recovery now distinguishes:
  - authentication failure
  - access outage
  - configuration failure
  - upstream unavailability
- `/portal/bootstrap` returns additive `reason`, `message`, and `retryable`
  fields so the browser shell can keep privileged actions fail-closed while
  surfacing recovery guidance.
- `/v1/*` preserves the existing error envelope shape while adding normalized
  `error.details.reason` and retryability metadata for auth/config/upstream
  failures.

### PR 4: Harden `/portal/assets/*` into a contract

- Shipped on `main`.
- The portal asset allowlist is now a checked-in manifest shared by the
  frontdoor proxy and contract tests.
- Drift detection fails when FastAPI `portal.html` asset references are no
  longer covered by the managed manifest.
- FastAPI and `portal.html` remain the operator-shell system of record for this
  roadmap horizon.

## Remaining Next Phase

### PR 5: Conditional state-scaling follow-up

- Keep SQLite single-instance deployment as the default supported production
  posture.
- Only promote shared/externalized session state if an actual deployment target
  requires multi-instance or ephemeral-runtime support.

## Acceptance Gates

- `make test-frontdoor-contract`
- `make validate-frontdoor-browser`
- `make test-orchestrator-contract`
- CI preflight classifies frontdoor changes as runtime-affecting
- FastAPI `portal.html` asset references remain covered by the checked-in portal
  asset manifest

## Explicit Non-Goals For This Horizon

- Replatforming the operator shell into Next.js
- Changing FastAPI `/v1/*` semantics
- Reopening the closed March 1, 2026 FastAPI/orchestrator roadmap
- Promoting shared session state before a real deployment requirement exists
