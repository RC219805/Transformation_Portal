# Portal Frontdoor Roadmap

Date: 2026-04-07
Scope: `web/secure-landing/` plus the managed browser boundary that fronts the FastAPI portal/orchestrator surfaces

## Objective

Track the managed frontdoor as its own roadmap lane now that the FastAPI
portal/orchestrator baseline is already re-baselined in
`docs/architecture/PORTAL_ORCHESTRATOR_ROADMAP.md`.

This roadmap is intentionally narrower than a full shell rewrite. The next
delivery horizon has already shipped; FastAPI, the generated root `portal.html`,
and the `web/secure-landing/portal-src/` template/deferred modules remain the
operator-shell system of record. This document now serves as a status and
validation record rather than an active feature-phase plan.

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

- SQLite-backed sessions remain intentionally single-instance for this roadmap
  horizon. Deployments that explicitly declare `multi_instance` or
  `ephemeral_runtime` session scaling now fail readiness until a real external
  session store exists.
- Managed browser validation still depends on live operator/browser smoke in
  addition to the now-normalized contract suite.

## Current Repo Status

- Rebaselined against `main` on April 7, 2026.
- Earlier roadmap drafts treated PR 3 as the next implementation target, but
  the repo now already contains the PR 1 through PR 4 deliverables listed
  below.

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
- Drift detection fails when FastAPI portal-shell asset references are no longer
  covered by the checked-in managed manifest.
- FastAPI, the generated root `portal.html`, and the `portal-src/` template and
  deferred modules remain the operator-shell system of record for this roadmap
  horizon.

### PR 5: Conditional state-scaling follow-up

- Implemented in PR `#1375`.
- `TP_FRONTDOOR_SESSION_SCALING_MODE` now makes the supported SQLite session
  posture explicit.
- `/healthz` fails closed when operators declare `multi_instance` or
  `ephemeral_runtime` scaling without a real external session store.
- Local launcher and frontdoor quickstart now pin the supported
  `single_instance` posture.

## Roadmap Status

- With PR `#1375` implemented, no queued phases remain for this roadmap horizon.
- The only UX-adjacent close-out lane was rerunning managed frontdoor
  contract/build/browser validation under the enforced Node `22.x` runtime.
- That close-out lane was completed on April 9, 2026:
  - `make test-frontdoor-contract` passed under Node `22.22.2`;
  - `make validate-frontdoor-browser` passed under the same Node `22.22.2`
    environment.
- Any local runtime outside the package contract `>=22 <23` is an unsupported
  toolchain posture, not a frontdoor product regression.

## Acceptance Gates

- `make test-frontdoor-contract`
- `make validate-frontdoor-browser`
- `make test-orchestrator-contract`
- CI preflight classifies frontdoor changes as runtime-affecting
- FastAPI portal-shell asset references remain covered by the checked-in portal
  asset manifest
- `/healthz` exposes the explicit `session_scaling` readiness check and fails
  when unsupported multi-instance or ephemeral-runtime modes are declared
- Local `make test-frontdoor-contract` verification must run under Node 22.x;
  the frontdoor package explicitly rejects unsupported runtimes outside
  `>=22 <23`.

## Explicit Non-Goals For This Horizon

- Replatforming the operator shell into Next.js
- Changing FastAPI `/v1/*` semantics
- Reopening the closed March 1, 2026 FastAPI/orchestrator roadmap
- Promoting a shared session backend before a real deployment requirement exists
