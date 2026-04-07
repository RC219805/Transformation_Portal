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

- Frontdoor validation is runtime-significant but was not previously a
  first-class CI surface with dedicated Node setup, `npm test`, and
  `npm run build`.
- Live frontdoor browser smoke exists in
  `scripts/validation/validate_frontdoor_browser_smoke.py`, but it was not
  wrapped by public Make targets.
- `/portal/assets/*` is still an interim bridge. It needs a checked-in manifest
  and drift detection so FastAPI asset references and frontdoor coverage cannot
  silently diverge.
- Session-store deployment assumptions were documented but not surfaced through
  structured readiness checks. Single-instance SQLite remains the default
  supported deployment posture for this horizon.

## Next 3-5 PRs

### PR 1: First-class frontdoor CI

- Add public Make targets for frontdoor contract and browser validation.
- Update GitHub Actions change classification so `web/secure-landing/**`,
  frontdoor smoke scripts, and the secure-frontdoor quickstart are treated as
  runtime-affecting changes.
- Run `npm ci`, `npm test`, and `npm run build` in `web/secure-landing` when
  frontdoor changes are present.

### PR 2: Readiness and deployment guardrails

- Extend `GET /healthz` to report structured checks for backend connectivity,
  Access configuration, user-source availability, and session-store readiness.
- Return `503` when required production checks fail while keeping the top-level
  `ok` contract simple.
- Keep `direct_debug` and local bypass behavior limited to explicit development
  flows.

### PR 3: Managed observability and recovery

- Normalize frontdoor audit coverage and operator-visible failure posture across
  `/portal`, `/portal/bootstrap`, `/portal/assets/*`, `/portal/video/*`, and
  `/v1/*`.
- Distinguish authentication failure, access outage, configuration failure, and
  upstream unavailability instead of treating all failures as a generic outage.

### PR 4: Harden `/portal/assets/*` into a contract

- Replace the ad hoc allowlist with a checked-in portal-asset manifest shared by
  the frontdoor proxy and contract tests.
- Add drift detection so `portal.html` asset references fail tests if the
  manifest is stale.
- Keep FastAPI and `portal.html` as the operator-shell system of record in this
  roadmap horizon.

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
