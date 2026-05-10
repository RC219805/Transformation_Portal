# Portal Operator Console Modernization RFC

**Status:** Approved
**Date:** 2026-04-14
**Owner:** Transformation Portal Architect
**Reviewers:** Frontdoor / Platform, Portal Frontend, Backend / Origin, Security / Privacy, QA / Validation
**Related:** `docs/decisions/ADR-050-portal-react-migration.md`, `docs/architecture/DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md`, `docs/architecture/PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md`, `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md`, `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md`

## Summary

This approved RFC records the portal modernization decision: continue improving the existing FastAPI-served operator console through bounded, low-risk tranches, keep the managed frontdoor as the authoritative browser boundary, and keep native web primitives as the default implementation path.

This RFC does not approve a React or Next.js migration of the operator console. `ADR-050` remains the separate future decision gate for any rewrite or boundary change.

## Decision

1. FastAPI remains private behind the managed frontdoor.
2. The managed frontdoor remains the authoritative browser boundary for `/`, `/login`, `/portal`, `/portal/bootstrap`, and same-origin `/v1/*` proxy behavior.
3. Native web primitives remain the default implementation path unless a targeted dependency has a clear operational benefit.
4. Material UX, runtime, review-surface, and observability changes ship behind deterministic feature flags with stable cohort assignment.
5. Milestones may be implemented without being formally closed; close status still requires the named sign-off and evidence gates.
6. Route, query-parameter, cache, auth, and selector contracts remain stable unless a separate approved decision changes them.
7. Any future React or Next.js migration remains blocked on the still-open evidence checklist in `ADR-050`.

## Normative Interpretation

The following sections are normative:

- `Decision`
- `Decision Drivers`
- `Non-Goals`
- `Architecture Constraints and Repo Truth`
- `Workstreams`
- `Feature Flag and Rollback Contract`
- `Milestones and Rollout`
- `Acceptance Criteria`
- `Risks and Future Decision Gates`
- any subsection labeled `In Scope`, `Out of Scope`, `Acceptance Gate`, `Rollback Posture`, `Required Validation Backbone`, or `Status`

The following sections are non-normative:

- `Implementation Status (2026-04-14)`
- `Outstanding TODOs`
- `Appendix C: Reference Notes`

## Implementation Status (2026-04-14)

### Implemented

- Deterministic rollout controls exist for portal RUM, the artifact viewer modal, and deferred review-surface loading.
- Portal RUM ingest, backend trace propagation, and the repo-owned summary path in `tools/portal_rum_summary.py` are implemented.
- Managed `returnTo` validation and transient build-draft restore behavior are implemented and covered in repo-native validation.
- Portal asset budget enforcement and deferred review loading are implemented.
- The in-portal artifact viewer, keyboard controls, fingerprint copy affordance, and explicit non-preview fallback behavior are implemented.
- M2's accepted contract is the current repo-native browser-probe accessibility coverage for `/login` and `/portal`, plus manual keyboard and reduced-motion validation. A broader automated suite is not required to keep this RFC approved.
- M3 resilience behavior is implemented for the currently supported managed recovery path.

### Partial

- M1 Measurement Foundation is implemented with bounded-pilot telemetry sign-off approved with conditions, but it is not formally closed until measured pilot evidence is attached.
- M4 Performance and Rendering is implemented but not formally closed pending measured pilot evidence against the provisional CWV, queue-latency, and SSE targets.
- M5 Artifact Review is implemented for deferred review loading, modal viewer behavior, metadata visibility, fingerprint copy, and explicit fallback states, but it is not formally closed pending measured pilot evidence.
- M5's optional segmentation refinement is not counted as shipped in this RFC. Current shipped scope is the viewer and fallback path only.

### Open Gate

- Repo-owned pilot evidence for M1, M4, and M5 remains open until measurements are captured and attached.
- Privacy packet revision remains required before any new RUM event family, metadata key, marker cookie, storage key, rollout knob, sink behavior, or retention posture.
- The ADR-050 rewrite evidence checklist remains open and is not resolved by this RFC.

## Decision Drivers

- Preserve the existing trust boundary and managed proxy model.
- Improve time to first useful action for operators.
- Increase resilience during authentication interruption and transient transport failures.
- Achieve WCAG 2.2 AA-oriented behavior for dense asynchronous workflows.
- Tie performance and review claims to measured evidence instead of anecdote.
- Avoid rewrite risk while targeted native-web improvements continue to land safely.

## Non-Goals

This RFC does not approve:

- a React or Next.js rewrite of the operator console
- a new identity model
- direct browser access to backend API keys or protected origin headers
- route or query-parameter contract changes unless separately approved
- broad product analytics or employee-performance analytics
- silent degraded review states

## Architecture Constraints and Repo Truth

The current browser experience spans three managed surfaces:

- `/` for the public frontdoor
- `/login` for managed authentication entry
- `/portal` for the operator console

The authoritative request path remains:

`public ingress/WAF -> web/secure-landing (Next.js managed frontdoor) -> app.py (FastAPI origin)`

Current-state boundary rules remain in force:

- the managed frontdoor owns the public homepage, login flow, portal proxy, `/portal/bootstrap`, and same-origin `/v1/*` proxy behavior
- FastAPI remains the origin for the operator shell and backend APIs
- browser-side code must not receive backend API keys in managed mode
- the operator console view contract remains `?view=overview|build|operate|review`
- route, cache, header, auth, and selector behavior must remain aligned with the managed frontdoor and edge hardening standard

Current repo-native validation remains authoritative:

- `make test-frontdoor-contract`
- `make test-portal-contract`
- `make validate-frontdoor-browser`
- `make validate-portal-browser`

## Workstreams

### 1. Telemetry and Observability

**Objective**
Move portal performance and reliability work from anecdotal assessment to measured frontend and backend evidence.

**In Scope**

- lightweight RUM for portal shell and key interaction milestones
- backend trace correlation for bootstrap and queue actions
- SSE health, reconnect counts, and bootstrap latency visibility
- configuration-level telemetry kill switches
- approved schemas that minimize operator-identifying data

**Out of Scope**

- employee-performance analytics
- broad product analytics unrelated to service health
- arbitrary client event collection without schema review

**Primary Owner**
Frontdoor / Platform

**Supporting Owners**

- Backend / Origin
- Security / Privacy

**Acceptance Gate**

- telemetry is visible in pilot or non-production environments
- collection can be disabled by configuration without code changes
- no plain-text usernames or email addresses are emitted in the approved schema
- measured pilot evidence is attached before M1 is marked closed

### 2. Accessibility Hardening

**Objective**
Achieve WCAG 2.2 AA-oriented hardening for dense, keyboard-first, asynchronous operator workflows.

**In Scope**

- focus visibility and sticky-shell offset handling
- modal focus trap and focus restore behavior
- target-size improvements for dense controls
- live-region semantics for asynchronous job and toast updates
- `prefers-reduced-motion` compliance
- repo-native browser-probe accessibility coverage for `/login` and `/portal`
- manual keyboard-only validation for review, modal, and queue flows

**Out of Scope**

- broad redesign unrelated to accessibility outcomes
- introduction of a new component framework
- a broader automated accessibility suite unless separately approved as follow-on work

**Primary Owner**
Portal Frontend

**Supporting Owners**

- QA / Validation

**Acceptance Gate**

- current browser-probe accessibility checks for `/login` and `/portal` stay green
- manual keyboard-only verification passes for modal open/close, queue interaction, and review flow
- reduced-motion behavior is preserved across touched surfaces

### 3. Session Resilience and State Preservation

**Objective**
Preserve operator context during 401s, session expiry, and transient infrastructure failures.

**In Scope**

- canonical route recovery via URL state and `returnTo`
- transient draft preservation in `sessionStorage`
- server-side validation of `returnTo`
- predictable recovery behavior for forced re-authentication and SSE interruption

**Out of Scope**

- new identity providers
- long-lived client-side persistence of sensitive drafts in `localStorage`
- route-contract changes that abandon the existing query-parameter model

**Primary Owner**
Frontdoor / Platform

**Supporting Owners**

- Portal Frontend
- Backend / Origin

**Acceptance Gate**

- forced 401 flows return operators to the same route context after re-authentication
- supported transient drafts recover from `sessionStorage`
- invalid `returnTo` values are rejected server-side and fail closed
- SSE recovery preserves operator context without destructive reset

### 4. Performance and Rendering

**Objective**
Reduce time to first useful action and preserve responsiveness during large queues, long logs, and continuous updates.

**In Scope**

- route- or view-based code splitting where the portal shell loads heavy non-critical review tooling too early
- native rendering optimizations such as `content-visibility`, reserved dimensions, and deferred non-critical work
- delegated event handling for large lists
- main-thread protection and watchdog refinement
- cache and header hardening for static versus dynamic surfaces

**Out of Scope**

- framework migration for performance alone
- heavyweight client-side state libraries as a first-line fix
- cache policies that weaken auth, bootstrap, or origin correctness

**Primary Owner**
Portal Frontend

**Supporting Owners**

- Frontdoor / Platform

**Acceptance Gate**

- the portal shell remains within a documented bundle budget enforced in CI
- queue interaction latency stays within the provisional threshold
- large queues and long logs remain usable without sustained responsiveness collapse
- targeted pilot evidence is attached before M4 is marked closed

### 5. Artifact Review and Operator Tooling

**Objective**
Improve operator review speed and precision without disrupting the main dispatch and monitoring flow.

**In Scope**

- an in-portal modal artifact viewer with keyboard support
- integrity metadata presentation for artifacts already emitted by the backend
- optional diagnostic-mode affordances where backend data already exists
- explicit degraded-state handling for non-previewable artifacts and missing managed URLs
- repo-owned telemetry for viewer opens and explicit fallback states

**Out of Scope**

- replacing the review workflow with a separate application
- mandatory interactive segmentation dependencies for all review cases
- claiming review-time segmentation refinement as shipped work in the current tranche

**Primary Owner**
Portal Frontend

**Supporting Owners**

- Backend / Origin
- QA / Validation

**Acceptance Gate**

- standard artifact review does not require a new browser tab for supported cases
- keyboard-only inspection is possible for open, close, next/previous, and zoom controls
- integrity metadata is visible and copyable for supported artifacts
- explicit non-preview fallback states remain visible and non-fatal
- targeted pilot evidence is attached before M5 is marked closed

## Feature Flag and Rollback Contract

The following controls are required for all material tranche rollouts:

- deterministic flag assignment derived from a stable documented cohort key
- a tranche-level kill switch or equivalent rollback control
- documented default-off posture for pilot release where feasible
- no user should flip between variants unpredictably across the supported life of the cohort key
- rollback steps must be recorded before production expansion beyond pilot cohorts

### Authenticated Operator Surfaces

For `/portal`, `/portal/bootstrap`, and other authenticated managed flows:

- assignment must derive from a stable internal operator identifier
- acceptable identifiers include a server-known operator ID, verified access identity, or another backend-controlled stable actor key
- raw browser-generated random assignment is not acceptable

### Pre-Auth Managed Surfaces

For `/` and `/login` before operator authentication:

- assignment must derive from a managed frontdoor-controlled anonymous session identifier or dedicated rollout cookie
- the assignment key must remain stable across reloads for the supported life of that anonymous session or rollout cookie
- IP address, user agent alone, or other unstable heuristics must not be used as the deterministic assignment key

## Milestones and Rollout

### M1: Measurement Foundation

**Status:** Partial - implemented but not formally closed
**Owner:** Frontdoor / Platform
**Depends On:** None

**Outputs**

- portal RUM milestones
- backend trace correlation for bootstrap and queue actions
- SSE reconnect and bootstrap latency visibility
- telemetry disable switch
- repo-owned evidence commands in `tools/portal_rum_summary.py` and `tools/portal_modernization_evidence.py`

**Acceptance Gate**

- Workstream 1 acceptance gate is met
- Security / Privacy sign-off is attached
- measured pilot evidence is attached

**Rollback Posture**

- feature-flag off
- ingestion disable by configuration

### M2: Accessibility Tranche

**Status:** Implemented
**Owner:** Portal Frontend
**Depends On:** M1 baseline instrumentation available for before/after comparison

**Outputs**

- focus visibility fixes
- modal focus discipline
- live-region semantics
- reduced-motion parity
- accepted repo-native browser-probe coverage for `/login` and `/portal`

**Acceptance Gate**

- Workstream 2 acceptance gate is met
- `make test-frontdoor-contract`, `make test-portal-contract`, `make validate-frontdoor-browser`, and `make validate-portal-browser` remain green for touched flows

**Rollback Posture**

- feature-flag off for tranche-specific UI deltas where feasible
- revert CSS or semantic changes that regress required validation

### M3: Resilience Tranche

**Status:** Implemented
**Owner:** Frontdoor / Platform
**Depends On:** M1 and M2 where recovery UI overlaps modal or focus behavior

**Outputs**

- `returnTo` recovery flow
- transient draft restore behavior
- redirect hardening
- documented auth interruption recovery behavior

**Acceptance Gate**

- Workstream 3 acceptance gate is met

**Rollback Posture**

- feature-flag off
- preserve prior login and portal recovery behavior until revert completes

### M4: Performance Tranche

**Status:** Partial - implemented but not formally closed
**Owner:** Portal Frontend
**Depends On:** M1 baseline telemetry

**Outputs**

- code splitting or deferred loading for heavy views
- delegated event handling
- native off-screen rendering optimizations
- cache and header improvements that preserve route ownership
- watchdog timing refinement

**Acceptance Gate**

- Workstream 4 acceptance gate is met
- measured pilot evidence is attached

**Rollback Posture**

- feature-flag off for view-level performance changes where feasible
- revert to prior loading strategy if required validation or operational behavior regresses

### M5: Artifact Review Tranche

**Status:** Partial - implemented but not formally closed
**Owner:** Portal Frontend
**Depends On:** M2 and M4

**Outputs**

- modal artifact viewer
- integrity dashboard surface
- diagnostic affordances for existing backend metadata
- explicit non-preview fallback states
- viewer open and fallback telemetry

**Acceptance Gate**

- Workstream 5 acceptance gate is met
- measured pilot evidence is attached
- segmentation refinement remains excluded from close criteria unless separately shipped and documented

**Rollback Posture**

- feature-flag off for new viewer and deferred review paths
- preserve the current review path as fallback until revert completes

## Acceptance Criteria

### Cross-Tranche Rules

- a milestone closes only when its acceptance gate is met
- acceptance criteria must be measurable and tied to repo-native validation where possible
- tracked metrics after rollout do not substitute for close criteria
- production expansion beyond pilot cohorts requires rollback instructions and feature-flag controls to be documented
- if there is uncertainty whether a change is browser-observable, run the relevant browser validation

### Required Validation Backbone

- `make test-frontdoor-contract`
- `make test-portal-contract`
- `make validate-frontdoor-browser`
- `make validate-portal-browser`

### Tranche-Specific Closure Checks

**Telemetry**

- metric ingestion is visible and disable-able by configuration
- bootstrap and queue timing can be correlated across frontend and backend surfaces

**Accessibility**

- current browser-probe accessibility checks stay green on `/login` and `/portal`
- focus, modal, live-region, keyboard, and reduced-motion behaviors pass required validation

**Resilience**

- same-route recovery works after forced authentication interruption
- `returnTo` validation fails closed for invalid inputs

**Performance**

- bundle budget is documented and enforced
- queue and review interactions stay within documented responsiveness expectations

**Artifact Review**

- the keyboard-only artifact review path is complete for supported cases
- integrity metadata is visible
- explicit fallback states are visible and non-fatal
- viewer pilot evidence includes open count, fallback count, and success rate

## Outstanding TODOs

Resolved during this refresh:

- M2 coverage uses the current browser-probe and manual keyboard contract; a broader automated suite is not required for this RFC.
- M5 shipped scope is the viewer and explicit fallback path; optional segmentation refinement is not treated as shipped work.

Still open:

- Security / Privacy must approve the telemetry schema, retention posture, and disposal procedure captured in `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md`.
- M1, M4, and M5 still need measured pilot evidence for CWV, queue latency, SSE reconnect rate, and artifact-viewer success before they can be marked closed.
- ADR-050 still needs its separate delivery, quality, and developer-experience evidence checklist before any rewrite decision can be made.

## Risks and Future Decision Gates

### Risks of Staying Native-Web Only

- portal complexity may continue to accumulate in bespoke client code
- some future interaction patterns may become more expensive to maintain without a framework
- tranche discipline must remain strong or the portal shell will continue to grow without clear module ownership

### Risks of Deferred Rewrite

- a later rewrite may carry higher migration cost if current contracts expand first
- split ownership between frontdoor and origin remains a coordination cost
- duplicated patterns may persist longer than ideal across browser surfaces

### Explicit Future Decision Gate

Any React or Next.js migration of the operator console remains subject to `ADR-050` or a successor architecture decision and must be approved through a separate, evidence-based decision. This RFC does not pre-approve that migration.

### Telemetry and Privacy Gate

Broad telemetry rollout must not proceed beyond pilot cohorts until Security / Privacy has reviewed the approved schema, retention posture, and data-minimization approach.

## Appendix A: Ownership Matrix

| Workstream | Primary Owner | Dependent Systems | Primary Metrics | Close Signal |
| --- | --- | --- | --- | --- |
| Telemetry and Observability | Frontdoor / Platform | managed frontdoor, FastAPI origin, telemetry ingestion | p75 CWV visibility, bootstrap latency, SSE reconnect counts | schema sign-off plus pilot evidence attached |
| Accessibility Hardening | Portal Frontend | portal shell, login shell, browser validation | browser-probe regressions, focus visibility, modal keyboard completion | validation remains green under accepted contract |
| Session Resilience and State Preservation | Frontdoor / Platform | login flow, portal routing, auth proxy, browser storage | recovery success, redirect rejection, draft restore success | recovery validation remains green |
| Performance and Rendering | Portal Frontend | portal shell, static assets, proxy and cache behavior | shell bundle budget, interaction latency, long-task behavior | pilot evidence attached |
| Artifact Review and Operator Tooling | Portal Frontend | review UI, artifact metadata | viewer success, keyboard-only review success, metadata visibility | pilot evidence attached |

## Appendix B: Initial Metric Targets

These are provisional targets and should be confirmed or adjusted after pilot capture.

| Metric | Initial Target |
| --- | --- |
| p75 LCP | <= 2.5s |
| p75 INP | <= 200ms |
| p75 CLS | <= 0.1 |
| Queue interaction latency | p75 <= 150ms for standard row selection and action affordances |
| SSE reconnect rate | < 1 reconnect per operator-hour in steady-state managed usage |
| Accessibility regressions | 0 blocking browser-probe regressions on `/login` and `/portal` |
| Artifact viewer interaction success | >= 95% successful open without fallback in pilot validation scenarios |

## Appendix C: Reference Notes

- `tools/portal_rum_summary.py` remains the contract-stable RUM-only summary path.
- `tools/portal_modernization_evidence.py` is the repo-owned pilot evidence path for M1, M4, and M5.
- `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md` records the evidence collection workflow and open gates.
- `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md` is the approved-with-conditions bounded-pilot sign-off record for the current telemetry schema, retention, and disposal posture.
