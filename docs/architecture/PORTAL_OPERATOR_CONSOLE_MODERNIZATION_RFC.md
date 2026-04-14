# Portal Operator Console Modernization RFC

**Status:** Proposed
**Date:** 2026-04-14
**Owner:** Transformation Portal Architect
**Reviewers:** Frontdoor / Platform, Portal Frontend, Backend / Origin, Security / Privacy, QA / Validation
**Related:** `docs/decisions/ADR-050-portal-react-migration.md`, `docs/architecture/DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md`, `docs/architecture/PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md`

## Summary

The Transformation Portal operator console should be modernized through bounded, low-risk tranches rather than a framework rewrite. This RFC preserves the current trust boundaries, keeps the FastAPI origin private behind the managed frontdoor, treats the native web platform as the default implementation path, and requires deterministic feature flags plus explicit validation gates for all material runtime and UX changes.

This RFC authorizes incremental modernization work. It does **not** approve a React or Next.js rewrite of the operator console, a new identity model, or any change that weakens the current browser-to-origin security boundary.

## Decision

The following decisions are adopted:

1. **FastAPI remains private behind the managed frontdoor.** Direct public exposure of the origin is out of scope for this RFC.
2. **The managed frontdoor remains the authoritative browser boundary.** Public homepage, login, proxying, route enforcement, and same-origin browser behavior continue to terminate there.
3. **Native web primitives remain the default implementation path.** Vanilla HTML, CSS, and JavaScript are preferred unless a targeted dependency is justified by a clear operational benefit.
4. **All material UX, runtime, and observability changes must ship behind deterministic feature flags.** Assignment must remain stable for a given viewer or operator over the supported lifetime of the assigned cohort key.
5. **Rollout proceeds by bounded tranches with explicit entry and exit criteria.** A tranche does not close merely because telemetry looks acceptable after deployment.
6. **This RFC is an implementation direction under `ADR-050`, not a competing architecture proposal.** Any future React or Next.js migration of the operator console requires a separate, evidence-based decision.

## Normative Interpretation

The following sections are normative and define binding requirements for implementation and approval:

- `Decision`
- `Decision Drivers`
- `Non-Goals`
- `Architecture Constraints and Repo Truth`
- `Workstreams`
- `Feature Flag and Rollback Contract`
- `Milestones and Rollout`
- `Acceptance Criteria`
- any subsection labeled `In Scope`, `Out of Scope`, `Exit Criteria`, `Close Conditions`, `Unconditional Validation`, `Conditional Browser Validation`, `Required Validation Backbone`, or `Rollback Posture`

The following content is non-normative:

- `Appendix C: Reference Implementation Notes`
- library examples, implementation patterns, and tooling suggestions unless they are explicitly restated in a normative section

## Decision Drivers

- Preserve the existing security and proxy boundary.
- Improve time to first useful action for operators.
- Increase resilience during authentication interruption and infrastructure turbulence.
- Achieve stronger WCAG 2.2 AA-oriented behavior for dense, asynchronous workflows.
- Establish measurable evidence before performance claims are accepted.
- Avoid the delivery risk of a framework rewrite when targeted native-web improvements can address immediate bottlenecks.

## Non-Goals

This RFC does not approve:

- a React or Next.js rewrite of the operator console in this tranche
- a new identity model or token model
- direct browser access to backend API keys or protected origin headers
- reopening the current route and query-parameter contract unless approved separately
- broad product analytics or employee-performance analytics
- broad visual redesign unrelated to the outcomes in this RFC

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
- route, cache, header, and auth behavior must remain aligned with the managed frontdoor and edge hardening standard

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

- lightweight Real User Monitoring for portal shell and key interaction milestones
- backend trace correlation for bootstrap and queue actions
- SSE health, reconnect counts, and bootstrap latency visibility
- a configuration-level kill switch for telemetry collection
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

**Dependencies**

- telemetry schema review
- ingestion path agreement across managed frontdoor and origin surfaces
- feature-flag wiring for observe-only rollout

**Exit Criteria**

- p75 LCP, INP, and CLS are visible by route and cohort
- `portal_shell_rendered`, `bootstrap_ready`, and `first_view_interactive` are measurable
- bootstrap and queue timings can be correlated across frontend and backend surfaces
- SSE reconnect counts and bootstrap latency are queryable
- collection can be disabled by configuration without code changes
- approved schemas emit no plain-text usernames or email addresses

### 2. Accessibility Hardening

**Objective**
Achieve WCAG 2.2 AA-oriented hardening for dense, keyboard-first, asynchronous operator workflows.

**In Scope**

- focus visibility and sticky-shell offset handling
- modal focus trap and focus restore behavior
- target-size improvements for dense controls
- live-region semantics for asynchronous job and toast updates
- `prefers-reduced-motion` compliance
- automated accessibility coverage for `/login` and `/portal`

**Out of Scope**

- broad redesign unrelated to accessibility outcomes
- introduction of a new component framework
- replacement of working keyboard shortcuts with a new command model

**Primary Owner**
Portal Frontend

**Supporting Owners**

- QA / Validation

**Dependencies**

- stable selectors and accessibility roles in current portal and frontdoor views
- managed browser validation coverage for `/login` and `/portal`

**Exit Criteria**

- no blocking accessibility regressions are present on `/login` or `/portal`
- focused controls remain visible under sticky UI
- modal interactions are keyboard-complete and restore focus correctly
- asynchronous status updates are announced without focus theft
- reduced-motion behavior is preserved across all touched surfaces
- manual keyboard-only verification passes for modal open/close, queue interaction, and review flow

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

**Dependencies**

- managed login and portal proxy remain authoritative
- origin and frontdoor agree on auth failure behavior and recovery contract

**Exit Criteria**

- forced 401 flows return operators to the same route context after re-authentication
- supported transient drafts recover from `sessionStorage`
- invalid `returnTo` values are rejected server-side and fail closed
- SSE recovery preserves operator context without destructive reset
- recovery behavior passes browser validation for supported flows

### 4. Performance and Rendering

**Objective**
Reduce time to first useful action and preserve responsiveness during large queues, long logs, and continuous updates.

**In Scope**

- route- or view-based code splitting where the current portal shell loads heavy, non-critical review tooling too early
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

**Dependencies**

- baseline telemetry from Workstream 1
- alignment with the portal edge hardening standard

**Exit Criteria**

- the portal shell remains within a documented bundle budget enforced in CI
- queue interaction latency stays within a documented provisional threshold
- large queues and long logs remain usable without sustained responsiveness collapse or uncontrolled long-task spikes
- static assets and bootstrap endpoints exhibit the intended cache behavior
- targeted performance scenarios pass required validation

### 5. Artifact Review and Operator Tooling

**Objective**
Improve operator review speed and precision without disrupting the main dispatch and monitoring flow.

**In Scope**

- an in-portal modal artifact viewer with keyboard support
- integrity metadata presentation for artifacts already emitted by the backend
- optional diagnostic-mode affordances where backend data already exists
- optional segmentation refinement when supported by the active backend
- explicit degraded-state handling when refinement or model assets are unavailable

**Out of Scope**

- replacing the review workflow with a separate application
- mandatory interactive segmentation dependencies for all review cases
- silent fallback behavior that hides degraded or stubbed review states

**Primary Owner**
Portal Frontend

**Supporting Owners**

- Backend / Origin
- QA / Validation

**Dependencies**

- integrity and artifact metadata are exposed by origin contracts
- viewer interactions comply with accessibility and reduced-motion requirements from earlier tranches

**Exit Criteria**

- standard artifact review does not require opening a new browser tab for supported cases
- keyboard-only inspection is possible for open, close, next/previous, and zoom controls
- integrity metadata is visible and copyable for supported artifacts
- degraded segmentation or refinement states are rendered explicitly and non-fatally

## Feature Flag and Rollback Contract

The following controls are required for all material tranche rollouts:

- deterministic flag assignment derived from a stable, documented cohort key
- a tranche-level kill switch or equivalent rollback control
- documented default-off posture for pilot release where feasible
- no user should flip between variants unpredictably across the supported life of the cohort key
- rollback steps must be recorded before production expansion beyond pilot cohorts

### Authenticated Operator Surfaces

For `/portal`, `/portal/bootstrap`, and other authenticated managed flows:

- assignment must derive from a stable internal operator identifier
- acceptable identifiers include a server-known operator ID, verified access identity, or another backend-controlled stable actor key
- raw browser-generated random assignment is not acceptable for authenticated operator cohorts

### Pre-Auth Managed Surfaces

For `/` and `/login` before operator authentication:

- assignment must derive from a managed frontdoor-controlled anonymous session identifier or a dedicated rollout cookie
- the assignment key must remain stable across reloads for the supported life of that anonymous session or rollout cookie
- IP address, user agent alone, or other unstable heuristics must not be used as the deterministic assignment key

### Required Implementation Contract

- if a pre-auth surface participates in a flagged rollout, the managed frontdoor must mint the anonymous session or rollout cookie before flag evaluation
- anonymous-session or rollout-cookie assignment must not leak backend secrets or operator identifiers to the browser
- if an experience spans pre-auth and post-auth surfaces, reassignment after login must be deterministic and documented
- production expansion beyond pilot requires:
  - the assignment key to be documented
  - the kill switch to be documented
  - the rollback owner to be documented
  - the cohort expansion criteria to be documented

## Milestones and Rollout

### M1: Measurement Foundation

**Owner:** Frontdoor / Platform
**Depends On:** None

**Outputs**

- portal RUM milestones
- backend trace correlation for bootstrap and queue actions
- SSE reconnect and bootstrap latency visibility
- telemetry disable switch

**Close Conditions**

- Workstream 1 exit criteria are met
- unconditional validation passes
- conditional browser validation passes only if M1 changes browser-observable frontdoor or portal behavior
- privacy and schema review is complete for pilot rollout

**Rollback Posture**

- feature-flag off
- ingestion disable by configuration

**Live Browser Validation Required Before Close**
Only if M1 changes browser-observable frontdoor or portal behavior

### M2: Accessibility Tranche

**Owner:** Portal Frontend
**Depends On:** M1 baseline instrumentation available for before/after comparison

**Outputs**

- focus visibility fixes
- modal focus discipline
- live-region semantics
- reduced-motion parity
- automated accessibility coverage for `/login` and `/portal`

**Close Conditions**

- Workstream 2 exit criteria are met
- unconditional validation passes
- conditional browser validation passes for each touched browser surface

**Rollback Posture**

- feature-flag off for tranche-specific UI deltas where feasible
- revert CSS or semantic changes that regress required validation

**Live Browser Validation Required Before Close**
Yes

### M3: Resilience Tranche

**Owner:** Frontdoor / Platform
**Depends On:** M1, and M2 where recovery UI overlaps modal or focus behavior

**Outputs**

- `returnTo` recovery flow
- transient draft restore behavior
- redirect hardening
- documented auth interruption recovery behavior

**Close Conditions**

- Workstream 3 exit criteria are met
- unconditional validation passes
- conditional browser validation passes for each touched browser surface

**Rollback Posture**

- feature-flag off
- preserve prior login and portal recovery behavior until tranche close

**Live Browser Validation Required Before Close**
Yes

### M4: Performance Tranche

**Owner:** Portal Frontend
**Depends On:** M1 baseline telemetry

**Outputs**

- code splitting or deferred loading for heavy views
- delegated event handling
- native off-screen rendering optimizations
- cache and header improvements that preserve route ownership
- watchdog timing refinement

**Close Conditions**

- Workstream 4 exit criteria are met
- unconditional validation passes
- conditional browser validation passes for each touched browser surface

**Rollback Posture**

- feature-flag off for view-level performance changes where feasible
- revert to prior loading strategy if required validation or operational behavior regresses

**Live Browser Validation Required Before Close**
Yes when frontdoor or portal browser-observable behavior changes

### M5: Artifact Review Tranche

**Owner:** Portal Frontend
**Depends On:** M2 and M4

**Outputs**

- modal artifact viewer
- integrity dashboard surface
- diagnostic affordances for existing backend metadata
- optional segmentation refinement path

**Close Conditions**

- Workstream 5 exit criteria are met
- unconditional validation passes
- `make validate-portal-browser` passes
- `make validate-frontdoor-browser` passes if M5 changes frontdoor-managed review entry or proxy behavior

**Rollback Posture**

- feature-flag off for new viewer and refinement paths
- preserve the current review path as fallback until tranche close

**Live Browser Validation Required Before Close**
Yes

## Acceptance Criteria

### Cross-Tranche Rules

- a milestone closes only when its close conditions are met
- acceptance criteria must be measurable and tied to repo-native validation where possible
- tracked metrics after rollout do not substitute for close criteria
- production expansion beyond pilot cohorts requires rollback instructions and feature-flag controls to be documented
- if there is uncertainty whether a change is browser-observable, the default is to run the relevant browser validation

### Required Validation Backbone

#### Unconditional Validation

The following checks are required before any milestone closes:

- `make test-frontdoor-contract`
- `make test-portal-contract`

#### Conditional Browser Validation

Run `make validate-frontdoor-browser` when a tranche changes any of:

- `/`
- `/login`
- managed auth behavior
- frontdoor proxy behavior
- cache or header behavior observable through the managed frontdoor
- feature-flagged UI on frontdoor-managed surfaces

Run `make validate-portal-browser` when a tranche changes any of:

- `/portal`
- `/portal/bootstrap`
- portal assets
- portal routing or view-state behavior
- portal review or artifact-inspection behavior
- SSE-driven UI behavior
- session recovery behavior observable in the portal

Browser validation is not required for telemetry-only or backend-correlation changes that do not change browser-observable behavior.

### Tranche-Specific Closure Checks

**Telemetry**

- metric ingestion is visible and disable-able by configuration
- bootstrap and queue timing can be correlated across frontend and backend surfaces

**Accessibility**

- no blocking accessibility regressions are present on `/login` or `/portal`
- focus, modal, and live-region behaviors pass required validation and manual keyboard review

**Resilience**

- same-route recovery works after forced authentication interruption
- `returnTo` validation fails closed for invalid inputs

**Performance**

- bundle budget is documented and enforced
- queue and review interactions stay within documented responsiveness expectations

**Artifact Review**

- the keyboard-only artifact review path is complete for supported cases
- integrity metadata is visible
- segmentation fallback states are explicit

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

Any React or Next.js migration of the operator console remains gated by `ADR-050` and must be re-opened as a separate evidence-based decision. This RFC does not pre-approve that migration.

### Telemetry and Privacy Gate

Broad telemetry rollout must not proceed beyond pilot cohorts until Security / Privacy has reviewed the approved schema, retention posture, and data-minimization approach.

## Appendix A: Ownership Matrix

| Workstream | Primary Owner | Dependent Systems | Primary Metrics | Closure Signal |
| --- | --- | --- | --- | --- |
| Telemetry and Observability | Frontdoor / Platform | managed frontdoor, FastAPI origin, telemetry ingestion | p75 CWV visibility, bootstrap latency, SSE reconnect counts | Workstream 1 exit criteria met |
| Accessibility Hardening | Portal Frontend | portal shell, login shell, browser validation | accessibility regressions, focus visibility, modal keyboard completion | Workstream 2 exit criteria met |
| Session Resilience and State Preservation | Frontdoor / Platform | login flow, portal routing, auth proxy, browser storage | recovery success, redirect rejection, draft restore success | Workstream 3 exit criteria met |
| Performance and Rendering | Portal Frontend | portal shell, static assets, proxy and cache behavior | shell bundle budget, interaction latency, long-task behavior | Workstream 4 exit criteria met |
| Artifact Review and Operator Tooling | Portal Frontend | review UI, artifact metadata, optional segmentation backends | viewer success, keyboard-only review success, metadata visibility | Workstream 5 exit criteria met |

## Appendix B: Initial Metric Targets

These are provisional targets and should be confirmed or adjusted after M1 baseline capture.

| Metric | Initial Target |
| --- | --- |
| p75 LCP | <= 2.5s |
| p75 INP | <= 200ms |
| p75 CLS | <= 0.1 |
| Queue interaction latency | p75 <= 150ms for standard row selection and action affordances |
| SSE reconnect rate | < 1 reconnect per operator-hour in steady-state managed usage |
| Accessibility regressions | 0 blocking accessibility regressions on `/login` and `/portal` |
| Artifact viewer interaction success | >= 95% successful open and basic inspect completion in pilot validation scenarios |

## Appendix C: Reference Implementation Notes (Non-Normative)

These notes are intentionally non-binding. They describe acceptable implementation paths without converting them into hard architectural commitments.

- `web-vitals`, `PerformanceObserver`, and OpenTelemetry-compatible traces are acceptable defaults for telemetry.
- `sessionStorage` is preferred over `localStorage` for transient draft recovery tied to a single browser tab.
- `encodeURIComponent()` is the correct primitive for encoding dynamic `returnTo` values; validation must still occur server-side.
- `content-visibility` and reserved dimensions are preferred before introducing complex JavaScript virtualization.
- delegated event handling is preferred for large, dynamic lists.
- a lightweight DOM-native viewer is preferred for artifact review; `panzoom` is an acceptable default if a dependency is needed.
- view-based dynamic imports are acceptable where the portal shell currently loads heavy review code too early.
- Playwright tests should prefer resilient user-facing locators such as `getByRole` and `getByLabel`.
