# ADR-050: Portal React/Next.js Migration Decision

## Status

**Proposed** - Pending evidence gathering and team discussion.

## Context

The operator console (`portal.html`, `portal.js`, `portal.css`) is currently a FastAPI-served HTML/JS application that provides the full operator workflow surface for the Transformation Portal. It supports:

- multi-view navigation (`?view=overview|build|operate|review`)
- real-time job updates via SSE
- managed authentication mode via the frontdoor proxy
- direct-debug mode
- keyboard accessibility
- reduced-motion support

The managed frontdoor (`web/secure-landing/`) is a separate Next.js application that handles:

- the public homepage
- authentication and login
- the managed portal proxy

This ADR remains the only decision gate for whether the operator console should migrate into the React/Next.js stack. The portal modernization RFC does not answer this ADR.

## Evidence Inventory

Current repo-backed sources:

- `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_RFC.md`
- `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md`
- `tools/portal_modernization_evidence.py`
- `scripts/validation/validate_portal_browser_smoke.py`
- `tests/test_app_orchestrator_runtime.py`

### 1. Delivery Velocity

| Evidence Item | State | Notes |
| --- | --- | --- |
| Average PR merge time for portal-related changes vs frontdoor changes | Still missing | Requires repository history analysis or external reporting |
| Lines of code changed per feature in each surface | Still missing | Requires historical sampling |
| Number of iterations required for typical portal changes | Still missing | Requires PR review-history analysis |

### 2. Quality

| Evidence Item | State | Notes |
| --- | --- | --- |
| Browser contract and runtime coverage for the current portal | Repo-backed now | Current portal validation is strong, but it is not a rewrite comparison by itself |
| Bug count per KLOC in portal vs frontdoor code | Still missing | Requires issue or incident tagging by surface |
| Test coverage comparison between portal and frontdoor surfaces | Still missing | Requires explicit comparative measurement |
| Regression frequency in each surface | Still missing | Requires incident or bug history by surface |

### 3. Developer Experience

| Evidence Item | State | Notes |
| --- | --- | --- |
| Current repo structure and workflow split are documented | Repo-backed now | The split boundary is clear, but this is not enough to justify migration |
| Developer friction survey results | Still missing | Needs human input |
| Onboarding time for new contributors to each surface | Still missing | Needs observed contributor data |
| IDE/tooling support comparison | Still missing | Needs an explicit comparison write-up |

### 4. Migration Requirements

| Current Capability | Next.js Equivalent | Complexity | State |
| --- | --- | --- | --- |
| `portal.html` static serve | `app/portal/page.js` | Low | Still hypothetical |
| `/portal/assets/*` | `public/` or static imports | Low | Still hypothetical |
| Query param routing `?view=` | client state or dynamic routes | Medium | Still hypothetical |
| `portal.js` state management | React state plus reducer or context | High | Still hypothetical |
| SSE via `EventSource` | same API or React data layer | Medium | Still hypothetical |
| Direct-debug bootstrap | API route plus middleware | Medium | Still hypothetical |
| Managed auth proxy | already exists in frontdoor | Low | Partial reuse available |
| Browser smoke tests | Playwright adaptation | Medium | Still hypothetical |

## Decision

*To be determined after evidence gathering.*

Options:

1. Proceed with migration.
2. Defer migration and continue with the native-web portal.
3. Reject migration and commit to the FastAPI/HTML architecture long term.

## Consequences

### If Migrating

**Benefits**

- unified browser stack
- React and Next.js tooling
- stronger component reuse opportunities

**Costs**

- migration delivery risk
- dual-surface maintenance during transition
- test rewrite and parity work
- regression risk against a stable validated portal

### If Not Migrating

**Benefits**

- no migration risk
- continued focus on operator workflow improvements
- simpler incremental change model

**Costs**

- continued maintenance of two browser codebases
- different implementation patterns across surfaces
- potential missed ecosystem benefits

## Go / No-Go Criteria

### Go if

- portal changes are materially slower than comparable frontdoor work
- portal defect rates are materially worse than comparable frontdoor work
- the team has clear React and Next.js migration ownership
- roadmap capacity exists for parity and migration hardening
- rewrite test coverage can reach current portal parity

### No-Go if

- current portal delivery remains adequate
- migration would block higher-priority operator work
- test parity cannot be reached in the migration window
- ownership of the migrated surface is unclear

## Acceptance Criteria for Migration

If migration is approved, the following must be true before deprecating the FastAPI-served portal:

- all existing browser smoke tests pass on the new surface
- direct-debug mode works identically
- managed auth mode works identically
- all four views remain fully functional
- the query-parameter routing contract is preserved
- keyboard accessibility and reduced-motion behavior do not regress
- CI coverage is equivalent or better
- page load and interaction latency remain at parity or better
- documentation is updated

## References

- `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_RFC.md`
- `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md`
- `web/secure-landing/`
- `scripts/validation/validate_portal_browser_smoke.py`
- `tests/test_app_orchestrator_runtime.py`
