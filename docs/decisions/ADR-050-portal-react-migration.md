# ADR-050: Portal React/Next.js Migration Decision

## Status

**Proposed** — Pending evidence gathering and team discussion.

## Context

The operator console (`portal.html`, `portal.js`, `portal.css`) is currently a FastAPI-served HTML/JS application that provides the full operator workflow surface for the Transformation Portal. It supports:

- Multi-view navigation (`?view=overview|build|operate|review`)
- Real-time job updates via SSE
- Managed authentication mode (via frontdoor proxy)
- Direct-debug mode (browser-side API key)
- Full keyboard accessibility
- Reduced-motion support

The managed frontdoor (`web/secure-landing/`) is a separate Next.js application that handles:

- Public homepage
- Authentication/login
- Portal proxy in managed mode

This ADR addresses whether the operator console should be migrated into the React/Next.js stack as part of the managed frontdoor, or whether it should remain as a FastAPI-served HTML/JS application.

## Evidence Required

Before making a decision, the following evidence should be gathered:

### 1. Delivery Velocity Metrics

- [ ] Average PR merge time for portal-related changes vs. frontdoor changes
- [ ] Lines of code changed per feature in each surface
- [ ] Number of iterations required for typical portal changes

### 2. Quality Metrics

- [ ] Bug count per KLOC in `portal.js` vs. frontdoor code
- [ ] Test coverage comparison (portal contract tests vs. frontdoor tests)
- [ ] Regression frequency in each surface

### 3. Developer Experience

- [ ] Developer friction survey results (if available)
- [ ] Onboarding time for new contributors to each surface
- [ ] IDE/tooling support comparison

### 4. Migration Requirements

| Current Capability | Next.js Equivalent | Complexity | Notes |
|-------------------|-------------------|------------|-------|
| `portal.html` static serve | `app/portal/page.js` | Low | Route handler required |
| `/portal/assets/*` | `public/` or Next.js static imports | Low | Path mapping needed |
| Query param routing `?view=` | Client state or dynamic routes | Medium | URL contract preservation |
| `portal.js` state management | React state + context/reducer | High | Significant refactor |
| SSE via EventSource | Same API or React Query | Medium | Connection management |
| Direct-debug bootstrap | API route + middleware | Medium | Auth flow adaptation |
| Managed auth proxy | Already exists in frontdoor | Low | Integration needed |
| Browser smoke tests | Playwright/Cypress adaptation | Medium | Test rewrite needed |

## Decision

*To be determined after evidence gathering.*

Options:

1. **Proceed with migration** — Move operator console into Next.js
2. **Defer migration** — Continue with Phase 5 modularization and revisit in 6 months
3. **Reject migration** — Commit to FastAPI/HTML architecture long-term

## Consequences

### If Migrating

**Benefits:**
- Unified codebase for all browser surfaces
- Modern React tooling and ecosystem
- Better component reusability
- Stronger type safety with TypeScript (optional)

**Costs:**
- 3-6 month migration timeline (estimated)
- Two UIs to maintain during transition
- Test coverage gap during migration
- Risk of regression in tested behavior

**Migration Phases (if approved):**

1. **Phase M1: Minimal Viable Portal** (4-6 weeks)
   - Basic route structure in Next.js
   - Static page rendering
   - Managed auth integration

2. **Phase M2: Feature Parity** (6-8 weeks)
   - All four views implemented
   - State management ported
   - SSE/real-time support

3. **Phase M3: Deprecation** (2-4 weeks)
   - Test coverage validation
   - FastAPI route deprecation
   - Documentation update

### If Not Migrating

**Benefits:**
- No migration risk
- Continued focus on operator workflow improvements
- Simpler architecture with clear separation

**Costs:**
- Continued maintenance of two codebases
- Different patterns between surfaces
- Potential missed ecosystem benefits

**Continued Improvement Path:**
- Complete Phase 5 modularization
- Phase 6 workflow acceleration
- Phase 7 shared token layer
- Revisit migration decision in 6 months

## Go/No-Go Criteria

### Go if:

- [ ] Portal changes taking >2x longer than similar complexity elsewhere
- [ ] Bug rate significantly higher than frontdoor (>2x per KLOC)
- [ ] Team has React expertise available for migration
- [ ] Clear 3-month window in roadmap for migration work
- [ ] Test coverage gap can be closed within timeline

### No-Go if:

- [ ] Current portal meeting delivery needs adequately
- [ ] Migration would block higher-priority work
- [ ] Test coverage gap cannot be closed in timeline
- [ ] No clear ownership of migrated surface
- [ ] Team prefers current architecture

## Acceptance Criteria for Migration

If migration is approved, the following must be met before deprecating the FastAPI-served portal:

- [ ] All existing browser smoke tests pass on new surface
- [ ] Direct-debug mode works identically
- [ ] Managed auth mode works identically
- [ ] All four views fully functional
- [ ] Query param routing contract preserved
- [ ] No regression in keyboard accessibility
- [ ] No regression in reduced-motion support
- [ ] CI coverage equivalent or better
- [ ] Performance parity (page load, interaction latency)
- [ ] Documentation updated

## References

- Phase 5 plan: Portal Shell Modularization
- `portal.html`, `portal.js`, `portal.css` — current implementation
- `web/secure-landing/` — managed frontdoor
- Browser smoke tests: `scripts/validation/validate_portal_browser_smoke.py`
- Contract tests: `tests/test_app_orchestrator_runtime.py`
