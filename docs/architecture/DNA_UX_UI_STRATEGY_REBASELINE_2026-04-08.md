# DNA UX/UI Strategy Re-baseline

Date: 2026-04-08
Scope: Dynamic Neural Access browser surfaces at `/`, `/login`, `/portal`, `/portal/assets/*`, and the operator-console view contract at `?view=overview|build|operate|review`

## Summary

This document reworks the external "UX/UI Audit and Enhancement Strategy" into
repo truth. The product is not a single React `App.jsx` surface with a Tailwind
token layer. It is a managed frontdoor and login experience implemented in
`web/secure-landing/`, plus a FastAPI operator console implemented through
`portal.html`, `public/portal-assets/portal.css`, and
`public/portal-assets/portal.js`.

The near-term recommendation is incremental improvement, not a shell rewrite.
The product should improve clarity, accessibility, and premium finish while
preserving the current managed-auth, asset-proxy, and contract-tested browser
boundary.

## Repo Truth

- `GET /` is the public DNA homepage rendered by `web/secure-landing/app/route.js`
  through `web/secure-landing/lib/homepage.js`.
- `GET /login` is a separate managed login flow rendered by
  `web/secure-landing/app/login/route.js` and styled by
  `web/secure-landing/public/login.css`.
- `GET /portal` remains the governed operator console. In managed mode the
  frontdoor proxies the upstream FastAPI HTML shell instead of replacing it with
  a React app.
- The operator console already includes client-side view routing for
  `overview`, `build`, `operate`, and `review`, plus a build stepper,
  disclosure panels, an effective-config drawer, and keyboard shortcuts.
- Near-term UX changes must preserve the existing contract coverage:
  `make test-frontdoor-contract`, `make test-portal-contract`,
  `make validate-frontdoor-browser`, and `make validate-portal-browser`.

## Correction Matrix

| Original draft claim | Status | Repo truth | Revised direction |
|---|---|---|---|
| "The current React landing page (`App.jsx`)..." | Remove | The public homepage is server-rendered from `web/secure-landing/lib/homepage.js`; there is no active root `App.jsx` for the DNA browser entry. | Describe the homepage as a managed frontdoor surface, not a React component tree. |
| "Update the design token system in Tailwind." | Remove | The current surfaces are styled through `frontdoor-homepage.css`, `login.css`, and `portal.css`, with CSS variables and repo-local assets. | Target CSS variables, shared color rules, and existing asset-driven CSS. |
| Progressive disclosure is needed in the operator UI. | Keep, but revise | The portal already ships a build stepper, `details` disclosures, and secondary drawers. | Refine the disclosure model, defaults, and hierarchy instead of proposing it as net-new. |
| Keyboard-first operator flows should be introduced. | Revise | The portal already supports a shortcuts modal, build-tab keyboard navigation, job-list keyboard navigation, and shortcut keys such as `Cmd/Ctrl+Enter`. | Extend discoverability and cross-view shortcuts before adding new abstractions. |
| Add `Cmd+K` as a power-user feature. | Revise | There is no command palette yet, but there is existing keyboard infrastructure and query-param console routing. | Treat a command palette as a later enhancement after information architecture cleanup. |
| Migrate `portal.html` into React as a roadmap phase. | Revise | The current frontdoor roadmap keeps FastAPI and `portal.html` as the operator-shell system of record for this horizon. | Move React/Next portal migration into a decision gate, not a near-term phase. |
| Accessibility and contrast need improvement. | Keep | Muted text tokens and dense dark surfaces across the homepage, login, and portal still warrant a structured audit. | Make contrast, focus, and motion parity the first UX priority. |
| Premium typography should be added. | Revise | The repo already uses different sanctioned stacks per surface: homepage Inter fallback, login Manrope plus IBM Plex Mono, portal Portal Sans plus Portal Mono. | Unify hierarchy and usage rules first; delay any greenfield font swap. |
| Motion polish should be added with a new library baseline. | Revise | All three surfaces already use CSS motion and reduced-motion rules. | Prefer CSS-first motion refinements unless a new runtime dependency is justified by a specific interaction. |
| The browser experience should become a seamless SPA. | Revise | The current split is intentional because the managed frontdoor proxies the FastAPI operator shell and keeps backend secrets off the browser. | Prioritize visual and interaction continuity across the split architecture rather than forcing SPA consolidation. |

## Current UX Audit By Surface

### 1. Public Frontdoor (`/`)

Strengths:
- Strong brand and mission framing around verification, provenance, and release
  proof.
- Effective visual storytelling through the release bundle preview and proof
  sections.
- Accessible foundations already exist: skip link, focus-visible styling,
  reduced-motion coverage, and a responsive mobile menu.

Issues to address:
- CTA hierarchy and section rhythm should more clearly separate "learn",
  "verify", and "operator access" flows.
- Contrast on secondary copy and panel borders should be audited against the
  actual video-backed background, not only static colors.
- Mobile spacing and touch-target sizing should be normalized across the header,
  menu summary, and CTA controls.
- Decorative glow and grid treatments should stay subordinate to proof and trust
  messaging.

### 2. Managed Login (`/login`)

Strengths:
- Focused single-task layout with minimal cognitive branching.
- Strong access-control framing with verified-access context and explicit error
  states.
- Managed-auth behaviors are already robust: CSRF binding, login throttling,
  and controlled session rotation.

Issues to address:
- The information sequence can better distinguish "Access identity verified"
  from "operator credentials required".
- Error banners, helper copy, and input spacing should be tuned for faster
  recovery under stress.
- Touch-target, focus, and reduced-motion behavior should stay aligned with the
  homepage and portal rather than feeling like a separate product.

### 3. Operator Console (`/portal`)

Strengths:
- Task-first structure is already strong: overview, build, operate, and review
  are distinct views with query-param routing.
- Progressive disclosure already exists through the build stepper, `details`
  sections, and secondary config tools.
- Power-user foundations are already present: shortcuts modal, keyboard
  navigation for build tabs and job lists, overlay focus trapping, and parity
  tooling for effective config and CLI preview.
- The console already uses clear shell boundaries that can evolve into a more
  polished "bento" presentation without a full rewrite.

Issues to address:
- Step 3 and Step 4 still concentrate a lot of dense operational detail into a
  small number of panels.
- Label/value hierarchy can be sharper, especially where descriptive copy,
  mono values, warnings, and controls compete in the same shell.
- Sticky context for selected job, current asset, or run posture should be more
  deliberate in `operate` and `review`.
- Mobile and narrow-width behavior should be audited across the stepper,
  side-by-side shells, and secondary tooling disclosures.

## Design Directives

### Information Architecture

- Treat the three browser surfaces as one ecosystem with different jobs:
  homepage for trust and orientation, login for controlled entry, and portal for
  high-frequency operation.
- Preserve the current view model in the operator console. Improve the clarity
  of existing `overview`, `build`, `operate`, and `review` surfaces instead of
  reopening the route contract.
- Keep secondary tools secondary. CLI preview, JSON import/export, and exact
  effective config should remain available but should not compete with the main
  dispatch and review path.

### Styling and Tokens

- The implementation layer for near-term visual work is CSS, not Tailwind.
- Use the current CSS sources of truth:
  - `web/secure-landing/public/frontdoor-homepage.css`
  - `web/secure-landing/public/login.css`
  - `public/portal-assets/portal.css`
- Normalize a small shared set of visual rules across all three surfaces:
  contrast-safe text roles, focus-ring behavior, panel borders, panel blur,
  shadow softness, and CTA emphasis.
- Treat video, glow, and glass effects as accent layers. They should support
  trust and legibility rather than become the dominant design signature.

### Typography

- Keep the current sanctioned stacks in the near term:
  - homepage: Inter fallback stack
  - login: Manrope plus IBM Plex Mono
  - portal: Portal Sans plus Portal Mono
- Standardize hierarchy and usage before changing font assets:
  - sans-serif for headings, labels, and explanatory copy
  - monospace only for hashes, IDs, paths, timestamps, and CLI/runtime values
  - larger contrast between headline, body, metadata, and eyebrow text
- Avoid introducing a new cross-product font family until the asset pipeline and
  shared browser architecture justify it.

### Motion and Interaction

- Preserve reduced-motion behavior on every enhancement.
- Prefer CSS-first improvements to easing, duration, hover response, and panel
  transitions before adding a new interaction dependency.
- Any cursor-reactive glow or premium hover treatment should be limited to
  non-critical decorative shells and must not interfere with text contrast or
  pointer precision.
- Improve keyboard discoverability now through visible shortcut hints and better
  modal/help copy; evaluate a command palette only after operator tasks are
  simplified.

### Responsive and Accessible Behavior

- Audit muted text and border tokens on live dark backgrounds across all three
  surfaces.
- Normalize interactive target sizing, especially on the homepage header,
  login form controls, and portal secondary buttons.
- Preserve or improve existing focus-visible coverage, overlay focus trapping,
  and keyboard navigation.
- Ensure design changes are validated in desktop, mobile, keyboard-only, and
  reduced-motion modes before they are considered complete.

## Incremental Delivery Roadmap

### Phase 1: Accessibility and Token Alignment

Scope:
- Contrast audit for frontdoor, login, and portal.
- Shared focus-ring and panel-border rules across the three surfaces.
- Touch-target normalization and type-scale cleanup.

Acceptance focus:
- No route or auth-flow changes.
- Existing contract coverage remains green.
- Browser smoke still passes on homepage, login, and portal entry flows.

### Phase 2: Operator Hierarchy and Disclosure Cleanup

Scope:
- Tighten label/value hierarchy in the portal shells.
- Revisit which `details` groups default open or closed based on task frequency.
- Improve selected-job, next-action, and review-context persistence in
  `operate` and `review`.

Acceptance focus:
- Preserve `?view=` routing and existing build-step semantics.
- Preserve shortcut, drawer, and CLI-parity flows.
- Keep direct-debug and managed mode behavior aligned with existing contracts.

### Phase 3: Cross-surface Visual Continuity

Scope:
- Harmonize CTA emphasis, empty/loading/error states, and premium polish across
  homepage, login, and portal.
- Evolve the existing portal shells toward a more deliberate bento-like visual
  rhythm without replacing the current structural model.
- Improve mobile compression and spacing consistency across all three surfaces.

Acceptance focus:
- Managed auth and proxy boundaries stay unchanged.
- Decorative motion remains optional and reduced-motion safe.
- Frontdoor and portal browser smokes both remain required.

### Phase 4: Power-user Enhancements

Scope:
- Expand shortcut discoverability and cross-view keyboard affordances.
- Tighten review and operate workflows for faster inspection and action.
- Prototype a command palette only if task analysis still shows high navigation
  friction after the earlier phases land.

Acceptance focus:
- Command palette work is optional and should not block the earlier phases.
- New keyboard flows must coexist with the current shortcuts modal and
  disclosure model.

## React/Next Migration Decision Gate

Replatforming the operator shell into React/Next is not a near-term phase. It
should only be opened as a deliberate decision gate.

Open the gate only when all of the following are true:
- Repeated portal changes are bottlenecked by the current `portal.html`,
  `portal.css`, and `portal.js` structure rather than by product ambiguity.
- A shared component system across homepage, login, and portal would materially
  reduce delivery risk or duplication.
- The managed proxy, asset manifest, and direct-debug contracts are explicitly
  mapped to a replacement plan.
- Browser-contract and runtime coverage are strong enough to absorb a shell
  rewrite without losing managed-mode guarantees.

Current blockers:
- `portal.html` remains the operator-shell system of record for the current
  roadmap horizon.
- Managed `/portal/assets/*` proxying and asset-manifest drift detection assume
  the current portal shell contract.
- Direct-debug and managed-mode parity still depend on the existing FastAPI
  shell semantics.

Expected benefits if the gate is opened later:
- Shared component primitives across all browser surfaces.
- Stronger asset and typography reuse.
- Better long-term maintainability for complex operator interactions.

Required output before implementation:
- a dedicated ADR or roadmap update that replaces the current non-goal posture
- explicit migration sequencing for managed mode, direct-debug mode, and test
  coverage

## Validation and Acceptance Criteria

The revised strategy is complete only if all of the following remain true:

- No stale architecture claims remain in the document.
- Every recommendation is mapped to a real current surface: homepage, login, or
  portal.
- Near-term work preserves managed-auth, proxy, and route contracts.
- Future-state items are clearly labeled as gated rather than implied defaults.
- Desktop, mobile, keyboard-only, reduced-motion, and managed-login flows are
  explicitly covered.

Recommended validation commands for any implementation derived from this
strategy:

```bash
make test-frontdoor-contract
make test-portal-contract
make validate-frontdoor-browser
make validate-portal-browser
```

Run `make test-orchestrator-contract` as well if a UX change alters `/v1/*`,
bootstrap behavior, or upstream portal semantics rather than pure presentation.

## Appendix: Already Present vs Proposed

### Already Present

- Separate managed surfaces for `/`, `/login`, and `/portal`
- Focus-visible styling across frontdoor, login, and portal
- Reduced-motion handling across the CSS surfaces
- Portal build stepper and query-param console routing
- Progressive disclosure via `details` sections and config drawers
- Keyboard support for build tabs, job list navigation, overlays, and existing
  shortcuts
- Managed login protections including CSRF, throttling, and session rotation
- Partial 44px control coverage in the portal

### Proposed

- Cross-surface contrast and token audit against live dark/video backgrounds
- Unified hierarchy rules for type, metadata, and monospace data presentation
- Better default disclosure states for dense operator settings
- Stronger sticky context in `operate` and `review`
- Mobile touch-target normalization for homepage and login to match portal
- Cross-surface polish for empty, loading, warning, and recovery states
- Optional command-palette evaluation only after earlier IA improvements land

## Source Notes and Assumptions

- The external UX/UI draft supplied by the user is treated as the source report
  being revised.
- The referenced PDF title matched that report, but this re-baseline is grounded
  primarily in current repository code and docs because full PDF extraction
  tooling was not available in this environment.
- This document is intentionally implementation-ready rather than executive-only
  narrative.
