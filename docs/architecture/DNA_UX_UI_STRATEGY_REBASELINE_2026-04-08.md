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

### Current Status as of April 9, 2026

The repo moved past the initial re-baseline quickly on April 8, 2026. The
strategy should therefore treat the following as already shipped rather than
still pending:

- Shared accessibility and token alignment across homepage, login, and portal
  landed in commit `42f7ae40` / PR `#1384`.
- The operator-console redesign and loading-polish work landed in commits
  `5a84289a` / PR `#1377` and `21e34e3c` / PR `#1378`.
- Backend-driven operator hints landed in commit `e4228f7b` / PR `#1379`.
- Review provenance and compare-surface accessibility landed in commits
  `304b3e86` / PR `#1380` and `57f30b2d` / PR `#1381`.
- Portal dispatch review now already includes `Next Operator Action`, pre-run
  checks, expected outputs, and a secondary CLI/config disclosure instead of
  leaving those as planned future concepts.
- Portal disclosure auto-open behavior is already state-driven from preview
  issues, research acknowledgments, and reconstruction/runtime posture.
- `operate` and `review` already ship a compact context ribbon for selected
  job, freshness, artifact, and compare state rather than leaving that as a
  future layout concept.
- Review deep links already extend the route contract through additive
  `artifact=<relative-path>` and `compare=1` params on top of the existing
  `view` and `job` state.
- Review compare surfaces already include paired-output summary behavior instead
  of treating compare state as an implicit thumbnail-only affordance.
- `operate` and `review` already preserve selected-job routing through
  `?view=operate|review&job=...` and reuse the last selected job across view
  changes.
- Portal runtime contracts and browser smoke already pin the context ribbon,
  additive review deep links, compare-summary behavior, selected-job reuse, and
  dispatch-tool disclosure so Phase 2B does not need to introduce them as
  net-new capabilities.
- Step 3 now keeps an always-visible posture band for reconstruction state,
  runtime workers, RAW ingest, debug-bundle posture, preview status, and
  estimate summary outside the contextual runtime disclosure.
- Step 3 disclosure badges and hint copy now frame advanced, governance, and
  reconstruction controls as contextual or attention-needed layers instead of
  peers to the primary posture band.
- Step 4 now groups `Next Operator Action` with a dispatch reason and primary
  execute CTA while keeping pre-run evidence and CLI/config parity tools
  visibly secondary.
- Operate/review freshness and paired-comparison copy now stay aligned across
  the context ribbon, selected-job inspector, and review compare summary.
- Phase 3 opened on April 9, 2026 as a bounded cross-surface continuity
  tranche across homepage, login, and portal rather than as a new portal-only
  feature lane.
- The active implementation focus is shared CTA hierarchy, shell/material
  consistency, loading/error-state polish, and stable homepage/login `data-ui`
  hooks for contract tests and browser smoke.
- The frontdoor roadmap remains closed. This strategy document is now the
  active implementation record for the UX tranche instead of opening a new
  frontdoor roadmap phase.
- The local verification baseline on April 9, 2026 is:
  - portal runtime/browser contract slice passes;
  - frontdoor Node contract still requires a Node `22.x` runtime, while the
    current local shell remains on Node `25.9.0`.

Phase 2B is now closed out. Phase 3 is the active UX lane rather than any
further portal-only hierarchy catch-up.

### Phase 1: Accessibility and Token Alignment (Completed April 8, 2026)

Scope:
- Contrast-safe text, focus-ring, panel-border, and touch-target alignment
  across frontdoor, login, and portal.
- Type-scale cleanup and focus-visible parity across the managed browser
  surfaces.

Acceptance focus:
- No route or auth-flow changes.
- Existing contract coverage remains green.
- Browser smoke still passes on homepage, login, and portal entry flows.

Status:
- Completed on April 8, 2026.

### Phase 2B: Portal Hierarchy and Context Close-out (Completed April 8, 2026)

Scope:
- Truth-sync the UX strategy to shipped portal work so the active lane no
  longer treats already-landed operator hints, disclosure defaults, selected-job
  routing, route-backed review context, compare-summary behavior, or dispatch
  parity tooling as pending.
- Tighten Step 3 and Step 4 hierarchy in the portal only by making output
  posture primary and keeping advanced/research controls visibly secondary.
- Keep the shipped compact operate/review context ribbon and shareable URL-backed
  review context, then only refine copy, layout, and consistency where the
  remaining close-out work still benefits from polish.
- Preserve the existing stale-route normalization behavior that reconciles
  invalid job/artifact/compare state back to the nearest valid client-derived
  selection without changing backend APIs.

Acceptance focus:
- Preserve `?view=` routing and existing build-step semantics.
- Preserve additive `job=` deep links plus the existing optional `artifact=`
  and `compare=1` review params without expanding the route contract further.
- Preserve shortcut, drawer, and CLI-parity flows.
- Keep direct-debug and managed mode behavior aligned with existing contracts.

Status:
- Completed on April 8, 2026.

### Phase 3: Cross-surface Visual Continuity (Active April 9, 2026)

Scope:
- Harmonize CTA emphasis, empty/loading/error states, and premium polish across
  homepage, login, and portal.
- Add stable homepage/login `data-ui` hooks so route tests and browser smoke
  key off durable selectors instead of exact marketing copy.
- Evolve the existing portal shells toward a more deliberate bento-like visual
  rhythm without replacing the current structural model.
- Improve mobile compression and spacing consistency across all three surfaces.
- Keep the implementation in the current CSS sources of truth:
  `frontdoor-homepage.css`, `login.css`, and the later custom override section
  of `portal.css`, without introducing a shared CSS asset or new runtime
  dependency.
- Keep the frontdoor roadmap closed and record this tranche here rather than
  opening a separate roadmap lane.

Acceptance focus:
- Managed auth and proxy boundaries stay unchanged.
- No route, auth, proxy, or `/v1/*` semantic changes are introduced.
- Homepage/login contract tests and browser smoke must be updated alongside the
  new selector hooks and continuity polish.
- Decorative motion remains optional and reduced-motion safe.
- Frontdoor and portal browser smokes both remain required.
- Frontdoor contract verification still runs only in a Node `22.x`
  environment.

Status:
- Active as of April 9, 2026.

### Phase 4: Power-user Enhancements (Deferred)

Scope:
- Expand shortcut discoverability and cross-view keyboard affordances.
- Tighten review and operate workflows for faster inspection and action.
- Prototype a command palette only if task analysis still shows high navigation
  friction after the earlier phases land.

Acceptance focus:
- Command palette work is optional and should not block the earlier phases.
- New keyboard flows must coexist with the current shortcuts modal and
  disclosure model.

Status:
- Deferred until later UX evidence proves that navigation friction remains high
  after the completed hierarchy work and any future Phase 3 polish.

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
- The completed Phase 2B close-out is recorded as truth-sync plus hierarchy
  polish rather than as a fresh feature tranche for already-shipped portal
  capabilities.
- Future-state items are clearly labeled as gated rather than implied defaults.
- Desktop, mobile, keyboard-only, reduced-motion, and managed-login flows are
  explicitly covered.

Recommended validation commands for any implementation derived from this
strategy:

```bash
make test-portal-contract
make validate-portal-browser
```

Run the frontdoor contract/browser checks only when homepage or login surfaces
change in the same tranche:

```bash
make test-frontdoor-contract
make validate-frontdoor-browser
```

`make test-frontdoor-contract` remains Node `22.x` only because the
frontdoor package explicitly rejects unsupported runtimes.

Run `make test-orchestrator-contract` as well only if a UX change alters
`/v1/*`, bootstrap behavior, or upstream portal semantics rather than pure
portal presentation and state handling.

## Appendix: Already Present vs Proposed

### Already Present

- Separate managed surfaces for `/`, `/login`, and `/portal`
- Focus-visible styling across frontdoor, login, and portal
- Reduced-motion handling across the CSS surfaces
- Portal build stepper and query-param console routing
- Progressive disclosure via `details` sections and config drawers
- Dispatch review with `Next Operator Action`, pre-run checks, expected outputs,
  and a secondary CLI/config disclosure
- State-driven disclosure defaults for advanced, governance, reconstruction,
  and dispatch-tool groupings
- An always-visible Step 3 posture band for runtime posture, preview status,
  research risk, and estimate summary outside the reconstruction disclosure
- Contextual Step 3 badge/hint copy that marks secondary controls as
  contextual or attention-needed instead of primary
- A compact operate/review context ribbon for selected job, freshness,
  artifact, and compare state
- A Step 4 primary dispatch lane that pairs `Next Operator Action` with a live
  dispatch reason and execute CTA while keeping evidence and parity tools
  secondary
- Consistent freshness and paired-comparison copy across the ribbon,
  selected-job inspector, and review compare summary
- Shareable additive `artifact=` and `compare=1` deep links for review context
  with stale-route normalization back to valid client state
- Compare-summary review behavior for paired outputs
- Selected-job route persistence across `operate` and `review`
- Keyboard support for build tabs, job list navigation, overlays, and existing
  shortcuts
- Managed login protections including CSRF, throttling, and session rotation
- Cross-surface token alignment and 44px target coverage for the managed
  browser surfaces
- Portal loading polish, backend-driven operator hints, review provenance, and
  review-compare accessibility improvements shipped on April 8, 2026
- Portal contract and browser coverage for the context ribbon, additive review
  deep links, compare-summary behavior, selected-job reuse, and dispatch-tool
  disclosure

### Proposed

- Active cross-surface polish across homepage, login, and portal, including
  stable homepage/login selector hooks for tests and browser smoke
- Optional command-palette evaluation only after earlier IA improvements land

## Source Notes and Assumptions

- The external UX/UI draft supplied by the user is treated as the source report
  being revised.
- The referenced PDF title matched that report, but this re-baseline is grounded
  primarily in current repository code and docs because full PDF extraction
  tooling was not available in this environment.
- This document is intentionally implementation-ready rather than executive-only
  narrative.
