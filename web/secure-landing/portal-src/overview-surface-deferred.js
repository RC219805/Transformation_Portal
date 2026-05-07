// Overview surface deferred bundle (intentional placeholder).
//
// AUDIT VERDICT: NO CARVE. The Overview surface remains rendered from
// portal-src/portal.template.js. This file is kept as a placeholder so
// the manifest, asset-budget, and loader plumbing established by the
// scaffolding commit (f5dd7ec) stay in place as a documented seam for
// any future Overview-only render code, but no production logic moves
// into this bundle in the current surface-split milestone.
//
// Audit findings (commit f1d3e3a + Overview audit):
//
//   - Only ~94 LOC are truly Overview-only render code:
//     renderCapabilityChips (37 LOC, writes only els.capabilityChips)
//     renderPresetIntelligence (57 LOC, writes 8 hero info elements).
//
//   - Larger apparent Overview candidates are not Overview-only:
//       * renderMissionControl is a parent dispatcher whose 7 callees
//         are mixed: renderCapabilityChips and renderPresetIntelligence
//         are Overview-only, but syncDisclosurePanels (346 LOC),
//         renderBuildStepPulse (32 LOC), and syncBuildSurfaceApplicability
//         are Build-only; renderRuntimeBriefing and renderGovernanceBanner
//         are dual-surface (Overview hero + Build form). Moving
//         renderMissionControl would re-monolithize Build.
//       * renderConsoleContextRibbon is a multi-view dispatcher with
//         explicit currentView branching across operate/review vs.
//         overview/build; it must stay in main.
//       * _syncOverviewBuildLoadingState atomically synchronizes
//         skeleton visibility and aria-busy across Overview AND Build
//         shells; splitting it would force dual-state-sync callbacks.
//
//   - Overview rendering is not on the bootstrap critical path.
//     renderMissionControl fires through renderReviewSurfaces on job
//     state changes, never at portal bootstrap. Lazy-loading the
//     Overview bundle would not reduce time-to-interactive, unlike
//     the Operate and Build carves.
//
//   - Signal-to-noise of a minimal 2-function carve is poor:
//     ~94 LOC moved, host of ~12–15 keys, expected payload reduction
//     of ~3 KB raw / ~0.7 KB gzip on portal.js. Compare with
//     Operate (17,508 raw / 4,110 gz, 7 functions, 28-key host) and
//     Build (5,523 raw / 1,219 gz, 8 functions, 9-key host).
//
// The placeholder factory preserves the loader contract:
//   - Default export: createDeferredOverviewSurfaceApi(host) -> {}
//   - The build pipeline (scripts/build-portal-bundle.mjs) still emits
//     public/portal-assets/portal-overview.js for the manifest.
//   - portal_asset_budgets.json keeps a tight placeholder budget
//     (1024 raw / 512 gzip) so any future Overview-only render code
//     that grows this bundle has to bump the budget explicitly.
//
// When Overview-only render code grows enough to justify a real carve
// (a clean entry point that does not entangle Build skeleton state or
// mission-control sub-renderers), this file is the place it lands.
// Until then, leaving Overview rendering inline avoids a symbolic
// micro-carve that would cost more in host complexity than it saves
// in bytes.
export function createDeferredOverviewSurfaceApi(_host) {
  return {};
}
