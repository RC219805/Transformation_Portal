// Overview surface deferred bundle (scaffolding placeholder).
//
// The Overview surface (dashboard hero, recent-job queue, readiness
// badges, capability chips, stats cards) will be carved out of
// portal.template.js into this module in a follow-up commit.
//
//   - Default export: createDeferredOverviewSurfaceApi(host)
//   - host: shared state (state, els, telemetry, formatters) provided
//     by the loader.
//   - Returns: an object whose keys are the surface entry points the
//     main bundle calls (e.g. renderOverviewSurface, hydrateOverview).
//
// Until the Overview carve lands the API is intentionally empty.
export function createDeferredOverviewSurfaceApi(_host) {
  return {};
}
