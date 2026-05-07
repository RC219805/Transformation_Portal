// Operate surface deferred bundle (scaffolding placeholder).
//
// The Operate surface (job queue + selected-job inspector) will be carved
// out of portal.template.js into this module in a follow-up commit. For
// now this file exposes the contract the upcoming carve must satisfy:
//
//   - Default export: createDeferredOperateSurfaceApi(host)
//   - host: a plain object containing references to shared state (state,
//     els, telemetry, dispatch helpers, and any utilities the surface
//     needs). The host is constructed at call time by the loader caller
//     so the deferred bundle never reaches into globals.
//   - Returns: an object whose keys are the surface entry points the
//     main bundle calls (e.g. renderOperateSurface, hydrateJobInspector).
//
// Until the Operate carve lands the API is intentionally empty so the
// loader path can be exercised without behavior changes.
export function createDeferredOperateSurfaceApi(_host) {
  return {};
}
