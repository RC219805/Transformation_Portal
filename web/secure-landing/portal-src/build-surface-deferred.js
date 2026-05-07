// Build surface deferred bundle (scaffolding placeholder).
//
// The Build surface (pipeline/preset selection, config form, dispatch)
// will be carved out of portal.template.js into this module in a
// follow-up commit. The contract for the carve mirrors the review and
// operate deferred bundles:
//
//   - Default export: createDeferredBuildSurfaceApi(host)
//   - host: shared state (state, els, telemetry, dispatch helpers,
//     config-preview machinery, validators) provided by the loader.
//   - Returns: an object whose keys are the surface entry points the
//     main bundle calls (e.g. renderBuildSurface, syncBuildControls).
//
// Until the Build carve lands the API is intentionally empty.
export function createDeferredBuildSurfaceApi(_host) {
  return {};
}
