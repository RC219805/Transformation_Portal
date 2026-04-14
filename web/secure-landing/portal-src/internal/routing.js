export function createPortalRouteHelpers(windowRef) {
  function build({
    viewName,
    jobId = "",
    artifactPath = "",
    compareEnabled = null,
    resolveView,
    normalizeSelectedJobId,
    normalizeArtifactRoutePath,
    activeContext
  }) {
    const url = new URL(windowRef.location.href);
    const resolvedView = resolveView(viewName);
    const resolvedJobId = normalizeSelectedJobId(jobId);
    url.searchParams.set("view", resolvedView);

    if ((resolvedView === "operate" || resolvedView === "review") && resolvedJobId) {
      const context = activeContext(resolvedJobId);
      const resolvedArtifactPath = normalizeArtifactRoutePath(artifactPath) || context.artifactPath;
      const resolvedCompareEnabled = compareEnabled === null ? context.compareEnabled : Boolean(compareEnabled);
      url.searchParams.set("job", resolvedJobId);
      if (resolvedArtifactPath) {
        url.searchParams.set("artifact", resolvedArtifactPath);
      } else {
        url.searchParams.delete("artifact");
      }
      if (resolvedCompareEnabled) {
        url.searchParams.set("compare", "1");
      } else {
        url.searchParams.delete("compare");
      }
    } else {
      url.searchParams.delete("job");
      url.searchParams.delete("artifact");
      url.searchParams.delete("compare");
    }

    return url;
  }

  function read({
    resolveView,
    normalizeSelectedJobId,
    normalizeArtifactRoutePath,
    normalizeCompareQueryValue
  }) {
    const url = new URL(windowRef.location.href);
    return {
      view: resolveView(url.searchParams.get("view")),
      jobId: normalizeSelectedJobId(url.searchParams.get("job")),
      artifactPath: normalizeArtifactRoutePath(url.searchParams.get("artifact")),
      compareEnabled: normalizeCompareQueryValue(url.searchParams.get("compare")),
      hasArtifact: url.searchParams.has("artifact"),
      hasCompare: url.searchParams.has("compare")
    };
  }

  return {
    build,
    read
  };
}
