export function finalizeManagedAssetUrl(parsedUrl) {
  return `${parsedUrl.pathname}${parsedUrl.search}`;
}

export function buildManagedArtifactUrl({
  job,
  artifact,
  apiBase,
  artifactLabel,
  sanitizeManagedAssetUrl
}) {
  const directUrl = String(artifact?.url || "").trim();
  if (directUrl) return sanitizeManagedAssetUrl(`${apiBase}${directUrl}`);
  if (!job || !artifact) return "";
  const relativePath = artifactLabel(artifact);
  if (!relativePath) return "";
  const encodedSegments = relativePath
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return sanitizeManagedAssetUrl(
    `${apiBase}/v1/jobs/${encodeURIComponent(String(job.id || ""))}/artifacts/${encodedSegments}`
  );
}
