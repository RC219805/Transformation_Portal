// Navigation metadata only. All photographic reads revalidate on the server.
export function photographicSelectionFor(state, job, artifactPath) {
  if (!state.auth?.features?.photographicPagedCatalogue || !job ||
    !(/lux[-_]depth|photograph/i.test(String(job.pipeline || job.pipeline_name || "")) ||
      (job.artifacts || []).some((item) => /(^|\/)delivery\.tiff?$/.test(item.relative_path || item.path || "")))) return null;
  const selected = state.artifactUi.photographicSelection;
  const actorScope = JSON.stringify([state.auth.mode, state.auth.actor, state.auth.tenant]);
  return selected?.jobId === job.id && selected.actorScope === actorScope &&
    [selected.artifactPath, selected.railPath].includes(artifactPath)
    ? selected : { artifactPath, railPath: "" };
}
