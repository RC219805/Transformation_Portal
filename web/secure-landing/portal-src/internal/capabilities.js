export const PORTAL_CAPABILITY_STATUSES = Object.freeze([
  "enabled",
  "available",
  "gated",
  "missing_runtime",
  "needs_ack",
  "blocked",
  "offline",
  "not_portal_controlled"
]);

const STATUS_SET = new Set(PORTAL_CAPABILITY_STATUSES);
const LUX_PIPELINE = "lux-depth-v3";
const ARCHIVE_PIPELINES = new Set(["archive-gate-a", "archive-gate-b", "archive-gate-c"]);

function boolLike(value, fallback = false) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized) return fallback;
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return fallback;
}

function text(value) {
  return String(value ?? "").trim();
}

function lower(value) {
  return text(value).toLowerCase();
}

function statusLabel(status) {
  return normalizeCapabilityStatus(status).replace(/_/g, " ");
}

function hasPreviewErrors(preview) {
  return Array.isArray(preview?.field_errors) && preview.field_errors.length > 0;
}

function hasBlockedReadiness(readiness, readinessIssues = []) {
  if (lower(readiness?.status) === "blocked") return true;
  return readinessIssues.some((issue) => lower(issue?.severity) === "blocked");
}

function hasArchivePrereqIssue(readinessIssues = []) {
  return readinessIssues.some((issue) => {
    const reason = lower(issue?.reason);
    return reason === "archive_index_required"
      || reason === "rights_manifest_required"
      || reason === "manifest_jsonl_required";
  });
}

function ackMissing(args, fields) {
  return fields.some((field) => !boolLike(args?.[field], false));
}

function backendStatus(backendOk, active, inactive = "available") {
  if (!backendOk) return "offline";
  return active ? "enabled" : inactive;
}

function makeRow({ id, group, label, status, summary, detail, nextAction = "" }) {
  const normalizedStatus = normalizeCapabilityStatus(status);
  return {
    id,
    group,
    label,
    status: normalizedStatus,
    statusLabel: statusLabel(normalizedStatus),
    summary,
    detail,
    nextAction
  };
}

export function normalizeCapabilityStatus(value, fallback = "available") {
  const normalized = lower(value);
  if (STATUS_SET.has(normalized)) return normalized;
  const normalizedFallback = lower(fallback);
  return STATUS_SET.has(normalizedFallback) ? normalizedFallback : "available";
}

export function buildPortalCapabilityCatalog(input = {}) {
  const pipeline = text(input.pipeline || LUX_PIPELINE);
  const args = input.args && typeof input.args === "object" ? input.args : {};
  const features = input.features && typeof input.features === "object" ? input.features : {};
  const readiness = input.readiness && typeof input.readiness === "object" ? input.readiness : null;
  const readinessIssues = Array.isArray(input.readinessIssues) ? input.readinessIssues : [];
  const preview = input.preview && typeof input.preview === "object" ? input.preview : null;
  const jobs = Array.isArray(input.jobs) ? input.jobs : [];
  const activeJob = input.activeJob && typeof input.activeJob === "object" ? input.activeJob : null;
  const reviewJob = input.reviewJob && typeof input.reviewJob === "object" ? input.reviewJob : null;
  const captioningRuntimeReadiness = input.captioningRuntimeReadiness
    && typeof input.captioningRuntimeReadiness === "object"
    ? input.captioningRuntimeReadiness
    : {};
  const backendOk = Boolean(input.backendOk);
  const bootstrapReady = Boolean(input.bootstrapReady);
  const authMode = lower(input.authMode || "managed_unavailable");
  const isLux = pipeline === LUX_PIPELINE;
  const isArchive = ARCHIVE_PIPELINES.has(pipeline);
  const previewBlocked = hasPreviewErrors(preview) || hasBlockedReadiness(readiness, readinessIssues);
  const archivePrereqIssue = hasArchivePrereqIssue(readinessIssues);
  const segmentationEnabled = boolLike(args.enable_segmentation, false);
  const segmentationBackend = lower(args.segmentation_backend || "efficientsam");
  const sam2Active = segmentationEnabled && segmentationBackend === "sam2";
  const reconstructionEnabled = boolLike(args.enable_reconstruction, false);
  const depthBackend = lower(args.depth_backend || "da3");
  const depthProActive = depthBackend === "depth_pro";
  const captioningEnabled = boolLike(args.vlm_captioning_enabled, false);
  const captioningRuntimeStatus = lower(captioningRuntimeReadiness.status || "off");
  const stagedUploadSupported = Boolean(input.stagedUploadSupported);
  const hasJobs = jobs.length > 0;
  const activeJobState = lower(activeJob?.state || activeJob?.status || "");
  const hasActiveStream = Boolean(activeJob && ["queued", "running", "ready"].includes(activeJobState));
  const hasReviewArtifacts = Boolean(
    reviewJob
    && (Array.isArray(reviewJob.artifacts) ? reviewJob.artifacts.length > 0 : reviewJob.artifacts)
  );

  const rows = [
    makeRow({
      id: "managed_access",
      group: "Entry",
      label: "Managed access",
      status: !bootstrapReady ? "gated" : authMode === "managed" ? "enabled" : "gated",
      summary: authMode === "managed" ? "Server-side credential handoff is active." : "Managed handoff is not active.",
      detail: "Sourced from /portal/bootstrap; browser API key entry stays disabled in managed mode.",
      nextAction: authMode === "managed" ? "Continue through the managed console." : "Recover access before privileged work."
    }),
    makeRow({
      id: "direct_debug",
      group: "Entry",
      label: "Direct debug",
      status: authMode === "direct_debug" && features.directDebug !== false ? "enabled" : "gated",
      summary: authMode === "direct_debug" ? "Direct-debug controls are available." : "Direct-debug entry is gated by auth mode.",
      detail: "This is a bootstrap-controlled fallback, not a managed operator default.",
      nextAction: authMode === "direct_debug" ? "Use session-only API key entry." : "Use managed login."
    }),
    makeRow({
      id: "lux_depth_v3",
      group: "Build",
      label: "Lux Depth v3",
      status: backendStatus(backendOk, isLux),
      summary: isLux ? "Current draft targets the Lux depth pipeline." : "Lux depth remains selectable from the pipeline menu.",
      detail: "Preset, depth backend, model, segmentation, deliverable, and preview contracts are configured in Build."
    }),
    makeRow({
      id: "archive_gates",
      group: "Build",
      label: "Archive gates",
      status: !backendOk ? "offline" : archivePrereqIssue ? "needs_ack" : isArchive ? "enabled" : "available",
      summary: isArchive ? "Current draft targets an archive gate." : "Fixity, BagIt, and METS gates remain portal-dispatchable.",
      detail: "Archive roots, indexes, and rights manifests are governed by readiness and config-preview validation.",
      nextAction: archivePrereqIssue ? "Supply the required archive index or manifest before dispatch." : ""
    }),
    makeRow({
      id: "da3_apache",
      group: "Build",
      label: "DA3 Apache path",
      status: backendStatus(backendOk, isLux && !depthProActive),
      summary: depthProActive ? "Depth Pro is selected instead of the Apache DA3 path." : "DA3 Apache-backed depth is selected.",
      detail: "The default DA3 path stays the primary governed Lux depth route."
    }),
    makeRow({
      id: "depth_pro",
      group: "Build",
      label: "Depth Pro",
      status: !backendOk ? "offline" : depthProActive && ackMissing(args, ["accept_apple_depth_pro_research_license"]) ? "needs_ack" : depthProActive ? "enabled" : "available",
      summary: depthProActive ? "Research-only depth backend is selected." : "Research-only depth backend is available as a governed option.",
      detail: "Depth Pro requires explicit Apple research-license acknowledgment before dispatch."
    }),
    makeRow({
      id: "materials_v3",
      group: "Build",
      label: "Materials V3",
      status: backendStatus(backendOk, isLux && boolLike(args.materials_v3, false)),
      summary: boolLike(args.materials_v3, false) ? "Material estimation outputs are enabled." : "Material estimation outputs are available.",
      detail: "Controlled by the Lux deliverables switch and reflected in config preview."
    }),
    makeRow({
      id: "pbr_generation",
      group: "Build",
      label: "PBR generation",
      status: backendStatus(backendOk, isLux && boolLike(args.pbr, false)),
      summary: boolLike(args.pbr, false) ? "PBR maps are enabled for the run." : "PBR maps can be enabled for Lux runs.",
      detail: "PBR is a dispatchable Lux output flag, not a separate route."
    }),
    makeRow({
      id: "segmentation",
      group: "Build",
      label: "Segmentation",
      status: backendStatus(backendOk, isLux && segmentationEnabled),
      summary: segmentationEnabled ? `Segmentation is enabled via ${segmentationBackend || "backend"}.` : "Segmentation is available for Lux runs.",
      detail: "Backend and strictness controls stay visible when segmentation is enabled."
    }),
    makeRow({
      id: "sam2_segmentation",
      group: "Build",
      label: "SAM2 segmentation",
      status: !backendOk ? "offline" : sam2Active && text(args.sam2_checkpoint_path) === "" ? "missing_runtime" : sam2Active ? "enabled" : "available",
      summary: sam2Active ? "SAM2 is the selected segmentation backend." : "SAM2 tuning is available when SAM2 is selected.",
      detail: "Checkpoint, tiling, generator, and concurrency controls are portal-configurable."
    }),
    makeRow({
      id: "reconstruction",
      group: "Build",
      label: "Reconstruction",
      status: !backendOk ? "offline" : reconstructionEnabled && ackMissing(args, ["accept_research_tools_license"]) ? "needs_ack" : reconstructionEnabled ? "enabled" : "available",
      summary: reconstructionEnabled ? "Scene reconstruction is enabled." : "Scene reconstruction is available as an experimental Lux option.",
      detail: "Grouping, sidecar, tier, iterations, and debug-bundle controls are portal-configurable."
    }),
    makeRow({
      id: "raw_ingest",
      group: "Build",
      label: "RAW ingest",
      status: backendStatus(backendOk, isLux && lower(args.raw_ingest_mode || "auto") !== "auto"),
      summary: `RAW ingest mode is ${lower(args.raw_ingest_mode || "auto") || "auto"}.`,
      detail: "RAW mode is configurable; white-balance and demosaic policy remain backend-locked."
    }),
    makeRow({
      id: "runtime_tuning",
      group: "Build",
      label: "Runtime tuning",
      status: !backendOk ? "offline" : isLux && (text(args.max_workers) || text(args.max_gpu_workers) || text(args.log_level)) ? "enabled" : "available",
      summary: "CPU/GPU worker caps and log level are configurable for Lux runs.",
      detail: "Auto remains the default unless the operator pins bounded runtime values."
    }),
    makeRow({
      id: "run_card",
      group: "Build",
      label: "Run-card proofs",
      status: backendStatus(backendOk, isLux && boolLike(args.emit_run_card, false)),
      summary: boolLike(args.run_card_include_proofs, false) ? "Run-card proof capture is enabled." : "Run-card emission is available; proof capture is optional.",
      detail: "Run cards are dispatchable outputs and proof inclusion is controlled in Build."
    }),
    makeRow({
      id: "staged_uploads",
      group: "Build",
      label: "Staged uploads",
      status: !stagedUploadSupported ? "not_portal_controlled" : !bootstrapReady || !features.stagedUploads ? "gated" : "available",
      summary: stagedUploadSupported ? "Staged upload controls are visible for supported pipelines." : "Current pipeline does not support portal staging.",
      detail: "The rollout flag controls whether the visible staged-upload controls are interactive."
    }),
    makeRow({
      id: "fastvlm_captioning",
      group: "Review",
      label: "FastVLM sidecars",
      status: !isLux ? "not_portal_controlled" : !bootstrapReady || !features.fastVlmCaptioning ? "gated" : captioningEnabled && captioningRuntimeStatus === "missing_runtime" ? "missing_runtime" : captioningEnabled && captioningRuntimeStatus === "invalid_config" ? "blocked" : captioningEnabled ? "enabled" : "available",
      summary: captioningEnabled ? "Advisory caption sidecars are enabled." : "Advisory caption controls are visible for Lux runs.",
      detail: "FastVLM remains advisory review metadata and never satisfies quality gates."
    }),
    makeRow({
      id: "job_queue",
      group: "Operate",
      label: "Queue",
      status: !backendOk ? "offline" : hasJobs || hasActiveStream ? "enabled" : "available",
      summary: hasJobs ? `${jobs.length} job${jobs.length === 1 ? "" : "s"} loaded.` : "Queue controls are ready when the backend is online.",
      detail: "Operate reflects queued, running, succeeded, failed, canceled, and partial jobs."
    }),
    makeRow({
      id: "sse_stream",
      group: "Operate",
      label: "SSE freshness",
      status: !backendOk ? "offline" : hasActiveStream ? "enabled" : "available",
      summary: hasActiveStream ? "Active job stream context is present." : "SSE reconnect and freshness monitoring are available.",
      detail: "Freshness and transport warnings stay observable in Operate and Review."
    }),
    makeRow({
      id: "artifact_review",
      group: "Review",
      label: "Artifact review",
      status: !backendOk ? "offline" : hasReviewArtifacts ? "enabled" : "available",
      summary: hasReviewArtifacts ? "Reviewable artifact context is loaded." : "Artifact review is available once jobs produce outputs.",
      detail: "Review surfaces align artifacts, provenance, thumbnails, compare state, and action buttons."
    }),
    makeRow({
      id: "artifact_viewer",
      group: "Review",
      label: "Artifact viewer modal",
      status: !bootstrapReady || !features.artifactViewerModal ? "gated" : hasReviewArtifacts ? "enabled" : "available",
      summary: features.artifactViewerModal ? "Modal artifact viewing is in this cohort." : "Modal artifact viewing is rollout-gated.",
      detail: "The fallback artifact panel remains available when the modal is gated."
    }),
    makeRow({
      id: "review_surface",
      group: "Review",
      label: "Deferred review surface",
      status: !bootstrapReady || !features.reviewSurfaceDeferred ? "gated" : "available",
      summary: features.reviewSurfaceDeferred ? "Deferred review assets may load on demand." : "Deferred review loading is rollout-gated.",
      detail: "The host keeps selector and fallback behavior stable while review assets load."
    }),
    makeRow({
      id: "portal_rum",
      group: "Operate",
      label: "Portal RUM",
      status: !bootstrapReady || !features.rumTelemetry ? "gated" : "enabled",
      summary: features.rumTelemetry ? "Portal RUM is enabled for this cohort." : "Portal RUM is rollout-gated.",
      detail: "No new event families are required for the capability catalog."
    }),
    makeRow({
      id: "plugin_trust",
      group: "Governance",
      label: "Plugin trust",
      status: "not_portal_controlled",
      summary: "Plugin trust policy is documented and governed outside this browser shell.",
      detail: "The portal should report this governance surface without inventing execution controls."
    })
  ];

  const blockedRows = rows.filter((row) => ["blocked", "needs_ack", "missing_runtime", "offline", "gated"].includes(row.status));
  const preferredNext = blockedRows.find((row) => row.status === "blocked")
    || blockedRows.find((row) => row.status === "needs_ack")
    || blockedRows.find((row) => row.status === "missing_runtime")
    || blockedRows.find((row) => row.status === "offline")
    || blockedRows.find((row) => row.status === "gated")
    || rows.find((row) => row.status === "available")
    || rows[0];

  return {
    rows,
    summary: {
      total: rows.length,
      enabled: rows.filter((row) => row.status === "enabled").length,
      actionable: blockedRows.length,
      previewBlocked,
      nextActionCapabilityId: preferredNext?.id || "",
      nextActionStatus: preferredNext?.status || "available",
      nextActionLabel: preferredNext
        ? `${preferredNext.label}: ${preferredNext.nextAction || preferredNext.summary}`
        : "Capability catalog is ready."
    }
  };
}
