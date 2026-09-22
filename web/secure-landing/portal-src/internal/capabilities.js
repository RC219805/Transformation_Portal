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
const ATTENTION_STATUSES = new Set(["blocked", "needs_ack", "missing_runtime", "offline", "gated"]);
const V3_ONLY_ROWS = new Set([
  "depth_pro", "materials_v3", "pbr_generation", "segmentation", "sam2_segmentation",
  "reconstruction", "raw_ingest", "runtime_tuning", "run_card", "fastvlm_captioning"
]);

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

function object(value, fallback = {}) {
  return value && typeof value === "object" ? value : fallback;
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

function makeRow({ id, group, label, status, summary, detail = "", nextAction = "" }) {
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
  const args = object(input.args);
  const features = object(input.features);
  const readiness = object(input.readiness, null);
  const readinessIssues = Array.isArray(input.readinessIssues) ? input.readinessIssues : [];
  const preview = object(input.preview, null);
  const jobs = Array.isArray(input.jobs) ? input.jobs : [];
  const activeJob = object(input.activeJob, null);
  const reviewJob = object(input.reviewJob, null);
  const captioningReadiness = object(input.captioningRuntimeReadiness);
  const backendOk = Boolean(input.backendOk);
  const bootstrapReady = Boolean(input.bootstrapReady);
  const authMode = lower(input.authMode || "managed_unavailable");
  const isLux = pipeline === LUX_PIPELINE;
  const isPhotography = pipeline === "lux-depth-v5";
  const isArchive = ARCHIVE_PIPELINES.has(pipeline);
  // The host supplies only the preview matched to the current draft. A failed
  // request may have no field errors (for example, an authentication failure).
  const previewBlocked = lower(preview?.status) === "error"
    || hasPreviewErrors(preview) || hasBlockedReadiness(readiness, readinessIssues);
  const previewReady = lower(preview?.status) === "ready"
    && (!text(preview?.pipeline) || preview.pipeline === pipeline)
    && !previewBlocked;
  const archivePrereqIssue = hasArchivePrereqIssue(readinessIssues);
  const segmentationEnabled = boolLike(args.enable_segmentation, false);
  const segmentationBackend = lower(args.segmentation_backend || "efficientsam");
  const sam2Active = segmentationEnabled && segmentationBackend === "sam2";
  const reconstructionEnabled = boolLike(args.enable_reconstruction, false);
  const depthBackend = lower(args.depth_backend || "da3");
  const depthProActive = depthBackend === "depth_pro";
  const captioningEnabled = boolLike(args.vlm_captioning_enabled, false);
  const captioningRuntimeStatus = lower(captioningReadiness.status || "off");
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
      summary: bootstrapReady && authMode === "managed" ? "Managed sign-in is active." : "Managed sign-in is not active.",
      detail: "No browser API key is needed.",
      nextAction: bootstrapReady && authMode === "managed" ? "" : "Recover managed access before starting a run."
    }),
    makeRow({
      id: "direct_debug",
      group: "Entry",
      label: "Direct debug",
      status: authMode === "direct_debug" && features.directDebug !== false ? "enabled" : "gated",
      summary: authMode === "direct_debug" ? "Direct-debug controls are available." : "Managed sign-in does not require direct debug.",
      detail: "For standalone troubleshooting only.",
      nextAction: authMode === "direct_debug" ? "Enter an API key for this page only." : ""
    }),
    makeRow({
      id: "lux_depth_v3",
      group: "Build",
      label: "Lux Depth v3",
      status: backendStatus(backendOk, isLux),
      summary: isLux ? "Selected for this draft." : "Select this pipeline in Build.",
      detail: "Production baseline; configure outputs in Build."
    }),
    makeRow({
      id: "lux_depth_v5",
      group: "Build",
      label: "LuxDepthV5 photography",
      status: !backendOk ? "offline" : isPhotography ? previewBlocked ? "blocked" : lower(readiness?.status) === "ready" && previewReady ? "enabled" : "gated" : "available",
      summary: !isPhotography
        ? "Opt-in photography, when enabled by the server."
        : previewBlocked
          ? "Resolve draft errors before starting a run."
          : !previewReady
            ? "Validate this draft in Build."
            : "Configuration preview passed.",
      detail: "Requires preview and readiness. Default size: 518; V3 remains the baseline.",
      nextAction: !isPhotography ? "" : !previewReady
          ? "Open Build to validate the current draft."
          : lower(readiness?.status) !== "ready"
            ? "Open Build to review photography readiness."
            : ""
    }),
    makeRow({
      id: "lux_depth_v4",
      group: "Build",
      label: "LuxDepthV4 foundation",
      status: "not_portal_controlled",
      summary: "Standalone V4 CLI foundation for V5."
    }),
    makeRow({
      id: "materials_v4",
      group: "Build",
      label: "MaterialsV4 evidence",
      status: !isPhotography ? "not_portal_controlled" : backendStatus(backendOk, Boolean(text(args.materials_manifest))),
      summary: "Use an existing MaterialsV4 evidence manifest.",
      detail: "Supply existing evidence; the server validates it."
    }),
    makeRow({
      id: "archive_gates",
      group: "Build",
      label: "Archive gates",
      status: !backendOk ? "offline" : isArchive && archivePrereqIssue ? "needs_ack" : isArchive ? "enabled" : "available",
      summary: isArchive ? "Archive gate selected." : "Fixity, BagIt, and METS workflows.",
      detail: "Validate archive paths and manifests in Build.",
      nextAction: archivePrereqIssue ? "Supply the required archive index or manifest before dispatch." : ""
    }),
    makeRow({
      id: "da3_apache",
      group: "Build",
      label: "DA3 Apache path",
      status: backendStatus(backendOk, (isLux || isPhotography) && !depthProActive),
      summary: isPhotography ? "DA3 Metric selected." : isLux
        ? depthProActive ? "Depth Pro selected instead." : "DA3 depth selected."
        : "Depth estimation for Lux workflows."
    }),
    makeRow({
      id: "depth_pro",
      group: "Build",
      label: "Depth Pro",
      status: !backendOk ? "offline" : depthProActive && ackMissing(args, ["accept_apple_depth_pro_research_license"]) ? "needs_ack" : depthProActive ? "enabled" : "available",
      summary: "Optional research depth backend.",
      detail: "Requires Apple research-license acknowledgment.",
      nextAction: depthProActive && ackMissing(args, ["accept_apple_depth_pro_research_license"])
        ? "Acknowledge the research license in Build or change backend." : ""
    }),
    makeRow({
      id: "materials_v3",
      group: "Build",
      label: "Materials V3",
      status: backendStatus(backendOk, isLux && boolLike(args.materials_v3, false)),
      summary: "Estimate materials for the run."
    }),
    makeRow({
      id: "pbr_generation",
      group: "Build",
      label: "PBR generation",
      status: backendStatus(backendOk, isLux && boolLike(args.pbr, false)),
      summary: "Generate PBR maps."
    }),
    makeRow({
      id: "segmentation",
      group: "Build",
      label: "Segmentation",
      status: backendStatus(backendOk, isLux && segmentationEnabled),
      summary: segmentationEnabled ? `Selected: ${segmentationBackend}.` : "Optional segmentation masks.",
      detail: "Choose a backend and strictness in Build."
    }),
    makeRow({
      id: "sam2_segmentation",
      group: "Build",
      label: "SAM2 segmentation",
      status: !backendOk ? "offline" : sam2Active && text(args.sam2_checkpoint_path) === "" ? "missing_runtime" : sam2Active ? "enabled" : "available",
      summary: "Segmentation with SAM2.",
      detail: "Set checkpoint and tiling in Build.",
      nextAction: sam2Active && !text(args.sam2_checkpoint_path)
        ? "Set the SAM2 checkpoint in Build or change backend." : ""
    }),
    makeRow({
      id: "reconstruction",
      group: "Build",
      label: "Reconstruction",
      status: !backendOk ? "offline" : reconstructionEnabled && ackMissing(args, ["accept_research_tools_license"]) ? "needs_ack" : reconstructionEnabled ? "enabled" : "available",
      summary: "Experimental scene reconstruction.",
      detail: "Configure scene grouping and cameras in Build.",
      nextAction: reconstructionEnabled && ackMissing(args, ["accept_research_tools_license"])
        ? "Acknowledge the research tools license in Build or disable reconstruction." : ""
    }),
    makeRow({
      id: "raw_ingest",
      group: "Build",
      label: "RAW ingest",
      status: backendStatus(backendOk, isLux && lower(args.raw_ingest_mode || "auto") !== "auto"),
      summary: `RAW ingest mode is ${lower(args.raw_ingest_mode || "auto") || "auto"}.`,
      detail: "White balance and demosaic are server-managed."
    }),
    makeRow({
      id: "runtime_tuning",
      group: "Build",
      label: "Runtime tuning",
      status: !backendOk ? "offline" : isLux && (text(args.max_workers) || text(args.max_gpu_workers) || text(args.log_level)) ? "enabled" : "available",
      summary: "Set worker limits and log level in Build.",
      detail: "Auto is the default."
    }),
    makeRow({
      id: "run_card",
      group: "Build",
      label: "Run-card proofs",
      status: backendStatus(backendOk, isLux && boolLike(args.emit_run_card, false)),
      summary: boolLike(args.run_card_include_proofs, false) ? "Proof capture enabled." : "Optional run cards and proofs."
    }),
    makeRow({
      id: "staged_uploads",
      group: "Build",
      label: "Staged uploads",
      status: !stagedUploadSupported ? "not_portal_controlled" : !bootstrapReady || !features.stagedUploads ? "gated" : "available",
      summary: stagedUploadSupported ? "Upload controls supported." : "Unavailable for this pipeline.",
      detail: "Availability depends on server configuration."
    }),
    makeRow({
      id: "fastvlm_captioning",
      group: "Review",
      label: "FastVLM sidecars",
      status: !isLux ? "not_portal_controlled" : !bootstrapReady || !features.fastVlmCaptioning ? "gated" : captioningEnabled && captioningRuntimeStatus === "missing_runtime" ? "missing_runtime" : captioningEnabled && captioningRuntimeStatus === "invalid_config" ? "blocked" : captioningEnabled ? "enabled" : "available",
      summary: captioningEnabled ? "Advisory captions enabled." : "Optional advisory captions.",
      detail: "Captions do not satisfy quality gates.",
      nextAction: captioningEnabled
        ? "Repair FastVLM configuration in Build or disable captions." : ""
    }),
    makeRow({
      id: "job_queue",
      group: "Operate",
      label: "Queue",
      status: !backendOk ? "offline" : hasJobs || hasActiveStream ? "enabled" : "available",
      summary: hasJobs ? `${jobs.length} job${jobs.length === 1 ? "" : "s"} loaded.` : "No jobs loaded.",
      detail: "Follow job progress in Operate."
    }),
    makeRow({
      id: "sse_stream",
      group: "Operate",
      label: "SSE freshness",
      status: !backendOk ? "offline" : hasActiveStream ? "enabled" : "available",
      summary: "Follow progress with live job updates.",
      detail: "Connection warnings appear in Operate and Review."
    }),
    makeRow({
      id: "artifact_review",
      group: "Review",
      label: "Artifact review",
      status: !backendOk ? "offline" : hasReviewArtifacts ? "enabled" : "available",
      summary: hasReviewArtifacts ? "Outputs loaded." : "Available after a run produces outputs.",
      detail: "Inspect and compare outputs in Review."
    }),
    makeRow({
      id: "artifact_viewer",
      group: "Review",
      label: "Artifact viewer modal",
      status: !bootstrapReady || !features.artifactViewerModal ? "gated" : hasReviewArtifacts ? "enabled" : "available",
      summary: "Inspect outputs in an expanded viewer.",
      detail: "The standard review panel remains available."
    }),
    makeRow({
      id: "review_surface",
      group: "Review",
      label: "Deferred review surface",
      status: !bootstrapReady || !features.reviewSurfaceDeferred ? "gated" : "available",
      summary: features.reviewSurfaceDeferred ? "Review loads on demand." : "Review loads with the portal."
    }),
    makeRow({
      id: "portal_rum",
      group: "Operate",
      label: "Portal RUM",
      status: !bootstrapReady || !features.rumTelemetry ? "gated" : "enabled",
      summary: "Session performance and reliability telemetry."
    }),
    makeRow({
      id: "plugin_trust",
      group: "Governance",
      label: "Plugin trust",
      status: "not_portal_controlled",
      summary: "Plugin trust is managed outside the portal.",
      detail: "Use the repository governance workflow."
    })
  ];

  const currentPipelineId = isPhotography ? "lux_depth_v5" : isLux ? "lux_depth_v3" : "archive_gates";
  for (const row of rows) {
    row.scope = "current";
    if (["lux_depth_v4", "plugin_trust"].includes(row.id)) {
      row.scope = "external";
      row.statusLabel = row.id === "lux_depth_v4" ? "CLI workflow" : "External governance";
    } else if (V3_ONLY_ROWS.has(row.id) && !isLux) {
      row.scope = "other_workflow";
      row.status = "not_portal_controlled";
      row.statusLabel = "LuxDepthV3 only";
      row.summary = "Available in LuxDepthV3.";
      row.detail = "Select Lux Depth v3 in Build.";
      row.nextAction = "";
    } else if (row.id === "materials_v4" && !isPhotography) {
      row.scope = "other_workflow";
      row.statusLabel = "LuxDepthV5 only";
    } else if (["lux_depth_v3", "lux_depth_v5", "archive_gates"].includes(row.id) && row.id !== currentPipelineId) {
      row.scope = "other_workflow";
    } else if (row.id === "da3_apache" && !isLux && !isPhotography) {
      row.scope = "other_workflow";
    } else if (row.id === "staged_uploads" && !stagedUploadSupported) {
      row.scope = "other_workflow";
      row.statusLabel = "Other pipelines";
    } else if (row.id === "direct_debug" && authMode !== "direct_debug") {
      row.scope = "other_workflow";
      row.statusLabel = "Not used in managed access";
    } else if (row.id === "managed_access" && authMode === "direct_debug") {
      row.scope = "other_workflow";
      row.statusLabel = "Managed entry";
    }
  }

  const currentPipelineRow = rows.find((row) => row.id === currentPipelineId);
  if (currentPipelineRow) {
    if (!backendOk) currentPipelineRow.nextAction = "Restore the backend connection before starting a run.";
    else if (previewBlocked && currentPipelineRow.status !== "needs_ack") {
      currentPipelineRow.status = "blocked";
      currentPipelineRow.statusLabel = statusLabel(currentPipelineRow.status);
      currentPipelineRow.nextAction = "Open Build to resolve the configuration preview or readiness error.";
    } else if (isPhotography && currentPipelineRow.status === "gated") {
      currentPipelineRow.statusLabel = previewReady ? "Readiness pending" : "Preview pending";
    }
  }

  // Rollout gates for optional controls are informational. Only the active entry
  // path, current pipeline, and selected options may interrupt the operator.
  const requiredIds = new Set([authMode === "direct_debug" ? "direct_debug" : "managed_access", currentPipelineId]);
  if (isLux) {
    if (depthProActive) requiredIds.add("depth_pro");
    if (sam2Active) requiredIds.add("sam2_segmentation");
    if (reconstructionEnabled) requiredIds.add("reconstruction");
    if (captioningEnabled) requiredIds.add("fastvlm_captioning");
  }
  const currentRows = rows.filter((row) => row.scope === "current");
  const blockedRows = currentRows.filter((row) => requiredIds.has(row.id) && ATTENTION_STATUSES.has(row.status));
  let preferredNext = blockedRows.find((row) => row.group === "Entry");
  for (const status of ["offline", "blocked", "needs_ack", "missing_runtime", "gated"]) {
    if (preferredNext) break;
    preferredNext = blockedRows.find((row) => row.status === status);
  }

  return {
    rows,
    summary: {
      total: rows.length,
      current: currentRows.length,
      enabled: currentRows.filter((row) => row.status === "enabled").length,
      available: currentRows.filter((row) => row.status === "available").length,
      outsideWorkflow: rows.length - currentRows.length,
      actionable: blockedRows.length,
      previewBlocked,
      nextActionCapabilityId: preferredNext?.id || "",
      nextActionStatus: preferredNext?.status || "enabled",
      nextActionLabel: preferredNext
        ? `${preferredNext.label}: ${preferredNext.nextAction || preferredNext.summary}`
        : "Current workflow capabilities are ready. Configure options in Build."
    }
  };
}
