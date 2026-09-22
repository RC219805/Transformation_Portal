import test from "node:test";
import assert from "node:assert/strict";

import {
  buildPortalCapabilityCatalog,
  normalizeCapabilityStatus,
  PORTAL_CAPABILITY_STATUSES
} from "../portal-src/internal/capabilities.js";
import { defaultPortalBootstrapPayload } from "../portal-src/internal/bootstrap-auth.js";

function rowById(catalog, id) {
  const row = catalog.rows.find((item) => item.id === id);
  assert.ok(row, `expected capability row ${id}`);
  return row;
}

test("portal capability catalog emits only the fixed internal status vocabulary", () => {
  const catalog = buildPortalCapabilityCatalog({
    pipeline: "lux-depth-v3",
    backendOk: false,
    bootstrapReady: false,
    authMode: "managed_unavailable",
    features: defaultPortalBootstrapPayload().features,
    args: {
      enable_segmentation: true,
      segmentation_backend: "sam2",
      emit_run_card: true
    },
    readiness: { status: "blocked" },
    readinessIssues: [{ severity: "blocked", reason: "archive_index_required" }]
  });

  assert.equal(normalizeCapabilityStatus("missing_runtime"), "missing_runtime");
  assert.equal(normalizeCapabilityStatus("unexpected", "blocked"), "blocked");
  const allowedStatuses = new Set(PORTAL_CAPABILITY_STATUSES);
  assert.deepEqual(
    catalog.rows.map((row) => row.status).filter((status) => !allowedStatuses.has(status)),
    []
  );
  assert.equal(rowById(catalog, "lux_depth_v3").status, "offline");
  assert.equal(rowById(catalog, "sam2_segmentation").status, "offline");
  assert.equal(catalog.summary.previewBlocked, true);
});

test("portal capability catalog derives rollout-gated controls from bootstrap flags", () => {
  const fallbackFeatures = defaultPortalBootstrapPayload().features;
  assert.equal(fallbackFeatures.stagedUploads, false);
  assert.equal(fallbackFeatures.fastVlmCaptioning, false);

  const gatedCatalog = buildPortalCapabilityCatalog({
    pipeline: "lux-depth-v3",
    backendOk: true,
    bootstrapReady: true,
    authMode: "managed",
    features: fallbackFeatures,
    stagedUploadSupported: true,
    args: {
      vlm_captioning_enabled: false
    }
  });

  assert.equal(rowById(gatedCatalog, "staged_uploads").status, "gated");
  assert.equal(rowById(gatedCatalog, "fastvlm_captioning").status, "gated");
  assert.equal(rowById(gatedCatalog, "artifact_viewer").status, "gated");
  assert.equal(rowById(gatedCatalog, "review_surface").status, "gated");

  const enabledCatalog = buildPortalCapabilityCatalog({
    pipeline: "lux-depth-v3",
    backendOk: true,
    bootstrapReady: true,
    authMode: "managed",
    features: {
      ...fallbackFeatures,
      artifactViewerModal: true,
      reviewSurfaceDeferred: true,
      stagedUploads: true,
      fastVlmCaptioning: true
    },
    stagedUploadSupported: true,
    args: {
      vlm_captioning_enabled: true
    },
    captioningRuntimeReadiness: {
      status: "missing_runtime"
    }
  });

  assert.equal(rowById(enabledCatalog, "staged_uploads").status, "available");
  assert.equal(rowById(enabledCatalog, "fastvlm_captioning").status, "missing_runtime");
  assert.equal(rowById(enabledCatalog, "artifact_viewer").status, "available");
  assert.equal(rowById(enabledCatalog, "review_surface").status, "available");
});

test("portal capability catalog marks acknowledgments and non-portal-controlled surfaces", () => {
  const catalog = buildPortalCapabilityCatalog({
    pipeline: "lux-depth-v3",
    backendOk: true,
    bootstrapReady: true,
    authMode: "managed",
    features: {
      ...defaultPortalBootstrapPayload().features,
      fastVlmCaptioning: true
    },
    args: {
      depth_backend: "depth_pro",
      enable_reconstruction: true,
      accept_apple_depth_pro_research_license: false,
      accept_research_tools_license: false
    }
  });

  assert.equal(rowById(catalog, "depth_pro").status, "needs_ack");
  assert.equal(rowById(catalog, "reconstruction").status, "needs_ack");
  assert.equal(rowById(catalog, "plugin_trust").status, "not_portal_controlled");
});

function managedInput(overrides = {}) {
  return {
    pipeline: "lux-depth-v5",
    backendOk: true,
    bootstrapReady: true,
    authMode: "managed",
    features: defaultPortalBootstrapPayload().features,
    readiness: { status: "ready" },
    ...overrides
  };
}

test("managed V5 without a successful preview directs the operator to Build, not login", () => {
  for (const preview of [null, { status: "loading" }, { status: "local_fallback" }]) {
    const catalog = buildPortalCapabilityCatalog(managedInput({ preview }));
    assert.equal(rowById(catalog, "managed_access").status, "enabled");
    assert.equal(rowById(catalog, "direct_debug").scope, "other_workflow");
    assert.equal(rowById(catalog, "direct_debug").nextAction, "");
    assert.equal(rowById(catalog, "lux_depth_v5").status, "gated");
    assert.equal(rowById(catalog, "lux_depth_v5").statusLabel, "Preview pending");
    assert.equal(catalog.summary.nextActionCapabilityId, "lux_depth_v5");
    assert.equal(catalog.summary.actionable, 1);
    assert.match(catalog.summary.nextActionLabel, /Open Build to validate the current draft/);
    assert.doesNotMatch(catalog.summary.nextActionLabel, /login|Direct debug/i);
  }
});

test("V5 requires both a successful current preview and ready server prerequisites", () => {
  for (const [preview, readiness, status] of [
    [{ status: "ready", pipeline: "lux-depth-v5", field_errors: [] }, { status: "ready" }, "enabled"],
    [{ status: "ready", pipeline: "lux-depth-v3", field_errors: [] }, { status: "ready" }, "gated"],
    [{ status: "ready", field_errors: [{ field: "input_dir", message: "Input is unavailable." }] }, { status: "ready" }, "blocked"],
    [{ status: "ready", field_errors: [] }, { status: "blocked" }, "blocked"],
    [{ status: "ready", field_errors: [] }, { status: "loading" }, "gated"]
  ]) {
    const catalog = buildPortalCapabilityCatalog(managedInput({ preview, readiness }));
    assert.equal(rowById(catalog, "lux_depth_v5").status, status);
  }
});

test("preview request errors block V5 even when no field errors were returned", () => {
  const catalog = buildPortalCapabilityCatalog(managedInput({
    preview: { status: "error", error: "auth_configuration_error", field_errors: [] }
  }));
  assert.equal(catalog.summary.previewBlocked, true);
  assert.equal(rowById(catalog, "lux_depth_v5").status, "blocked");
  assert.equal(catalog.summary.nextActionCapabilityId, "lux_depth_v5");
  assert.match(catalog.summary.nextActionLabel, /resolve the configuration preview or readiness error/);
});

test("capability counts describe the current workflow separately from other workflows and CLI tools", () => {
  const catalog = buildPortalCapabilityCatalog(managedInput({
    preview: { status: "ready", field_errors: [] }
  }));
  assert.equal(catalog.summary.total, 26);
  assert.equal(catalog.summary.current, 10);
  assert.equal(catalog.summary.enabled, 3);
  assert.equal(catalog.summary.available, 4);
  assert.equal(catalog.summary.outsideWorkflow, 16);
  assert.equal(catalog.summary.current + catalog.summary.outsideWorkflow, catalog.summary.total);
  assert.equal(catalog.summary.actionable, 0);
  assert.equal(catalog.summary.nextActionCapabilityId, "");
  assert.match(catalog.summary.nextActionLabel, /Current workflow capabilities are ready/);
  for (const id of ["depth_pro", "materials_v3", "pbr_generation", "segmentation", "sam2_segmentation", "reconstruction", "raw_ingest", "runtime_tuning", "run_card", "fastvlm_captioning"]) {
    const row = rowById(catalog, id);
    assert.equal(row.scope, "other_workflow", id);
    assert.equal(row.statusLabel, "LuxDepthV3 only", id);
    assert.equal(row.status, "not_portal_controlled", id);
  }
  assert.equal(rowById(catalog, "lux_depth_v4").scope, "external");
  assert.equal(rowById(catalog, "lux_depth_v4").statusLabel, "CLI workflow");
  assert.equal(rowById(catalog, "plugin_trust").statusLabel, "External governance");
});

test("optional rollout gates do not masquerade as required next actions", () => {
  const catalog = buildPortalCapabilityCatalog(managedInput({ pipeline: "lux-depth-v3", stagedUploadSupported: true }));
  for (const id of ["staged_uploads", "fastvlm_captioning", "artifact_viewer", "review_surface", "portal_rum"]) {
    assert.equal(rowById(catalog, id).status, "gated", id);
  }
  assert.equal(catalog.summary.actionable, 0);
  assert.equal(catalog.summary.nextActionStatus, "enabled");

  const selectedCaptioning = buildPortalCapabilityCatalog(managedInput({
    pipeline: "lux-depth-v3", args: { vlm_captioning_enabled: true }
  }));
  assert.equal(selectedCaptioning.summary.actionable, 1);
  assert.equal(selectedCaptioning.summary.nextActionCapabilityId, "fastvlm_captioning");
});

test("selected V3 options still require their acknowledgments and runtimes", () => {
  for (const [args, expected] of [
    [{ depth_backend: "depth_pro" }, "depth_pro"],
    [{ enable_reconstruction: true }, "reconstruction"],
    [{ enable_segmentation: true, segmentation_backend: "sam2" }, "sam2_segmentation"]
  ]) {
    const catalog = buildPortalCapabilityCatalog(managedInput({ pipeline: "lux-depth-v3", args }));
    assert.equal(catalog.summary.nextActionCapabilityId, expected);
    assert.equal(catalog.summary.actionable, 1);
  }
});

test("archive workflows do not surface stale Lux configuration as blockers", () => {
  const catalog = buildPortalCapabilityCatalog(managedInput({
    pipeline: "archive-gate-a",
    args: { depth_backend: "depth_pro", enable_reconstruction: true, enable_segmentation: true, segmentation_backend: "sam2" }
  }));
  assert.equal(rowById(catalog, "archive_gates").scope, "current");
  assert.equal(rowById(catalog, "da3_apache").scope, "other_workflow");
  assert.equal(rowById(catalog, "materials_v4").scope, "other_workflow");
  assert.equal(catalog.summary.actionable, 0);

  const missingIndex = buildPortalCapabilityCatalog(managedInput({
    pipeline: "archive-gate-a",
    readiness: { status: "blocked" },
    readinessIssues: [{ severity: "blocked", reason: "archive_index_required" }]
  }));
  assert.equal(missingIndex.summary.nextActionCapabilityId, "archive_gates");
  assert.match(missingIndex.summary.nextActionLabel, /required archive index or manifest/);
});

test("entry recovery and backend connectivity take precedence over draft validation", () => {
  const recovery = buildPortalCapabilityCatalog(managedInput({ bootstrapReady: false }));
  assert.equal(recovery.summary.nextActionCapabilityId, "managed_access");
  assert.match(recovery.summary.nextActionLabel, /Recover managed access/);

  const offline = buildPortalCapabilityCatalog(managedInput({ backendOk: false }));
  assert.equal(offline.summary.nextActionCapabilityId, "lux_depth_v5");
  assert.equal(offline.summary.nextActionStatus, "offline");
  assert.match(offline.summary.nextActionLabel, /Restore the backend connection/);

  const direct = buildPortalCapabilityCatalog(managedInput({
    pipeline: "lux-depth-v3", authMode: "direct_debug", features: { directDebug: true }
  }));
  assert.equal(rowById(direct, "managed_access").scope, "other_workflow");
  assert.equal(rowById(direct, "direct_debug").scope, "current");
  assert.equal(direct.summary.actionable, 0);
});
