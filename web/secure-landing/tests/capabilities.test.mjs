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
