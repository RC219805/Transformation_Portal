import test from "node:test";
import assert from "node:assert/strict";
import { buildUnifiedPhotographyArgs, createUnifiedPhotographyConfig, isCompositePhotographyPipeline, unifiedPhotographyConfigFromArgs } from "../portal-src/internal/photography-unified.js";
import { buildPhotographyArgs, isLuxPipeline, photographyLabel } from "../portal-src/internal/photography.js";
import { buildPhotographyV6Args } from "../portal-src/internal/photography-v6.js";
import { createPortalConfigState } from "../portal-src/internal/state.js";
import { buildPortalCapabilityCatalog } from "../portal-src/internal/capabilities.js";

test("Unified process and infer use closed requests with independent workflow defaults", () => {
  assert.equal(isLuxPipeline("lux-depth"), true);
  assert.equal(photographyLabel("lux-depth"), "Lux Depth Unified");
  assert.equal(isCompositePhotographyPipeline("lux-depth", "process"), true);
  assert.equal(isCompositePhotographyPipeline("lux-depth", "infer"), false);
  assert.deepEqual(buildUnifiedPhotographyArgs(), { workflow: "process", ...buildPhotographyV6Args() });
  assert.deepEqual(buildUnifiedPhotographyArgs({ workflow: "infer" }), { workflow: "infer", ...buildPhotographyArgs() });
  for (const workflow of ["process", "infer"]) {
    const config = createUnifiedPhotographyConfig({ workflow });
    Object.assign(config[workflow], { runtime_python: "/untrusted/python", cache_dir: "/untrusted/cache", enable_segmentation: true, depth_maps: false });
    assert.deepEqual(buildUnifiedPhotographyArgs(config), { workflow, ...(workflow === "process" ? buildPhotographyV6Args() : buildPhotographyArgs()) });
  }
});

test("Unified imports preserve all selected workflow fields and isolate prior drafts", () => {
  const previous = createUnifiedPhotographyConfig();
  previous.process.exposureStops = 1.25;
  const inferArgs = { workflow: "infer", ...buildPhotographyArgs(), strength: 0, preview_maps: true, materials_manifest: "/input/materials.json", companions_manifest: "/input/camera.json" };
  const infer = unifiedPhotographyConfigFromArgs(inferArgs, previous);
  assert.deepEqual(buildUnifiedPhotographyArgs(infer), inferArgs);
  assert.equal(infer.process.exposureStops, 1.25);
  assert.equal(previous.workflow, "process");
  const processArgs = { workflow: "process", ...buildPhotographyV6Args(), white_balance: [1.1, 1, 0.9], saturation: 0, max_pixels: 20000000 };
  const process = unifiedPhotographyConfigFromArgs(processArgs, infer);
  assert.deepEqual(buildUnifiedPhotographyArgs(process), processArgs);
  assert.equal(process.infer.materialsManifest, "/input/materials.json");
  const state = createPortalConfigState();
  state.photographyUnified.process.strength = 1;
  state.photographyUnified.infer.strength = 0;
  assert.equal(state.photographyV6.strength, 0.25);
  assert.equal(state.photography.strength, 0.25);
  assert.equal(createPortalConfigState().photographyUnified.process.strength, 0.25);
});

test("Unified forwards invalid workflow values and invalid numeric controls to rejection", () => {
  for (const workflow of ["unknown", "", null, 0]) {
    assert.deepEqual(buildUnifiedPhotographyArgs(unifiedPhotographyConfigFromArgs({ workflow })), { workflow });
  }
  assert.ok(Number.isNaN(buildUnifiedPhotographyArgs({ workflow: "process", process: { exposureStops: "" } }).exposure_stops));
  assert.ok(Number.isNaN(buildUnifiedPhotographyArgs({ workflow: "infer" }, { infer: { strength: { value: "bad" } } }).strength));
});

test("Unified capability status requires the selected workflow preview and scopes materials to infer", () => {
  for (const workflow of ["process", "infer"]) {
    for (const [previewWorkflow, readinessStatus, expected] of [[workflow, "ready", "enabled"], [workflow, "blocked", "blocked"], [workflow === "infer" ? "process" : "infer", "ready", "gated"]]) {
      const catalog = buildPortalCapabilityCatalog({
        pipeline: "lux-depth", args: { workflow, materials_manifest: "/input/materials.json" },
        preview: { pipeline: "lux-depth", status: "ready", normalized_args: { workflow: previewWorkflow } },
        readiness: { status: readinessStatus }, backendOk: true, bootstrapReady: true, authMode: "managed"
      });
      const row = (id) => catalog.rows.find((item) => item.id === id);
      assert.equal(row("lux_depth_unified").status, expected);
      assert.equal(row("lux_depth_unified").scope, "current");
      assert.equal(row("materials_v4").status, workflow === "infer" ? "enabled" : "not_portal_controlled");
      for (const id of ["lux_depth_v3", "lux_depth_v5", "lux_depth_v6", "segmentation"]) assert.equal(row(id).scope, "other_workflow");
    }
  }
});
