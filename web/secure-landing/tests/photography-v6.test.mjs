import test from "node:test";
import assert from "node:assert/strict";
import { buildPhotographyV6Args, createPhotographyV6Config, photographyV6ConfigFromArgs } from "../portal-src/internal/photography-v6.js";
import { buildPhotographyArgs, isLuxPipeline } from "../portal-src/internal/photography.js";
import { createPortalConfigState } from "../portal-src/internal/state.js";
import { buildPortalCapabilityCatalog } from "../portal-src/internal/capabilities.js";

const DEFAULT_ARGS = {
  model_key: "da3-metric", input_color: "auto", device: "cpu", precision: "fp32", target_size: 518,
  refinement: "guided_bilinear", strength: 0.25, clarity: 0, exposure_stops: 0,
  white_balance: [1, 1, 1], contrast: 1, pivot: 0.18, saturation: 1,
  render: "perceptual_srgb", shoulder: 0.8, depth_refinement: "guided_bilinear_v4",
  max_pixels: 100000000, max_input_bytes: 1073741824, max_output_bytes: 68719476736,
  wall_time_seconds: 3600, memory_mib: 16384
};

test("V6 uses a closed typed request with neutral grade and mandatory server-owned outputs", () => {
  const args = buildPhotographyV6Args({
    ...createPhotographyV6Config(), materialsManifest: "/inputs/materials.json", materials_policy: {},
    runtime_python: "/untrusted/python", cache_dir: "/untrusted/cache", preview_maps: false,
    depth_maps: false, archive_command: "fixity-scan", enable_segmentation: true,
  });
  assert.deepEqual(args, DEFAULT_ARGS);
  assert.equal(isLuxPipeline("lux-depth-v6"), true);
});

test("V6 round trips every grade, renderer, depth, input and resource control without losing zeros", () => {
  const requested = {
    ...DEFAULT_ARGS, input_color: "srgb", device: "mps", precision: "fp16", target_size: 1008,
    refinement: "bilinear", strength: 0, clarity: 0.2, exposure_stops: -1.25,
    white_balance: [1.1, 0.9, 1.2], contrast: 1.2, pivot: 0.2, saturation: 0,
    render: "soft_srgb", shoulder: 0.75, depth_refinement: "bilinear", companions_manifest: "/inputs/camera.json",
    max_pixels: 20000000, max_input_bytes: 209715200, max_output_bytes: 4294967296,
    wall_time_seconds: 7200, memory_mib: 8192
  };
  const config = photographyV6ConfigFromArgs({ ...requested, cache_dir: "/not-imported" });
  assert.equal(config.cache_dir, undefined);
  assert.deepEqual(buildPhotographyV6Args(config), requested);
  const controls = Object.fromEntries(Object.entries(config).map(([key, value]) => [key, { value: String(value) }]));
  assert.deepEqual(buildPhotographyV6Args(createPhotographyV6Config(), controls), requested);
});

test("V6 preserves invalid numeric controls for rejection and omits only an empty companion path", () => {
  const args = buildPhotographyV6Args({ exposureStops: "", targetSize: "bad", whiteBalanceR: " ", companionsManifest: " " });
  assert.ok(Number.isNaN(args.exposure_stops));
  assert.ok(Number.isNaN(args.target_size));
  assert.ok(Number.isNaN(args.white_balance[0]));
  assert.equal(args.companions_manifest, undefined);
  assert.equal(buildPhotographyV6Args({ saturation: -1 }).saturation, -1);
  const imported = buildPhotographyV6Args(photographyV6ConfigFromArgs({ white_balance: [1] }));
  assert.ok(Number.isNaN(imported.white_balance[1]));
  assert.ok(Number.isNaN(imported.white_balance[2]));
  assert.ok(buildPhotographyV6Args(photographyV6ConfigFromArgs({ white_balance: [1, 1, 1, 1] })).white_balance.every(Number.isNaN));
});

test("V6 draft mutations remain isolated from V5 and newly created defaults", () => {
  const state = createPortalConfigState();
  const originalV5 = buildPhotographyArgs(state.photography);
  state.photographyV6.targetSize = 1008;
  state.photographyV6.exposureStops = 2;
  assert.deepEqual(buildPhotographyArgs(state.photography), originalV5);
  assert.deepEqual(buildPhotographyV6Args(createPortalConfigState().photographyV6), DEFAULT_ARGS);
});

test("V6 capability status requires the matching preview and readiness, and excludes V3 and MaterialsV4", () => {
  const base = { pipeline: "lux-depth-v6", backendOk: true, bootstrapReady: true, authMode: "managed", stagedUploadSupported: true, features: { stagedUploads: true } };
  for (const [preview, readiness, status] of [
    [null, { status: "ready" }, "gated"],
    [{ pipeline: "lux-depth-v5", status: "ready" }, { status: "ready" }, "gated"],
    [{ pipeline: "lux-depth-v6", status: "ready" }, { status: "ready" }, "enabled"],
    [{ pipeline: "lux-depth-v6", status: "ready" }, { status: "blocked" }, "blocked"],
    [{ pipeline: "lux-depth-v6", status: "error" }, { status: "ready" }, "blocked"],
  ]) {
    const catalog = buildPortalCapabilityCatalog({ ...base, preview, readiness });
    const row = (id) => catalog.rows.find((item) => item.id === id);
    assert.equal(row("lux_depth_v6").status, status);
    assert.equal(row("lux_depth_v6").scope, "current");
    assert.equal(row("lux_depth_v5").scope, "other_workflow");
    assert.equal(row("staged_uploads").status, "available");
    for (const id of ["materials_v4", "materials_v3", "pbr_generation", "segmentation", "reconstruction", "runtime_tuning", "run_card"]) {
      assert.equal(row(id).status, "not_portal_controlled", id);
      assert.equal(row(id).scope, "other_workflow", id);
    }
    assert.equal(catalog.summary.actionable, status === "enabled" ? 0 : 1);
  }
});
