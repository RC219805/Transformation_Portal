import test from "node:test";
import assert from "node:assert/strict";
import { buildPhotographyArgs, createPhotographyConfig, isLuxPipeline, photographyConfigFromArgs } from "../portal-src/internal/photography.js";
import { createPortalConfigState } from "../portal-src/internal/state.js";
import { buildPortalCapabilityCatalog } from "../portal-src/internal/capabilities.js";

// This is the browser-owned subset of PhotographyJobArgs. Physical runtime,
// cache, V3, archive, and request-only resource policies cannot leak through.
const EXPECTED_FIELDS = ["clarity", "device", "input_color", "model_key", "precision", "preview_maps", "refinement", "strength", "target_size"];

test("photography payload preserves the 518/FP32 baseline and isolates the V5 contract", () => {
  const args = buildPhotographyArgs({ ...createPortalConfigState(), runtime_python: "/untrusted/python", cache_dir: "/untrusted/cache", archive_command: "fixity-scan", materials_policy: { arbitrary: true } });
  assert.deepEqual(Object.keys(args).sort(), EXPECTED_FIELDS);
  assert.deepEqual(args, { model_key: "da3-metric", input_color: "auto", device: "cpu", precision: "fp32", target_size: 518, refinement: "guided_bilinear", strength: 0.25, clarity: 0, preview_maps: false });
  assert.equal(isLuxPipeline("lux-depth-v5"), true);
  assert.equal(isLuxPipeline("lux-depth-v4"), false);
  assert.equal(isLuxPipeline("archive-gate-a"), false);
});

test("photography controls override saved values using strict numeric and boolean JSON types", () => {
  const args = buildPhotographyArgs(createPhotographyConfig(), {
    targetSize: { value: "1008" }, strength: { value: "0.35" }, clarity: { value: "0.1" },
    inputColor: { value: "srgb" }, device: { value: "mps" }, precision: { value: "fp16" },
    previewMaps: { checked: true }, materialsManifest: { value: " /inputs/materials.json " },
    companionsManifest: { value: " /inputs/companions.json " }
  });
  assert.equal(args.target_size, 1008);
  assert.equal(args.strength, 0.35);
  assert.equal(args.clarity, 0.1);
  assert.equal(args.preview_maps, true);
  assert.equal(args.materials_manifest, "/inputs/materials.json");
  assert.equal(args.companions_manifest, "/inputs/companions.json");
  assert.equal(args.precision, "fp16");
});

test("photography validation preserves invalid input for server rejection instead of silently changing it", () => {
  assert.equal(buildPhotographyArgs({ strength: 9 }).strength, 9);
  assert.ok(Number.isNaN(buildPhotographyArgs({ strength: " " }).strength));
  assert.ok(Number.isNaN(buildPhotographyArgs({ clarity: "" }).clarity));
  assert.ok(Number.isNaN(buildPhotographyArgs({ targetSize: "invalid" }).target_size));
  assert.equal(buildPhotographyArgs({ materialsManifest: "   " }).materials_manifest, undefined);
});

test("V5 capability rows expose readiness and do not advertise V3-only output controls", () => {
  for (const status of ["ready", "blocked"]) {
    const catalog = buildPortalCapabilityCatalog({ pipeline: "lux-depth-v5", backendOk: true, bootstrapReady: true, readiness: { status }, args: { materials_manifest: "/inputs/materials.json" } });
    const row = (id) => catalog.rows.find((item) => item.id === id);
    assert.equal(row("lux_depth_v5").status, status === "ready" ? "enabled" : "blocked");
    assert.equal(row("lux_depth_v4").status, "not_portal_controlled");
    assert.equal(row("materials_v4").status, "enabled");
    for (const id of ["materials_v3", "pbr_generation", "segmentation", "reconstruction", "runtime_tuning", "run_card"]) {
      assert.equal(row(id).status, "not_portal_controlled", id);
    }
  }
});


test("V5 exported config import preserves zero/false values and optional manifest paths", () => {
  const args = { target_size: 1008, strength: 0, clarity: 0, preview_maps: false, materials_manifest: "/inputs/materials.json", cache_dir: "/not-imported" };
  const config = photographyConfigFromArgs(args, { strength: 1, clarity: 1, previewMaps: true });
  const rebuilt = buildPhotographyArgs(config);
  assert.equal(rebuilt.target_size, 1008);
  assert.equal(rebuilt.strength, 0);
  assert.equal(rebuilt.clarity, 0);
  assert.equal(rebuilt.preview_maps, false);
  assert.equal(rebuilt.materials_manifest, "/inputs/materials.json");
  assert.equal(config.cache_dir, undefined);
});
