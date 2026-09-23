// Managed photography uses its own closed request contract. V3 settings must
// never leak into this payload; runtime and cache bindings remain server-owned.
export function isLuxPipeline(pipeline) {
  return pipeline === "lux-depth-v3" || isPhotographyPipeline(pipeline);
}

export function isPhotographyPipeline(pipeline) {
  return pipeline === "lux-depth-v5" || pipeline === "lux-depth-v6";
}

export function photographyVersion(pipeline) {
  return pipeline === "lux-depth-v6" ? "V6" : "V5";
}

export function createPhotographyConfig() {
  return {
    inputColor: "auto", device: "cpu", precision: "fp32", targetSize: 518,
    refinement: "guided_bilinear", strength: 0.25, clarity: 0,
    previewMaps: false, materialsManifest: "", companionsManifest: ""
  };
}

export function buildPhotographyArgs(config = {}, controls = {}) {
  const defaults = createPhotographyConfig();
  const value = (key) => controls[key]?.value ?? config[key] ?? defaults[key];
  const number = (key) => String(value(key)).trim() === "" ? Number.NaN : Number(value(key));
  const args = {
    model_key: "da3-metric",
    input_color: String(value("inputColor")),
    device: String(value("device")),
    precision: String(value("precision")),
    target_size: number("targetSize"),
    refinement: String(value("refinement")),
    strength: number("strength"),
    clarity: number("clarity"),
    preview_maps: Boolean(controls.previewMaps?.checked ?? config.previewMaps ?? defaults.previewMaps)
  };
  for (const [key, field] of [["materialsManifest", "materials_manifest"], ["companionsManifest", "companions_manifest"]]) {
    const path = String(value(key)).trim();
    if (path) args[field] = path;
  }
  return args;
}

export function photographyConfigFromArgs(args = {}, previous = {}) {
  const config = { ...createPhotographyConfig(), ...previous };
  const fields = {
    inputColor: "input_color", device: "device", precision: "precision", targetSize: "target_size",
    refinement: "refinement", strength: "strength", clarity: "clarity", previewMaps: "preview_maps",
    materialsManifest: "materials_manifest", companionsManifest: "companions_manifest"
  };
  for (const [key, field] of Object.entries(fields)) {
    if (Object.prototype.hasOwnProperty.call(args, field)) config[key] = args[field] ?? "";
  }
  return config;
}
