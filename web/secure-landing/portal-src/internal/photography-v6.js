// The managed V6 request admits photographs and freezes both inference and
// finishing. Runtime paths, cache locations, and materials stay server-owned.
export const PHOTOGRAPHY_V6_FIELDS = Object.freeze({
  inputColor: "input_color", device: "device", precision: "precision", targetSize: "target_size",
  refinement: "refinement", strength: "strength", clarity: "clarity", companionsManifest: "companions_manifest",
  exposureStops: "exposure_stops", contrast: "contrast", pivot: "pivot", saturation: "saturation",
  render: "render", shoulder: "shoulder", depthRefinement: "depth_refinement",
  maxPixels: "max_pixels", maxInputBytes: "max_input_bytes", maxOutputBytes: "max_output_bytes",
  wallTimeSeconds: "wall_time_seconds", memoryMib: "memory_mib"
});

export function createPhotographyV6Config() {
  return {
    inputColor: "auto", device: "cpu", precision: "fp32", targetSize: 518,
    refinement: "guided_bilinear", strength: 0.25, clarity: 0, companionsManifest: "",
    exposureStops: 0, whiteBalanceR: 1, whiteBalanceG: 1, whiteBalanceB: 1,
    contrast: 1, pivot: 0.18, saturation: 1, render: "perceptual_srgb", shoulder: 0.8,
    depthRefinement: "guided_bilinear_v4", maxPixels: 100000000, maxInputBytes: 1073741824,
    maxOutputBytes: 68719476736, wallTimeSeconds: 3600, memoryMib: 16384
  };
}

export function buildPhotographyV6Args(config = {}, controls = {}) {
  const defaults = createPhotographyV6Config();
  const value = (key) => controls[key]?.value ?? config[key] ?? defaults[key];
  const number = (key) => String(value(key)).trim() === "" ? Number.NaN : Number(value(key));
  const textKeys = new Set(["inputColor", "device", "precision", "refinement", "render", "depthRefinement"]);
  const args = { model_key: "da3-metric" };
  for (const [key, field] of Object.entries(PHOTOGRAPHY_V6_FIELDS)) {
    if (key === "companionsManifest") {
      const path = String(value(key)).trim();
      if (path) args[field] = path;
    } else {
      args[field] = textKeys.has(key) ? String(value(key)) : number(key);
    }
  }
  args.white_balance = ["whiteBalanceR", "whiteBalanceG", "whiteBalanceB"].map(number);
  return args;
}

export function photographyV6ConfigFromArgs(args = {}, previous = {}) {
  const config = { ...createPhotographyV6Config(), ...previous };
  for (const [key, field] of Object.entries(PHOTOGRAPHY_V6_FIELDS)) {
    if (Object.prototype.hasOwnProperty.call(args, field)) config[key] = args[field] ?? "";
  }
  if (Object.prototype.hasOwnProperty.call(args, "white_balance")) {
    for (const [index, key] of ["whiteBalanceR", "whiteBalanceG", "whiteBalanceB"].entries()) {
      config[key] = Array.isArray(args.white_balance) && args.white_balance.length === 3 ? args.white_balance[index] ?? "" : "";
    }
  }
  return config;
}
