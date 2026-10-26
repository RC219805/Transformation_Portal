// Browser-local processing intent maps onto existing canonical pipeline flags.
// Delivery export presets remain a separate, exact-hash approved Review action.
export const PROCESSING_PROFILES = Object.freeze({
  photographic_master: Object.freeze({ label: "Photographic master", outputBitDepth: 16 }),
  web_delivery: Object.freeze({ label: "Web delivery", outputBitDepth: 8 }),
  retouch_print: Object.freeze({ label: "Retouch / print", outputBitDepth: 16 })
});

export function resolveProcessingProfile(config = {}) {
  // Explicit historical 8-bit choices are authoritative. Never upgrade them silently.
  if (Number(config.outputBitDepth ?? config.output_bit_depth) === 8) return "web_delivery";
  return Object.hasOwn(PROCESSING_PROFILES, config.processingProfile) && config.processingProfile !== "web_delivery" ? config.processingProfile : "photographic_master";
}

export function applyProcessingProfile(config, profile) {
  if (!config || !Object.hasOwn(PROCESSING_PROFILES, profile)) throw new TypeError("Choose a known photographic processing profile");
  config.processingProfile = profile;
  config.outputBitDepth = PROCESSING_PROFILES[profile].outputBitDepth;
  delete config.emit_master16;
  return config;
}

export function processingProfileGuidance(profile, pipeline) {
  const name = PROCESSING_PROFILES[profile]?.label || "Photographic master";
  if (pipeline !== "lux-depth-v3") {
    return `${name} intent. Unified/V5/V6 retain their verified full-precision masters; choose named JPEG, print, or float TIFF exports after approval in Review. Delivery TIFF is 16-bit sRGB; retained master arrays remain float32 extended linear sRGB.`;
  }
  if (profile === "web_delivery") return "V3 requests an explicit 8-bit encoded-sRGB PNG for enabled photographic finishing stages. Keep a separate full-precision source master; an 8-bit output cannot recover discarded precision. Website/MLS/social JPEG delivery follows approval in Review.";
  if (profile === "retouch_print") return "V3 requests a 16-bit encoded-sRGB TIFF for enabled photographic finishing stages, limited to bounded SDR and 16-bit sources. Use Unified/V5/V6 and the float32 retouch export for extended/HDR or float masters. Printer-specific profiles remain an explicit external conversion.";
  return "V3 requests a 16-bit encoded-sRGB TIFF for enabled photographic finishing stages and leaves sources unchanged. Promoting an 8-bit source does not restore precision. For extended/HDR or float masters, choose Unified/V5/V6; validate source color before finishing.";
}
