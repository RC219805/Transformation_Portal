const SHA256 = /^[a-f0-9]{64}$/;
const PRESET_KEYS = ["name", "format", "intended_use", "long_edge", "jpeg_quality", "sharpen_amount", "render", "shoulder"];
const STORAGE_KEY = "tp.photo.delivery_recovery.v1";
const MAX_STORAGE_CHARACTERS = 128 * 1024;

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  return value;
}

export function photographicDeliveryAttempt(body, variants, defaults) {
  return {
    body: JSON.parse(JSON.stringify(body)),
    intent: {
      property_name: body.property_name,
      sources: body.approved_sources.map((source) => ({ ...source,
        presentation_policy_sha256: variants.find((variant) => variant.artifact_path === source.artifact_path)?.presentation_policy_sha256 || null })),
      scene_labels: { ...body.scene_labels },
      presets: body.presets.map((selection) => {
        const name = typeof selection === "string" ? selection : selection.name;
        const preset = defaults.find((row) => row?.name === name);
        if (!preset || PRESET_KEYS.some((key) => !Object.hasOwn(preset, key))) return null;
        return Object.fromEntries([["schema", "tp.photo.export_preset.v1"], ...PRESET_KEYS.map((key) =>
          [key, typeof selection === "object" && Object.hasOwn(selection, key) ? selection[key] : preset[key]])]);
      }),
    },
  };
}

export function photographicDeliveryRecovered(entry, attempt) {
  return Boolean(entry && attempt && entry.request_correlation_id === attempt.body.request_correlation_id
    && entry.approval_version === attempt.body.expected_version
    && entry.request_fingerprint_schema === "tp.photo.delivery_intent.v1" && SHA256.test(entry.request_fingerprint_sha256 || "")
    && JSON.stringify(stable(entry.intent)) === JSON.stringify(stable(attempt.intent)));
}

export function photographicDeliveryRecoveryStorage(storage) {
  function read() {
    try {
      const raw = storage?.getItem(STORAGE_KEY);
      if (!raw || raw.length > MAX_STORAGE_CHARACTERS) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) && parsed.length <= 16 ? parsed.filter((entry) => entry && typeof entry.scope === "string"
        && entry.attempt?.body && Number.isInteger(entry.attempt.body.expected_version) && entry.attempt.body.expected_version >= 0
        && /^[A-Za-z0-9._:-]{1,64}$/.test(entry.attempt.body.request_correlation_id || "")) : [];
    } catch { return []; }
  }
  return {
    get(scope) { return read().find((entry) => entry.scope === scope)?.attempt || null; },
    set(scope, attempt) {
      try {
        const entries = read().filter((entry) => entry.scope !== scope);
        if (attempt) entries.push({ scope, attempt });
        const raw = JSON.stringify(entries);
        if (entries.length > 16 || raw.length > MAX_STORAGE_CHARACTERS) return false;
        storage?.setItem(STORAGE_KEY, raw);
        return Boolean(storage) && JSON.stringify(read().find((entry) => entry.scope === scope)?.attempt || null) === JSON.stringify(attempt);
      } catch { return false; }
    },
  };
}
