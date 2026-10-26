const SHA256 = /^[a-f0-9]{64}$/;
const PRESET_KEYS = ["name", "format", "intended_use", "long_edge", "jpeg_quality", "sharpen_amount", "render", "shoulder"];
const STORAGE_KEY = "tp.photo.delivery_recovery.v1";
const MAX_STORAGE_CHARACTERS = 128 * 1024;
const object = (value) => Boolean(value && typeof value === "object" && !Array.isArray(value));
const source = (value) => object(value) && typeof value.artifact_path === "string" && value.artifact_path.length > 0
  && SHA256.test(value.sha256 || "") && SHA256.test(value.source_identity_sha256 || "");

function validScope(value) {
  if (typeof value !== "string" || value.length > 8192) return false;
  try {
    const scope = JSON.parse(value);
    const actor = Array.isArray(scope) && typeof scope[1] === "string" ? JSON.parse(scope[1]) : null;
    return scope.length === 2 && typeof scope[0] === "string" && Boolean(scope[0].trim())
      && Array.isArray(actor) && actor.length === 3 && typeof actor[0] === "string" && Boolean(actor[0])
      && JSON.stringify(scope) === value && JSON.stringify(actor) === scope[1];
  } catch { return false; }
}

function validAttempt(value) {
  const body = value?.body, intent = value?.intent;
  return object(value) && object(body) && Number.isInteger(body.expected_version) && body.expected_version >= 0
    && typeof body.request_correlation_id === "string" && /^[A-Za-z0-9._:-]{1,64}$/.test(body.request_correlation_id) && typeof body.property_name === "string"
    && Array.isArray(body.approved_sources) && body.approved_sources.length > 0 && body.approved_sources.every(source)
    && Array.isArray(body.presets) && body.presets.length > 0 && object(body.scene_labels)
    && object(intent) && typeof intent.property_name === "string" && object(intent.scene_labels)
    && Array.isArray(intent.sources) && intent.sources.length === body.approved_sources.length && intent.sources.every((row) =>
      source(row) && (row.presentation_policy_sha256 === null || SHA256.test(row.presentation_policy_sha256 || "")))
    && Array.isArray(intent.presets) && intent.presets.length === body.presets.length
    && intent.presets.every((row) => object(row) && row.schema === "tp.photo.export_preset.v1");
}

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
      if (typeof storage?.getItem !== "function") return { status: "unavailable", entries: [] };
      const raw = storage.getItem(STORAGE_KEY);
      if (raw === null || raw === undefined) return { status: "available", entries: [] };
      if (typeof raw !== "string" || !raw || raw.length > MAX_STORAGE_CHARACTERS) return { status: "invalid", entries: [] };
      let parsed;
      try { parsed = JSON.parse(raw); } catch { return { status: "invalid", entries: [] }; }
      if (!Array.isArray(parsed) || parsed.length > 16 || !parsed.every((entry) => object(entry) && validScope(entry.scope)
        && validAttempt(entry.attempt)) || new Set(parsed.map((entry) => entry.scope)).size !== parsed.length) return { status: "invalid", entries: [] };
      return { status: "available", entries: parsed };
    } catch { return { status: "unavailable", entries: [] }; }
  }
  return {
    inspect(scope) { const snapshot = read(); return { status: snapshot.status, attempt: snapshot.entries.find((entry) => entry.scope === scope)?.attempt || null }; },
    get(scope) { return this.inspect(scope).attempt; },
    set(scope, attempt, expectedAttempt = undefined) {
      try {
        const snapshot = read();
        if (snapshot.status !== "available" || !validScope(scope) || (attempt !== null && !validAttempt(attempt))) return false;
        const current = snapshot.entries.find((entry) => entry.scope === scope)?.attempt || null;
        if ((expectedAttempt !== undefined && JSON.stringify(current) !== JSON.stringify(expectedAttempt))
          || (attempt && current && JSON.stringify(current) !== JSON.stringify(attempt))) return false;
        const entries = snapshot.entries.filter((entry) => entry.scope !== scope);
        if (attempt) entries.push({ scope, attempt });
        const raw = JSON.stringify(entries);
        if (entries.length > 16 || raw.length > MAX_STORAGE_CHARACTERS) return false;
        storage?.setItem(STORAGE_KEY, raw);
        const observed = read();
        return observed.status === "available" && JSON.stringify(observed.entries.find((entry) => entry.scope === scope)?.attempt || null) === JSON.stringify(attempt);
      } catch { return false; }
    },
  };
}
