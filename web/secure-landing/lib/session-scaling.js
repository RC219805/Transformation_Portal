export const SESSION_SCALING_MODE = Object.freeze({
  SINGLE_INSTANCE: "single_instance",
  MULTI_INSTANCE: "multi_instance",
  EPHEMERAL_RUNTIME: "ephemeral_runtime"
});

export const SESSION_STORE_BACKEND = Object.freeze({
  SQLITE: "sqlite",
  REDIS: "redis"
});

function normalizeSessionScalingMode(value) {
  return String(value || "").trim().toLowerCase().replace(/-/g, "_");
}

function normalizeSessionStoreBackend(value) {
  return String(value || SESSION_STORE_BACKEND.SQLITE).trim().toLowerCase();
}

export function evaluateSessionScaling(config) {
  const requestedMode = normalizeSessionScalingMode(config.sessionScalingMode);
  const requestedBackend = normalizeSessionStoreBackend(config.sessionStoreBackend);
  const externalStoreConfigured = requestedBackend === SESSION_STORE_BACKEND.REDIS;
  const base = {
    backend: externalStoreConfigured ? SESSION_STORE_BACKEND.REDIS : SESSION_STORE_BACKEND.SQLITE,
    required: true
  };

  if (!requestedMode) {
    return {
      ok: true,
      mode: SESSION_SCALING_MODE.SINGLE_INSTANCE,
      reason: null,
      ...base
    };
  }

  if (requestedMode === SESSION_SCALING_MODE.SINGLE_INSTANCE) {
    return {
      ok: true,
      mode: requestedMode,
      reason: null,
      ...base
    };
  }

  if (requestedMode === SESSION_SCALING_MODE.MULTI_INSTANCE) {
    if (externalStoreConfigured) {
      return {
        ok: true,
        mode: requestedMode,
        reason: null,
        ...base
      };
    }
    return {
      ok: false,
      mode: requestedMode,
      reason: "multi_instance_requires_external_session_store",
      ...base
    };
  }

  if (requestedMode === SESSION_SCALING_MODE.EPHEMERAL_RUNTIME) {
    if (externalStoreConfigured) {
      return {
        ok: true,
        mode: requestedMode,
        reason: null,
        ...base
      };
    }
    return {
      ok: false,
      mode: requestedMode,
      reason: "ephemeral_runtime_requires_external_session_store",
      ...base
    };
  }

  return {
    ok: false,
    mode: requestedMode,
    reason: "invalid_session_scaling_mode",
    ...base
  };
}
