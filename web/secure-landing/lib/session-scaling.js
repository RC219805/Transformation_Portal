export const SESSION_SCALING_MODE = Object.freeze({
  SINGLE_INSTANCE: "single_instance",
  MULTI_INSTANCE: "multi_instance",
  EPHEMERAL_RUNTIME: "ephemeral_runtime"
});

function normalizeSessionScalingMode(value) {
  return String(value || "").trim().toLowerCase().replace(/-/g, "_");
}

export function evaluateSessionScaling(config) {
  const requestedMode = normalizeSessionScalingMode(config.sessionScalingMode);
  const base = {
    backend: "sqlite",
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
    return {
      ok: false,
      mode: requestedMode,
      reason: "multi_instance_requires_external_session_store",
      ...base
    };
  }

  if (requestedMode === SESSION_SCALING_MODE.EPHEMERAL_RUNTIME) {
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
