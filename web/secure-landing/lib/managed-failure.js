import { audit } from "./audit.js";

export const MANAGED_FAILURE_REASON = Object.freeze({
  AUTH_FAILURE: "auth_failure",
  ACCESS_OUTAGE: "access_outage",
  CONFIG_FAILURE: "config_failure",
  UPSTREAM_UNAVAILABLE: "upstream_unavailable"
});

const ACCESS_OUTAGE_ERROR_CODES = new Set(["jwks_invalid", "jwks_unreachable"]);
const CONFIG_ERROR_CODES = new Set(["configuration", "missing_backend_api_key"]);

const DEFAULT_FAILURE_MESSAGES = Object.freeze({
  [MANAGED_FAILURE_REASON.AUTH_FAILURE]: "Managed authentication is required before this action can continue.",
  [MANAGED_FAILURE_REASON.ACCESS_OUTAGE]: "Managed access verification is temporarily unavailable. Retry once Access validation recovers.",
  [MANAGED_FAILURE_REASON.CONFIG_FAILURE]:
    "Managed front door configuration is incomplete. Resolve the front door configuration before retrying.",
  [MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE]:
    "The upstream portal service is temporarily unavailable. Retry once the backend recovers."
});

const SURFACE_FAILURE_MESSAGES = Object.freeze({
  portal: Object.freeze({
    [MANAGED_FAILURE_REASON.ACCESS_OUTAGE]: "Managed access verification unavailable",
    [MANAGED_FAILURE_REASON.CONFIG_FAILURE]: "Managed front door configuration unavailable",
    [MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE]: "Portal upstream unavailable"
  }),
  portal_bootstrap: Object.freeze({
    [MANAGED_FAILURE_REASON.AUTH_FAILURE]:
      "Managed authentication is required before portal bootstrap can continue.",
    [MANAGED_FAILURE_REASON.ACCESS_OUTAGE]:
      "Managed access verification is temporarily unavailable. Portal bootstrap will recover once Access validation returns.",
    [MANAGED_FAILURE_REASON.CONFIG_FAILURE]:
      "Managed front door configuration is incomplete. Portal bootstrap remains blocked until configuration is fixed."
  }),
  portal_asset: Object.freeze({
    [MANAGED_FAILURE_REASON.CONFIG_FAILURE]: "Portal asset proxy configuration unavailable",
    [MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE]: "Portal asset upstream unavailable"
  }),
  portal_video: Object.freeze({
    [MANAGED_FAILURE_REASON.CONFIG_FAILURE]: "Portal video proxy configuration unavailable",
    [MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE]: "Portal video upstream unavailable"
  }),
  v1_proxy: Object.freeze({
    [MANAGED_FAILURE_REASON.ACCESS_OUTAGE]: "managed access unavailable",
    [MANAGED_FAILURE_REASON.CONFIG_FAILURE]: "managed proxy misconfigured",
    [MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE]: "upstream request failed"
  })
});

function resolveActorFields(actor) {
  if (!actor || typeof actor !== "object") {
    return {};
  }
  return {
    ...(actor.username ? { username: actor.username } : {}),
    ...(actor.accessEmail ? { accessEmail: actor.accessEmail } : {})
  };
}

export function isRetryableManagedFailure(reason) {
  return (
    reason === MANAGED_FAILURE_REASON.ACCESS_OUTAGE ||
    reason === MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE
  );
}

export function classifyManagedAccessFailure(errorCode) {
  const normalized = String(errorCode || "").trim().toLowerCase();
  if (CONFIG_ERROR_CODES.has(normalized)) {
    return MANAGED_FAILURE_REASON.CONFIG_FAILURE;
  }
  if (ACCESS_OUTAGE_ERROR_CODES.has(normalized)) {
    return MANAGED_FAILURE_REASON.ACCESS_OUTAGE;
  }
  return MANAGED_FAILURE_REASON.AUTH_FAILURE;
}

export function classifyUpstreamFailureStatus(status, { clientErrorIsConfig = false } = {}) {
  const normalizedStatus = Number.parseInt(String(status), 10);
  if (!Number.isFinite(normalizedStatus) || normalizedStatus <= 0) {
    return MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE;
  }
  if (normalizedStatus === 401 || normalizedStatus === 403) {
    return MANAGED_FAILURE_REASON.CONFIG_FAILURE;
  }
  if (normalizedStatus >= 500) {
    return MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE;
  }
  if (clientErrorIsConfig && normalizedStatus >= 400) {
    return MANAGED_FAILURE_REASON.CONFIG_FAILURE;
  }
  return null;
}

export function getManagedFailureMessage(surface, reason) {
  return (
    SURFACE_FAILURE_MESSAGES[surface]?.[reason] ||
    DEFAULT_FAILURE_MESSAGES[reason] ||
    DEFAULT_FAILURE_MESSAGES[MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE]
  );
}

export function buildManagedBootstrapFailure({ status, reason }) {
  return {
    error: reason === MANAGED_FAILURE_REASON.AUTH_FAILURE ? "authentication required" : "managed access unavailable",
    reason,
    message: getManagedFailureMessage("portal_bootstrap", reason),
    retryable: status === 503 && isRetryableManagedFailure(reason)
  };
}

export function auditManagedSurfaceFailure(surface, details = {}) {
  const status = Number.isFinite(Number(details.status)) ? Math.trunc(Number(details.status)) : 0;
  const reason = String(details.reason || "").trim().toLowerCase() || MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE;
  const path = String(details.path || "").trim();
  return audit("managed_surface_failure", {
    surface,
    path,
    reason,
    status,
    retryable: isRetryableManagedFailure(reason),
    ...(details.errorCode ? { errorCode: String(details.errorCode) } : {}),
    ...(Number.isFinite(Number(details.upstreamStatus))
      ? { upstreamStatus: Math.trunc(Number(details.upstreamStatus)) }
      : {}),
    ...(details.remoteAddr ? { remoteAddr: String(details.remoteAddr) } : {}),
    ...(details.message ? { message: String(details.message) } : {}),
    ...resolveActorFields(details.actor),
    ...Object.fromEntries(
      Object.entries(details.extra || {}).filter(([, value]) => value !== undefined && value !== null && value !== "")
    )
  });
}

export function buildManagedV1ErrorDetails(path, reason, details = {}) {
  return {
    path,
    reason,
    retryable: isRetryableManagedFailure(reason),
    ...details
  };
}
