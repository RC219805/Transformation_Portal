// Server-side RUM emitter for the front-door's POST /login handler.
//
// Fire-and-forget POSTs to FastAPI's /v1/portal/rum using the same proxy
// helpers the browser-facing /v1/portal/rum proxy route uses. Bypasses the
// browser proxy because server-internal calls have no Origin/Referrer headers
// to validate.
//
// Gate contract:
//   - The CALLER is the sole feature-flag gate. This emitter does NOT consult
//     `isPortalRumEnabled()` so paired events (attempt + terminal) cannot
//     diverge if the env flag flips between calls. The route handler captures
//     `isPortalRumEnabled()` once per request and short-circuits its own
//     wrapper before reaching this function.
//
// Safety contract:
//   - Returns synchronously; the caller never awaits.
//   - Catches every promise rejection internally so no unhandled rejection
//     can escape the emitter.
//   - Times out via `AbortSignal.timeout(...)`, which uses an internally
//     `unref`'d timer so a slow upstream cannot extend Node process lifetime
//     past the response.
//   - Emission failure must never alter login response status, redirect,
//     cookie, session rotation, CSRF handling, or response timing.

import { getConfig } from "./config.js";
import { buildUpstreamHeaders, buildUpstreamUrl } from "./proxy.js";
import { generateTraceparent, normalizeTraceparent } from "./trace.js";

const RUM_PATH = "/v1/portal/rum";
const RUM_EMIT_TIMEOUT_MS = 2000;

const ALLOWED_FAILURE_CODES = new Set([
  "csrf",
  "configuration",
  "access",
  "throttled",
  "invalid",
]);

const LOGIN_ATTEMPT_EVENT = "login_submit_attempt";
const LOGIN_SUCCESS_EVENT = "login_submit_success";
const LOGIN_FAILURE_EVENT = "login_submit_failure";

const LOGIN_EVENT_TYPES = new Set([
  LOGIN_ATTEMPT_EVENT,
  LOGIN_SUCCESS_EVENT,
  LOGIN_FAILURE_EVENT,
]);

export const LOGIN_RUM_EVENT_TYPES = Object.freeze({
  ATTEMPT: LOGIN_ATTEMPT_EVENT,
  SUCCESS: LOGIN_SUCCESS_EVENT,
  FAILURE: LOGIN_FAILURE_EVENT,
});

export const LOGIN_RUM_FAILURE_CODES = Object.freeze({
  CSRF: "csrf",
  CONFIGURATION: "configuration",
  ACCESS: "access",
  THROTTLED: "throttled",
  INVALID: "invalid",
});

export function emitLoginRumEvent({
  eventType,
  value,
  failureCode = null,
  traceparent = "",
}) {
  if (!LOGIN_EVENT_TYPES.has(eventType)) return;

  const config = getConfig();
  if (!config.backendApiKey) return;

  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return;
  }

  const metric = eventType === LOGIN_ATTEMPT_EVENT ? "count" : "duration";
  const unit = eventType === LOGIN_ATTEMPT_EVENT ? "count" : "ms";

  const metadata = {};
  if (
    eventType === LOGIN_FAILURE_EVENT
    && typeof failureCode === "string"
    && ALLOWED_FAILURE_CODES.has(failureCode)
  ) {
    metadata.failure_code = failureCode;
  }

  const tp = normalizeTraceparent(traceparent) || generateTraceparent();
  const headers = buildUpstreamHeaders(new Headers(), {
    backendApiKey: config.backendApiKey,
    actor: null,
    traceparent: tp,
  });
  headers.set("Content-Type", "application/json");

  const body = JSON.stringify({
    event_type: eventType,
    route: "/login",
    view: "login",
    metric,
    value,
    unit,
    metadata,
  });

  // AbortSignal.timeout uses an internally unref'd timer so it cannot keep
  // the Node event loop alive past the response. Detach from the event loop
  // via Promise.resolve().then so the caller never awaits and unhandled
  // rejections cannot escape.
  void Promise.resolve()
    .then(() =>
      fetch(buildUpstreamUrl(RUM_PATH), {
        method: "POST",
        headers,
        body,
        cache: "no-store",
        keepalive: true,
        signal: AbortSignal.timeout(RUM_EMIT_TIMEOUT_MS),
        redirect: "manual",
      })
    )
    .catch(() => {
      // best-effort telemetry; never propagate
    });
}
