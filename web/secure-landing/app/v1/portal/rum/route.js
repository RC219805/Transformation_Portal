import { NextResponse } from "next/server.js";

import { getConfig, isFrontdoorRumTelemetryEnabled, isPortalRumEnabled } from "../../../../lib/config.js";
import { applySecurityHeaders } from "../../../../lib/http.js";
import {
  auditManagedSurfaceFailure,
  buildManagedV1ErrorDetails,
  classifyUpstreamFailureStatus,
  getManagedFailureMessage,
  MANAGED_FAILURE_REASON
} from "../../../../lib/managed-failure.js";
import { buildUpstreamHeaders, buildUpstreamUrl, copyUpstreamResponseHeaders } from "../../../../lib/proxy.js";
import { validateOriginAndReferrer } from "../../../../lib/request-security.js";
import { resolveRequestTraceparent, traceIdFromTraceparent, normalizeTraceparent } from "../../../../lib/trace.js";

export const runtime = "nodejs";

const PUBLIC_RUM_PATH = "/v1/portal/rum";
const FRONTDOOR_RUM_EVENT_TYPES = new Set([
  "landing_rendered",
  "login_rendered",
  "login_submit_attempt",
  "login_submit_success",
  "login_submit_failure",
  "logout_submit_attempt",
  "logout_submit_success",
  "logout_submit_failure",
]);
const FRONTDOOR_RUM_ROUTES = new Set(["/", "/login"]);
const FRONTDOOR_RUM_VIEWS = new Set(["landing", "login"]);
// RUM samples are a handful of short scalar fields plus token-only metadata;
// this unauthenticated route must never buffer or re-sign arbitrary bodies.
const MAX_RUM_BODY_BYTES = 16 * 1024;
// Only these client headers are forwarded; the backend auth, forwarding and
// traceparent headers are set by buildUpstreamHeaders itself.
const FORWARDED_RUM_REQUEST_HEADERS = ["accept", "content-type"];

function successEnvelope(data, traceparent = "") {
  return applySecurityHeaders(
    NextResponse.json(
      {
        schema: "tp.orchestrator.portal_rum_ingest.v1",
        success: true,
        data,
        error: null
      },
      {
        status: 200,
        headers: {
          "Cache-Control": "no-store",
          ...(traceparent ? { traceparent } : {})
        }
      }
    )
  );
}

function errorEnvelope(status, code, message, details = {}, traceparent = "") {
  return applySecurityHeaders(
    NextResponse.json(
      {
        schema: "tp.orchestrator.error.v1",
        success: false,
        data: null,
        error: {
          code,
          message,
          details
        }
      },
      {
        status,
        headers: {
          "Cache-Control": "no-store",
          ...(traceparent ? { traceparent } : {})
        }
      }
    )
  );
}

function requestTooLargeEnvelope(traceparent) {
  return errorEnvelope(
    413,
    "REQUEST_TOO_LARGE",
    "portal rum payload exceeds the maximum request size",
    { path: PUBLIC_RUM_PATH, max_request_bytes: MAX_RUM_BODY_BYTES },
    traceparent
  );
}

function declaredContentLengthExceedsLimit(headers) {
  const raw = String(headers.get("content-length") || "").trim();
  // A malformed value is left to the capped reader below.
  if (!/^\d+$/.test(raw)) return false;
  return Number(raw) > MAX_RUM_BODY_BYTES;
}

// Returns the decoded body, or null when it exceeds MAX_RUM_BODY_BYTES. Stops
// reading (and cancels the stream) as soon as the cap is crossed.
async function readCappedBodyText(request) {
  if (!request.body) return "";
  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_RUM_BODY_BYTES) {
      await reader.cancel().catch(() => {});
      return null;
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(bytes);
}

function pickForwardedRumHeaders(sourceHeaders) {
  const headers = new Headers();
  for (const name of FORWARDED_RUM_REQUEST_HEADERS) {
    const value = sourceHeaders.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

function isFrontdoorRumPayload(bodyText) {
  try {
    const parsed = JSON.parse(bodyText || "{}");
    const eventType = String(parsed?.event_type || "").trim();
    const route = String(parsed?.route || "").trim();
    const view = String(parsed?.view || "").trim().toLowerCase();
    return (
      FRONTDOOR_RUM_EVENT_TYPES.has(eventType)
      || FRONTDOOR_RUM_ROUTES.has(route)
      || FRONTDOOR_RUM_VIEWS.has(view)
    );
  } catch {
    return false;
  }
}

export async function POST(request) {
  const requestTraceparent = resolveRequestTraceparent(request);
  const traceId = traceIdFromTraceparent(requestTraceparent);

  if (!validateOriginAndReferrer(request)) {
    return errorEnvelope(
      403,
      "INVALID_CSRF",
      "origin validation failed",
      { path: PUBLIC_RUM_PATH },
      requestTraceparent
    );
  }

  if (!isPortalRumEnabled()) {
    return successEnvelope({ accepted: false, disabled: true }, requestTraceparent);
  }

  if (declaredContentLengthExceedsLimit(request.headers)) {
    return requestTooLargeEnvelope(requestTraceparent);
  }
  const bodyText = await readCappedBodyText(request);
  if (bodyText === null) {
    return requestTooLargeEnvelope(requestTraceparent);
  }
  if (
    isFrontdoorRumPayload(bodyText)
    && !isFrontdoorRumTelemetryEnabled({ traceparent: requestTraceparent })
  ) {
    return successEnvelope({ accepted: false, disabled: true }, requestTraceparent);
  }

  const config = getConfig();
  if (!config.backendApiKey) {
    auditManagedSurfaceFailure("v1_proxy", {
      extra: { env: "TP_BACKEND_API_KEY", ...(traceId ? { traceId } : {}) },
      path: PUBLIC_RUM_PATH,
      reason: MANAGED_FAILURE_REASON.CONFIG_FAILURE,
      status: 503
    });
    return errorEnvelope(
      503,
      "AUTH_CONFIGURATION_ERROR",
      "TP_BACKEND_API_KEY is not configured",
      buildManagedV1ErrorDetails(PUBLIC_RUM_PATH, MANAGED_FAILURE_REASON.CONFIG_FAILURE, {
        env: "TP_BACKEND_API_KEY"
      }),
      requestTraceparent
    );
  }

  const upstreamHeaders = buildUpstreamHeaders(pickForwardedRumHeaders(request.headers), {
    backendApiKey: config.backendApiKey,
    actor: null,
    traceparent: requestTraceparent,
    forwarding: {
      clientIp: String(request.headers.get("cf-connecting-ip") || "").trim() || null,
      host: request.headers.get("host") || request.nextUrl.host,
      proto: request.nextUrl.protocol.replace(":", "")
    }
  });

  let upstream;
  try {
    upstream = await fetch(buildUpstreamUrl(PUBLIC_RUM_PATH, request.nextUrl.search), {
      method: "POST",
      headers: upstreamHeaders,
      body: bodyText,
      cache: "no-store",
      redirect: "manual",
      duplex: "half"
    });
  } catch (error) {
    auditManagedSurfaceFailure("v1_proxy", {
      message: error instanceof Error ? error.message : String(error),
      path: PUBLIC_RUM_PATH,
      reason: MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE,
      status: 502,
      extra: traceId ? { traceId } : {}
    });
    return errorEnvelope(
      502,
      "UPSTREAM_UNAVAILABLE",
      getManagedFailureMessage("v1_proxy", MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE),
      buildManagedV1ErrorDetails(PUBLIC_RUM_PATH, MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE),
      requestTraceparent
    );
  }

  const upstreamFailureReason = classifyUpstreamFailureStatus(upstream.status);
  if (upstreamFailureReason) {
    const status = upstreamFailureReason === MANAGED_FAILURE_REASON.CONFIG_FAILURE ? 503 : 502;
    const code =
      upstreamFailureReason === MANAGED_FAILURE_REASON.CONFIG_FAILURE
        ? "AUTH_CONFIGURATION_ERROR"
        : "UPSTREAM_UNAVAILABLE";
    auditManagedSurfaceFailure("v1_proxy", {
      path: PUBLIC_RUM_PATH,
      reason: upstreamFailureReason,
      status,
      upstreamStatus: upstream.status,
      extra: traceId ? { traceId } : {}
    });
    return errorEnvelope(
      status,
      code,
      getManagedFailureMessage("v1_proxy", upstreamFailureReason),
      buildManagedV1ErrorDetails(PUBLIC_RUM_PATH, upstreamFailureReason, {
        upstreamStatus: upstream.status
      }),
      requestTraceparent
    );
  }

  const responseHeaders = copyUpstreamResponseHeaders(upstream.headers);
  responseHeaders.set("Cache-Control", "no-store");
  responseHeaders.set("traceparent", normalizeTraceparent(upstream.headers.get("traceparent")) || requestTraceparent);

  return applySecurityHeaders(
    new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders
    })
  );
}
