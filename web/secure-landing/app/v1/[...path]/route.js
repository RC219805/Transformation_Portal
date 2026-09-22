import { NextResponse } from "next/server.js";

import { resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../../lib/access.js";
import { audit } from "../../../lib/audit.js";
import { getConfig } from "../../../lib/config.js";
import { hasIdentitySecret } from "../../../lib/frontdoor-identity.js";
import { applySecurityHeaders } from "../../../lib/http.js";
import {
  auditManagedSurfaceFailure,
  buildManagedV1ErrorDetails,
  classifyManagedAccessFailure,
  classifyUpstreamFailureStatus,
  getManagedFailureMessage,
  MANAGED_FAILURE_REASON
} from "../../../lib/managed-failure.js";
import { buildUpstreamHeaders, buildUpstreamUrl, copyUpstreamResponseHeaders, isSsePath } from "../../../lib/proxy.js";
import { isUnsafeMethod, validateOriginAndReferrer } from "../../../lib/request-security.js";
import { clearSessionCookie, getRemoteAddress, validateCsrfToken } from "../../../lib/sessions.js";
import { normalizeTraceparent, resolveRequestTraceparent, traceIdFromTraceparent } from "../../../lib/trace.js";

export const runtime = "nodejs";

// Canonical fields checked by the backend's tenant path admission. Only this
// typed denial is operator-correctable; unknown 401/403 responses still mean
// that the server-to-server authentication boundary failed.
const TENANT_PATH_FIELDS = new Set([
  "input_dir", "output_dir", "companions_manifest", "materials_manifest",
  "sam2_checkpoint_path", "cameras_sidecar_path", "fastvlm_python_executable",
  "fastvlm_mlx_vlm_dir", "vlm_captioning_model", "archive_index", "manifest_jsonl",
  "archive_root", "out_dir", "hash_manifest", "report_path", "out_jsonl",
  "out_summary", "policy_yaml", "bag_dir", "report_json", "out_ledger", "out_xml",
  "out_prov_jsonld", "out_stac_catalog", "out_stac_items_dir", "rights_jsonl"
]);
const TENANT_PATH_DENIAL_TIMEOUT_MS = 1000;

async function readTenantPathDenial(upstream) {
  if (!upstream.body || !/^application\/json(?:\s*;|$)/i.test(upstream.headers.get("content-type") || "")) {
    return null;
  }
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const expiresAt = performance.now() + TENANT_PATH_DENIAL_TIMEOUT_MS;
  let timeoutId;
  // This bounds only error classification, not the existing upstream request
  // or SSE lifetime. One deadline covers the entire body, including slow drip.
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error("tenant path denial read timed out")), TENANT_PATH_DENIAL_TIMEOUT_MS);
  });
  let text = "";
  let bytes = 0;
  try {
    for (;;) {
      // Already-queued empty chunks can otherwise starve the timer callback.
      if (performance.now() >= expiresAt) return null;
      const { value, done } = await Promise.race([reader.read(), deadline]);
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 4096) return null;
      text += decoder.decode(value, { stream: true });
    }
    const envelope = JSON.parse(text + decoder.decode());
    const error = envelope?.error;
    if (
      envelope?.schema !== "tp.orchestrator.error.v1" || envelope.success !== false || envelope.data !== null
      || error?.code !== "FORBIDDEN" || error?.details?.reason !== "tenant_path_outside_workspace"
      || !TENANT_PATH_FIELDS.has(error.details.field)
    ) return null;
    // Reconstruct the envelope; never forward arbitrary upstream messages,
    // filesystem paths, or extra diagnostic fields across the frontdoor.
    return { field: error.details.field, reason: "tenant_path_outside_workspace" };
  } catch {
    return null;
  } finally {
    clearTimeout(timeoutId);
    // A broken upstream cancel hook must not keep the managed error pending.
    void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
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

function classifyAuditEvent(method, pathname) {
  if (method === "POST" && pathname === "/v1/jobs") return "job_submit";
  if (method === "POST" && /\/v1\/jobs\/[^/]+\/cancel$/.test(pathname)) return "job_cancel";
  return null;
}

async function streamSse(upstream, session, traceparent) {
  const headers = copyUpstreamResponseHeaders(upstream.headers);
  headers.set("Content-Type", upstream.headers.get("content-type") || "text/event-stream");
  headers.set("Cache-Control", "no-store, no-transform");
  headers.set("traceparent", normalizeTraceparent(upstream.headers.get("traceparent")) || traceparent);

  audit("sse_proxy_open", {
    username: session.username,
    accessEmail: session.accessEmail
  });

  if (!upstream.body) {
    audit("sse_proxy_close", {
      username: session.username,
      accessEmail: session.accessEmail,
      reason: "no_upstream_body"
    });
    return new Response(null, { status: upstream.status, headers });
  }

  let terminalSeen = false;
  let bufferedTail = "";
  const decoder = new TextDecoder("utf-8", { fatal: false });

  const { readable, writable } = new TransformStream({
    transform(value, controller) {
      try {
        const chunk = typeof value === "string" ? value : decoder.decode(value, { stream: true });
        const combined = bufferedTail + chunk;
        // The backend signals job termination with an `event: done` SSE frame
        // (app.py:_job_events). Detecting it lets the frontdoor distinguish
        // a clean job-end close from a transport disconnect.
        if (/(?:^|\n)event:\s*done\s*(?:\r?\n|\r|$)/.test(combined)) {
          terminalSeen = true;
        }
        bufferedTail = combined.slice(-128);
      } catch {
        // Best-effort terminal detection; never let it interfere with proxying.
      }
      controller.enqueue(value);
    }
  });

  // Native piping propagates cancellation to the backend, errors to the
  // browser, and releases both locks even while the backend is idle.
  void upstream.body.pipeTo(writable)
    .then(() => {
      audit("sse_proxy_close", {
        username: session.username,
        accessEmail: session.accessEmail,
        reason: terminalSeen ? "terminal_event" : "upstream_eof"
      });
    })
    .catch((error) => {
      const isAbort = error?.name === "AbortError" || /aborted/i.test(error?.message || "");
      const message = error instanceof Error ? error.message : String(error);
      if (isAbort) {
        audit("sse_proxy_close", {
          username: session.username,
          accessEmail: session.accessEmail,
          reason: "client_abort"
        });
      } else if (!message || !message.trim()) {
        // Empty error payloads typically come from an upstream that closed
        // mid-frame; this is not actionable for operators.
        audit("sse_proxy_close", {
          username: session.username,
          accessEmail: session.accessEmail,
          reason: "upstream_disconnect"
        });
      } else {
        audit("sse_proxy_error", {
          username: session.username,
          accessEmail: session.accessEmail,
          message
        });
      }
    });

  return new Response(readable, {
    status: upstream.status,
    headers
  });
}

async function handleProxy(request, { params }) {
  const resolvedParams = typeof params?.then === "function" ? await params : params;
  const pathSegments = Array.isArray(resolvedParams?.path) ? resolvedParams.path : [];
  // Next supplies decoded segments. Re-encode filenames before URL parsing so
  // literal query/fragment characters cannot change the upstream target.
  let pathname = "/v1/";
  let invalidPath = pathSegments.length === 0;
  try {
    invalidPath ||= pathSegments.some((segment) => (
      typeof segment !== "string" || !segment || segment === "." || segment === ".."
      || /[\\/\u0000-\u001f\u007f]/.test(segment)
    ));
    if (!invalidPath) pathname += pathSegments.map(encodeURIComponent).join("/");
  } catch {
    invalidPath = true;
  }
  const sseRequest = isSsePath(pathname);
  const requestTraceparent = resolveRequestTraceparent(request);
  const traceId = traceIdFromTraceparent(requestTraceparent);
  const authState = await resolveAuthenticatedAccessSession(request, { touch: !sseRequest });

  if (!authState.ok) {
    const reason = classifyManagedAccessFailure(authState.errorCode);
    if (authState.revokeSession) {
      await revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }
    auditManagedSurfaceFailure("v1_proxy", {
      actor: authState.session,
      errorCode: authState.errorCode,
      path: pathname,
      remoteAddr: getRemoteAddress(request),
      reason,
      status: authState.status,
      extra: traceId ? { traceId } : {}
    });
    const code = reason === MANAGED_FAILURE_REASON.CONFIG_FAILURE
      ? "AUTH_CONFIGURATION_ERROR"
      : authState.status === 503
        ? "ACCESS_UNAVAILABLE"
        : authState.status === 403
          ? "FORBIDDEN"
          : "UNAUTHORIZED";
    const message =
      authState.status === 503 || reason === MANAGED_FAILURE_REASON.CONFIG_FAILURE
        ? getManagedFailureMessage("v1_proxy", reason)
        : authState.status === 403
          ? "forbidden"
          : "authentication required";
    const response = errorEnvelope(
      authState.status,
      code,
      message,
      buildManagedV1ErrorDetails(pathname, reason),
      requestTraceparent
    );
    if (authState.revokeSession) {
      clearSessionCookie(response);
    }
    return response;
  }
  const { session } = authState;

  if (isUnsafeMethod(request.method)) {
    if (!validateOriginAndReferrer(request)) {
      audit("csrf_failure", {
        path: pathname,
        username: session.username,
        traceId
      });
      return errorEnvelope(403, "INVALID_CSRF", "origin validation failed", { path: pathname }, requestTraceparent);
    }

    const csrfToken = request.headers.get("x-csrf-token") || "";
    if (!validateCsrfToken(session, csrfToken)) {
      audit("csrf_failure", {
        path: pathname,
        username: session.username,
        traceId
      });
      return errorEnvelope(403, "INVALID_CSRF", "csrf token validation failed", { path: pathname }, requestTraceparent);
    }
  }

  if (invalidPath) {
    return errorEnvelope(400, "INVALID_PATH", "invalid API path", {}, requestTraceparent);
  }

  const config = getConfig();
  if (!config.backendApiKey) {
    auditManagedSurfaceFailure("v1_proxy", {
      actor: session,
      extra: { env: "TP_BACKEND_API_KEY", ...(traceId ? { traceId } : {}) },
      path: pathname,
      reason: MANAGED_FAILURE_REASON.CONFIG_FAILURE,
      status: 503
    });
    return errorEnvelope(
      503,
      "AUTH_CONFIGURATION_ERROR",
      "TP_BACKEND_API_KEY is not configured",
      buildManagedV1ErrorDetails(pathname, MANAGED_FAILURE_REASON.CONFIG_FAILURE, {
        env: "TP_BACKEND_API_KEY"
      }),
      requestTraceparent
    );
  }

  if (config.pilotControlPlaneEnabled && (
    !hasIdentitySecret(config.frontdoorIdentitySecret) || config.frontdoorIdentitySecret === config.backendApiKey
  )) {
    return errorEnvelope(
      503, "AUTH_CONFIGURATION_ERROR", "tenant identity configuration unavailable",
      buildManagedV1ErrorDetails(pathname, MANAGED_FAILURE_REASON.CONFIG_FAILURE, {
        env: "TP_FRONTDOOR_IDENTITY_SECRET"
      }), requestTraceparent
    );
  }
  const upstreamUrl = buildUpstreamUrl(pathname, request.nextUrl.search);
  const upstreamTarget = new URL(upstreamUrl);
  const upstreamHeaders = buildUpstreamHeaders(request.headers, {
    backendApiKey: config.backendApiKey,
    actor: session,
    identity: config.pilotControlPlaneEnabled ? {
      secret: config.frontdoorIdentitySecret,
      method: request.method,
      target: upstreamTarget.pathname + upstreamTarget.search
    } : null,
    preferIdentityEncoding: sseRequest,
    traceparent: requestTraceparent,
    forwarding: {
      clientIp: String(request.headers.get("cf-connecting-ip") || "").trim() || null,
      host: request.headers.get("host") || request.nextUrl.host,
      proto: request.nextUrl.protocol.replace(":", "")
    }
  });

  const fetchOptions = {
    method: request.method,
    headers: upstreamHeaders,
    cache: "no-store",
    redirect: "manual",
    signal: request.signal
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    fetchOptions.body = request.body;
    fetchOptions.duplex = "half";
  }

  let upstream;
  try {
    upstream = await fetch(upstreamUrl, fetchOptions);
  } catch (error) {
    auditManagedSurfaceFailure("v1_proxy", {
      actor: session,
      message: error instanceof Error ? error.message : String(error),
      path: pathname,
      reason: MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE,
      status: 502,
      extra: traceId ? { traceId } : {}
    });
    return errorEnvelope(
      502,
      "UPSTREAM_UNAVAILABLE",
      getManagedFailureMessage("v1_proxy", MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE),
      buildManagedV1ErrorDetails(pathname, MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE),
      requestTraceparent
    );
  }

  if (
    config.pilotControlPlaneEnabled && upstream.status === 403 && request.method === "POST"
    && (pathname === "/v1/config-preview" || pathname === "/v1/jobs")
  ) {
    const pathDenial = await readTenantPathDenial(upstream);
    if (pathDenial) {
      return errorEnvelope(403, "FORBIDDEN", "tenant admission failed", pathDenial, requestTraceparent);
    }
  }

  const upstreamFailureReason = classifyUpstreamFailureStatus(upstream.status);
  if (upstreamFailureReason) {
    try {
      void upstream.body?.cancel().catch(() => {});
    } catch {
      // The body is discarded; retain the managed error envelope if closing fails.
    }
    const status = upstreamFailureReason === MANAGED_FAILURE_REASON.CONFIG_FAILURE ? 503 : 502;
    const code =
      upstreamFailureReason === MANAGED_FAILURE_REASON.CONFIG_FAILURE
        ? "AUTH_CONFIGURATION_ERROR"
        : "UPSTREAM_UNAVAILABLE";
    auditManagedSurfaceFailure("v1_proxy", {
      actor: session,
      path: pathname,
      reason: upstreamFailureReason,
      status,
      upstreamStatus: upstream.status,
      extra: traceId ? { traceId } : {}
    });
    return errorEnvelope(
      status,
      code,
      getManagedFailureMessage("v1_proxy", upstreamFailureReason),
      buildManagedV1ErrorDetails(pathname, upstreamFailureReason, {
        upstreamStatus: upstream.status
      }),
      requestTraceparent
    );
  }

  if (sseRequest) {
    return applySecurityHeaders(await streamSse(upstream, session, requestTraceparent));
  }

  const auditEvent = classifyAuditEvent(request.method, pathname);
  if (auditEvent && upstream.status < 400) {
    audit(auditEvent, {
      username: session.username,
      accessEmail: session.accessEmail,
      path: pathname,
      traceId
    });
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

export const GET = handleProxy;
export const POST = handleProxy;
export const PUT = handleProxy;
export const PATCH = handleProxy;
export const DELETE = handleProxy;
