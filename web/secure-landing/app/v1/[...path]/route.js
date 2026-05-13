import { NextResponse } from "next/server.js";

import { resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../../lib/access.js";
import { audit } from "../../../lib/audit.js";
import { getConfig } from "../../../lib/config.js";
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

  const { readable, writable } = new TransformStream();
  const reader = upstream.body?.getReader();
  const writer = writable.getWriter();

  audit("sse_proxy_open", {
    username: session.username,
    accessEmail: session.accessEmail
  });

  void (async () => {
    let terminalSeen = false;
    let bufferedTail = "";
    const decoder = new TextDecoder("utf-8", { fatal: false });

    const inspectChunk = (value) => {
      try {
        const chunk = typeof value === "string" ? value : decoder.decode(value, { stream: true });
        if (!chunk) return;
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
    };

    try {
      if (!reader) {
        await writer.close();
        audit("sse_proxy_close", {
          username: session.username,
          accessEmail: session.accessEmail,
          reason: "no_upstream_body"
        });
        return;
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        inspectChunk(value);
        await writer.write(value);
      }
      audit("sse_proxy_close", {
        username: session.username,
        accessEmail: session.accessEmail,
        reason: terminalSeen ? "terminal_event" : "upstream_eof"
      });
    } catch (error) {
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
    } finally {
      try {
        await writer.close();
      } catch {
        // Ignore double-close during error shutdown.
      }
      reader?.releaseLock();
    }
  })();

  return new Response(readable, {
    status: upstream.status,
    headers
  });
}

async function handleProxy(request, { params }) {
  const resolvedParams = typeof params?.then === "function" ? await params : params;
  const pathSegments = Array.isArray(resolvedParams?.path) ? resolvedParams.path : [];
  const pathname = `/v1/${pathSegments.join("/")}`;
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

  const upstreamHeaders = buildUpstreamHeaders(request.headers, {
    backendApiKey: config.backendApiKey,
    actor: session,
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
    redirect: "manual"
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    fetchOptions.body = request.body;
    fetchOptions.duplex = "half";
  }

  const upstreamUrl = buildUpstreamUrl(pathname, request.nextUrl.search);

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

  const upstreamFailureReason = classifyUpstreamFailureStatus(upstream.status);
  if (upstreamFailureReason) {
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
