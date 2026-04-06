import { NextResponse } from "next/server.js";

import { resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../../lib/access.js";
import { audit } from "../../../lib/audit.js";
import { getConfig } from "../../../lib/config.js";
import { applySecurityHeaders } from "../../../lib/http.js";
import { buildUpstreamHeaders, buildUpstreamUrl, copyUpstreamResponseHeaders, isSsePath } from "../../../lib/proxy.js";
import { isUnsafeMethod, validateOriginAndReferrer } from "../../../lib/request-security.js";
import { clearSessionCookie, getRemoteAddress, validateCsrfToken } from "../../../lib/sessions.js";

export const runtime = "nodejs";

function errorEnvelope(status, code, message, details = {}) {
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
          "Cache-Control": "no-store"
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

async function streamSse(upstream, session) {
  const headers = copyUpstreamResponseHeaders(upstream.headers);
  headers.set("Content-Type", upstream.headers.get("content-type") || "text/event-stream");
  headers.set("Cache-Control", "no-store, no-transform");

  const { readable, writable } = new TransformStream();
  const reader = upstream.body?.getReader();
  const writer = writable.getWriter();

  audit("sse_proxy_open", {
    username: session.username,
    accessEmail: session.accessEmail
  });

  void (async () => {
    try {
      if (!reader) {
        await writer.close();
        audit("sse_proxy_close", {
          username: session.username,
          accessEmail: session.accessEmail,
          empty: true
        });
        return;
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        await writer.write(value);
      }
      audit("sse_proxy_close", {
        username: session.username,
        accessEmail: session.accessEmail
      });
    } catch (error) {
      audit("sse_proxy_error", {
        username: session.username,
        accessEmail: session.accessEmail,
        message: error instanceof Error ? error.message : String(error)
      });
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
  const authState = await resolveAuthenticatedAccessSession(request, { touch: !sseRequest });

  if (!authState.ok) {
    if (authState.revokeSession) {
      revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }
    audit("authorization_denied", {
      path: pathname,
      remoteAddr: getRemoteAddress(request),
      errorCode: authState.errorCode
    });
    const code =
      authState.status === 503 ? "ACCESS_UNAVAILABLE" : authState.status === 403 ? "FORBIDDEN" : "UNAUTHORIZED";
    const message =
      authState.status === 503
        ? "managed access unavailable"
        : authState.status === 403
          ? "forbidden"
          : "authentication required";
    const response = errorEnvelope(authState.status, code, message, { path: pathname });
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
        username: session.username
      });
      return errorEnvelope(403, "INVALID_CSRF", "origin validation failed", { path: pathname });
    }

    const csrfToken = request.headers.get("x-csrf-token") || "";
    if (!validateCsrfToken(session, csrfToken)) {
      audit("csrf_failure", {
        path: pathname,
        username: session.username
      });
      return errorEnvelope(403, "INVALID_CSRF", "csrf token validation failed", { path: pathname });
    }
  }

  const config = getConfig();
  if (!config.backendApiKey) {
    return errorEnvelope(503, "AUTH_CONFIGURATION_ERROR", "TP_BACKEND_API_KEY is not configured", {
      env: "TP_BACKEND_API_KEY",
      path: pathname
    });
  }

  const upstreamHeaders = buildUpstreamHeaders(request.headers, {
    backendApiKey: config.backendApiKey,
    actor: session,
    preferIdentityEncoding: sseRequest,
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
    audit("proxy_auth_failure", {
      path: pathname,
      username: session.username,
      message: error instanceof Error ? error.message : String(error)
    });
    return errorEnvelope(502, "UPSTREAM_UNAVAILABLE", "upstream request failed", {
      path: pathname
    });
  }

  const auditEvent = classifyAuditEvent(request.method, pathname);
  if (auditEvent) {
    audit(auditEvent, {
      username: session.username,
      accessEmail: session.accessEmail,
      path: pathname
    });
  }

  if (upstream.status === 401 || upstream.status === 403) {
    audit("proxy_auth_failure", {
      path: pathname,
      username: session.username,
      status: upstream.status
    });
  }

  if (sseRequest) {
    return applySecurityHeaders(await streamSse(upstream, session));
  }

  const responseHeaders = copyUpstreamResponseHeaders(upstream.headers);
  responseHeaders.set("Cache-Control", "no-store");

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
