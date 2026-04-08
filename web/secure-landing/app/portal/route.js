import { NextResponse } from "next/server.js";

import { resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../lib/access.js";
import { applySecurityHeaders } from "../../lib/http.js";
import { copyUpstreamResponseHeaders } from "../../lib/proxy.js";
import { getConfig } from "../../lib/config.js";
import {
  auditManagedSurfaceFailure,
  classifyManagedAccessFailure,
  classifyUpstreamFailureStatus,
  getManagedFailureMessage,
  MANAGED_FAILURE_REASON
} from "../../lib/managed-failure.js";
import { clearSessionCookie } from "../../lib/sessions.js";

export const runtime = "nodejs";

export async function GET(request) {
  const authState = await resolveAuthenticatedAccessSession(request, { touch: true });
  if (!authState.ok) {
    const reason = classifyManagedAccessFailure(authState.errorCode);
    auditManagedSurfaceFailure("portal", {
      actor: authState.session,
      errorCode: authState.errorCode,
      path: "/portal",
      reason,
      status: authState.status
    });
    if (reason !== MANAGED_FAILURE_REASON.AUTH_FAILURE) {
      const headers = new Headers();
      headers.set("Cache-Control", "no-store");
      return applySecurityHeaders(
        new Response(getManagedFailureMessage("portal", reason), {
          status: 503,
          headers
        })
      );
    }

    if (authState.revokeSession) {
      revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }

    const response = applySecurityHeaders(NextResponse.redirect(new URL("/login", request.url), 302));
    if (authState.revokeSession) {
      clearSessionCookie(response);
    }
    return response;
  }
  const { session } = authState;

  let upstream;
  try {
    upstream = await fetch(new URL("/", getConfig().fastapiOrigin), {
      headers: {
        "Accept": "text/html"
      },
      cache: "no-store"
    });
  } catch (error) {
    auditManagedSurfaceFailure("portal", {
      actor: session,
      message: error instanceof Error ? error.message : String(error),
      path: "/portal",
      reason: MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE,
      status: 503
    });
    const headers = new Headers();
    headers.set("Cache-Control", "no-store");
    return applySecurityHeaders(
      new Response(getManagedFailureMessage("portal", MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE), {
        status: 503,
        headers
      })
    );
  }

  if (!upstream.ok) {
    const reason = classifyUpstreamFailureStatus(upstream.status, { clientErrorIsConfig: true });
    if (reason) {
      auditManagedSurfaceFailure("portal", {
        actor: session,
        path: "/portal",
        reason,
        status: 503,
        upstreamStatus: upstream.status
      });
      const headers = new Headers();
      headers.set("Cache-Control", "no-store");
      return applySecurityHeaders(
        new Response(getManagedFailureMessage("portal", reason), {
          status: 503,
          headers
        })
      );
    }
  }

  const headers = copyUpstreamResponseHeaders(upstream.headers);
  headers.set("Cache-Control", "no-store");
  return applySecurityHeaders(
    new Response(upstream.body, {
      status: upstream.status,
      headers
    })
  );
}
