import { NextResponse } from "next/server.js";

import { resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../lib/access.js";
import { applySecurityHeaders } from "../../lib/http.js";
import { copyUpstreamResponseHeaders } from "../../lib/proxy.js";
import { getConfig } from "../../lib/config.js";
import { audit } from "../../lib/audit.js";
import { clearSessionCookie } from "../../lib/sessions.js";

export const runtime = "nodejs";

export async function GET(request) {
  const authState = await resolveAuthenticatedAccessSession(request, { touch: true });
  if (!authState.ok) {
    if (authState.status === 503) {
      const headers = new Headers();
      headers.set("Cache-Control", "no-store");
      return applySecurityHeaders(
        new Response("Managed front door unavailable", {
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
    audit("proxy_auth_failure", {
      path: "/portal",
      username: session.username,
      message: error instanceof Error ? error.message : String(error),
      status: 503
    });
    const headers = new Headers();
    headers.set("Cache-Control", "no-store");
    return applySecurityHeaders(
      new Response("Upstream service unavailable", {
        status: 503,
        headers
      })
    );
  }

  if (!upstream.ok) {
    audit("authorization_denied", {
      path: "/portal",
      username: session.username,
      status: upstream.status
    });
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
