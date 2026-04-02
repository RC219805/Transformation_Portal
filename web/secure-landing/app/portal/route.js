import { NextResponse } from "next/server.js";

import { applySecurityHeaders } from "../../lib/http.js";
import { copyUpstreamResponseHeaders } from "../../lib/proxy.js";
import { getConfig } from "../../lib/config.js";
import { audit } from "../../lib/audit.js";
import { getSessionFromRequest } from "../../lib/sessions.js";

export const runtime = "nodejs";

export async function GET(request) {
  const session = getSessionFromRequest(request, { touch: true });
  if (!session?.authenticated) {
    return applySecurityHeaders(NextResponse.redirect(new URL("/login", request.url), 302));
  }

  const upstream = await fetch(new URL("/", getConfig().fastapiOrigin), {
    headers: {
      "Accept": "text/html"
    },
    cache: "no-store"
  });

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
