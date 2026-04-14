import { NextResponse } from "next/server.js";

import { applySecurityHeaders, buildRequestUrl } from "../../lib/http.js";
import {
  clearSessionCookie,
  destroySession,
  getSessionFromRequest,
  validateCsrfToken
} from "../../lib/sessions.js";
import { validateOriginAndReferrer } from "../../lib/request-security.js";
import { audit } from "../../lib/audit.js";

export const runtime = "nodejs";

export async function POST(request) {
  const session = getSessionFromRequest(request, { touch: false });
  if (!validateOriginAndReferrer(request)) {
    audit("csrf_failure", { path: "/logout" });
    const response = applySecurityHeaders(NextResponse.redirect(buildRequestUrl(request, "/login?error=csrf"), 303));
    clearSessionCookie(response);
    return response;
  }

  const csrfToken = request.headers.get("x-csrf-token") || "";
  if (session && !validateCsrfToken(session, csrfToken)) {
    audit("csrf_failure", { path: "/logout", username: session.username });
    const response = applySecurityHeaders(NextResponse.redirect(buildRequestUrl(request, "/login?error=csrf"), 303));
    clearSessionCookie(response);
    return response;
  }

  if (session?.id) {
    destroySession(session.id, "logout");
  }

  const response = applySecurityHeaders(NextResponse.redirect(buildRequestUrl(request, "/login"), 303));
  clearSessionCookie(response);
  return response;
}
