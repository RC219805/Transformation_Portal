import { NextResponse } from "next/server.js";

import { getSessionFromRequest } from "../lib/sessions.js";
import { applySecurityHeaders, FRONTDOOR_CSP } from "../lib/http.js";
import { renderHomepage } from "../lib/homepage.js";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request) {
  const session = getSessionFromRequest(request, { touch: false });
  const hasAuthenticatedSessionHint = Boolean(session?.authenticated);
  const html = renderHomepage({ hasAuthenticatedSessionHint });
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store"
    }
  });
  return applySecurityHeaders(response, { csp: FRONTDOOR_CSP });
}
