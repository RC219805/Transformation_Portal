import { NextResponse } from "next/server.js";

import { getSessionFromRequest } from "../lib/sessions.js";
import { applySecurityHeaders } from "../lib/http.js";

export const runtime = "nodejs";

export async function GET(request) {
  const session = getSessionFromRequest(request, { touch: false });
  const destination = session?.authenticated ? "/portal" : "/login";
  return applySecurityHeaders(NextResponse.redirect(new URL(destination, request.url), 302));
}
