import { NextResponse } from "next/server.js";

import { applySecurityHeaders, FRONTDOOR_CSP } from "../lib/http.js";
import { renderHomepage } from "../lib/homepage.js";

export const runtime = "nodejs";
export const dynamic = "force-static";

const HOMEPAGE_CACHE_CONTROL = "public, max-age=300, must-revalidate";

export async function GET() {
  const html = renderHomepage();
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": HOMEPAGE_CACHE_CONTROL
    }
  });
  return applySecurityHeaders(response, { csp: FRONTDOOR_CSP });
}
