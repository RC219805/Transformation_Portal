import { NextResponse } from "next/server.js";

import { applySecurityHeaders, FRONTDOOR_CSP, generateScriptNonce } from "../lib/http.js";
import { renderHomepage } from "../lib/homepage.js";
import { renderRumClientScript } from "../lib/rum-client.js";
import { generateTraceparent } from "../lib/trace.js";

export const runtime = "nodejs";
// force-dynamic ensures GET re-runs per request so each response carries a
// fresh CSP nonce when RUM is enabled. When RUM is disabled the response body
// is identical and the Cache-Control hint below still allows downstream
// caching; when RUM is enabled the response is marked no-store so a per-user
// nonce cannot leak across requests via shared caches.
export const dynamic = "force-dynamic";

const HOMEPAGE_CACHE_CONTROL_STATIC = "public, max-age=300, must-revalidate";
const HOMEPAGE_CACHE_CONTROL_DYNAMIC = "no-store";

function _isRumEnabled() {
  return String(process.env.TP_PORTAL_RUM_ENABLED || "").trim().toLowerCase() === "true";
}

export async function GET() {
  const rumEnabled = _isRumEnabled();
  const scriptNonce = rumEnabled ? generateScriptNonce() : null;
  const rumScript = rumEnabled
    ? renderRumClientScript({
        route: "/",
        view: "landing",
        traceparent: generateTraceparent(),
      })
    : "";
  const html = renderHomepage({ rumScript, scriptNonce });
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": rumEnabled ? HOMEPAGE_CACHE_CONTROL_DYNAMIC : HOMEPAGE_CACHE_CONTROL_STATIC
    }
  });
  return applySecurityHeaders(response, { csp: FRONTDOOR_CSP, scriptNonce });
}
