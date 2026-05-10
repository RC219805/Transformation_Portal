import { NextResponse } from "next/server.js";

import {
  isFrontdoorRumTelemetryEnabled,
  shouldDisableFrontdoorRumHomepageCache
} from "../lib/config.js";
import { applySecurityHeaders, FRONTDOOR_CSP, generateScriptNonce } from "../lib/http.js";
import { renderHomepage } from "../lib/homepage.js";
import { renderRumClientScript } from "../lib/rum-client.js";
import { resolveRequestTraceparent } from "../lib/trace.js";

export const runtime = "nodejs";
// force-dynamic ensures GET re-runs per request so each response carries a
// fresh CSP nonce when RUM is enabled. When front-door RUM can vary by rollout
// sampling the response is marked no-store for every request, so shared caches
// cannot pin a sampled-out body for the same URL.
export const dynamic = "force-dynamic";

const HOMEPAGE_CACHE_CONTROL_STATIC = "public, max-age=300, must-revalidate";
const HOMEPAGE_CACHE_CONTROL_DYNAMIC = "no-store";

export async function GET(request) {
  const rumTraceparent = resolveRequestTraceparent(request);
  const rumEnabled = isFrontdoorRumTelemetryEnabled({ traceparent: rumTraceparent });
  const disableRumHomepageCache = shouldDisableFrontdoorRumHomepageCache();
  const scriptNonce = rumEnabled ? generateScriptNonce() : null;
  const rumScript = rumEnabled
    ? renderRumClientScript({
        route: "/",
        view: "landing",
        traceparent: rumTraceparent,
      })
    : "";
  const html = renderHomepage({ rumScript, scriptNonce });
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": disableRumHomepageCache
        ? HOMEPAGE_CACHE_CONTROL_DYNAMIC
        : HOMEPAGE_CACHE_CONTROL_STATIC
    }
  });
  return applySecurityHeaders(response, { csp: FRONTDOOR_CSP, scriptNonce });
}
