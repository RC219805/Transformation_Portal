import { NextResponse } from "next/server.js";

import { applySecurityHeaders, buildRequestUrl } from "../../lib/http.js";
import { isFrontdoorRumTelemetryEnabled } from "../../lib/config.js";
import {
  emitLogoutRumEvent,
  LOGOUT_RUM_EVENT_TYPES,
  LOGOUT_RUM_FAILURE_CODES
} from "../../lib/rum-emitter.js";
import {
  clearSessionCookie,
  destroySession,
  getSessionFromRequest,
  validateCsrfToken
} from "../../lib/sessions.js";
import { resolveRequestTraceparent } from "../../lib/trace.js";
import { validateOriginAndReferrer } from "../../lib/request-security.js";
import { audit } from "../../lib/audit.js";

export const runtime = "nodejs";

export async function POST(request) {
  // Capture the RUM enable decision once at handler entry. This is the SOLE
  // gate for paired emissions: the emitter intentionally does not re-check
  // isFrontdoorRumTelemetryEnabled() so a flag flip mid-request cannot produce a partial
  // attempt/terminal pair (one captured but not the other). Mirrors the
  // /login handler's contract from #1684.
  const rumTraceparent = resolveRequestTraceparent(request);
  const rumActive = isFrontdoorRumTelemetryEnabled({ traceparent: rumTraceparent });
  const attemptStart = Date.now();
  const emitLogoutRum = (eventType, failureCode = null) => {
    if (!rumActive) return;
    const value = eventType === LOGOUT_RUM_EVENT_TYPES.ATTEMPT
      ? 1
      : Math.max(0, Date.now() - attemptStart);
    emitLogoutRumEvent({
      eventType,
      value,
      failureCode,
      traceparent: rumTraceparent
    });
  };

  emitLogoutRum(LOGOUT_RUM_EVENT_TYPES.ATTEMPT);

  const session = await getSessionFromRequest(request, { touch: false });
  if (!validateOriginAndReferrer(request)) {
    audit("csrf_failure", { path: "/logout" });
    emitLogoutRum(LOGOUT_RUM_EVENT_TYPES.FAILURE, LOGOUT_RUM_FAILURE_CODES.CSRF);
    const response = applySecurityHeaders(NextResponse.redirect(buildRequestUrl(request, "/login?error=csrf"), 303));
    clearSessionCookie(response);
    return response;
  }

  const csrfToken = request.headers.get("x-csrf-token") || "";
  if (session && !validateCsrfToken(session, csrfToken)) {
    audit("csrf_failure", { path: "/logout", username: session.username });
    emitLogoutRum(LOGOUT_RUM_EVENT_TYPES.FAILURE, LOGOUT_RUM_FAILURE_CODES.CSRF);
    const response = applySecurityHeaders(NextResponse.redirect(buildRequestUrl(request, "/login?error=csrf"), 303));
    clearSessionCookie(response);
    return response;
  }

  if (session?.id) {
    await destroySession(session.id, "logout");
  }

  emitLogoutRum(LOGOUT_RUM_EVENT_TYPES.SUCCESS);

  const response = applySecurityHeaders(NextResponse.redirect(buildRequestUrl(request, "/login"), 303));
  clearSessionCookie(response);
  return response;
}
