import { NextResponse } from "next/server.js";

import { resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../lib/access.js";
import { escapeHtml, FRONTDOOR_ASSETS, renderBrandAsset } from "../../lib/brand.js";
import { applySecurityHeaders, buildRequestUrl, LOGIN_CSP } from "../../lib/http.js";
import { copyUpstreamResponseHeaders } from "../../lib/proxy.js";
import { applyPortalReturnTo, currentPortalReturnToFromRequest } from "../../lib/return-to.js";
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

function resolveManagedRecoveryContent(reason, message) {
  const detail = String(message || "").trim();
  if (reason === MANAGED_FAILURE_REASON.ACCESS_OUTAGE) {
    return {
      title: "Managed entry is waiting on access recovery.",
      label: "Retry when Access recovers",
      detail: detail || "Managed access verification is temporarily unavailable.",
      nextStep: "Refresh the verified access session, then return to the portal once Access validation responds again."
    };
  }
  if (reason === MANAGED_FAILURE_REASON.CONFIG_FAILURE) {
    return {
      title: "Managed entry is blocked by configuration.",
      label: "Configuration required",
      detail: detail || "Managed front door configuration is unavailable.",
      nextStep: "Restore the managed front door configuration before retrying portal entry."
    };
  }
  return {
    title: "Portal handoff is waiting on backend recovery.",
    label: "Backend recovery",
    detail: detail || "The upstream portal service is temporarily unavailable.",
    nextStep: "Wait for the FastAPI operator shell to recover, then retry portal entry from the managed boundary."
  };
}

function renderManagedPortalRecoveryPage({ reason, message, loginHref }) {
  const resolvedReason = reason || MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE;
  const content = resolveManagedRecoveryContent(resolvedReason, message);
  const safeReason = escapeHtml(resolvedReason);
  const safeLabel = escapeHtml(content.label);
  const safeTitle = escapeHtml(content.title);
  const safeDetail = escapeHtml(content.detail);
  const safeNextStep = escapeHtml(content.nextStep);
  const recoveryTone = resolvedReason === MANAGED_FAILURE_REASON.ACCESS_OUTAGE ? "waiting" : "blocked";

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dynamic Neural Access | Managed Recovery</title>
    <link rel="stylesheet" href="/login.css" />
  </head>
  <body>
    <a class="skip-link" href="#main-content">Skip to recovery</a>
    <main class="shell">
      <video
        class="hero-video"
        autoplay
        muted
        loop
        playsinline
        preload="metadata"
        disablePictureInPicture
        disableRemotePlayback
        poster=""
        aria-hidden="true"
      >
        <source src="${FRONTDOOR_ASSETS.loopVideo}" type="video/mp4" />
      </video>
      <div class="login-vignette" aria-hidden="true"></div>
      <section id="main-content" class="content" tabindex="-1" data-ui="managed-recovery-shell" data-reason="${safeReason}">
        <div class="login-stage">
          <a class="brand-lockup brand-lockup--stacked" href="/" aria-label="Dynamic Neural Access home">
            <span class="brand-asset-frame brand-asset-frame--login">
              ${renderBrandAsset({
                kind: "lockup",
                variant: "dark",
                alt: "Dynamic Neural Access",
                className: "brand-asset"
              })}
            </span>
            <span class="brand-subtitle">Managed portal recovery</span>
          </a>
          <div class="card card--login" data-ui="managed-recovery-card">
            <p class="eyebrow">Managed portal entry</p>
            <h1>${safeTitle}</h1>
            <p class="lede">The managed boundary stays closed until the blocking condition is resolved; dispatch, artifacts, archive gates, and optional runtime controls remain fail-closed.</p>
            <div class="login-entry-state">
              <article class="login-status-card login-status-card--full" data-state="${recoveryTone}">
                <p class="login-status-card-kicker">Current state</p>
                <p class="login-status-card-title">${safeLabel}</p>
                <p class="login-status-card-detail">${safeDetail}</p>
              </article>
            </div>
            <div class="login-next-step" data-state="${recoveryTone}">
              <p class="login-next-step-kicker">Next step</p>
              <p class="login-next-step-title">Recover, then retry</p>
              <p class="login-next-step-detail">${safeNextStep}</p>
            </div>
            <div class="login-status-stack">
              <div class="login-recovery-card" data-ui="managed-recovery-capabilities" data-state="${safeReason}">
                <p class="login-recovery-card-title">Capability posture</p>
                <p class="login-recovery-card-detail">Queue, artifact viewer, staged uploads, FastVLM sidecars, archive gates, and run-card proof controls stay read-only or unavailable until managed recovery succeeds.</p>
              </div>
              <div class="login-recovery-card" data-ui="managed-recovery-guidance" data-state="${safeReason}">
                <p class="login-recovery-card-title">Managed boundary stays fail-closed</p>
                <p class="login-recovery-card-detail">Browser-side API key entry remains unavailable in managed mode while portal recovery is pending.</p>
              </div>
            </div>
            <div class="login-actions" data-ui="managed-recovery-actions">
              <a class="login-secondary-link" href="${escapeHtml(loginHref)}">Return to login</a>
              <a class="login-secondary-link" href="/">Review public proof surface</a>
            </div>
          </div>
        </div>
      </section>
    </main>
  </body>
</html>`;
}

export async function GET(request) {
  const currentReturnTo = currentPortalReturnToFromRequest(request) || "/portal";
  const loginUrl = buildRequestUrl(request, "/login");
  applyPortalReturnTo(loginUrl, currentReturnTo);
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
      headers.set("Content-Type", "text/html; charset=utf-8");
      return applySecurityHeaders(
        new Response(renderManagedPortalRecoveryPage({
          reason,
          message: getManagedFailureMessage("portal", reason),
          loginHref: `${loginUrl.pathname}${loginUrl.search}`
        }), {
          status: 503,
          headers
        }),
        { csp: LOGIN_CSP }
      );
    }

    if (authState.revokeSession) {
      await revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }

    const response = applySecurityHeaders(NextResponse.redirect(loginUrl, 302));
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
    headers.set("Content-Type", "text/html; charset=utf-8");
    return applySecurityHeaders(
        new Response(renderManagedPortalRecoveryPage({
          reason: MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE,
          message: getManagedFailureMessage("portal", MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE),
          loginHref: `${loginUrl.pathname}${loginUrl.search}`
        }), {
          status: 503,
          headers
      }),
      { csp: LOGIN_CSP }
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
      headers.set("Content-Type", "text/html; charset=utf-8");
      return applySecurityHeaders(
        new Response(renderManagedPortalRecoveryPage({
          reason,
          message: getManagedFailureMessage("portal", reason),
          loginHref: `${loginUrl.pathname}${loginUrl.search}`
        }), {
          status: 503,
          headers
        }),
        { csp: LOGIN_CSP }
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
