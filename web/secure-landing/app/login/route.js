import { NextResponse } from "next/server.js";

import { resolveAccessContext, resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../lib/access.js";
import { escapeHtml, FRONTDOOR_ASSETS, renderBrandAsset } from "../../lib/brand.js";
import { audit } from "../../lib/audit.js";
import { applySecurityHeaders, FRONTDOOR_CSP } from "../../lib/http.js";
import {
  clearSessionCookie,
  createAnonymousSession,
  destroySession,
  getRemoteAddress,
  getSessionFromRequest,
  isLoginThrottled,
  recordLoginAttempt,
  rotateAuthenticatedSession,
  setSessionCookie,
  validateCsrfToken
} from "../../lib/sessions.js";
import { getConfig } from "../../lib/config.js";
import { validateOriginAndReferrer } from "../../lib/request-security.js";
import { verifyUserCredentials } from "../../lib/users.js";

export const runtime = "nodejs";

function resolveLoginMessage(code) {
  if (code === "access") return "Access verification is required before sign-in can continue. Refresh your Access session and try again.";
  if (code === "csrf") return "Your session could not be verified. Refresh the page and submit the form again.";
  if (code === "throttled") return "Too many sign-in attempts. Wait a few minutes before trying again.";
  if (code === "configuration") return "Operator access is temporarily unavailable. Contact an administrator if this persists.";
  return "Invalid username or password.";
}

function resolveRecoveryGuidance(code) {
  if (code === "access") return "Re-open the verified access path, then return here after Access identity is restored.";
  if (code === "csrf") return "Refresh the page to mint a clean session, then retry the operator credential handoff.";
  if (code === "throttled") return "Wait for the throttle window to clear before attempting the operator credential handoff again.";
  if (code === "configuration") return "Managed entry is fail-closed until the front door configuration is restored.";
  return "Re-enter operator credentials after verifying the managed access context above.";
}

function resolveEntryState({ accessEmail, errorCode }) {
  const hasVerifiedAccess = Boolean(accessEmail);
  const hasRecoveryIssue = Boolean(errorCode);
  return {
    accessState: hasVerifiedAccess ? "verified" : "required",
    credentialState: hasRecoveryIssue ? "blocked" : hasVerifiedAccess ? "ready" : "waiting",
    accessLabel: hasVerifiedAccess ? "Verified access ready" : "Managed access required",
    accessDetail: hasVerifiedAccess
      ? `Cloudflare Access is already verified for <strong>${escapeHtml(accessEmail)}</strong>.`
      : "Managed access verification completes before operator credential handoff opens.",
    credentialLabel: hasRecoveryIssue ? "Recovery required" : hasVerifiedAccess ? "Credential handoff ready" : "Waiting on verified access",
    credentialDetail: hasRecoveryIssue
      ? escapeHtml(resolveRecoveryGuidance(errorCode))
      : hasVerifiedAccess
        ? "Continue with operator credentials to rotate into the governed portal session."
        : "Operator credential handoff stays blocked until the verified access context is present.",
    recoveryState: hasRecoveryIssue ? String(errorCode || "").trim().toLowerCase() || "invalid" : "clear",
  };
}

function renderLoginPage({ csrfToken, accessEmail, errorCode }) {
  const errorMessage = errorCode ? resolveLoginMessage(errorCode) : "";
  const entryState = resolveEntryState({ accessEmail, errorCode });
  const escapedAccessEmail = accessEmail ? escapeHtml(accessEmail) : "";
  const accessSequenceDetail = accessEmail
    ? `Managed access already verified for <strong>${escapedAccessEmail}</strong>.`
    : "Managed access is verified before operator credentials are accepted.";
  const accessText = accessEmail
    ? `Access identity verified for <strong>${escapedAccessEmail}</strong>. Credential entry is now available.`
    : "";
  const recoveryCard = errorMessage
    ? {
      title: "Recovery path",
      detail: resolveRecoveryGuidance(errorCode)
    }
    : accessEmail
      ? {
        title: "Verified access context",
        detail: `Managed access has already been verified for ${accessEmail}. Successful sign-in rotates this browser into the governed portal session.`
      }
      : null;

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dynamic Neural Access | Transformation Portal</title>
    <link rel="stylesheet" href="/login.css" />
  </head>
  <body>
    <a class="skip-link" href="#main-content">Skip to sign-in</a>
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
      <section
        id="main-content"
        class="content"
        tabindex="-1"
        data-ui="login-shell"
        data-access-state="${escapeHtml(entryState.accessState)}"
        data-credential-state="${escapeHtml(entryState.credentialState)}"
        data-recovery-state="${escapeHtml(entryState.recoveryState)}"
      >
        <div class="login-stage" data-ui="login-stage">
          <a class="brand-lockup brand-lockup--stacked" href="/" aria-label="Dynamic Neural Access home">
            <span class="brand-asset-frame brand-asset-frame--login">
              ${renderBrandAsset({
                kind: "lockup",
                variant: "dark",
                alt: "Dynamic Neural Access",
                className: "brand-asset"
              })}
            </span>
            <span class="brand-subtitle">Transformation Portal operator console</span>
          </a>
          <div class="card card--login" data-ui="login-card">
            <p class="eyebrow" data-ui="login-eyebrow">Managed operator access</p>
            <h1 data-ui="login-title">Continue to the operator console.</h1>
            <p class="lede" data-ui="login-lede">
              Managed access and operator credentials stay separate on purpose. Confirm the verified access context first, then complete credential handoff into the governed console.
            </p>
            <div class="login-entry-state" data-ui="login-entry-state">
              <article class="login-status-card" data-ui="login-access-status" data-state="${escapeHtml(entryState.accessState)}">
                <p class="login-status-card-kicker">Verified access</p>
                <p class="login-status-card-title">${entryState.accessLabel}</p>
                <p class="login-status-card-detail">${entryState.accessDetail}</p>
              </article>
              <article class="login-status-card" data-ui="login-credential-status" data-state="${escapeHtml(entryState.credentialState)}">
                <p class="login-status-card-kicker">Credential handoff</p>
                <p class="login-status-card-title">${entryState.credentialLabel}</p>
                <p class="login-status-card-detail">${entryState.credentialDetail}</p>
              </article>
            </div>
            <div class="login-sequence" data-ui="login-sequence">
              <article class="login-sequence-step${accessEmail ? " login-sequence-step--ready" : ""}">
                <p class="login-sequence-step-kicker">Step 1</p>
                <p class="login-sequence-step-title">Verified access</p>
                <p class="login-sequence-step-detail">${accessSequenceDetail}</p>
              </article>
              <article class="login-sequence-step login-sequence-step--active">
                <p class="login-sequence-step-kicker">Step 2</p>
                <p class="login-sequence-step-title">Operator credentials</p>
                <p class="login-sequence-step-detail">Use your portal username and password to rotate into the governed build, operate, and review session.</p>
              </article>
            </div>
            ${errorMessage || accessText ? `<div class="login-status-stack" data-ui="login-status-stack">
              ${errorMessage ? `<div class="banner" data-ui="login-error-banner" role="alert">
                <p class="banner-title">Sign-in needs attention</p>
                <p class="banner-detail">${escapeHtml(errorMessage)}</p>
              </div>` : ""}
              ${accessText ? `<p class="card-meta card-meta--verified" data-ui="login-access-context">${accessText}</p>` : ""}
              ${recoveryCard ? `<div class="login-recovery-card" data-ui="login-recovery-card" data-state="${escapeHtml(entryState.recoveryState)}">
                <p class="login-recovery-card-title">${escapeHtml(recoveryCard.title)}</p>
                <p class="login-recovery-card-detail">${escapeHtml(recoveryCard.detail)}</p>
              </div>` : ""}
            </div>` : ""}
            <form method="post" action="/login" autocomplete="on" data-ui="login-form">
              <input type="hidden" name="csrf_token" value="${escapeHtml(csrfToken)}" />
              <label data-ui="login-username-field">
                Username
                <input type="text" name="username" autocomplete="username" required />
              </label>
              <label data-ui="login-password-field">
                Password
                <input type="password" name="password" autocomplete="current-password" required />
              </label>
              <p class="login-helper" data-ui="login-helper">
                Use your operator credentials. Successful sign-in rotates the session before handoff to the governed portal.
              </p>
              <div class="login-actions" data-ui="login-actions">
                <button type="submit" data-ui="login-submit">Sign in</button>
                <a class="login-secondary-link" href="/" data-ui="login-secondary-link">Review public proof surface</a>
              </div>
            </form>
          </div>
        </div>
      </section>
    </main>
  </body>
</html>`;
}

function redirectToLogin(request, errorCode, session) {
  const url = new URL("/login", request.url);
  if (errorCode) url.searchParams.set("error", errorCode);
  const response = applySecurityHeaders(NextResponse.redirect(url, 303));
  if (session?.id) {
    setSessionCookie(response, session.id);
  }
  return response;
}

export async function GET(request) {
  const currentSession = getSessionFromRequest(request, { touch: false });
  let session = currentSession;
  if (currentSession?.authenticated) {
    const authState = await resolveAuthenticatedAccessSession(request, { touch: false });
    if (authState.ok) {
      return applySecurityHeaders(NextResponse.redirect(new URL("/portal", request.url), 302));
    }
    if (authState.revokeSession) {
      revokeSessionOnAccessFailure(currentSession, authState.errorCode);
      session = null;
    }
  }

  // Mint an anonymous session before credentials are posted so the hidden CSRF
  // token on the login form is bound to a server-side session from the start.
  session = session || createAnonymousSession();
  const accessContext = await resolveAccessContext(request);
  const html = renderLoginPage({
    csrfToken: session.csrfToken,
    accessEmail: accessContext.accessEmail,
    errorCode: request.nextUrl.searchParams.get("error")
  });
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store"
    }
  });
  setSessionCookie(response, session.id);
  return applySecurityHeaders(response, { csp: FRONTDOOR_CSP });
}

export async function POST(request) {
  let session = getSessionFromRequest(request, { touch: false });
  if (!session) {
    session = createAnonymousSession();
  }

  if (!validateOriginAndReferrer(request)) {
    audit("csrf_failure", {
      path: "/login",
      remoteAddr: getRemoteAddress(request)
    });
    return redirectToLogin(request, "csrf", session);
  }

  const formData = await request.formData();
  const csrfToken = String(formData.get("csrf_token") || "");
  if (!validateCsrfToken(session, csrfToken)) {
    audit("csrf_failure", {
      path: "/login",
      remoteAddr: getRemoteAddress(request)
    });
    return redirectToLogin(request, "csrf", session);
  }

  const config = getConfig();
  if (!config.users.length) {
    return redirectToLogin(request, "configuration", session);
  }

  const accessContext = await resolveAccessContext(request);
  if (accessContext.errorCode === "configuration") {
    return redirectToLogin(request, "configuration", session);
  }
  if (!accessContext.accessEmail && !accessContext.bypass) {
    audit("access_validation_failure", {
      path: "/login",
      remoteAddr: getRemoteAddress(request),
      assertedEmail: accessContext.assertedEmail,
      errorCode: accessContext.errorCode
    });
    return redirectToLogin(request, "access", session);
  }

  const username = String(formData.get("username") || "").trim();
  const password = String(formData.get("password") || "");
  const remoteAddr = getRemoteAddress(request);
  const throttleKey = `${accessContext.accessEmail || "local"}:${username.toLowerCase()}:${remoteAddr}`;

  if (isLoginThrottled(throttleKey)) {
    audit("login_throttle", {
      username: username.toLowerCase(),
      accessEmail: accessContext.accessEmail,
      remoteAddr
    });
    return redirectToLogin(request, "throttled", session);
  }

  const user = await verifyUserCredentials({
    username,
    password,
    accessEmail: accessContext.accessEmail,
    allowAccessBypass: accessContext.bypass
  });
  recordLoginAttempt({
    throttleKey,
    success: Boolean(user),
    remoteAddr
  });

  if (!user) {
    audit("login_failure", {
      username: username.toLowerCase(),
      accessEmail: accessContext.accessEmail,
      remoteAddr,
      bypass: accessContext.bypass
    });
    return redirectToLogin(request, "invalid", session);
  }

  const authenticatedSession = rotateAuthenticatedSession(session, user);
  if (session?.id) {
    destroySession(session.id, "rotation_cleanup");
  }

  audit("login_success", {
    username: user.username,
    accessEmail: user.accessEmail,
    role: user.role,
    remoteAddr,
    bypass: accessContext.bypass
  });

  const response = applySecurityHeaders(NextResponse.redirect(new URL("/portal", request.url), 303));
  setSessionCookie(response, authenticatedSession.id);
  return response;
}
