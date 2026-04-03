import { NextResponse } from "next/server.js";

import { resolveAccessContext, resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../lib/access.js";
import { audit } from "../../lib/audit.js";
import { applySecurityHeaders, LOGIN_CSP } from "../../lib/http.js";
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

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function resolveLoginMessage(code) {
  if (code === "access") return "Cloudflare Access identity is required before login can succeed.";
  if (code === "csrf") return "Your session could not be verified. Refresh and try again.";
  if (code === "throttled") return "Too many login attempts. Wait a few minutes and try again.";
  if (code === "configuration") return "The front door is not fully configured yet.";
  return "Invalid username or password.";
}

function renderLoginPage({ csrfToken, accessEmail, errorCode, allowLocalBypass }) {
  const errorMessage = errorCode ? resolveLoginMessage(errorCode) : "";
  const accessText = accessEmail
    ? `Cloudflare Access identity detected for <strong>${escapeHtml(accessEmail)}</strong>.`
    : allowLocalBypass
      ? "Local development bypass is enabled. Username and password are still required."
      : "Cloudflare Access identity is required before credentials can be accepted.";

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Transformation Portal Login</title>
    <link rel="stylesheet" href="/login.css" />
  </head>
  <body>
    <main class="shell">
      <video
        class="hero-video"
        autoplay
        muted
        loop
        playsinline
        preload="auto"
        disablePictureInPicture
        disableRemotePlayback
        poster=""
        aria-hidden="true"
      >
        <source src="/video/login-loop.mp4" type="video/mp4" />
      </video>
      <section class="content">
        <div class="card">
          <p class="eyebrow">Transformation Portal</p>
          <h1>Operator Login</h1>
          <p class="lede">Authenticate to the front door. The browser never receives the backend orchestration API key.</p>
          ${errorMessage ? `<div class="banner" role="alert">${escapeHtml(errorMessage)}</div>` : ""}
          <div class="meta">${accessText}</div>
          <form method="post" action="/login" autocomplete="on">
            <input type="hidden" name="csrf_token" value="${escapeHtml(csrfToken)}" />
            <label>
              Username
              <input type="text" name="username" autocomplete="username" required />
            </label>
            <label>
              Password
              <input type="password" name="password" autocomplete="current-password" required />
            </label>
            <button type="submit">Sign in</button>
          </form>
          <p class="footnote">Production / Cloudflare Access env vars: <code>TP_FASTAPI_ORIGIN</code>, <code>TP_BACKEND_API_KEY</code>, <code>TP_FRONTDOOR_USERS_FILE</code>, <code>TP_FRONTDOOR_SESSION_DB</code>, <code>TP_CF_ACCESS_TEAM_DOMAIN</code>, <code>TP_CF_ACCESS_AUD</code>.</p>
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

  session = session || createAnonymousSession();
  const accessContext = await resolveAccessContext(request);
  const html = renderLoginPage({
    csrfToken: session.csrfToken,
    accessEmail: accessContext.accessEmail,
    errorCode: request.nextUrl.searchParams.get("error"),
    allowLocalBypass: accessContext.bypass
  });
  const response = new NextResponse(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store"
    }
  });
  setSessionCookie(response, session.id);
  return applySecurityHeaders(response, { csp: LOGIN_CSP });
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
