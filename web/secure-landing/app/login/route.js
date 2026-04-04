import { NextResponse } from "next/server.js";

import { resolveAccessContext, resolveAuthenticatedAccessSession, revokeSessionOnAccessFailure } from "../../lib/access.js";
import { escapeHtml, renderLanternMark } from "../../lib/brand.js";
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

function resolveLoginMessage(code) {
  if (code === "access") return "Access verification is required before sign-in can continue.";
  if (code === "csrf") return "Your session could not be verified. Refresh and try again.";
  if (code === "throttled") return "Too many login attempts. Wait a few minutes and try again.";
  if (code === "configuration") return "Operator access is temporarily unavailable.";
  return "Invalid username or password.";
}

function renderLoginPage({ csrfToken, accessEmail, errorCode }) {
  const errorMessage = errorCode ? resolveLoginMessage(errorCode) : "";
  const accessText = accessEmail
    ? `Access identity verified for <strong>${escapeHtml(accessEmail)}</strong>.`
    : "Authorized operators only.";

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
        preload="metadata"
        disablePictureInPicture
        disableRemotePlayback
        poster=""
        aria-hidden="true"
      >
        <source src="/video/login-loop.mp4" type="video/mp4" />
      </video>
      <div class="login-vignette" aria-hidden="true"></div>
      <section class="content">
        <div class="login-stage">
          <a class="brand-lockup brand-lockup--centered" href="/" aria-label="Dynamic Neural Access home">
            <span class="brand-mark-shell">${renderLanternMark("Transformation Portal brand mark")}</span>
            <span class="brand-copy brand-copy--centered">
              <span class="brand-kicker">Dynamic Neural Access</span>
              <span class="brand-title">Transformation Portal</span>
            </span>
          </a>
          <div class="card card--login">
          <p class="eyebrow">Transformation Portal</p>
          <h1>Operator Login</h1>
          <p class="lede">Secure operator access to governed orchestration.</p>
          ${errorMessage ? `<div class="banner" role="alert">${escapeHtml(errorMessage)}</div>` : ""}
          <p class="card-meta">${accessText}</p>
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
          <div class="login-footer">
            <p class="footnote">Governed access, protected session, premium review flow.</p>
            <a class="tertiary-link" href="/#final-cta">Need access?</a>
          </div>
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
