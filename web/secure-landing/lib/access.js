import { verifyAccessJwt } from "./access-jwt.js";
import { getConfig, normalizeAccessEmail } from "./config.js";
import { destroySession, getSessionFromRequest } from "./sessions.js";

export async function resolveAccessContext(request) {
  const config = getConfig();
  const assertedEmail = normalizeAccessEmail(
    request.headers.get("cf-access-authenticated-user-email") ||
      request.headers.get("x-access-email")
  );
  const accessJwtAssertion = String(request.headers.get("Cf-Access-Jwt-Assertion") || "").trim();

  if (config.allowLocalAccessBypass) {
    return {
      accessEmail: null,
      assertedEmail,
      bypass: true,
      verified: false,
      errorCode: null
    };
  }

  if (!config.cfAccessTeamDomain || !config.cfAccessAud) {
    return {
      accessEmail: null,
      assertedEmail,
      bypass: false,
      verified: false,
      errorCode: "configuration"
    };
  }

  if (!accessJwtAssertion) {
    return {
      accessEmail: null,
      assertedEmail,
      bypass: false,
      verified: false,
      errorCode: "missing_assertion"
    };
  }

  try {
    const verified = await verifyAccessJwt(accessJwtAssertion, {
      teamDomain: config.cfAccessTeamDomain,
      audience: config.cfAccessAud
    });

    return {
      accessEmail: verified.accessEmail,
      assertedEmail,
      bypass: false,
      verified: true,
      errorCode: null
    };
  } catch (error) {
    return {
      accessEmail: null,
      assertedEmail,
      bypass: false,
      verified: false,
      errorCode: error?.code || "invalid_assertion"
    };
  }
}

function classifyAuthenticatedAccessFailure(errorCode) {
  if (errorCode === "configuration" || errorCode === "jwks_unreachable" || errorCode === "jwks_invalid") {
    return {
      status: 503,
      revokeSession: false
    };
  }

  if (errorCode === "access_mismatch") {
    return {
      status: 403,
      revokeSession: true
    };
  }

  return {
    status: 401,
    revokeSession: true
  };
}

export async function resolveAuthenticatedAccessSession(request, { touch = false } = {}) {
  const session = await getSessionFromRequest(request, { touch });
  if (!session?.authenticated) {
    return {
      ok: false,
      session: null,
      accessContext: null,
      errorCode: "authentication_required",
      status: 401,
      revokeSession: false
    };
  }

  const accessContext = await resolveAccessContext(request);
  if (accessContext.bypass) {
    return {
      ok: true,
      session,
      accessContext,
      errorCode: null,
      status: 200,
      revokeSession: false
    };
  }

  if (!accessContext.verified || !accessContext.accessEmail) {
    const failure = classifyAuthenticatedAccessFailure(accessContext.errorCode);
    return {
      ok: false,
      session,
      accessContext,
      errorCode: accessContext.errorCode || "invalid_assertion",
      ...failure
    };
  }

  const sessionAccessEmail = normalizeAccessEmail(session.accessEmail);
  if (!sessionAccessEmail || sessionAccessEmail !== accessContext.accessEmail) {
    const failure = classifyAuthenticatedAccessFailure("access_mismatch");
    return {
      ok: false,
      session,
      accessContext,
      errorCode: "access_mismatch",
      ...failure
    };
  }

  return {
    ok: true,
    session,
    accessContext,
    errorCode: null,
    status: 200,
    revokeSession: false
  };
}

export async function revokeSessionOnAccessFailure(session, errorCode) {
  if (!session?.id) return;
  await destroySession(session.id, errorCode || "access_validation_failure");
}
