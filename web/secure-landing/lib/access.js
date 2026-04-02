import { verifyAccessJwt } from "./access-jwt.js";
import { getConfig, normalizeAccessEmail } from "./config.js";

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
