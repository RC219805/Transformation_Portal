import { createPublicKey, verify as verifySignature } from "node:crypto";

import { normalizeAccessEmail, normalizeAccessTeamDomain } from "./config.js";

const ACCESS_CERT_CACHE_TTL_MS = 5 * 60 * 1000;
const ACCESS_CERT_CACHE = new Map();

class AccessJwtError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "AccessJwtError";
    this.code = code;
  }
}

function decodeBase64UrlJson(segment, code) {
  try {
    return JSON.parse(Buffer.from(String(segment || ""), "base64url").toString("utf-8"));
  } catch {
    throw new AccessJwtError(code, "Malformed Access JWT segment.");
  }
}

function parseAccessJwt(token) {
  const rawToken = String(token || "").trim();
  const segments = rawToken.split(".");
  if (segments.length !== 3) {
    throw new AccessJwtError("invalid_assertion", "Malformed Access JWT.");
  }

  const [encodedHeader, encodedPayload, encodedSignature] = segments;
  const header = decodeBase64UrlJson(encodedHeader, "invalid_header");
  const payload = decodeBase64UrlJson(encodedPayload, "invalid_payload");

  return {
    header,
    payload,
    signature: Buffer.from(encodedSignature, "base64url"),
    signingInput: Buffer.from(`${encodedHeader}.${encodedPayload}`, "utf-8")
  };
}

function resolveAccessSignatureAlgorithm(alg) {
  if (alg === "RS256") return "RSA-SHA256";
  if (alg === "RS384") return "RSA-SHA384";
  if (alg === "RS512") return "RSA-SHA512";
  throw new AccessJwtError("unsupported_algorithm", `Unsupported Access JWT algorithm: ${String(alg || "")}`);
}

function buildAccessCertsUrl(teamDomain) {
  return `${normalizeAccessTeamDomain(teamDomain)}/cdn-cgi/access/certs`;
}

async function fetchAccessCerts(teamDomain, { forceRefresh = false } = {}) {
  const certsUrl = buildAccessCertsUrl(teamDomain);
  const cached = ACCESS_CERT_CACHE.get(certsUrl);
  if (!forceRefresh && cached && cached.expiresAt > Date.now()) {
    return cached.keys;
  }

  let response;
  try {
    response = await fetch(certsUrl, {
      headers: { Accept: "application/json" },
      cache: "no-store"
    });
  } catch (error) {
    throw new AccessJwtError("jwks_unreachable", error instanceof Error ? error.message : "Access cert fetch failed.");
  }

  if (!response.ok) {
    throw new AccessJwtError("jwks_unreachable", `Access cert fetch failed with status ${response.status}.`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new AccessJwtError("jwks_invalid", "Access cert response was not valid JSON.");
  }

  const keys = Array.isArray(payload?.keys) ? payload.keys : [];
  if (!keys.length) {
    throw new AccessJwtError("jwks_invalid", "Access cert response did not contain signing keys.");
  }

  ACCESS_CERT_CACHE.set(certsUrl, {
    expiresAt: Date.now() + ACCESS_CERT_CACHE_TTL_MS,
    keys
  });

  return keys;
}

function verifyWithJwk({ alg, signingInput, signature, jwk }) {
  const key = createPublicKey({ key: jwk, format: "jwk" });
  return verifySignature(resolveAccessSignatureAlgorithm(alg), signingInput, key, signature);
}

function normalizeAccessIssuer(value) {
  return normalizeAccessTeamDomain(value);
}

function validateAccessClaims(payload, { expectedIssuer, expectedAudience }) {
  const issuer = normalizeAccessIssuer(payload?.iss);
  if (!issuer || issuer !== expectedIssuer) {
    throw new AccessJwtError("invalid_issuer", "Access JWT issuer did not match the configured team domain.");
  }

  const audiences = Array.isArray(payload?.aud) ? payload.aud : [payload?.aud];
  if (!audiences.includes(expectedAudience)) {
    throw new AccessJwtError("invalid_audience", "Access JWT audience did not match the configured application audience.");
  }

  const now = Math.floor(Date.now() / 1000);
  if (typeof payload?.nbf === "number" && payload.nbf > now) {
    throw new AccessJwtError("token_not_yet_valid", "Access JWT is not valid yet.");
  }
  if (typeof payload?.exp !== "number" || payload.exp <= now) {
    throw new AccessJwtError("token_expired", "Access JWT has expired.");
  }

  const email = normalizeAccessEmail(payload?.email);
  if (!email) {
    throw new AccessJwtError("missing_email", "Access JWT did not contain an email claim.");
  }

  return email;
}

export async function verifyAccessJwt(token, { teamDomain, audience }) {
  const expectedIssuer = normalizeAccessTeamDomain(teamDomain);
  const expectedAudience = String(audience || "").trim();

  if (!expectedIssuer || !expectedAudience) {
    throw new AccessJwtError("configuration", "Cloudflare Access verification is not fully configured.");
  }

  const parsed = parseAccessJwt(token);
  let candidateKeys = await fetchAccessCerts(expectedIssuer);
  let matchingKeys = parsed.header?.kid
    ? candidateKeys.filter((candidate) => candidate?.kid === parsed.header.kid)
    : candidateKeys;

  if (parsed.header?.kid && !matchingKeys.length) {
    candidateKeys = await fetchAccessCerts(expectedIssuer, { forceRefresh: true });
    matchingKeys = candidateKeys.filter((candidate) => candidate?.kid === parsed.header.kid);
  }

  if (!matchingKeys.length) {
    throw new AccessJwtError("unknown_key", "Access JWT signing key did not match the published certs.");
  }

  const signatureValid = matchingKeys.some((candidate) => {
    try {
      return verifyWithJwk({
        alg: parsed.header?.alg,
        signingInput: parsed.signingInput,
        signature: parsed.signature,
        jwk: candidate
      });
    } catch {
      return false;
    }
  });

  if (!signatureValid) {
    throw new AccessJwtError("invalid_signature", "Access JWT signature verification failed.");
  }

  const accessEmail = validateAccessClaims(parsed.payload, {
    expectedIssuer,
    expectedAudience
  });

  return {
    accessEmail,
    issuer: expectedIssuer,
    audience: expectedAudience,
    header: parsed.header,
    payload: parsed.payload
  };
}
