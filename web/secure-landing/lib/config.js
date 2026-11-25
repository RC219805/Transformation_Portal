import { readFileSync } from "node:fs";

import { stableRolloutBucket } from "./rollout.js";

const DEFAULT_SESSION_DB_PATH = "/tmp/transformation-portal-frontdoor-sessions.db";
const DEFAULT_SESSION_SCALING_MODE = "single_instance";
const DEFAULT_SESSION_STORE_BACKEND = "sqlite";
const DEFAULT_FRONTDOOR_RUM_ROLLOUT_PERCENT = 100;
const SESSION_IDLE_TIMEOUT_MS = 8 * 60 * 60 * 1000;
const SESSION_ABSOLUTE_TIMEOUT_MS = 24 * 60 * 60 * 1000;
const LOGIN_WINDOW_MS = 15 * 60 * 1000;
const LOGIN_ATTEMPT_LIMIT = 5;

export function isLocalAccessBypassEnabled(env = process.env) {
  return (env.NODE_ENV || "development") === "development" && env.TP_ALLOW_LOCAL_ACCESS_BYPASS === "1";
}

export function isTruthyEnvFlag(value) {
  return ["1", "true", "yes", "on"].includes(String(value || "").trim().toLowerCase());
}

export function isPortalRumEnabled(env = process.env) {
  return isTruthyEnvFlag(env.TP_PORTAL_RUM_ENABLED);
}

function parseFrontdoorRumRolloutPercent(rawValue) {
  const raw = String(rawValue ?? "").trim();
  if (!raw) {
    return DEFAULT_FRONTDOOR_RUM_ROLLOUT_PERCENT;
  }
  if (!/^\d+$/u.test(raw)) {
    return 0;
  }
  const parsed = Number(raw);
  if (!Number.isSafeInteger(parsed)) {
    return 0;
  }
  return Math.max(0, Math.min(100, parsed));
}

export function resolveFrontdoorRumRolloutPercent(env = process.env) {
  return parseFrontdoorRumRolloutPercent(env.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT);
}

export function shouldDisableFrontdoorRumHomepageCache(env = process.env) {
  if (!isPortalRumEnabled(env)) {
    return false;
  }
  if (!isTruthyEnvFlag(env.TP_FRONTDOOR_RUM_ENABLED)) {
    return false;
  }
  return resolveFrontdoorRumRolloutPercent(env) > 0;
}

export function isFrontdoorRumTelemetryEnabled({ traceparent = "", env = process.env } = {}) {
  if (!isPortalRumEnabled(env)) {
    return false;
  }
  if (!isTruthyEnvFlag(env.TP_FRONTDOOR_RUM_ENABLED)) {
    return false;
  }
  const rolloutPercent = resolveFrontdoorRumRolloutPercent(env);
  if (rolloutPercent <= 0) {
    return false;
  }
  if (rolloutPercent >= 100) {
    return true;
  }
  return stableRolloutBucket(traceparent) < rolloutPercent;
}

export function normalizeUsername(value) {
  return String(value || "").trim().toLowerCase();
}

export function normalizeAccessEmail(value) {
  return String(value || "").trim().toLowerCase();
}

export function normalizeAccessTeamDomain(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";

  const normalized = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;

  try {
    const url = new URL(normalized);
    if (url.protocol !== "https:") {
      return "";
    }
    url.pathname = "";
    url.search = "";
    url.hash = "";
    return `${url.protocol}//${url.host}`;
  } catch {
    return "";
  }
}

// TP_FRONTDOOR_PUBLIC_ORIGIN names the one origin browsers reach this front
// door on. The accepted form is exactly what a browser serialises in the
// Origin header, so nothing is normalised: a value that would need it is
// rejected. See "Trusted Host And Proxy Trust" in
// docs/architecture/PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md.
export const PUBLIC_ORIGIN_INVALID_REASON = "invalid_public_origin";
export const PUBLIC_ORIGIN_BYPASS_CONFLICT_REASON = "public_origin_conflicts_with_local_access_bypass";

const PUBLIC_ORIGIN_MAX_HOST_LENGTH = 253;
const PUBLIC_ORIGIN_MAX_PORT = 65535;
const PUBLIC_ORIGIN_DEFAULT_PORT = 443;
// https, a lower-case DNS name of at least two labels whose last label starts
// with a letter (so no IP literal), and an optional port without leading zero.
const PUBLIC_ORIGIN_PATTERN =
  /^https:\/\/((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)(?::([1-9][0-9]{0,4}))?$/;

export function isCanonicalPublicOrigin(value) {
  if (typeof value !== "string") return false;

  const match = PUBLIC_ORIGIN_PATTERN.exec(value);
  if (!match) return false;

  const [, host, port] = match;
  if (host.length > PUBLIC_ORIGIN_MAX_HOST_LENGTH || host.endsWith(".localhost")) return false;
  if (port !== undefined) {
    const portNumber = Number(port);
    if (portNumber > PUBLIC_ORIGIN_MAX_PORT || portNumber === PUBLIC_ORIGIN_DEFAULT_PORT) return false;
  }

  // The grammar already implies this; the URL parser is kept as a second opinion.
  try {
    return new URL(value).origin === value;
  } catch {
    return false;
  }
}

// Never throws. An origin is returned only when it is in force; `reason` says
// why a configured value is not honoured, and readiness reports it.
export function resolvePublicOrigin(env = process.env) {
  const raw = String(env.TP_FRONTDOOR_PUBLIC_ORIGIN ?? "").replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "");
  if (!raw) {
    return { configured: false, origin: "", reason: null };
  }
  if (!isCanonicalPublicOrigin(raw)) {
    return { configured: true, origin: "", reason: PUBLIC_ORIGIN_INVALID_REASON };
  }
  // The bypass is for non-public local development only, so a bypass-mode
  // front door must stay unusable from a public origin.
  if (isLocalAccessBypassEnabled(env)) {
    return { configured: true, origin: "", reason: PUBLIC_ORIGIN_BYPASS_CONFLICT_REASON };
  }
  return { configured: true, origin: raw, reason: null };
}

function parseUsersJson(raw) {
  if (!raw) return [];

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }

  if (!Array.isArray(parsed)) return [];

  return parsed
    .map((item) => ({
      username: normalizeUsername(item?.username),
      passwordHash: String(item?.password_hash || ""),
      accessEmail: normalizeAccessEmail(item?.access_email),
      role: String(item?.role || "admin").trim() || "admin"
    }))
    .filter((item) => item.username && item.passwordHash && item.accessEmail);
}

function parseUsersFile(filePath) {
  if (!filePath) return [];

  try {
    const raw = readFileSync(String(filePath), "utf-8");
    return parseUsersJson(raw);
  } catch {
    return [];
  }
}

export function getConfig() {
  const nodeEnv = process.env.NODE_ENV || "development";
  const isProduction = nodeEnv === "production";
  const allowLocalAccessBypass = isLocalAccessBypassEnabled();
  const usersFilePath = String(process.env.TP_FRONTDOOR_USERS_FILE || "").trim();
  const users = usersFilePath
    ? parseUsersFile(usersFilePath)
    : parseUsersJson(process.env.TP_FRONTDOOR_USERS_JSON || "[]");

  return {
    nodeEnv,
    isProduction,
    fastapiOrigin: String(process.env.TP_FASTAPI_ORIGIN || "http://127.0.0.1:8000").trim(),
    backendApiKey: String(process.env.TP_BACKEND_API_KEY || "").trim(),
    pilotControlPlaneEnabled: isTruthyEnvFlag(process.env.TP_PILOT_CONTROL_PLANE_ENABLED),
    pilotTenantHeader: String(process.env.TP_PILOT_TENANT_HEADER || "x-tp-tenant-id").trim().toLowerCase(),
    frontdoorIdentitySecret: String(process.env.TP_FRONTDOOR_IDENTITY_SECRET || "").trim(),
    cfAccessTeamDomain: normalizeAccessTeamDomain(process.env.TP_CF_ACCESS_TEAM_DOMAIN),
    cfAccessAud: String(process.env.TP_CF_ACCESS_AUD || "").trim(),
    sessionDbPath: String(process.env.TP_FRONTDOOR_SESSION_DB || DEFAULT_SESSION_DB_PATH).trim(),
    sessionScalingMode: String(
      process.env.TP_FRONTDOOR_SESSION_SCALING_MODE || DEFAULT_SESSION_SCALING_MODE
    ).trim(),
    // Phase 3 - session-store backend selector. ``sqlite`` keeps the
    // existing ``better-sqlite3`` path (default). ``redis`` activates
    // the cross-instance store backed by TP_FRONTDOOR_REDIS_URL and
    // flips ``evaluateSessionScaling`` to ``ok: true`` for
    // ``multi_instance`` / ``ephemeral_runtime`` deployments.
    sessionStoreBackend: String(
      process.env.TP_FRONTDOOR_SESSION_STORE || DEFAULT_SESSION_STORE_BACKEND
    )
      .trim()
      .toLowerCase(),
    sessionStoreRedisUrl: String(process.env.TP_FRONTDOOR_REDIS_URL || "").trim(),
    sessionStoreRedisKeyPrefix: String(
      process.env.TP_FRONTDOOR_REDIS_KEY_PREFIX || "tp:frontdoor:"
    ).trim(),
    users,
    usersFilePath,
    sessionCookieName: isProduction ? "__Host-tp_session" : "tp_session",
    sessionCookieSecure: isProduction,
    sessionIdleTimeoutMs: SESSION_IDLE_TIMEOUT_MS,
    sessionAbsoluteTimeoutMs: SESSION_ABSOLUTE_TIMEOUT_MS,
    loginWindowMs: LOGIN_WINDOW_MS,
    loginAttemptLimit: LOGIN_ATTEMPT_LIMIT,
    allowLocalAccessBypass
  };
}
