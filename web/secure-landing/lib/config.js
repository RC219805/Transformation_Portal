import { readFileSync } from "node:fs";

const DEFAULT_SESSION_DB_PATH = "/tmp/transformation-portal-frontdoor-sessions.db";
const DEFAULT_SESSION_SCALING_MODE = "single_instance";
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
    cfAccessTeamDomain: normalizeAccessTeamDomain(process.env.TP_CF_ACCESS_TEAM_DOMAIN),
    cfAccessAud: String(process.env.TP_CF_ACCESS_AUD || "").trim(),
    sessionDbPath: String(process.env.TP_FRONTDOOR_SESSION_DB || DEFAULT_SESSION_DB_PATH).trim(),
    sessionScalingMode: String(
      process.env.TP_FRONTDOOR_SESSION_SCALING_MODE || DEFAULT_SESSION_SCALING_MODE
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
