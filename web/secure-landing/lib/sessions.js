import { randomBytes, timingSafeEqual } from "node:crypto";

import { getConfig } from "./config.js";
import { getDb } from "./db.js";
import { audit } from "./audit.js";

function nowMs() {
  return Date.now();
}

function getSessionDb() {
  return getDb(getConfig().sessionDbPath);
}

function generateId() {
  return randomBytes(32).toString("hex");
}

function generateCsrfToken() {
  return randomBytes(24).toString("hex");
}

function rowToSession(row) {
  if (!row) return null;
  return {
    id: row.id,
    createdAt: row.created_at,
    lastSeenAt: row.last_seen_at,
    idleExpiresAt: row.idle_expires_at,
    absoluteExpiresAt: row.absolute_expires_at,
    csrfToken: row.csrf_token,
    authenticated: Boolean(row.authenticated),
    username: row.username || null,
    accessEmail: row.access_email || null,
    role: row.role || null
  };
}

function persistSession(record) {
  const db = getSessionDb();
  db.prepare(
    `
      INSERT INTO sessions (
        id,
        created_at,
        last_seen_at,
        idle_expires_at,
        absolute_expires_at,
        csrf_token,
        authenticated,
        username,
        access_email,
        role,
        rotated_from
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `
  ).run(
    record.id,
    record.createdAt,
    record.lastSeenAt,
    record.idleExpiresAt,
    record.absoluteExpiresAt,
    record.csrfToken,
    record.authenticated ? 1 : 0,
    record.username,
    record.accessEmail,
    record.role,
    record.rotatedFrom || null
  );
  return record;
}

function isExpired(row, currentTime) {
  return row.idle_expires_at <= currentTime || row.absolute_expires_at <= currentTime;
}

function deleteSessionRecord(sessionId) {
  if (!sessionId) return;
  getSessionDb().prepare("DELETE FROM sessions WHERE id = ?").run(sessionId);
}

export function getRemoteAddress(request) {
  const cfIp = String(request.headers.get("cf-connecting-ip") || "").trim();
  if (cfIp) return cfIp;

  const forwardedFor = String(request.headers.get("x-forwarded-for") || "").trim();
  if (!forwardedFor) return "unknown";
  return forwardedFor.split(",")[0].trim() || "unknown";
}

export function createAnonymousSession() {
  const config = getConfig();
  const currentTime = nowMs();
  const record = persistSession({
    id: generateId(),
    createdAt: currentTime,
    lastSeenAt: currentTime,
    idleExpiresAt: currentTime + config.sessionIdleTimeoutMs,
    absoluteExpiresAt: currentTime + config.sessionAbsoluteTimeoutMs,
    csrfToken: generateCsrfToken(),
    authenticated: false,
    username: null,
    accessEmail: null,
    role: null
  });
  audit("session_created", { authenticated: false });
  return record;
}

export function getSessionById(sessionId, { touch = false } = {}) {
  if (!sessionId) return null;
  const db = getSessionDb();
  const row = db.prepare("SELECT * FROM sessions WHERE id = ?").get(sessionId);
  if (!row) return null;

  const currentTime = nowMs();
  if (isExpired(row, currentTime)) {
    deleteSessionRecord(sessionId);
    audit("session_expired", {
      authenticated: Boolean(row.authenticated),
      username: row.username || null,
      reason: row.absolute_expires_at <= currentTime ? "absolute" : "idle"
    });
    return null;
  }

  if (touch) {
    db.prepare(
      `
        UPDATE sessions
        SET last_seen_at = ?, idle_expires_at = ?
        WHERE id = ?
      `
    ).run(currentTime, currentTime + getConfig().sessionIdleTimeoutMs, sessionId);
    row.last_seen_at = currentTime;
    row.idle_expires_at = currentTime + getConfig().sessionIdleTimeoutMs;
  }

  return rowToSession(row);
}

export function getSessionFromRequest(request, { touch = false } = {}) {
  const sessionId = request.cookies.get(getConfig().sessionCookieName)?.value || "";
  return getSessionById(sessionId, { touch });
}

export function rotateAuthenticatedSession(existingSession, user) {
  if (existingSession?.id) {
    deleteSessionRecord(existingSession.id);
  }

  const config = getConfig();
  const currentTime = nowMs();
  const record = persistSession({
    id: generateId(),
    createdAt: currentTime,
    lastSeenAt: currentTime,
    idleExpiresAt: currentTime + config.sessionIdleTimeoutMs,
    absoluteExpiresAt: currentTime + config.sessionAbsoluteTimeoutMs,
    csrfToken: generateCsrfToken(),
    authenticated: true,
    username: user.username,
    accessEmail: user.accessEmail,
    role: user.role,
    rotatedFrom: existingSession?.id || null
  });

  audit("session_rotated", {
    username: user.username,
    accessEmail: user.accessEmail,
    role: user.role
  });
  audit("session_created", {
    authenticated: true,
    username: user.username,
    accessEmail: user.accessEmail,
    role: user.role
  });
  return record;
}

export function destroySession(sessionId, reason = "logout") {
  if (!sessionId) return;
  const existing = getSessionById(sessionId, { touch: false });
  deleteSessionRecord(sessionId);
  audit("session_destroyed", {
    reason,
    authenticated: Boolean(existing?.authenticated),
    username: existing?.username || null
  });
}

export function setSessionCookie(response, sessionId) {
  const config = getConfig();
  response.cookies.set(config.sessionCookieName, sessionId, {
    httpOnly: true,
    secure: config.sessionCookieSecure,
    sameSite: "lax",
    path: "/",
    maxAge: Math.floor(config.sessionAbsoluteTimeoutMs / 1000)
  });
  return response;
}

export function clearSessionCookie(response) {
  const config = getConfig();
  response.cookies.set(config.sessionCookieName, "", {
    httpOnly: true,
    secure: config.sessionCookieSecure,
    sameSite: "lax",
    path: "/",
    expires: new Date(0)
  });
  return response;
}

export function validateCsrfToken(session, token) {
  const sessionToken = String(session?.csrfToken || "");
  const candidate = String(token || "");
  if (!sessionToken || !candidate || sessionToken.length !== candidate.length) {
    return false;
  }
  return timingSafeEqual(Buffer.from(sessionToken), Buffer.from(candidate));
}

function pruneLoginAttempts(currentTime) {
  const cutoff = currentTime - getConfig().loginWindowMs;
  getSessionDb().prepare("DELETE FROM login_attempts WHERE attempted_at < ?").run(cutoff);
}

export function isLoginThrottled(throttleKey) {
  const currentTime = nowMs();
  pruneLoginAttempts(currentTime);
  const row = getSessionDb()
    .prepare(
      `
        SELECT COUNT(*) AS failures
        FROM login_attempts
        WHERE throttle_key = ?
          AND success = 0
          AND attempted_at >= ?
      `
    )
    .get(throttleKey, currentTime - getConfig().loginWindowMs);
  return Number(row?.failures || 0) >= getConfig().loginAttemptLimit;
}

export function recordLoginAttempt({ throttleKey, success, remoteAddr }) {
  const currentTime = nowMs();
  pruneLoginAttempts(currentTime);
  getSessionDb()
    .prepare(
      `
        INSERT INTO login_attempts (throttle_key, attempted_at, success, remote_addr)
        VALUES (?, ?, ?, ?)
      `
    )
    .run(throttleKey, currentTime, success ? 1 : 0, remoteAddr || null);

  if (success) {
    getSessionDb().prepare("DELETE FROM login_attempts WHERE throttle_key = ?").run(throttleKey);
  }
}
