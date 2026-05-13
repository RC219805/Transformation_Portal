import { randomBytes, timingSafeEqual } from "node:crypto";

import { getConfig } from "./config.js";
import { getSessionStore } from "./session-store/index.js";
import { audit } from "./audit.js";

/**
 * @fileoverview Frontdoor session helpers (Phase 3.B cut-over).
 *
 * Before Phase 3.B these functions called ``better-sqlite3`` directly through
 * ``./db.js``. Phase 3.B delegates every persistence-touching helper to the
 * factory-returned ``SessionStore`` (``./session-store/index.js``) so the
 * ``sqlite`` and ``redis`` backends become interchangeable at deploy time.
 *
 * Sync vs async surface:
 *
 *   - Sync (no I/O): ``getRemoteAddress``, ``setSessionCookie``,
 *     ``clearSessionCookie``, ``validateCsrfToken``. These never touched the
 *     store and keep their pre-Phase-3 signatures.
 *   - Async (delegates to the store): ``createAnonymousSession``,
 *     ``getSessionById``, ``getSessionFromRequest``, ``rotateAuthenticatedSession``,
 *     ``destroySession``, ``isLoginThrottled``, ``recordLoginAttempt``.
 *     Callers must ``await`` these — Phase 3.B updated every call site
 *     under ``app/`` and ``lib/access.js`` to match.
 *
 * The audit-event payloads, CSRF-token shape, and cookie semantics are
 * intentionally byte-identical to the pre-Phase-3 implementation so the
 * cut-over is observable only via the new backend selection and the new
 * async signatures.
 */

function nowMs() {
  return Date.now();
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

function isExpired(row, currentTime) {
  return row.idle_expires_at <= currentTime || row.absolute_expires_at <= currentTime;
}

export function getRemoteAddress(request) {
  const cfIp = String(request.headers.get("cf-connecting-ip") || "").trim();
  if (cfIp) return cfIp;

  const forwardedFor = String(request.headers.get("x-forwarded-for") || "").trim();
  if (!forwardedFor) return "unknown";
  return forwardedFor.split(",")[0].trim() || "unknown";
}

export async function createAnonymousSession() {
  const config = getConfig();
  const store = getSessionStore();
  const currentTime = nowMs();
  const record = {
    id: generateId(),
    created_at: currentTime,
    last_seen_at: currentTime,
    idle_expires_at: currentTime + config.sessionIdleTimeoutMs,
    absolute_expires_at: currentTime + config.sessionAbsoluteTimeoutMs,
    csrf_token: generateCsrfToken(),
    authenticated: false,
    username: null,
    access_email: null,
    role: null,
    rotated_from: null
  };
  await store.persistSession(record);
  audit("session_created", { authenticated: false });
  return rowToSession(record);
}

export async function getSessionById(sessionId, { touch = false } = {}) {
  if (!sessionId) return null;
  const store = getSessionStore();
  const row = await store.getRawSessionRow(sessionId);
  if (!row) return null;

  const currentTime = nowMs();
  if (isExpired(row, currentTime)) {
    await store.deleteSession(sessionId);
    audit("session_expired", {
      authenticated: Boolean(row.authenticated),
      username: row.username || null,
      reason: row.absolute_expires_at <= currentTime ? "absolute" : "idle"
    });
    return null;
  }

  if (touch) {
    const newIdle = currentTime + getConfig().sessionIdleTimeoutMs;
    await store.touchSession(sessionId, currentTime, newIdle);
    row.last_seen_at = currentTime;
    row.idle_expires_at = newIdle;
  }

  return rowToSession(row);
}

export async function getSessionFromRequest(request, { touch = false } = {}) {
  const sessionId = request.cookies.get(getConfig().sessionCookieName)?.value || "";
  return getSessionById(sessionId, { touch });
}

export async function rotateAuthenticatedSession(existingSession, user) {
  const store = getSessionStore();
  if (existingSession?.id) {
    await store.deleteSession(existingSession.id);
  }

  const config = getConfig();
  const currentTime = nowMs();
  const record = {
    id: generateId(),
    created_at: currentTime,
    last_seen_at: currentTime,
    idle_expires_at: currentTime + config.sessionIdleTimeoutMs,
    absolute_expires_at: currentTime + config.sessionAbsoluteTimeoutMs,
    csrf_token: generateCsrfToken(),
    authenticated: true,
    username: user.username,
    access_email: user.accessEmail,
    role: user.role,
    rotated_from: existingSession?.id || null
  };
  await store.persistSession(record);

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
  return rowToSession(record);
}

export async function destroySession(sessionId, reason = "logout") {
  if (!sessionId) return;
  const store = getSessionStore();
  const existing = await getSessionById(sessionId, { touch: false });
  await store.deleteSession(sessionId);
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

async function pruneLoginAttempts(currentTime) {
  const store = getSessionStore();
  // Phase 3.B perf note: the Redis backend sets a per-bucket TTL inside
  // ``recordLoginAttempt`` (PEXPIRE = absoluteTimeoutMs), so old failure
  // buckets auto-expire without an inline sweep. The pre-Phase-3 helper
  // delegated unconditionally and the Redis implementation backs that with
  // a ``SCAN ${prefix}login:*`` of every bucket — twice per login attempt
  // (once from ``isLoginThrottled``, once from ``recordLoginAttempt``).
  // That's O(N-throttle-keys) on every credential POST in a multi-host
  // fleet. Short-circuit here so the Redis path relies on TTL +
  // ``ZCOUNT(sinceMs, +inf)``'s built-in window filter (already used by
  // ``countLoginFailures``) and never pays the global scan. SQLite keeps
  // the inline prune because its ``login_attempts`` table has no TTL and
  // would otherwise accumulate forever.
  if (store.backend === "redis") {
    return;
  }
  const cutoff = currentTime - getConfig().loginWindowMs;
  await store.pruneLoginAttempts(cutoff);
}

export async function isLoginThrottled(throttleKey) {
  const currentTime = nowMs();
  await pruneLoginAttempts(currentTime);
  const failures = await getSessionStore().countLoginFailures(
    throttleKey,
    currentTime - getConfig().loginWindowMs
  );
  return failures >= getConfig().loginAttemptLimit;
}

export async function recordLoginAttempt({ throttleKey, success, remoteAddr }) {
  const currentTime = nowMs();
  await pruneLoginAttempts(currentTime);
  const store = getSessionStore();
  await store.recordLoginAttempt({
    throttle_key: throttleKey,
    attempted_at: currentTime,
    success: success ? 1 : 0,
    remote_addr: remoteAddr || null
  });

  if (success) {
    await store.resetLoginAttempts(throttleKey);
  }
}
