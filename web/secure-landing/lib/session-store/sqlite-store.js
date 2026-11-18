/**
 * @fileoverview ``better-sqlite3``-backed ``SessionStore`` (Phase 3).
 *
 * Wraps the pre-existing SQLite tables (``sessions`` + ``login_attempts``)
 * behind the ``SessionStore`` contract documented in ``./contract.js`` so
 * ``sessions.js`` can swap between SQLite and Redis without branching at
 * every call site. The schema and statements are byte-identical to the
 * pre-Phase-3 implementation living in ``../sessions.js`` — this module is
 * a refactor of that code, not a rewrite.
 */

import { getConfig } from "../config.js";
import { getDb } from "../db.js";

function getSessionDb() {
  return getDb(getConfig().sessionDbPath);
}

function normalizeThrottleKeys(throttleKeys) {
  const keys = Array.isArray(throttleKeys) ? throttleKeys.map((key) => String(key || "")) : [];
  if (!keys.length || keys.some((key) => !key)) {
    throw new Error("reserveLoginAttempt requires at least one non-empty throttle key");
  }
  return keys;
}

export class SqliteSessionStore {
  /** @type {"sqlite"} */
  get backend() {
    return "sqlite";
  }

  persistSession(record) {
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
      record.created_at,
      record.last_seen_at,
      record.idle_expires_at,
      record.absolute_expires_at,
      record.csrf_token,
      record.authenticated ? 1 : 0,
      record.username,
      record.access_email,
      record.role,
      record.rotated_from || null
    );
    return record;
  }

  getRawSessionRow(sessionId) {
    if (!sessionId) return null;
    const row = getSessionDb().prepare("SELECT * FROM sessions WHERE id = ?").get(sessionId);
    return row || null;
  }

  touchSession(sessionId, lastSeenAt, idleExpiresAt) {
    if (!sessionId) return;
    getSessionDb()
      .prepare(
        `
          UPDATE sessions
          SET last_seen_at = ?, idle_expires_at = ?
          WHERE id = ?
        `
      )
      .run(lastSeenAt, idleExpiresAt, sessionId);
  }

  deleteSession(sessionId) {
    if (!sessionId) return;
    getSessionDb().prepare("DELETE FROM sessions WHERE id = ?").run(sessionId);
  }

  countLoginFailures(throttleKey, sinceMs) {
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
      .get(throttleKey, sinceMs);
    return Number(row?.failures || 0);
  }

  recordLoginAttempt(attempt) {
    getSessionDb()
      .prepare(
        `
          INSERT INTO login_attempts (throttle_key, attempted_at, success, remote_addr)
          VALUES (?, ?, ?, ?)
        `
      )
      .run(attempt.throttle_key, attempt.attempted_at, attempt.success, attempt.remote_addr || null);
  }

  reserveLoginAttempt(throttleKeys, { attemptedAt, sinceMs, limit, remoteAddr = null } = {}) {
    const keys = normalizeThrottleKeys(throttleKeys);
    const db = getSessionDb();
    const countFailures = db.prepare(
      `
        SELECT COUNT(*) AS failures
        FROM login_attempts
        WHERE throttle_key = ?
          AND success = 0
          AND attempted_at >= ?
      `
    );
    const insertFailure = db.prepare(
      `
        INSERT INTO login_attempts (throttle_key, attempted_at, success, remote_addr)
        VALUES (?, ?, 0, ?)
      `
    );
    // BEGIN IMMEDIATE takes the write lock before the count, so the check and
    // the insert are atomic even across processes sharing the database file.
    const reserve = db.transaction(() => {
      for (const key of keys) {
        if (Number(countFailures.get(key, sinceMs)?.failures || 0) >= limit) {
          return false;
        }
      }
      for (const key of keys) {
        insertFailure.run(key, attemptedAt, remoteAddr || null);
      }
      return true;
    });
    return reserve.immediate();
  }

  resetLoginAttempts(throttleKey) {
    getSessionDb().prepare("DELETE FROM login_attempts WHERE throttle_key = ?").run(throttleKey);
  }

  pruneLoginAttempts(cutoffMs) {
    getSessionDb().prepare("DELETE FROM login_attempts WHERE attempted_at < ?").run(cutoffMs);
  }
}
