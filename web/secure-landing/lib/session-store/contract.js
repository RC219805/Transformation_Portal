/**
 * @fileoverview Frontdoor session-store contract (Phase 3).
 *
 * Defines the surface area every backend implementation must satisfy so the
 * caller in ``sessions.js`` can swap between ``sqlite-store`` (default) and
 * ``redis-store`` without conditional code at every call site. The Phase 2
 * worker-split track established the same pattern for the orchestrator's
 * ``QueueBroker`` Protocol; this is the frontdoor analogue.
 *
 * Records are plain dicts using the snake_case column names that the legacy
 * SQLite schema exposes (``id``, ``created_at``, ``last_seen_at``,
 * ``idle_expires_at``, ``absolute_expires_at``, ``csrf_token``,
 * ``authenticated``, ``username``, ``access_email``, ``role``,
 * ``rotated_from``). New backends MUST round-trip the same keys so
 * ``sessions.js`` does not branch on backend identity downstream.
 *
 * @typedef {object} SessionRow
 * @property {string} id
 * @property {number} created_at
 * @property {number} last_seen_at
 * @property {number} idle_expires_at
 * @property {number} absolute_expires_at
 * @property {string} csrf_token
 * @property {number|boolean} authenticated     `0`/`1` accepted on read, written as `1` or `0`
 * @property {string|null} username
 * @property {string|null} access_email
 * @property {string|null} role
 * @property {string|null} rotated_from
 *
 * @typedef {object} LoginAttempt
 * @property {string} throttle_key
 * @property {number} attempted_at              Unix epoch ms.
 * @property {0|1} success
 * @property {string|null} remote_addr
 *
 * @typedef {object} SessionStore
 * @property {string} backend                   ``"sqlite"`` or ``"redis"``.
 * @property {(row: SessionRow) => SessionRow} persistSession
 *     Insert (or replace) the session row.
 * @property {(sessionId: string) => SessionRow|null} getRawSessionRow
 *     Fetch the raw row by id. Returns ``null`` when absent. Callers handle
 *     expiry — the store does not.
 * @property {(sessionId: string, lastSeenAt: number, idleExpiresAt: number) => void} touchSession
 *     Update ``last_seen_at`` + ``idle_expires_at`` for a leased session.
 * @property {(sessionId: string) => void} deleteSession
 *     Idempotent. ``sessionId`` falsy is a no-op.
 * @property {(throttleKey: string, sinceMs: number) => number} countLoginFailures
 *     How many failed login attempts for this throttle key since ``sinceMs``.
 * @property {(attempt: LoginAttempt) => void} recordLoginAttempt
 * @property {(throttleKey: string) => void} resetLoginAttempts
 *     Called after a successful login to clear the throttle.
 * @property {(cutoffMs: number) => void} pruneLoginAttempts
 *     Drop login_attempts rows older than ``cutoffMs``.
 */

export const SESSION_STORE_CONTRACT_VERSION = "tp.frontdoor.session_store.v1";
