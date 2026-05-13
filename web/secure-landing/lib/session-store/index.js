/**
 * @fileoverview Session-store factory (Phase 3).
 *
 * Keyed off the ``sessionStoreBackend`` field of the resolved frontdoor
 * config (``TP_FRONTDOOR_SESSION_STORE=sqlite|redis``, default ``sqlite``).
 * Returns a singleton implementation that satisfies the ``SessionStore``
 * contract documented in ``./contract.js``.
 *
 * The factory caches the constructed store across the process lifetime so
 * ``sessions.js`` does not pay the import/connection cost on every call.
 * Tests that need a fresh store call ``resetSessionStoreSingleton()``
 * between cases.
 */

import { getConfig } from "../config.js";

import { RedisSessionStore } from "./redis-store.js";
import { SqliteSessionStore } from "./sqlite-store.js";

let _cachedStore = null;
let _cachedBackend = null;

function buildStore(config) {
  const backend = String(config.sessionStoreBackend || "sqlite").trim().toLowerCase();
  if (backend === "redis") {
    if (!config.sessionStoreRedisUrl) {
      throw new Error(
        "TP_FRONTDOOR_SESSION_STORE=redis requires TP_FRONTDOOR_REDIS_URL to be set."
      );
    }
    // ``RedisSessionStore`` constructs without touching ``ioredis`` — the
    // dynamic ``import("ioredis")`` happens lazily on first ``_client_()``,
    // so sqlite-only deployments that import this factory keep working
    // even when ``ioredis`` is not installed.
    return new RedisSessionStore({
      redisUrl: config.sessionStoreRedisUrl,
      keyPrefix: config.sessionStoreRedisKeyPrefix,
      idleTimeoutMs: config.sessionIdleTimeoutMs,
      absoluteTimeoutMs: config.sessionAbsoluteTimeoutMs
    });
  }
  return new SqliteSessionStore();
}

export function getSessionStore() {
  const config = getConfig();
  const backend = String(config.sessionStoreBackend || "sqlite").trim().toLowerCase();
  if (_cachedStore !== null && _cachedBackend === backend) {
    return _cachedStore;
  }
  _cachedStore = buildStore(config);
  _cachedBackend = backend;
  return _cachedStore;
}

export function resetSessionStoreSingleton() {
  _cachedStore = null;
  _cachedBackend = null;
}

// Re-export the canonical backend list so callers can compare against
// stable string constants instead of magic strings.
export { SESSION_STORE_BACKEND } from "../session-scaling.js";
