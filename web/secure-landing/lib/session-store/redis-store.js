/**
 * @fileoverview Redis-backed ``SessionStore`` (Phase 3).
 *
 * Activated by ``TP_FRONTDOOR_SESSION_STORE=redis`` +
 * ``TP_FRONTDOOR_REDIS_URL``. Stores each session as a single Redis JSON
 * blob keyed by ``${prefix}session:${id}`` so reads/writes are O(1) and
 * session expiry can be enforced by Redis TTL (set to
 * ``absolute_expires_at - now`` on every persist/touch). Login attempts
 * live in a per-throttle-key sorted set keyed by ``${prefix}login:${key}``
 * with the timestamp as score, so the throttle-window count is a
 * ``ZCOUNT`` against ``(now - window, now)``.
 *
 * The module uses ``ioredis`` lazily so the bundle still ships before a Redis
 * connection is needed. When Redis is explicitly selected, a missing package or
 * unreachable Redis endpoint fails the session path instead of silently
 * downgrading a hosted deployment to local SQLite.
 *
 * Schema invariants:
 *
 *   - The session JSON blob carries snake_case keys identical to the
 *     ``SessionStore`` contract so ``sessions.js`` does not branch on
 *     backend identity downstream.
 *   - ``authenticated`` is stored as ``0`` / ``1`` to match the SQLite
 *     wire shape; the rowToSession converter in ``sessions.js`` already
 *     coerces it with ``Boolean(row.authenticated)``.
 *   - ``null`` fields are persisted as JSON ``null`` (not the literal
 *     string "null").
 */

let _Redis = null;
const DEFAULT_REDIS_CONNECT_TIMEOUT_MS = 1000;

export function buildRedisClientOptions(connectTimeoutMs) {
  return {
    lazyConnect: false,
    connectTimeout: connectTimeoutMs,
    maxRetriesPerRequest: 1,
    // ioredis 6 defaults to RESP3. Preserve the established RESP2 wire
    // contract until the managed-provider lane deliberately validates and
    // adopts RESP3 semantics.
    protocol: 2
  };
}

async function loadRedisClient() {
  if (_Redis !== null) return _Redis;
  try {
    const mod = await import("ioredis");
    _Redis = mod.default || mod.Redis || mod;
  } catch (err) {
    throw new Error(
      "Redis session store requires the 'ioredis' package. Install it with " +
        "`npm install ioredis` in web/secure-landing/ or fall back to " +
        "TP_FRONTDOOR_SESSION_STORE=sqlite. Original error: " +
        (err && err.message ? err.message : String(err))
    );
  }
  return _Redis;
}

export class RedisSessionStore {
  constructor({
    redisUrl,
    keyPrefix,
    idleTimeoutMs,
    absoluteTimeoutMs,
    client = null,
    connectTimeoutMs = DEFAULT_REDIS_CONNECT_TIMEOUT_MS
  } = {}) {
    if (!redisUrl) {
      throw new Error("RedisSessionStore requires a non-empty redisUrl (TP_FRONTDOOR_REDIS_URL).");
    }
    this._redisUrl = redisUrl;
    this._keyPrefix = keyPrefix || "tp:frontdoor:";
    this._idleTimeoutMs = idleTimeoutMs;
    this._absoluteTimeoutMs = absoluteTimeoutMs;
    this._connectTimeoutMs = connectTimeoutMs;
    this._client = client;
    this._clientPromise = null;
  }

  get backend() {
    return "redis";
  }

  async _client_() {
    if (this._client) return this._client;
    if (!this._clientPromise) {
      this._clientPromise = (async () => {
        const Redis = await loadRedisClient();
        this._client = new Redis(
          this._redisUrl,
          buildRedisClientOptions(this._connectTimeoutMs)
        );
        return this._client;
      })();
    }
    return this._clientPromise;
  }

  async ping() {
    const client = await this._client_();
    return client.ping();
  }

  _sessionKey(sessionId) {
    return `${this._keyPrefix}session:${sessionId}`;
  }

  _loginKey(throttleKey) {
    return `${this._keyPrefix}login:${throttleKey}`;
  }

  async persistSession(record) {
    const client = await this._client_();
    const payload = JSON.stringify({
      id: record.id,
      created_at: record.created_at,
      last_seen_at: record.last_seen_at,
      idle_expires_at: record.idle_expires_at,
      absolute_expires_at: record.absolute_expires_at,
      csrf_token: record.csrf_token,
      // Match SQLite wire shape: integer 0/1 so ``Boolean()`` coercion in
      // ``sessions.js:rowToSession`` works identically for both backends.
      authenticated: record.authenticated ? 1 : 0,
      username: record.username || null,
      access_email: record.access_email || null,
      role: record.role || null,
      rotated_from: record.rotated_from || null
    });
    const ttlSeconds = Math.max(1, Math.ceil((record.absolute_expires_at - Date.now()) / 1000));
    await client.set(this._sessionKey(record.id), payload, "PX", ttlSeconds * 1000);
    return record;
  }

  async getRawSessionRow(sessionId) {
    if (!sessionId) return null;
    const client = await this._client_();
    const raw = await client.get(this._sessionKey(sessionId));
    if (!raw) return null;
    return JSON.parse(raw);
  }

  async touchSession(sessionId, lastSeenAt, idleExpiresAt) {
    if (!sessionId) return;
    const existing = await this.getRawSessionRow(sessionId);
    if (!existing) return;
    existing.last_seen_at = lastSeenAt;
    existing.idle_expires_at = idleExpiresAt;
    // Repersist so TTL refreshes against the absolute deadline.
    await this.persistSession(existing);
  }

  async deleteSession(sessionId) {
    if (!sessionId) return;
    const client = await this._client_();
    await client.del(this._sessionKey(sessionId));
  }

  async countLoginFailures(throttleKey, sinceMs) {
    const client = await this._client_();
    const count = await client.zcount(this._loginKey(throttleKey), sinceMs, "+inf");
    return Number(count || 0);
  }

  async recordLoginAttempt(attempt) {
    const client = await this._client_();
    if (attempt.success) {
      // The throttle bucket only tracks failures, so a successful login is
      // a tombstone: leave the bucket as-is unless ``resetLoginAttempts``
      // is called explicitly.
      return;
    }
    const throttleKey = attempt.throttle_key;
    if (!throttleKey) {
      throw new Error("recordLoginAttempt requires attempt.throttle_key");
    }
    // Score = attempted_at ms; member = unique attempt id (epoch ms +
    // random suffix so duplicates within the same millisecond don't
    // collapse — ZADD uses member identity).
    const member = `${attempt.attempted_at}:${Math.random().toString(36).slice(2, 10)}`;
    await client.zadd(this._loginKey(throttleKey), attempt.attempted_at, member);
    // TTL the bucket so old failures auto-prune even without an explicit
    // pruneLoginAttempts call.
    await client.pexpire(this._loginKey(throttleKey), this._absoluteTimeoutMs);
  }

  async resetLoginAttempts(throttleKey) {
    const client = await this._client_();
    await client.del(this._loginKey(throttleKey));
  }

  async pruneLoginAttempts(cutoffMs) {
    // Best-effort prune: iterate every ``login:`` bucket and drop scores
    // older than ``cutoffMs``. Production sweepers typically rely on the
    // TTL set in ``recordLoginAttempt`` instead.
    const client = await this._client_();
    const pattern = `${this._keyPrefix}login:*`;
    let cursor = "0";
    do {
      const result = await client.scan(cursor, "MATCH", pattern, "COUNT", 200);
      cursor = result[0];
      for (const key of result[1] || []) {
        await client.zremrangebyscore(key, "-inf", `(${cutoffMs}`);
      }
    } while (cursor !== "0");
  }

  async close() {
    this._clientPromise = null;
    if (this._client) {
      await this._client.quit().catch(() => undefined);
      this._client = null;
    }
  }
}
