import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";

import { evaluateSessionScaling } from "../lib/session-scaling.js";
import { RedisSessionStore } from "../lib/session-store/redis-store.js";

const redisUrl = process.env.TP_FRONTDOOR_REDIS_URL;

function requireRedisUrl() {
  if (!redisUrl) {
    throw new Error(
      "TP_FRONTDOOR_REDIS_URL is required for the live Redis SessionStore contract",
    );
  }
  return redisUrl;
}

function makeKeyPrefix() {
  const basePrefix =
    process.env.TP_FRONTDOOR_REDIS_KEY_PREFIX || "tp:frontdoor:test:";
  const normalizedBase = basePrefix.endsWith(":") ? basePrefix : `${basePrefix}:`;
  return `${normalizedBase}live:${process.pid}:${Date.now()}:${randomUUID()}:`;
}

async function deleteKeysWithPrefix(client, prefix) {
  let cursor = "0";
  do {
    const [nextCursor, keys] = await client.scan(
      cursor,
      "MATCH",
      `${prefix}*`,
      "COUNT",
      100,
    );
    if (keys.length > 0) {
      await client.del(...keys);
    }
    cursor = nextCursor;
  } while (cursor !== "0");
}

function sessionRow(sessionId) {
  const now = new Date();
  return {
    session_id: sessionId,
    username: "operator",
    access_email: "operator@example.test",
    role: "admin",
    created_at: now.toISOString(),
    expires_at: new Date(now.getTime() + 60_000).toISOString(),
    last_seen_at: now.toISOString(),
    revoked_at: null,
    csrf_token: "csrf-token",
    user_agent: "node:test",
    ip_hash: "ip-hash",
  };
}

test("RedisSessionStore live contract persists sessions and throttle state", async () => {
  const keyPrefix = makeKeyPrefix();
  const store = new RedisSessionStore({
    redisUrl: requireRedisUrl(),
    keyPrefix,
    sessionTtlSeconds: 120,
    loginAttemptTtlSeconds: 120,
  });
  const client = await store._client_();

  try {
    await deleteKeysWithPrefix(client, keyPrefix);

    const sessionId = `session-${randomUUID()}`;
    await store.persistSession(sessionRow(sessionId));

    const persisted = await store.getRawSessionRow(sessionId);
    assert.equal(persisted?.session_id, sessionId);
    assert.equal(persisted?.username, "operator");

    const initialTtl = await client.pttl(`${keyPrefix}session:${sessionId}`);
    assert.ok(initialTtl > 0, "persisted session should have a Redis TTL");

    const touchedAt = new Date(Date.now() + 1_000).toISOString();
    const touched = await store.touchSession(sessionId, touchedAt);
    assert.equal(touched, true);
    const touchedRow = await store.getRawSessionRow(sessionId);
    assert.equal(touchedRow?.last_seen_at, touchedAt);

    assert.equal(await store.countLoginFailures("operator"), 0);
    await store.recordLoginAttempt({ throttle_key: "operator", success: false });
    await store.recordLoginAttempt({ throttle_key: "operator", success: false });
    assert.equal(await store.countLoginFailures("operator"), 2);
    const throttleTtl = await client.pttl(`${keyPrefix}login:operator`);
    assert.ok(throttleTtl > 0, "login throttle counter should have a Redis TTL");

    await store.resetLoginAttempts("operator");
    assert.equal(await store.countLoginFailures("operator"), 0);

    const deleted = await store.deleteSession(sessionId);
    assert.equal(deleted, true);
    assert.equal(await store.getRawSessionRow(sessionId), null);
  } finally {
    await deleteKeysWithPrefix(client, keyPrefix);
    await store.close();
  }
});

test("Redis SessionStore satisfies multi-instance readiness", () => {
  const readiness = evaluateSessionScaling({
    sessionScalingMode: "multi_instance",
    sessionStoreBackend: "redis",
  });

  assert.equal(readiness.ok, true);
  assert.equal(readiness.mode, "multi_instance");
  assert.equal(readiness.backend, "redis");
});
