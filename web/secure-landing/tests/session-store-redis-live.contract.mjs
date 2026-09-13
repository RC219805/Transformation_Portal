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
  const now = Date.now();
  return {
    id: sessionId,
    created_at: now,
    last_seen_at: now,
    idle_expires_at: now + 60_000,
    absolute_expires_at: now + 120_000,
    csrf_token: "csrf-token",
    authenticated: 1,
    username: "operator",
    access_email: "operator@example.test",
    role: "admin",
    rotated_from: null,
  };
}

test("RedisSessionStore live contract persists sessions and throttle state", async () => {
  const keyPrefix = makeKeyPrefix();
  const store = new RedisSessionStore({
    redisUrl: requireRedisUrl(),
    keyPrefix,
    idleTimeoutMs: 60_000,
    absoluteTimeoutMs: 120_000,
  });
  const client = await store._client_();

  try {
    await deleteKeysWithPrefix(client, keyPrefix);

    const sessionId = `session-${randomUUID()}`;
    await store.persistSession(sessionRow(sessionId));

    const persisted = await store.getRawSessionRow(sessionId);
    assert.equal(persisted?.id, sessionId);
    assert.equal(persisted?.username, "operator");
    assert.equal(persisted?.authenticated, 1);

    const initialTtl = await client.pttl(`${keyPrefix}session:${sessionId}`);
    assert.ok(initialTtl > 0, "persisted session should have a Redis TTL");

    const touchedAt = Date.now() + 1_000;
    const idleExpiresAt = touchedAt + 60_000;
    await store.touchSession(sessionId, touchedAt, idleExpiresAt);
    const touchedRow = await store.getRawSessionRow(sessionId);
    assert.equal(touchedRow?.last_seen_at, touchedAt);
    assert.equal(touchedRow?.idle_expires_at, idleExpiresAt);

    const attemptedAt = Date.now();
    const sinceMs = attemptedAt - 60_000;
    assert.equal(await store.countLoginFailures("operator", sinceMs), 0);
    await store.recordLoginAttempt({
      throttle_key: "operator",
      attempted_at: attemptedAt - 1_000,
      success: 0,
      remote_addr: "127.0.0.1",
    });
    await store.recordLoginAttempt({
      throttle_key: "operator",
      attempted_at: attemptedAt - 500,
      success: 0,
      remote_addr: "127.0.0.1",
    });
    await store.recordLoginAttempt({
      throttle_key: "operator",
      attempted_at: attemptedAt - 70_000,
      success: 0,
      remote_addr: "127.0.0.1",
    });
    await store.recordLoginAttempt({
      throttle_key: "operator",
      attempted_at: attemptedAt,
      success: 1,
      remote_addr: "127.0.0.1",
    });
    assert.equal(await store.countLoginFailures("operator", sinceMs), 2);
    const throttleTtl = await client.pttl(`${keyPrefix}login:operator`);
    assert.ok(throttleTtl > 0, "login throttle counter should have a Redis TTL");

    await store.resetLoginAttempts("operator");
    assert.equal(await store.countLoginFailures("operator", sinceMs), 0);

    await store.deleteSession(sessionId);
    assert.equal(await store.getRawSessionRow(sessionId), null);
    await store.deleteSession(sessionId);
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

function pauseSessionTouch(client) {
  const reached = Promise.withResolvers();
  const released = Promise.withResolvers();
  return {
    reached: reached.promise,
    release: released.resolve,
    client: {
      // Capture a stale read for the former GET/SET implementation. The
      // atomic implementation instead pauses before executing its script.
      async get(...args) {
        const snapshot = await client.get(...args);
        reached.resolve();
        await released.promise;
        return snapshot;
      },
      async eval(...args) {
        reached.resolve();
        await released.promise;
        return client.eval(...args);
      },
      set: client.set.bind(client)
    }
  };
}

for (const change of ["delete", "rotation", "expiry"]) {
  test(`a delayed Redis session touch cannot undo ${change}`, { timeout: 10_000 }, async () => {
    const keyPrefix = makeKeyPrefix();
    const store = new RedisSessionStore({ redisUrl: requireRedisUrl(), keyPrefix });
    const client = await store._client_();
    const paused = pauseSessionTouch(client);
    const touchingStore = new RedisSessionStore({
      redisUrl: requireRedisUrl(),
      keyPrefix,
      client: paused.client
    });
    let pendingTouch;

    try {
      const original = sessionRow(`session-${randomUUID()}`);
      await store.persistSession(original);
      pendingTouch = touchingStore.touchSession(original.id, Date.now() + 1_000, Date.now() + 61_000);
      await paused.reached;

      if (change === "expiry") {
        await client.pexpire(`${keyPrefix}session:${original.id}`, 0);
      } else {
        await store.deleteSession(original.id);
      }
      const rotated = sessionRow(`rotated-${randomUUID()}`);
      if (change === "rotation") {
        await store.persistSession(rotated);
      }

      paused.release();
      await pendingTouch;

      assert.equal(await store.getRawSessionRow(original.id), null);
      if (change === "rotation") {
        assert.deepEqual(await store.getRawSessionRow(rotated.id), rotated);
      }
    } finally {
      paused.release();
      await pendingTouch?.catch(() => undefined);
      await deleteKeysWithPrefix(client, keyPrefix);
      await store.close();
    }
  });
}

test("Redis session touch updates only current timing fields and preserves the existing TTL", { timeout: 10_000 }, async () => {
  const keyPrefix = makeKeyPrefix();
  const store = new RedisSessionStore({ redisUrl: requireRedisUrl(), keyPrefix });
  const client = await store._client_();
  const paused = pauseSessionTouch(client);
  const touchingStore = new RedisSessionStore({
    redisUrl: requireRedisUrl(),
    keyPrefix,
    client: paused.client
  });
  let pendingTouch;

  try {
    const original = sessionRow(`session-${randomUUID()}`);
    await store.persistSession(original);
    const touchedAt = Date.now() + 1_000;
    const idleExpiresAt = touchedAt + 60_000;
    pendingTouch = touchingStore.touchSession(original.id, touchedAt, idleExpiresAt);
    await paused.reached;

    const current = { ...original, csrf_token: "current-csrf-token", role: "operator", authenticated: 0 };
    await store.persistSession(current);
    const key = `${keyPrefix}session:${original.id}`;
    await client.pexpire(key, 30_000);
    const initialTtl = await client.pttl(key);

    paused.release();
    await pendingTouch;

    assert.deepEqual(await store.getRawSessionRow(original.id), {
      ...current,
      last_seen_at: touchedAt,
      idle_expires_at: idleExpiresAt
    });
    const touchedTtl = await client.pttl(key);
    assert.ok(touchedTtl > 0 && touchedTtl <= initialTtl, "touch must preserve the existing expiration");
    await store.touchSession("missing-session", touchedAt, idleExpiresAt);
    assert.equal(await store.getRawSessionRow("missing-session"), null);
  } finally {
    paused.release();
    await pendingTouch?.catch(() => undefined);
    await deleteKeysWithPrefix(client, keyPrefix);
    await store.close();
  }
});
