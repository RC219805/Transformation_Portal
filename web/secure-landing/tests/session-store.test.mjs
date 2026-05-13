/**
 * Phase 3.A — basic contract coverage for the new ``SessionStore``
 * abstraction. Asserts that ``SqliteSessionStore`` (the default) round-trips
 * the snake_case ``SessionRow`` shape the contract documents, that
 * ``getSessionStore()`` returns a singleton across calls, and that
 * ``resetSessionStoreSingleton()`` lets tests rotate it. The
 * ``RedisSessionStore`` constructor's contract (URL validation) is covered
 * separately so the suite stays runnable without ``ioredis`` installed.
 *
 * Phase 3.B will cut ``sessions.js`` over to consume the factory; until
 * then this test exists to pin the new module's wire shape and prevent
 * drift while the factory is wired in.
 */

import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync } from "node:fs";

import { SqliteSessionStore } from "../lib/session-store/sqlite-store.js";
import { RedisSessionStore } from "../lib/session-store/redis-store.js";
import {
  getSessionStore,
  resetSessionStoreSingleton
} from "../lib/session-store/index.js";

function withTempSessionDb(callback) {
  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-session-store-"));
  const dbPath = path.join(tmpDir, "sessions.db");
  const previousDbPath = process.env.TP_FRONTDOOR_SESSION_DB;
  const previousStore = process.env.TP_FRONTDOOR_SESSION_STORE;
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  process.env.TP_FRONTDOOR_SESSION_STORE = "sqlite";
  resetSessionStoreSingleton();
  try {
    callback(dbPath);
  } finally {
    process.env.TP_FRONTDOOR_SESSION_DB = previousDbPath;
    process.env.TP_FRONTDOOR_SESSION_STORE = previousStore;
    resetSessionStoreSingleton();
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

function _sessionRow(overrides = {}) {
  const now = Date.now();
  return {
    id: "session-fixture-id",
    created_at: now,
    last_seen_at: now,
    idle_expires_at: now + 60_000,
    absolute_expires_at: now + 3_600_000,
    csrf_token: "csrf-fixture-token",
    authenticated: false,
    username: null,
    access_email: null,
    role: null,
    rotated_from: null,
    ...overrides
  };
}

test("SqliteSessionStore round-trips a session row through the contract shape", () => {
  withTempSessionDb(() => {
    const store = new SqliteSessionStore();
    const record = _sessionRow();
    store.persistSession(record);
    const fetched = store.getRawSessionRow(record.id);
    assert.ok(fetched, "row must be present after persist");
    assert.equal(fetched.id, record.id);
    assert.equal(fetched.csrf_token, record.csrf_token);
    // SQLite stores ``authenticated`` as 0/1, contract says either is OK.
    assert.equal(fetched.authenticated, 0);
    assert.equal(fetched.username, null);
  });
});

test("SqliteSessionStore touchSession updates last_seen_at without changing other fields", () => {
  withTempSessionDb(() => {
    const store = new SqliteSessionStore();
    const record = _sessionRow();
    store.persistSession(record);
    const newSeen = record.last_seen_at + 5_000;
    const newIdle = record.idle_expires_at + 5_000;
    store.touchSession(record.id, newSeen, newIdle);
    const fetched = store.getRawSessionRow(record.id);
    assert.equal(fetched.last_seen_at, newSeen);
    assert.equal(fetched.idle_expires_at, newIdle);
    // Untouched fields stay put.
    assert.equal(fetched.absolute_expires_at, record.absolute_expires_at);
    assert.equal(fetched.csrf_token, record.csrf_token);
  });
});

test("SqliteSessionStore deleteSession is idempotent", () => {
  withTempSessionDb(() => {
    const store = new SqliteSessionStore();
    const record = _sessionRow();
    store.persistSession(record);
    store.deleteSession(record.id);
    assert.equal(store.getRawSessionRow(record.id), null);
    // Second delete must not raise.
    store.deleteSession(record.id);
    // Falsy id is a no-op.
    store.deleteSession("");
  });
});

test("SqliteSessionStore login-attempt counters match the SQL surface", () => {
  withTempSessionDb(() => {
    const store = new SqliteSessionStore();
    const now = Date.now();
    const windowStart = now - 60_000;
    store.recordLoginAttempt({ throttle_key: "user-a", attempted_at: now - 1_000, success: 0, remote_addr: "127.0.0.1" });
    store.recordLoginAttempt({ throttle_key: "user-a", attempted_at: now - 2_000, success: 0, remote_addr: "127.0.0.1" });
    store.recordLoginAttempt({ throttle_key: "user-a", attempted_at: now - 70_000, success: 0, remote_addr: "127.0.0.1" });
    // Successful login does not count as a failure.
    store.recordLoginAttempt({ throttle_key: "user-a", attempted_at: now - 500, success: 1, remote_addr: "127.0.0.1" });
    assert.equal(store.countLoginFailures("user-a", windowStart), 2);
    store.resetLoginAttempts("user-a");
    assert.equal(store.countLoginFailures("user-a", windowStart), 0);
  });
});

test("getSessionStore returns the same singleton until reset", () => {
  withTempSessionDb(() => {
    const first = getSessionStore();
    const second = getSessionStore();
    assert.equal(first, second, "factory must cache per-backend");
    assert.equal(first.backend, "sqlite");
    resetSessionStoreSingleton();
    const third = getSessionStore();
    assert.notEqual(first, third, "reset must drop the cached instance");
    assert.equal(third.backend, "sqlite");
  });
});

test("RedisSessionStore constructor rejects missing redisUrl", () => {
  assert.throws(() => new RedisSessionStore({ redisUrl: "" }), /TP_FRONTDOOR_REDIS_URL/);
});

test("RedisSessionStore exposes the redis backend identifier", () => {
  const store = new RedisSessionStore({
    redisUrl: "redis://127.0.0.1:6379/0",
    keyPrefix: "tp:test:",
    idleTimeoutMs: 60_000,
    absoluteTimeoutMs: 3_600_000
  });
  assert.equal(store.backend, "redis");
});
