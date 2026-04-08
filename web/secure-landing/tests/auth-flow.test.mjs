import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";

import argon2 from "argon2";

import { resetDbCache } from "../lib/db.js";

const ENV_KEYS = [
  "NODE_ENV",
  "TP_FRONTDOOR_USERS_FILE",
  "TP_FRONTDOOR_USERS_JSON",
  "TP_FRONTDOOR_SESSION_DB",
  "TP_FRONTDOOR_SESSION_SCALING_MODE",
  "TP_ALLOW_LOCAL_ACCESS_BYPASS"
];

let envSnapshot = null;

function snapshotEnv() {
  return new Map(ENV_KEYS.map((key) => [key, process.env[key]]));
}

function clearEnv() {
  for (const key of ENV_KEYS) {
    delete process.env[key];
  }
}

function restoreEnv(snapshot) {
  for (const key of ENV_KEYS) {
    const previous = snapshot.get(key);
    if (typeof previous === "string") {
      process.env[key] = previous;
    } else {
      delete process.env[key];
    }
  }
}

function withTempDb() {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-"));
  const dbPath = path.join(tempDir, "sessions.sqlite");
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  resetDbCache();
  return {
    cleanup() {
      resetDbCache();
      rmSync(tempDir, { recursive: true, force: true });
    }
  };
}

test.beforeEach(() => {
  envSnapshot = snapshotEnv();
  clearEnv();
  resetDbCache();
});

test.afterEach(() => {
  resetDbCache();
  if (envSnapshot) {
    restoreEnv(envSnapshot);
    envSnapshot = null;
  }
});

test("verifyUserCredentials requires matching Access email unless local bypass is enabled", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  process.env.TP_FRONTDOOR_USERS_JSON = JSON.stringify([
    {
      username: "admin",
      password_hash: passwordHash,
      access_email: "admin@example.com",
      role: "admin"
    }
  ]);

  const { verifyUserCredentials } = await import(`../lib/users.js?case=${Date.now()}`);

  const mismatch = await verifyUserCredentials({
    username: "admin",
    password: "correct horse battery staple",
    accessEmail: "other@example.com",
    allowAccessBypass: false
  });
  assert.equal(mismatch, null);

  const match = await verifyUserCredentials({
    username: "admin",
    password: "correct horse battery staple",
    accessEmail: "admin@example.com",
    allowAccessBypass: false
  });
  assert.equal(match?.username, "admin");
});

test("json-backed credential fixtures stay stable unless file precedence is explicitly enabled", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-users-precedence-"));
  const usersFile = path.join(tempDir, "users.json");
  const passwordHash = await argon2.hash("correct horse battery staple");
  writeFileSync(
    usersFile,
    JSON.stringify([
      {
        username: "file-admin",
        password_hash: passwordHash,
        access_email: "file-admin@example.com",
        role: "admin"
      }
    ]),
    "utf-8"
  );

  process.env.TP_FRONTDOOR_USERS_FILE = usersFile;
  process.env.TP_FRONTDOOR_USERS_JSON = JSON.stringify([
    {
      username: "json-admin",
      password_hash: passwordHash,
      access_email: "json-admin@example.com",
      role: "admin"
    }
  ]);

  try {
    const { getConfig: getConfigWithFile } = await import(`../lib/config.js?case=${Date.now()}-file`);
    assert.equal(getConfigWithFile().users[0]?.username, "file-admin");

    delete process.env.TP_FRONTDOOR_USERS_FILE;
    const { getConfig: getJsonConfig } = await import(`../lib/config.js?case=${Date.now()}-json`);
    assert.equal(getJsonConfig().users[0]?.username, "json-admin");
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("config loads users from TP_FRONTDOOR_USERS_FILE before JSON fallback", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-users-"));
  const usersFile = path.join(tempDir, "users.json");
  const passwordHash = await argon2.hash("correct horse battery staple");
  writeFileSync(
    usersFile,
    JSON.stringify([
      {
        username: "file-admin",
        password_hash: passwordHash,
        access_email: "file-admin@example.com",
        role: "admin"
      }
    ]),
    "utf-8"
  );

  process.env.TP_FRONTDOOR_USERS_FILE = usersFile;
  process.env.TP_FRONTDOOR_USERS_JSON = JSON.stringify([
    {
      username: "json-admin",
      password_hash: passwordHash,
      access_email: "json-admin@example.com",
      role: "admin"
    }
  ]);

  try {
    const { getConfig } = await import(`../lib/config.js?case=${Date.now()}`);
    const config = getConfig();

    assert.equal(config.usersFilePath, usersFile);
    assert.equal(config.users.length, 1);
    assert.equal(config.users[0].username, "file-admin");
  } finally {
    delete process.env.TP_FRONTDOOR_USERS_FILE;
    delete process.env.TP_FRONTDOOR_USERS_JSON;
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("session rotation creates a new authenticated session id", async () => {
  const temp = withTempDb();
  try {
    const sessions = await import(`../lib/sessions.js?case=${Date.now()}`);
    const anon = sessions.createAnonymousSession();
    const rotated = sessions.rotateAuthenticatedSession(anon, {
      username: "admin",
      accessEmail: "admin@example.com",
      role: "admin"
    });

    assert.notEqual(rotated.id, anon.id);
    assert.equal(rotated.authenticated, true);
    assert.equal(sessions.getSessionById(rotated.id, { touch: false })?.username, "admin");
  } finally {
    temp.cleanup();
  }
});

test("login throttling trips after repeated failures in the same window", async () => {
  const temp = withTempDb();
  try {
    const sessions = await import(`../lib/sessions.js?case=${Date.now()}`);
    const key = "admin@example.com:admin:127.0.0.1";

    for (let index = 0; index < 5; index += 1) {
      sessions.recordLoginAttempt({
        throttleKey: key,
        success: false,
        remoteAddr: "127.0.0.1"
      });
    }

    assert.equal(sessions.isLoginThrottled(key), true);
  } finally {
    temp.cleanup();
  }
});

test("buildUpstreamHeaders strips browser auth and injects backend auth", async () => {
  const { buildUpstreamHeaders } = await import(`../lib/proxy.js?case=${Date.now()}`);
  const headers = buildUpstreamHeaders(
    new Headers({
      Authorization: "Bearer browser-token",
      Forwarded: 'for="198.51.100.9"',
      "x-api-key": "browser-token",
      "x-csrf-token": "csrf-token",
      "x-forwarded-for": "198.51.100.9",
      Accept: "application/json"
    }),
    {
      backendApiKey: "backend-secret",
      actor: {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      },
      forwarding: {
        clientIp: "203.0.113.5",
        host: "portal.example.com",
        proto: "https"
      }
    }
  );

  assert.equal(headers.get("Authorization"), "Bearer backend-secret");
  assert.equal(headers.get("Forwarded"), 'for="203.0.113.5";host="portal.example.com";proto="https"');
  assert.equal(headers.get("x-api-key"), "backend-secret");
  assert.equal(headers.has("x-csrf-token"), false);
  assert.equal(headers.get("x-forwarded-for"), "203.0.113.5");
  assert.equal(headers.get("x-forwarded-host"), "portal.example.com");
  assert.equal(headers.get("x-forwarded-proto"), "https");
  assert.equal(headers.get("x-real-ip"), "203.0.113.5");
  assert.equal(headers.get("x-tp-actor"), "admin");
});

test("getDb creates the session database parent directory automatically", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-db-"));
  const dbPath = path.join(tempDir, "nested", "sessions.sqlite");

  try {
    const { getDb, resetDbCache } = await import(`../lib/db.js?case=${Date.now()}`);
    const db = getDb(dbPath);
    const row = db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sessions'").get();
    assert.equal(row?.name, "sessions");
    resetDbCache();
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("validateOriginAndReferrer rejects cross-origin unsafe requests", async () => {
  const { validateOriginAndReferrer } = await import(`../lib/request-security.js?case=${Date.now()}`);
  const request = {
    method: "POST",
    nextUrl: { origin: "https://portal.example.com" },
    headers: new Headers({
      origin: "https://evil.example.com"
    })
  };

  assert.equal(validateOriginAndReferrer(request), false);
});
