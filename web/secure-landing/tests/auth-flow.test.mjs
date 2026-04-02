import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync } from "node:fs";

import argon2 from "argon2";

import { resetDbCache } from "../lib/db.js";

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
      "x-api-key": "browser-token",
      "x-csrf-token": "csrf-token",
      Accept: "application/json"
    }),
    {
      backendApiKey: "backend-secret",
      actor: {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    }
  );

  assert.equal(headers.get("Authorization"), "Bearer backend-secret");
  assert.equal(headers.get("x-api-key"), "backend-secret");
  assert.equal(headers.has("x-csrf-token"), false);
  assert.equal(headers.get("x-tp-actor"), "admin");
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
