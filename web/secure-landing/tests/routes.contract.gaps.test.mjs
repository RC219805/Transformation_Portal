import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";

import { NextRequest, NextResponse } from "next/server.js";

import { resetDbCache } from "../lib/db.js";

const ENV_KEYS = [
  "NODE_ENV",
  "TP_FASTAPI_ORIGIN",
  "TP_BACKEND_API_KEY",
  "TP_FRONTDOOR_USERS_FILE",
  "TP_FRONTDOOR_USERS_JSON",
  "TP_FRONTDOOR_SESSION_DB",
  "TP_CF_ACCESS_TEAM_DOMAIN",
  "TP_CF_ACCESS_AUD",
  "TP_ALLOW_LOCAL_ACCESS_BYPASS",
  "TP_PORTAL_RUM_ENABLED",
  "TP_PORTAL_RUM_ROLLOUT_PERCENT",
  "TP_FRONTDOOR_RUM_ENABLED",
  "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT"
];

const TEST_CF_ACCESS_TEAM_DOMAIN = "https://tp-frontdoor-tests.cloudflareaccess.com";
const TEST_CF_ACCESS_AUD = "tp-frontdoor-aud";

function snapshotEnv() {
  return new Map(ENV_KEYS.map((key) => [key, process.env[key]]));
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

function withTempEnvironment(overrides = {}) {
  const snapshot = snapshotEnv();
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-contract-gaps-"));
  const dbPath = path.join(tempDir, "sessions.sqlite");
  const usersFilePath = path.join(tempDir, "frontdoor-users.json");

  process.env.NODE_ENV = overrides.NODE_ENV ?? "production";
  process.env.TP_FASTAPI_ORIGIN = overrides.TP_FASTAPI_ORIGIN ?? "http://127.0.0.1:8000";
  process.env.TP_BACKEND_API_KEY = overrides.TP_BACKEND_API_KEY ?? "backend-secret";
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  process.env.TP_CF_ACCESS_TEAM_DOMAIN = overrides.TP_CF_ACCESS_TEAM_DOMAIN ?? TEST_CF_ACCESS_TEAM_DOMAIN;
  process.env.TP_CF_ACCESS_AUD = overrides.TP_CF_ACCESS_AUD ?? TEST_CF_ACCESS_AUD;

  if (Array.isArray(overrides.usersFileEntries)) {
    writeFileSync(usersFilePath, JSON.stringify(overrides.usersFileEntries), "utf-8");
    process.env.TP_FRONTDOOR_USERS_FILE = usersFilePath;
  } else {
    delete process.env.TP_FRONTDOOR_USERS_FILE;
  }

  process.env.TP_FRONTDOOR_USERS_JSON = overrides.TP_FRONTDOOR_USERS_JSON ?? "[]";

  if (typeof overrides.TP_ALLOW_LOCAL_ACCESS_BYPASS === "string") {
    process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS = overrides.TP_ALLOW_LOCAL_ACCESS_BYPASS;
  } else {
    delete process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS;
  }

  delete process.env.TP_PORTAL_RUM_ENABLED;
  delete process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT;
  delete process.env.TP_FRONTDOOR_RUM_ENABLED;
  delete process.env.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT;

  resetDbCache();

  return {
    cleanup() {
      resetDbCache();
      restoreEnv(snapshot);
      rmSync(tempDir, { recursive: true, force: true });
    }
  };
}

async function importFresh(relativePath) {
  return import(`${relativePath}?case=${Date.now()}-${Math.random()}`);
}

function buildRequest(url, options = {}) {
  return new NextRequest(url, options);
}

test("production login POST ignores the local bypass flag unless development mode is enabled", async () => {
  const env = withTempEnvironment({
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "placeholder-hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = await sessions.createAnonymousSession();
    const request = buildRequest("https://portal.example.com/login", {
      method: "POST",
      headers: new Headers({
        origin: "https://portal.example.com",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `__Host-tp_session=${anonymousSession.id}`
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "irrelevant",
        csrf_token: anonymousSession.csrfToken
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login?error=access");
    assert.equal((await sessions.getSessionById(anonymousSession.id, { touch: false }))?.authenticated, false);
  } finally {
    env.cleanup();
  }
});

test("production session cookie omits Domain while preserving host-scoped managed cookie semantics", async () => {
  const env = withTempEnvironment();

  try {
    const { setSessionCookie } = await importFresh("../lib/sessions.js");
    const response = NextResponse.json({ ok: true });

    setSessionCookie(response, "production-session-id");

    const cookieHeader = response.headers.get("set-cookie") || "";
    assert.match(cookieHeader, /^__Host-tp_session=/);
    assert.match(cookieHeader, /HttpOnly/i);
    assert.match(cookieHeader, /Secure/i);
    assert.match(cookieHeader, /SameSite=lax/i);
    assert.match(cookieHeader, /Path=\//i);
    assert.doesNotMatch(cookieHeader, /\bDomain=/i);
  } finally {
    env.cleanup();
  }
});

test("development session cookie also omits Domain while relaxing production-only host transport requirements", async () => {
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1"
  });

  try {
    const { setSessionCookie } = await importFresh("../lib/sessions.js");
    const response = NextResponse.json({ ok: true });

    setSessionCookie(response, "development-session-id");

    const cookieHeader = response.headers.get("set-cookie") || "";
    assert.match(cookieHeader, /^tp_session=/);
    assert.match(cookieHeader, /HttpOnly/i);
    assert.match(cookieHeader, /SameSite=lax/i);
    assert.match(cookieHeader, /Path=\//i);
    assert.doesNotMatch(cookieHeader, /\bDomain=/i);
    assert.doesNotMatch(cookieHeader, /__Host-tp_session/);
    assert.doesNotMatch(cookieHeader, /Secure/i);
  } finally {
    env.cleanup();
  }
});
