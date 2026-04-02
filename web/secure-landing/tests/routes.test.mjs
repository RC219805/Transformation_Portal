import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync } from "node:fs";

import argon2 from "argon2";
import { NextRequest } from "next/server.js";

import { getDb, resetDbCache } from "../lib/db.js";

const ENV_KEYS = [
  "NODE_ENV",
  "TP_FASTAPI_ORIGIN",
  "TP_BACKEND_API_KEY",
  "TP_FRONTDOOR_USERS_JSON",
  "TP_FRONTDOOR_SESSION_DB",
  "TP_ALLOW_LOCAL_ACCESS_BYPASS"
];

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
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-routes-"));
  const dbPath = path.join(tempDir, "sessions.sqlite");

  process.env.NODE_ENV = overrides.NODE_ENV ?? "production";
  process.env.TP_FASTAPI_ORIGIN = overrides.TP_FASTAPI_ORIGIN ?? "http://127.0.0.1:8000";
  process.env.TP_BACKEND_API_KEY = overrides.TP_BACKEND_API_KEY ?? "backend-secret";
  process.env.TP_FRONTDOOR_USERS_JSON = overrides.TP_FRONTDOOR_USERS_JSON ?? "[]";
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;

  if (typeof overrides.TP_ALLOW_LOCAL_ACCESS_BYPASS === "string") {
    process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS = overrides.TP_ALLOW_LOCAL_ACCESS_BYPASS;
  } else {
    delete process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS;
  }

  resetDbCache();

  return {
    dbPath,
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

function extractSessionCookie(response) {
  const cookieHeader = response.headers.get("set-cookie") || "";
  const match = cookieHeader.match(/__Host-tp_session=([^;]+)/);
  return {
    raw: cookieHeader,
    value: match?.[1] || ""
  };
}

function buildRequest(url, options = {}) {
  return new NextRequest(url, options);
}

test("login POST rotates the session and redirects authenticated users to /portal", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    TP_FRONTDOOR_USERS_JSON: JSON.stringify([
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ])
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = sessions.createAnonymousSession();
    const form = new URLSearchParams({
      username: "admin",
      password: "correct horse battery staple",
      csrf_token: anonymousSession.csrfToken
    });

    const request = buildRequest("https://portal.example.com/login", {
      method: "POST",
      headers: new Headers({
        origin: "https://portal.example.com",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `__Host-tp_session=${anonymousSession.id}`,
        "cf-access-authenticated-user-email": "admin@example.com"
      }),
      body: form
    });

    const response = await POST(request);
    const rotatedCookie = extractSessionCookie(response);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/portal");
    assert.ok(rotatedCookie.value);
    assert.notEqual(rotatedCookie.value, anonymousSession.id);
    assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false }), null);

    const authenticatedSession = sessions.getSessionById(rotatedCookie.value, { touch: false });
    assert.equal(authenticatedSession?.authenticated, true);
    assert.equal(authenticatedSession?.username, "admin");
    assert.match(rotatedCookie.raw, /HttpOnly/);
    assert.match(rotatedCookie.raw, /Secure/);
    assert.match(rotatedCookie.raw, /SameSite=lax/i);
  } finally {
    env.cleanup();
  }
});

test("login POST keeps failures generic when Access email does not match the configured account", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    TP_FRONTDOOR_USERS_JSON: JSON.stringify([
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ])
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = sessions.createAnonymousSession();
    const request = buildRequest("https://portal.example.com/login", {
      method: "POST",
      headers: new Headers({
        origin: "https://portal.example.com",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `__Host-tp_session=${anonymousSession.id}`,
        "cf-access-authenticated-user-email": "other@example.com"
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login?error=invalid");
    assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false })?.authenticated, false);
  } finally {
    env.cleanup();
  }
});

test("login POST rejects invalid CSRF before credential verification", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    TP_FRONTDOOR_USERS_JSON: JSON.stringify([
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ])
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = sessions.createAnonymousSession();
    const request = buildRequest("https://portal.example.com/login", {
      method: "POST",
      headers: new Headers({
        origin: "https://portal.example.com",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `__Host-tp_session=${anonymousSession.id}`,
        "cf-access-authenticated-user-email": "admin@example.com"
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: "wrong-token"
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login?error=csrf");
    assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false })?.authenticated, false);
  } finally {
    env.cleanup();
  }
});

test("logout POST invalidates the authenticated session and clears the cookie", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");

    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/logout", {
      method: "POST",
      headers: new Headers({
        origin: "https://portal.example.com",
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "x-csrf-token": authenticatedSession.csrfToken
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
  } finally {
    env.cleanup();
  }
});

test("expired sessions are removed for both idle and absolute timeout breaches", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const idleExpired = sessions.createAnonymousSession();
    const absoluteExpired = sessions.createAnonymousSession();
    const db = getDb(env.dbPath);
    const now = Date.now();

    db.prepare("UPDATE sessions SET idle_expires_at = ? WHERE id = ?").run(now - 1_000, idleExpired.id);
    db.prepare("UPDATE sessions SET absolute_expires_at = ? WHERE id = ?").run(now - 1_000, absoluteExpired.id);

    assert.equal(sessions.getSessionById(idleExpired.id, { touch: false }), null);
    assert.equal(sessions.getSessionById(absoluteExpired.id, { touch: false }), null);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap returns actor metadata and CSRF for authenticated sessions", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");

    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.authMode, "managed");
    assert.equal(body.actor.username, "admin");
    assert.equal(body.actor.accessEmail, "admin@example.com");
    assert.equal(body.features.apiKeyInput, false);
    assert.equal(body.features.directDebug, false);
    assert.equal(body.csrfToken, authenticatedSession.csrfToken);
  } finally {
    env.cleanup();
  }
});

test("v1 POST rejects requests missing valid same-origin CSRF protections", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/v1/jobs", {
      method: "POST",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`
      }),
      body: JSON.stringify({ pipeline: "lux-depth-v3" })
    });

    const response = await route.POST(request, {
      params: { path: ["jobs"] }
    });
    const body = await response.json();

    assert.equal(response.status, 403);
    assert.equal(body.error.code, "INVALID_CSRF");
  } finally {
    env.cleanup();
  }
});

test("v1 SSE proxy preserves event-stream framing while injecting backend auth server-side", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const upstreamEvents = [
      'event: state\ndata: {"state":"running"}\n\n',
      'event: log\ndata: {"message":"still working"}\n\n',
      'event: done\ndata: {"ok":true}\n\n'
    ].join("");

    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs/job-123/events");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      assert.equal(init.headers.get("Accept-Encoding"), "identity");
      assert.equal(init.headers.has("cookie"), false);
      assert.equal(init.headers.has("x-csrf-token"), false);

      return new Response(upstreamEvents, {
        status: 200,
        headers: {
          "content-type": "text/event-stream; charset=utf-8",
          "cache-control": "private"
        }
      });
    };

    try {
      const request = buildRequest("https://portal.example.com/v1/jobs/job-123/events", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          accept: "text/event-stream"
        })
      });

      const response = await route.GET(request, {
        params: { path: ["jobs", "job-123", "events"] }
      });

      assert.equal(response.status, 200);
      assert.match(response.headers.get("content-type") || "", /text\/event-stream/);
      assert.equal(response.headers.get("cache-control"), "no-store, no-transform");
      assert.equal(await response.text(), upstreamEvents);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});
