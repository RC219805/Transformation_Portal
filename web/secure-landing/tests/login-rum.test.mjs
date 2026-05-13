import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createSign, generateKeyPairSync } from "node:crypto";

import argon2 from "argon2";
import { NextRequest } from "next/server.js";

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
  "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT",
];

const TEST_CF_ACCESS_TEAM_DOMAIN = "https://tp-frontdoor-tests.cloudflareaccess.com";
const TEST_CF_ACCESS_AUD = "tp-frontdoor-aud";
const TEST_CF_ACCESS_KID = "tp-frontdoor-key";
const TEST_CF_ACCESS_KEYS = generateKeyPairSync("rsa", { modulusLength: 2048 });
const TEST_CF_ACCESS_PUBLIC_JWK = {
  ...TEST_CF_ACCESS_KEYS.publicKey.export({ format: "jwk" }),
  kid: TEST_CF_ACCESS_KID,
  alg: "RS256",
  use: "sig",
};

const FASTAPI_ORIGIN = "http://127.0.0.1:8000";

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

function withTestEnvironment({
  rumEnabled = true,
  frontdoorRumEnabled = rumEnabled,
  frontdoorRolloutPercent = "100",
  usersFileEntries = []
} = {}) {
  const snapshot = snapshotEnv();
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-login-rum-"));
  const dbPath = path.join(tempDir, "sessions.sqlite");
  const usersFilePath = path.join(tempDir, "frontdoor-users.json");
  writeFileSync(usersFilePath, JSON.stringify(usersFileEntries), "utf-8");

  process.env.NODE_ENV = "production";
  process.env.TP_FASTAPI_ORIGIN = FASTAPI_ORIGIN;
  process.env.TP_BACKEND_API_KEY = "backend-secret";
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  process.env.TP_FRONTDOOR_USERS_FILE = usersFilePath;
  process.env.TP_FRONTDOOR_USERS_JSON = "[]";
  process.env.TP_CF_ACCESS_TEAM_DOMAIN = TEST_CF_ACCESS_TEAM_DOMAIN;
  process.env.TP_CF_ACCESS_AUD = TEST_CF_ACCESS_AUD;
  delete process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS;

  if (rumEnabled) {
    process.env.TP_PORTAL_RUM_ENABLED = "1";
    process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT = "100";
  } else {
    delete process.env.TP_PORTAL_RUM_ENABLED;
    delete process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT;
  }

  if (frontdoorRumEnabled) {
    process.env.TP_FRONTDOOR_RUM_ENABLED = "1";
    process.env.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT = frontdoorRolloutPercent;
  } else {
    delete process.env.TP_FRONTDOOR_RUM_ENABLED;
    delete process.env.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT;
  }

  resetDbCache();
  return {
    dbPath,
    cleanup() {
      resetDbCache();
      restoreEnv(snapshot);
      rmSync(tempDir, { recursive: true, force: true });
    },
  };
}

async function importFresh(relativePath) {
  return import(`${relativePath}?case=${Date.now()}-${Math.random()}`);
}

function buildRequest(url, options = {}) {
  return new NextRequest(url, options);
}

function createAccessJwt(overrides = {}) {
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", kid: TEST_CF_ACCESS_KID, typ: "JWT" };
  const payload = {
    aud: [TEST_CF_ACCESS_AUD],
    email: "admin@example.com",
    exp: now + 300,
    iat: now - 5,
    iss: TEST_CF_ACCESS_TEAM_DOMAIN,
    nbf: now - 5,
    ...overrides,
  };
  const encodedHeader = Buffer.from(JSON.stringify(header)).toString("base64url");
  const encodedPayload = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signingInput = `${encodedHeader}.${encodedPayload}`;
  const signature = createSign("RSA-SHA256")
    .update(signingInput)
    .end()
    .sign(TEST_CF_ACCESS_KEYS.privateKey)
    .toString("base64url");
  return `${signingInput}.${signature}`;
}

function installFetchMock({ rumCalls, rumFetchImpl } = {}) {
  const originalFetch = global.fetch;
  global.fetch = async (url, init) => {
    const target = String(url);
    if (target === `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`) {
      return Response.json({ keys: [TEST_CF_ACCESS_PUBLIC_JWK] }, { status: 200 });
    }
    if (target === `${FASTAPI_ORIGIN}/v1/portal/rum`) {
      const headers = init?.headers instanceof Headers ? init.headers : new Headers(init?.headers || {});
      const headerEntries = {};
      for (const [k, v] of headers.entries()) headerEntries[k] = v;
      const body = init?.body ? JSON.parse(init.body) : null;
      rumCalls?.push({ url: target, body, headers: headerEntries });
      if (rumFetchImpl) return rumFetchImpl(url, init);
      return Response.json(
        {
          schema: "tp.orchestrator.portal_rum_ingest.v1",
          success: true,
          data: { accepted: true, event: body },
          error: null,
        },
        { status: 200 }
      );
    }
    throw new Error(`unmocked fetch: ${target}`);
  };
  return () => {
    global.fetch = originalFetch;
  };
}

async function flushFireAndForget() {
  // Drain the microtask + macrotask queue so emitter Promise.resolve().then(...)
  // chains complete before assertions run.
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

async function buildAdminUserFixture() {
  const passwordHash = await argon2.hash("correct horse battery staple");
  return {
    username: "admin",
    password_hash: passwordHash,
    access_email: "admin@example.com",
    role: "admin",
  };
}

function buildPostRequest({
  session,
  csrfToken,
  password = "correct horse battery staple",
  username = "admin",
  accessEmail = "admin@example.com",
  origin = "https://portal.example.com",
  extraCookies = [],
}) {
  const form = new URLSearchParams({
    username,
    password,
    csrf_token: csrfToken,
  });
  const cookie = [`__Host-tp_session=${session.id}`, ...extraCookies].join("; ");
  return buildRequest(`${origin}/login`, {
    method: "POST",
    headers: new Headers({
      origin,
      "content-type": "application/x-www-form-urlencoded",
      cookie,
      "Cf-Access-Jwt-Assertion": createAccessJwt({ email: accessEmail }),
      "cf-access-authenticated-user-email": accessEmail,
    }),
    body: form,
  });
}

const PII_FORBIDDEN_KEYS = ["username", "accessEmail", "access_email", "role", "session_id", "sessionId", "throttle_key", "throttleKey", "remote_addr", "remoteAddr"];

function assertLoginFailureMarkerCookie(response, failureCode) {
  const cookieHeader = response.headers.get("set-cookie") || "";
  assert.match(cookieHeader, new RegExp(`\\btp_login_submit_failure=${failureCode}\\b`));
  assert.match(cookieHeader, /\bMax-Age=60\b/);
  assert.match(cookieHeader, /\bPath=\/login\b/);
  assert.match(cookieHeader, /\bSameSite=Lax\b/i);
  assert.doesNotMatch(cookieHeader, /HttpOnly[^,]*tp_login_submit_failure/i);
}

function assertClearsLoginFailureMarkerCookie(response) {
  const cookieHeader = response.headers.get("set-cookie") || "";
  assert.match(cookieHeader, /\btp_login_submit_failure=;/);
  assert.match(cookieHeader, /\bPath=\/login\b/);
}

function assertLoginSuccessMarkerCookie(response) {
  const cookieHeader = response.headers.get("set-cookie") || "";
  // Path=/ (NOT /login) so the cookie crosses to /portal where the
  // portal bundle reads it; the value is a fixed presence marker.
  assert.match(cookieHeader, /\btp_login_submit_success=1\b/);
  assert.match(cookieHeader, /\bMax-Age=60\b/);
  assert.match(cookieHeader, /\btp_login_submit_success=[^;]*;[^,]*\bPath=\/(?!login)/);
  assert.match(cookieHeader, /\bSameSite=Lax\b/i);
  // The cookie must be readable from JS so the portal bundle can
  // observe and clear it.
  assert.doesNotMatch(cookieHeader, /HttpOnly[^,]*tp_login_submit_success/i);
}

function assertClearsLoginSuccessMarkerCookie(response) {
  const cookieHeader = response.headers.get("set-cookie") || "";
  assert.match(cookieHeader, /\btp_login_submit_success=;/);
  assert.match(cookieHeader, /\btp_login_submit_success=[^;]*;[^,]*\bPath=\/(?!login)/);
}

function assertNoPiiInRumPosts(rumCalls) {
  for (const call of rumCalls) {
    const flat = JSON.stringify(call.body);
    for (const key of PII_FORBIDDEN_KEYS) {
      assert.ok(!flat.includes(`"${key}"`), `RUM body should not contain ${key}: ${flat}`);
    }
    assert.ok(!flat.includes("admin@example.com"), `RUM body should not contain access email: ${flat}`);
    assert.ok(!flat.includes("correct horse battery staple"), `RUM body should not contain password: ${flat}`);
    // metadata must only carry failure_code (when present)
    if (call.body?.metadata) {
      const allowedKeys = new Set(["failure_code"]);
      for (const key of Object.keys(call.body.metadata)) {
        assert.ok(allowedKeys.has(key), `RUM metadata key should be in closed enum: ${key}`);
      }
    }
  }
}

test("login POST emits no RUM events when the shared RUM master flag is disabled", async () => {
  const env = withTestEnvironment({ rumEnabled: false, usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 0, `expected no RUM POSTs, got ${rumCalls.length}`);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits no RUM events when the front-door RUM flag is disabled", async () => {
  const env = withTestEnvironment({
    rumEnabled: true,
    frontdoorRumEnabled: false,
    usersFileEntries: [await buildAdminUserFixture()]
  });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 0, `expected no RUM POSTs, got ${rumCalls.length}`);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits no RUM events when front-door rollout percent samples out", async () => {
  const env = withTestEnvironment({
    rumEnabled: true,
    frontdoorRumEnabled: true,
    frontdoorRolloutPercent: "0",
    usersFileEntries: [await buildAdminUserFixture()]
  });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 0, `expected sampled-out RUM POSTs, got ${rumCalls.length}`);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits attempt + success events on successful credential rotation", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 2, `expected attempt + success, got ${rumCalls.length}`);
    const [attempt, success] = rumCalls;
    assert.equal(attempt.body.event_type, "login_submit_attempt");
    assert.equal(attempt.body.route, "/login");
    assert.equal(attempt.body.view, "login");
    assert.equal(attempt.body.metric, "count");
    assert.equal(attempt.body.unit, "count");
    assert.equal(attempt.body.value, 1);
    assert.deepEqual(attempt.body.metadata, {});

    assert.equal(success.body.event_type, "login_submit_success");
    assert.equal(success.body.metric, "duration");
    assert.equal(success.body.unit, "ms");
    assert.ok(typeof success.body.value === "number" && success.body.value >= 0);
    assert.deepEqual(success.body.metadata, {});
    assert.doesNotMatch(response.headers.get("set-cookie") || "", /\btp_login_submit_failure=/);
    assertLoginSuccessMarkerCookie(response);

    // Backend auth + traceparent are forwarded.
    assert.equal(attempt.headers["authorization"], "Bearer backend-secret");
    assert.equal(attempt.headers["x-api-key"], "backend-secret");
    assert.match(attempt.headers["traceparent"] || "", /^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$/);
    // Both events share the same traceparent — single decision, single trace per submission.
    assert.equal(attempt.headers["traceparent"], success.headers["traceparent"]);

    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits attempt + failure(csrf) when the CSRF token mismatches", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: "wrong-token" });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[0].body.event_type, "login_submit_attempt");
    assert.equal(rumCalls[1].body.event_type, "login_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "csrf" });
    assertLoginFailureMarkerCookie(response, "csrf");
    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits attempt + failure(configuration) when no users are configured", async () => {
  const env = withTestEnvironment({ usersFileEntries: [] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[1].body.event_type, "login_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "configuration" });
    assertLoginFailureMarkerCookie(response, "configuration");
    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits attempt + failure(access) when CF Access verification fails", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const form = new URLSearchParams({
      username: "admin",
      password: "correct horse battery staple",
      csrf_token: session.csrfToken,
    });
    const request = buildRequest("https://portal.example.com/login", {
      method: "POST",
      headers: new Headers({
        origin: "https://portal.example.com",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `__Host-tp_session=${session.id}`,
        // No Cf-Access-Jwt-Assertion → access verification fails.
      }),
      body: form,
    });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[1].body.event_type, "login_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "access" });
    assertLoginFailureMarkerCookie(response, "access");
    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits attempt + failure(invalid) when the password is wrong", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({
      session,
      csrfToken: session.csrfToken,
      password: "this-is-wrong",
    });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[1].body.event_type, "login_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "invalid" });
    assertLoginFailureMarkerCookie(response, "invalid");
    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST emits attempt + failure(throttled) when the throttle limit is reached", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    // Drive recordLoginAttempt past LOGIN_ATTEMPT_LIMIT (5) so the next POST is
    // throttled. The route handler builds the key from the request's remoteAddr
    // which getRemoteAddress() returns as "unknown" when no proxy headers are
    // set on the test request.
    const remoteAddr = "unknown";
    const throttleKey = `admin@example.com:admin:${remoteAddr}`;
    for (let i = 0; i < 6; i += 1) {
      await sessions.recordLoginAttempt({ throttleKey, success: false, remoteAddr });
    }

    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[1].body.event_type, "login_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "throttled" });
    assertLoginFailureMarkerCookie(response, "throttled");
    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST success clears a stale client failure marker without setting a new one", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({
      session,
      csrfToken: session.csrfToken,
      extraCookies: ["tp_login_submit_failure=invalid"],
    });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/portal");
    assert.equal(rumCalls[1].body.event_type, "login_submit_success");
    assertClearsLoginFailureMarkerCookie(response);
    // The same response also installs the success marker so the portal
    // bundle can recognize the redirect target as a real submit.
    assertLoginSuccessMarkerCookie(response);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST failure clears a stale client success marker without setting a new one", async () => {
  // Cross-marker hygiene: a failed submission must not leave a stale
  // tp_login_submit_success cookie alive on /portal. The two markers
  // are mutually exclusive — at most one can be live for any given
  // submission outcome — so the failure path explicitly clears the
  // opposite cookie when one is observed in the request.
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({
      session,
      csrfToken: "wrong-token",
      extraCookies: ["tp_login_submit_success=1"],
    });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(rumCalls[1].body.event_type, "login_submit_failure");
    assertLoginFailureMarkerCookie(response, "csrf");
    assertClearsLoginSuccessMarkerCookie(response);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("login POST still returns the normal redirect when the RUM emitter fetch rejects", async () => {
  const env = withTestEnvironment({ usersFileEntries: [await buildAdminUserFixture()] });
  const rumCalls = [];
  const restoreFetch = installFetchMock({
    rumCalls,
    rumFetchImpl: async () => {
      throw new Error("simulated upstream RUM outage");
    },
  });
  // Track unhandled rejections — emitter must catch its own.
  const unhandled = [];
  const onUnhandled = (err) => unhandled.push(err);
  process.on("unhandledRejection", onUnhandled);

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const session = await sessions.createAnonymousSession();
    const request = buildPostRequest({ session, csrfToken: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/portal");
    assert.equal(rumCalls.length, 2, "fetch was attempted for both events");
    assert.equal(unhandled.length, 0, "emitter must catch all rejections");
  } finally {
    process.off("unhandledRejection", onUnhandled);
    restoreFetch();
    env.cleanup();
  }
});
