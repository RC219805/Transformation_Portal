import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";

import { NextRequest } from "next/server.js";

import { resetDbCache } from "../lib/db.js";

const ENV_KEYS = [
  "NODE_ENV",
  "TP_FASTAPI_ORIGIN",
  "TP_BACKEND_API_KEY",
  "TP_FRONTDOOR_USERS_FILE",
  "TP_FRONTDOOR_USERS_JSON",
  "TP_FRONTDOOR_SESSION_DB",
  "TP_PORTAL_RUM_ENABLED",
  "TP_PORTAL_RUM_ROLLOUT_PERCENT",
  "TP_FRONTDOOR_RUM_ENABLED",
  "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT",
];

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
  frontdoorRolloutPercent = "100"
} = {}) {
  const snapshot = snapshotEnv();
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-logout-rum-"));
  const dbPath = path.join(tempDir, "sessions.sqlite");
  const usersFilePath = path.join(tempDir, "frontdoor-users.json");
  writeFileSync(usersFilePath, JSON.stringify([]), "utf-8");

  process.env.NODE_ENV = "production";
  process.env.TP_FASTAPI_ORIGIN = FASTAPI_ORIGIN;
  process.env.TP_BACKEND_API_KEY = "backend-secret";
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  process.env.TP_FRONTDOOR_USERS_FILE = usersFilePath;
  process.env.TP_FRONTDOOR_USERS_JSON = "[]";

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

function installFetchMock({ rumCalls, rumFetchImpl } = {}) {
  const originalFetch = global.fetch;
  global.fetch = async (url, init) => {
    const target = String(url);
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

async function buildAuthenticatedSession(sessions) {
  return await sessions.rotateAuthenticatedSession(
    await sessions.createAnonymousSession(),
    {
      username: "admin",
      accessEmail: "admin@example.com",
      role: "admin",
    }
  );
}

function buildLogoutPostRequest({
  session = null,
  csrfHeader = null,
  // The URL the route sees (validateOriginAndReferrer derives the
  // expected origin from this URL, not from the Origin header).
  requestUrlOrigin = "https://portal.example.com",
  // The Origin header the route compares against the URL origin. Pass a
  // distinct value to simulate a cross-origin POST.
  originHeader = "https://portal.example.com",
  extraHeaders = {},
}) {
  const headers = new Headers({
    origin: originHeader,
    "content-type": "application/x-www-form-urlencoded",
    ...extraHeaders,
  });
  if (session) {
    headers.set("cookie", `__Host-tp_session=${session.id}`);
  }
  if (csrfHeader !== null) {
    headers.set("x-csrf-token", csrfHeader);
  }
  return buildRequest(`${requestUrlOrigin}/logout`, { method: "POST", headers });
}

const PII_FORBIDDEN_KEYS = [
  "username",
  "accessEmail",
  "access_email",
  "role",
  "session_id",
  "sessionId",
  "throttle_key",
  "throttleKey",
  "remote_addr",
  "remoteAddr",
];

function assertNoPiiInRumPosts(rumCalls) {
  for (const call of rumCalls) {
    const flat = JSON.stringify(call.body);
    for (const key of PII_FORBIDDEN_KEYS) {
      assert.ok(!flat.includes(`"${key}"`), `RUM body should not contain ${key}: ${flat}`);
    }
    assert.ok(!flat.includes("admin@example.com"), `RUM body should not contain access email: ${flat}`);
    assert.ok(!flat.includes("admin"), `RUM body should not contain username: ${flat}`);
    // Logout failures only carry failure_code; success/attempt carry no metadata at all.
    if (call.body?.metadata) {
      const allowedKeys = new Set(["failure_code"]);
      for (const key of Object.keys(call.body.metadata)) {
        assert.ok(allowedKeys.has(key), `RUM metadata key should be in closed enum: ${key}`);
      }
    }
  }
}

test("logout POST emits no RUM events when the shared RUM master flag is disabled", async () => {
  const env = withTestEnvironment({ rumEnabled: false });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(rumCalls.length, 0, `expected zero RUM calls when disabled, got ${rumCalls.length}`);
    // Session was destroyed even though RUM was off — RUM must never alter behavior.
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST emits no RUM events when the front-door RUM flag is disabled", async () => {
  const env = withTestEnvironment({ rumEnabled: true, frontdoorRumEnabled: false });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(rumCalls.length, 0, `expected zero RUM calls when disabled, got ${rumCalls.length}`);
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST emits no RUM events when front-door rollout percent samples out", async () => {
  const env = withTestEnvironment({
    rumEnabled: true,
    frontdoorRumEnabled: true,
    frontdoorRolloutPercent: "0"
  });
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(rumCalls.length, 0, `expected sampled-out RUM calls, got ${rumCalls.length}`);
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST emits attempt + success on the happy path with an authenticated session", async () => {
  const env = withTestEnvironment();
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(rumCalls.length, 2, `expected attempt + success, got ${rumCalls.length}`);

    const [attempt, success] = rumCalls;
    assert.equal(attempt.body.event_type, "logout_submit_attempt");
    assert.equal(attempt.body.route, "/logout");
    assert.equal(attempt.body.view, "login");
    assert.equal(attempt.body.metric, "count");
    assert.equal(attempt.body.unit, "count");
    assert.equal(attempt.body.value, 1);
    assert.deepEqual(attempt.body.metadata, {});

    assert.equal(success.body.event_type, "logout_submit_success");
    assert.equal(success.body.route, "/logout");
    assert.equal(success.body.view, "login");
    assert.equal(success.body.metric, "duration");
    assert.equal(success.body.unit, "ms");
    assert.ok(typeof success.body.value === "number" && success.body.value >= 0);
    assert.deepEqual(success.body.metadata, {});

    // Session is destroyed on the success path.
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);

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

test("logout POST emits attempt + success even when the request carries no session", async () => {
  // An anonymous logout (no __Host-tp_session cookie) is still a successful
  // no-op: the route returns the same 303 to /login whether or not there was
  // a session to destroy. RUM must capture both branches uniformly so a
  // dashboard delta isn't biased by the cookie's presence.
  const env = withTestEnvironment();
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const request = buildLogoutPostRequest({ session: null, csrfHeader: null });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[0].body.event_type, "logout_submit_attempt");
    assert.equal(rumCalls[1].body.event_type, "logout_submit_success");
    assert.deepEqual(rumCalls[1].body.metadata, {});

    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST emits attempt + failure(csrf) when Origin/Referrer validation fails", async () => {
  const env = withTestEnvironment();
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    // Cross-origin POST: the route's validateOriginAndReferrer rejects
    // this (Origin header doesn't match the URL's origin) before even
    // looking at x-csrf-token.
    const request = buildLogoutPostRequest({
      session,
      csrfHeader: session.csrfToken,
      requestUrlOrigin: "https://portal.example.com",
      originHeader: "https://attacker.example.com",
    });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.match(response.headers.get("location") || "", /\/login\?error=csrf$/);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[1].body.event_type, "logout_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "csrf" });

    // Existing route behavior: Origin/Referrer failure does NOT destroy the
    // session in storage, only clears the cookie. RUM emit must not change
    // that.
    assert.notEqual(await sessions.getSessionById(session.id, { touch: false }), null);

    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST emits attempt + failure(csrf) when the x-csrf-token header mismatches", async () => {
  const env = withTestEnvironment();
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: "wrong-token" });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.match(response.headers.get("location") || "", /\/login\?error=csrf$/);
    assert.equal(rumCalls.length, 2);
    assert.equal(rumCalls[0].body.event_type, "logout_submit_attempt");
    assert.equal(rumCalls[1].body.event_type, "logout_submit_failure");
    assert.deepEqual(rumCalls[1].body.metadata, { failure_code: "csrf" });

    // CSRF token failure does NOT destroy the session in storage either.
    assert.notEqual(await sessions.getSessionById(session.id, { touch: false }), null);

    assertNoPiiInRumPosts(rumCalls);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST still completes the redirect when the RUM emitter fetch rejects", async () => {
  const env = withTestEnvironment();
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
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    // Session is still destroyed even when RUM fetches reject — RUM is
    // strictly best-effort and cannot change the session-destruction path.
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);
    assert.equal(rumCalls.length, 2, "fetch was attempted for both events");
    assert.equal(unhandled.length, 0, "emitter must catch all rejections");
  } finally {
    process.off("unhandledRejection", onUnhandled);
    restoreFetch();
    env.cleanup();
  }
});

test("logout POST does not emit RUM events when no backend api key is configured", async () => {
  const env = withTestEnvironment();
  // Simulate a misconfigured deployment where TP_BACKEND_API_KEY is unset.
  // The emitter short-circuits on missing config.backendApiKey, so no events
  // should reach FastAPI even though the front-door RUM env flag is on.
  delete process.env.TP_BACKEND_API_KEY;
  const rumCalls = [];
  const restoreFetch = installFetchMock({ rumCalls });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");
    const session = await buildAuthenticatedSession(sessions);
    const request = buildLogoutPostRequest({ session, csrfHeader: session.csrfToken });

    const response = await POST(request);
    await flushFireAndForget();

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.equal(rumCalls.length, 0);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});
