import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";

import { NextRequest } from "next/server.js";

import { getDb, resetDbCache } from "../lib/db.js";

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
  "TP_PORTAL_RUM_ROLLOUT_PERCENT"
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

function withRumEnvironment({ rumEnabled = false, rumFlagValue = "1" } = {}) {
  const snapshot = snapshotEnv();
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-rum-"));
  const dbPath = path.join(tempDir, "sessions.sqlite");
  const usersFilePath = path.join(tempDir, "frontdoor-users.json");
  writeFileSync(usersFilePath, JSON.stringify([]), "utf-8");

  process.env.NODE_ENV = "production";
  process.env.TP_FASTAPI_ORIGIN = "http://127.0.0.1:8000";
  process.env.TP_BACKEND_API_KEY = "backend-secret";
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  process.env.TP_FRONTDOOR_USERS_FILE = usersFilePath;
  process.env.TP_FRONTDOOR_USERS_JSON = "[]";
  process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS = "1";
  delete process.env.TP_CF_ACCESS_TEAM_DOMAIN;
  delete process.env.TP_CF_ACCESS_AUD;

  if (rumEnabled) {
    process.env.TP_PORTAL_RUM_ENABLED = rumFlagValue;
    process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT = "100";
  } else {
    delete process.env.TP_PORTAL_RUM_ENABLED;
    delete process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT;
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

function buildRequest(url, options = {}) {
  return new NextRequest(url, options);
}

function withMockedFetch(handler) {
  const originalFetch = global.fetch;
  global.fetch = handler;
  return () => {
    global.fetch = originalFetch;
  };
}

function extractScriptNonce(html) {
  const match = html.match(/<script\s+nonce="([^"]+)"/);
  return match ? match[1] : null;
}

function extractCspScriptNonce(csp) {
  const match = String(csp || "").match(/script-src 'nonce-([^']+)'/);
  return match ? match[1] : null;
}

test("homepage GET omits the RUM script tag and keeps script-src 'none' when RUM is disabled", async () => {
  const env = withRumEnvironment({ rumEnabled: false });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const response = await GET(request);
    const html = await response.text();
    const csp = response.headers.get("content-security-policy") || "";

    assert.equal(response.status, 200);
    assert.match(csp, /script-src 'none'/);
    assert.doesNotMatch(csp, /script-src 'nonce-/);
    assert.doesNotMatch(html, /<script\b/);
    assert.match(response.headers.get("cache-control") || "", /\bpublic\b/i);
  } finally {
    env.cleanup();
  }
});

test("homepage GET injects a nonced inline RUM script and matching CSP when RUM is enabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const response = await GET(request);
    const html = await response.text();
    const csp = response.headers.get("content-security-policy") || "";
    const cacheControl = response.headers.get("cache-control") || "";
    const htmlNonce = extractScriptNonce(html);
    const cspNonce = extractCspScriptNonce(csp);

    assert.equal(response.status, 200);
    assert.match(cacheControl, /no-store/);
    assert.ok(htmlNonce, "expected an inline <script nonce=\"...\"> tag");
    assert.ok(cspNonce, "expected CSP to carry script-src 'nonce-...'");
    assert.equal(htmlNonce, cspNonce);
    assert.doesNotMatch(csp, /script-src 'none'/);
    assert.match(html, /event_type:\s*sample\.event_type/);
    assert.match(html, /"landing_rendered"/);
    assert.match(html, /ROUTE\s*=\s*"\/"/);
    assert.match(html, /VIEW\s*=\s*"landing"/);
    assert.match(html, /"\/v1\/portal\/rum"/);
  } finally {
    env.cleanup();
  }
});

test("login GET injects a nonced inline RUM script with /login + login view when RUM is enabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/login/route.js");
    const request = buildRequest("https://portal.example.com/login");

    const response = await GET(request);
    const html = await response.text();
    const csp = response.headers.get("content-security-policy") || "";
    const htmlNonce = extractScriptNonce(html);
    const cspNonce = extractCspScriptNonce(csp);

    assert.equal(response.status, 200);
    assert.ok(htmlNonce, "expected an inline <script nonce=\"...\"> tag");
    assert.ok(cspNonce, "expected CSP to carry script-src 'nonce-...'");
    assert.equal(htmlNonce, cspNonce);
    assert.doesNotMatch(csp, /script-src 'none'/);
    assert.match(html, /"login_rendered"/);
    assert.match(html, /ROUTE\s*=\s*"\/login"/);
    assert.match(html, /VIEW\s*=\s*"login"/);
    assert.match(html, /"\/v1\/portal\/rum"/);
  } finally {
    env.cleanup();
  }
});

test("renderRumClientScript embeds caller-supplied route, view, and traceparent verbatim", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

  const body = renderRumClientScript({
    route: "/",
    view: "landing",
    traceparent,
  });

  assert.equal(typeof body, "string");
  assert.doesNotMatch(body, /^<script/);
  assert.match(body, /ROUTE\s*=\s*"\/"/);
  assert.match(body, /VIEW\s*=\s*"landing"/);
  assert.match(body, /RENDERED_EVENT\s*=\s*"landing_rendered"/);
  assert.match(body, new RegExp(`TRACEPARENT\\s*=\\s*"${traceparent}"`));
  assert.match(body, /"\/v1\/portal\/rum"/);
  assert.doesNotMatch(body, /\beval\s*\(/);
  assert.doesNotMatch(body, /\bnew\s+Function\b/);
});

test("renderRumClientScript omits invalid or missing traceparent values", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");

  const missingTraceparentBody = renderRumClientScript({
    route: "/",
    view: "landing",
    traceparent: undefined,
  });
  const invalidTraceparentBody = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "undefined",
  });

  assert.match(missingTraceparentBody, /TRACEPARENT\s*=\s*""/);
  assert.match(invalidTraceparentBody, /TRACEPARENT\s*=\s*""/);
  assert.doesNotMatch(missingTraceparentBody, /TRACEPARENT\s*=\s*"undefined"/);
  assert.doesNotMatch(invalidTraceparentBody, /TRACEPARENT\s*=\s*"undefined"/);
  assert.doesNotMatch(invalidTraceparentBody, /TRACEPARENT\s*=\s*"null"/);
});

test("homepage GET mints a fresh nonce on every request when RUM is enabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");

    const firstResponse = await GET(buildRequest("https://portal.example.com/"));
    const firstNonce = extractCspScriptNonce(firstResponse.headers.get("content-security-policy") || "");
    const secondResponse = await GET(buildRequest("https://portal.example.com/"));
    const secondNonce = extractCspScriptNonce(secondResponse.headers.get("content-security-policy") || "");

    assert.ok(firstNonce);
    assert.ok(secondNonce);
    assert.notEqual(firstNonce, secondNonce);
  } finally {
    env.cleanup();
  }
});

test("renderRumClientScript wires a login-form submit listener only for view=login", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const loginBody = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });
  const landingBody = renderRumClientScript({
    route: "/",
    view: "landing",
    traceparent: "",
  });

  // The login script registers a submit listener against the stable
  // data-ui="login-form" selector with { once: true } so re-submits
  // within the same page render don't double-count.
  assert.match(loginBody, /scheduleLoginSubmitListener\s*\(\s*\)/);
  assert.match(loginBody, /\[data-ui="login-form"\]/);
  assert.match(loginBody, /addEventListener\("submit",[\s\S]+?\{\s*once:\s*true\s*\}\)/);

  // The view-conditional gate keeps the listener body inert on the
  // landing view: scheduleLoginSubmitListener is still defined in the
  // bundled IIFE for both views, but the function returns early when
  // VIEW !== "login".
  assert.match(loginBody, /if\s*\(VIEW\s*!==\s*"login"\)\s*return/);
  assert.match(landingBody, /if\s*\(VIEW\s*!==\s*"login"\)\s*return/);
  // The landing script's VIEW literal pins it away from "login" so the
  // form-submit branch is unreachable there.
  assert.match(landingBody, /VIEW\s*=\s*"landing"/);
  assert.doesNotMatch(landingBody, /VIEW\s*=\s*"login"/);
});

test("renderRumClientScript emits login_submit_attempt with count/count/1 + safe metadata", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });

  // Locate the submit listener body so we assert against the exact
  // emission shape, not unrelated parts of the script.
  const listenerStart = body.indexOf('addEventListener("submit"');
  assert.ok(listenerStart >= 0, "expected a submit listener in the login script");
  const listenerSection = body.slice(listenerStart);

  // Backend allowlist contract — count/count/1 only.
  assert.match(listenerSection, /event_type:\s*"login_submit_attempt"/);
  assert.match(listenerSection, /metric:\s*"count"/);
  assert.match(listenerSection, /unit:\s*"count"/);
  assert.match(listenerSection, /value:\s*1\b/);

  // Sanitized metadata: only the two allowed keys.
  assert.match(listenerSection, /source:\s*"client"/);
  assert.match(listenerSection, /duration_ms:\s*elapsedMs/);
  // duration_ms is computed once at submit time and clamped to a
  // non-negative integer millisecond count.
  assert.match(listenerSection, /Math\.max\(0,\s*Math\.round\(nowMs\(\)\)\)/);
});

test("renderRumClientScript login submit listener never preventDefaults or leaks PII", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });
  const listenerStart = body.indexOf('addEventListener("submit"');
  const listenerSection = body.slice(listenerStart);

  // The listener must not block native form submission; the form
  // proceeds to the server unchanged.
  assert.doesNotMatch(listenerSection, /preventDefault\s*\(/);
  assert.doesNotMatch(listenerSection, /stopPropagation\s*\(/);

  // No PII / secret tokens may flow through the emitted metadata.
  for (const piiToken of [
    "username",
    "password",
    "email",
    "csrf",
    "session",
    "throttle",
    "accessEmail",
    "Authorization",
  ]) {
    assert.ok(
      !listenerSection.includes(piiToken),
      `submit listener must not reference ${piiToken}: ${listenerSection.slice(0, 200)}`
    );
  }
});

test("renderRumClientScript routes the submit emission through keepalive fetch", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });
  const listenerStart = body.indexOf('addEventListener("submit"');
  const listenerSection = body.slice(listenerStart);

  // The listener uses the existing enqueue/postSample path with
  // keepalive: true so the request survives the form's page navigation.
  assert.match(listenerSection, /enqueue\(\{[\s\S]+?\},\s*\{\s*keepalive:\s*true\s*\}\)/);
  // The postSample helper (shared with the existing observers) is the
  // single transport — no separate fetch in the submit branch.
  assert.doesNotMatch(listenerSection, /fetch\s*\(/);
});

test("login GET HTML omits the submit-listener selector when RUM is disabled", async () => {
  const env = withRumEnvironment({ rumEnabled: false });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/login/route.js");
    const request = buildRequest("https://portal.example.com/login");

    const response = await GET(request);
    const html = await response.text();

    assert.equal(response.status, 200);
    // No inline RUM script means no submit listener is registered.
    assert.doesNotMatch(html, /<script\b/);
    assert.doesNotMatch(html, /data-ui="login-form"[\s\S]*?login_submit_attempt/);
  } finally {
    env.cleanup();
  }
});

test("login GET HTML embeds the submit listener payload when RUM is enabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/login/route.js");
    const request = buildRequest("https://portal.example.com/login");

    const response = await GET(request);
    const html = await response.text();

    assert.equal(response.status, 200);
    // The login form's stable data-ui selector is present in BOTH the
    // form markup and the inline RUM script's listener registration.
    assert.match(html, /<form[^>]*data-ui="login-form"/);
    assert.match(html, /\[data-ui="login-form"\]/);
    assert.match(html, /event_type:\s*"login_submit_attempt"/);
    assert.match(html, /source:\s*"client"/);
  } finally {
    env.cleanup();
  }
});

test("applySecurityHeaders inserts script nonces into the script-src directive", async () => {
  const { applySecurityHeaders } = await importFresh("../lib/http.js");
  const response = new Response("ok");

  applySecurityHeaders(response, {
    csp: "default-src 'self'; object-src 'none'; script-src 'self' https://cdn.example.test",
    scriptNonce: "nonce-value"
  });

  assert.equal(
    response.headers.get("content-security-policy"),
    "default-src 'self'; object-src 'none'; script-src 'nonce-nonce-value' 'self' https://cdn.example.test"
  );
});

test("applySecurityHeaders fails closed when a script nonce cannot be inserted", async () => {
  const { applySecurityHeaders } = await importFresh("../lib/http.js");
  const response = new Response("ok");

  assert.throws(
    () => applySecurityHeaders(response, {
      csp: "default-src 'self'; object-src 'none'",
      scriptNonce: "nonce-value"
    }),
    /script-src directive required/
  );
});

test("public RUM route proxies same-origin unauthenticated samples with backend auth", async () => {
  const env = withRumEnvironment({ rumEnabled: true });
  const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
  const upstreamTraceparent = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01";

  try {
    const route = await importFresh("../app/v1/portal/rum/route.js");
    const restoreFetch = withMockedFetch(async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/portal/rum");
      assert.equal(init.method, "POST");
      assert.equal(init.headers.get("authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      assert.equal(init.headers.get("traceparent"), traceparent);
      assert.equal(init.headers.get("x-forwarded-host"), "portal.example.com");
      assert.equal(init.headers.get("x-forwarded-proto"), "https");
      assert.equal(init.headers.has("cookie"), false);
      assert.equal(init.headers.has("x-csrf-token"), false);
      assert.equal(await new Response(init.body).text(), JSON.stringify({
        event_type: "landing_rendered",
        route: "/",
        view: "landing",
        metric: "duration",
        value: 25,
        unit: "ms"
      }));
      return Response.json(
        {
          schema: "tp.orchestrator.portal_rum_ingest.v1",
          success: true,
          data: { accepted: true },
          error: null
        },
        {
          headers: {
            traceparent: upstreamTraceparent
          }
        }
      );
    });

    try {
      const request = buildRequest("https://portal.example.com/v1/portal/rum", {
        method: "POST",
        headers: new Headers({
          cookie: "__Host-tp_session=browser-session",
          "content-type": "application/json",
          origin: "https://portal.example.com",
          referer: "https://portal.example.com/",
          traceparent,
          "x-csrf-token": "browser-csrf-token"
        }),
        body: JSON.stringify({
          event_type: "landing_rendered",
          route: "/",
          view: "landing",
          metric: "duration",
          value: 25,
          unit: "ms"
        })
      });

      const response = await route.POST(request);
      const body = await response.json();

      assert.equal(response.status, 200);
      assert.equal(response.headers.get("cache-control"), "no-store");
      assert.equal(response.headers.get("traceparent"), upstreamTraceparent);
      assert.deepEqual(body.data, { accepted: true });
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("public RUM route rejects cross-origin samples before proxying", async () => {
  const env = withRumEnvironment({ rumEnabled: true });

  try {
    const route = await importFresh("../app/v1/portal/rum/route.js");
    const restoreFetch = withMockedFetch(async () => {
      assert.fail("cross-origin public RUM requests must not reach the backend");
    });

    try {
      const request = buildRequest("https://portal.example.com/v1/portal/rum", {
        method: "POST",
        headers: new Headers({
          "content-type": "application/json",
          origin: "https://attacker.example.com"
        }),
        body: JSON.stringify({
          event_type: "landing_rendered",
          route: "/",
          view: "landing",
          metric: "duration",
          value: 25,
          unit: "ms"
        })
      });

      const response = await route.POST(request);
      const body = await response.json();

      assert.equal(response.status, 403);
      assert.equal(body.error.code, "INVALID_CSRF");
      assert.equal(body.error.details.path, "/v1/portal/rum");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("public RUM route noops without proxying when RUM is disabled", async () => {
  const env = withRumEnvironment({ rumEnabled: false });

  try {
    const route = await importFresh("../app/v1/portal/rum/route.js");
    const restoreFetch = withMockedFetch(async () => {
      assert.fail("disabled public RUM requests must not reach the backend");
    });

    try {
      const request = buildRequest("https://portal.example.com/v1/portal/rum", {
        method: "POST",
        headers: new Headers({
          "content-type": "application/json",
          origin: "https://portal.example.com"
        }),
        body: JSON.stringify({
          event_type: "landing_rendered",
          route: "/",
          view: "landing",
          metric: "duration",
          value: 25,
          unit: "ms"
        })
      });

      const response = await route.POST(request);
      const body = await response.json();

      assert.equal(response.status, 200);
      assert.deepEqual(body.data, { accepted: false, disabled: true });
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});
