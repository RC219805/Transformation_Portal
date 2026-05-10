import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";

import { NextRequest, NextResponse } from "next/server.js";

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
  "TP_PORTAL_RUM_ROLLOUT_PERCENT",
  "TP_FRONTDOOR_RUM_ENABLED",
  "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT"
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

function withRumEnvironment({
  rumEnabled = false,
  rumFlagValue = "1",
  frontdoorRumEnabled = rumEnabled,
  frontdoorRolloutPercent = "100"
} = {}) {
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

test("homepage GET omits front-door RUM when only the front-door flag is disabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true, frontdoorRumEnabled: false });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const response = await GET(request);
    const html = await response.text();
    const csp = response.headers.get("content-security-policy") || "";

    assert.equal(response.status, 200);
    assert.match(csp, /script-src 'none'/);
    assert.doesNotMatch(html, /landing_rendered/);
    assert.match(response.headers.get("cache-control") || "", /\bpublic\b/i);
  } finally {
    env.cleanup();
  }
});

test("homepage GET samples out front-door RUM when rollout percent is zero", async () => {
  const env = withRumEnvironment({
    rumEnabled: true,
    frontdoorRumEnabled: true,
    frontdoorRolloutPercent: "0"
  });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const response = await GET(request);
    const html = await response.text();
    const csp = response.headers.get("content-security-policy") || "";

    assert.equal(response.status, 200);
    assert.match(csp, /script-src 'none'/);
    assert.doesNotMatch(html, /landing_rendered/);
    assert.match(response.headers.get("cache-control") || "", /\bpublic\b/i);
  } finally {
    env.cleanup();
  }
});

test("homepage GET treats invalid front-door RUM rollout values as sampled out", async () => {
  const env = withRumEnvironment({
    rumEnabled: true,
    frontdoorRumEnabled: true,
    frontdoorRolloutPercent: "not-a-number"
  });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const response = await GET(request);
    const html = await response.text();
    const csp = response.headers.get("content-security-policy") || "";

    assert.equal(response.status, 200);
    assert.match(csp, /script-src 'none'/);
    assert.doesNotMatch(html, /landing_rendered/);
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

test("front-door RUM rollout config defaults, clamps, and fails closed on invalid values", async () => {
  const {
    isFrontdoorRumTelemetryEnabled,
    resolveFrontdoorRumRolloutPercent
  } = await importFresh("../lib/config.js");
  const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

  assert.equal(
    resolveFrontdoorRumRolloutPercent({}),
    100,
    "front-door rollout defaults to 100 when the env value is omitted"
  );
  assert.equal(resolveFrontdoorRumRolloutPercent({ TP_FRONTDOOR_RUM_ROLLOUT_PERCENT: "150" }), 100);
  assert.equal(resolveFrontdoorRumRolloutPercent({ TP_FRONTDOOR_RUM_ROLLOUT_PERCENT: "-10" }), 0);
  assert.equal(resolveFrontdoorRumRolloutPercent({ TP_FRONTDOOR_RUM_ROLLOUT_PERCENT: "bogus" }), 0);

  assert.equal(isFrontdoorRumTelemetryEnabled({
    traceparent,
    env: {
      TP_PORTAL_RUM_ENABLED: "1",
      TP_FRONTDOOR_RUM_ENABLED: "1"
    }
  }), true);
  assert.equal(isFrontdoorRumTelemetryEnabled({
    traceparent,
    env: {
      TP_PORTAL_RUM_ENABLED: "0",
      TP_FRONTDOOR_RUM_ENABLED: "1"
    }
  }), false);
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
  assert.ok(listenerStart >= 0, "expected a submit listener in the login script");
  // Narrow to the submit-handler body specifically. The slice from
  // listenerStart to end-of-script otherwise picks up downstream
  // helpers like scheduleLoginSubmitFailureListener whose legitimate
  // failure-code allowlist contains the string "csrf".
  const onceMarker = body.indexOf("{ once: true }", listenerStart);
  assert.ok(onceMarker > listenerStart, "expected { once: true } closer for submit listener");
  const listenerSection = body.slice(listenerStart, onceMarker);

  // The listener must not block native form submission; the form
  // proceeds to the server unchanged.
  assert.doesNotMatch(listenerSection, /preventDefault\s*\(/);
  assert.doesNotMatch(listenerSection, /stopPropagation\s*\(/);

  // No PII / secret tokens may flow through the emitted metadata. Use
  // word-boundary regex so the legitimate Web Storage API name
  // ("sessionStorage", which the breadcrumb write needs) is not
  // mistakenly flagged as a session-ID leak.
  const piiPatterns = [
    /\busername\b/i,
    /\bpassword\b/i,
    /\bemail\b/i,
    /\bcsrf\b/i,
    /\bsession_?(id|token|hash|key)\b/i,
    /\bthrottle_?(key|id|token)\b/i,
    /\baccessEmail\b/i,
    /\bAuthorization\b/i,
  ];
  for (const pattern of piiPatterns) {
    assert.doesNotMatch(
      listenerSection,
      pattern,
      `submit listener must not match ${pattern}: ${listenerSection.slice(0, 200)}`
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
  assert.ok(listenerStart >= 0, "expected a submit listener in the login script");
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

test("renderRumClientScript writes the tpLoginSubmitStartedAt breadcrumb at submit time", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });

  // Locate the submit-listener body so the breadcrumb assertion is
  // anchored to the actual submit handler (not, e.g., the failure
  // listener's removeItem call below).
  const listenerStart = body.indexOf('addEventListener("submit"');
  assert.ok(listenerStart >= 0, "expected a submit listener in the login script");
  // The submit listener body ends at "{ once: true }". Slice to that
  // closing token so the assertion only inspects the submit branch.
  const onceMarker = body.indexOf("{ once: true }", listenerStart);
  assert.ok(onceMarker > listenerStart, "expected { once: true } closer for submit listener");
  const listenerSection = body.slice(listenerStart, onceMarker);

  // Date.now() epoch-ms breadcrumb (NOT performance.now() — that resets
  // on each navigation and would always read 0 on the redirect target).
  assert.match(listenerSection, /sessionStorage\.setItem\(/);
  assert.match(listenerSection, /LOGIN_SUBMIT_BREADCRUMB_KEY/);
  assert.match(listenerSection, /String\(Date\.now\(\)\)/);
  assert.match(body, /LOGIN_SUBMIT_BREADCRUMB_KEY\s*=\s*"tpLoginSubmitStartedAt"/);
  // Storage failure (private mode) must not abort the submit handler.
  assert.match(listenerSection, /catch\s*\(\s*_storageErr\s*\)/);
});

test("renderRumClientScript exposes a scheduleLoginSubmitFailureListener gated on view=login", async () => {
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

  // The failure listener is wired into bootstrap() alongside the
  // existing login_submit_attempt listener.
  assert.match(loginBody, /scheduleLoginSubmitFailureListener\s*\(\s*\)/);
  assert.match(loginBody, /function\s+scheduleLoginSubmitFailureListener\s*\(\s*\)/);
  // The same view-conditional gate applies — landing view short-circuits.
  assert.match(landingBody, /function\s+scheduleLoginSubmitFailureListener\s*\(\s*\)/);
  assert.match(loginBody, /if\s*\(VIEW\s*!==\s*"login"\)\s*return/);
  // Landing view's VIEW literal pins it away from "login" so the failure
  // branch is unreachable there.
  assert.doesNotMatch(landingBody, /VIEW\s*=\s*"login"/);
});

test("renderRumClientScript failure listener emits login_submit_failure with duration/ms + safe metadata", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });

  const fnStart = body.indexOf("function scheduleLoginSubmitFailureListener");
  assert.ok(fnStart >= 0, "expected scheduleLoginSubmitFailureListener function");
  // Slice to the next top-level `function bootstrap()` so the assertions
  // only inspect the failure-listener body.
  const fnEnd = body.indexOf("function bootstrap", fnStart);
  assert.ok(fnEnd > fnStart, "expected bootstrap() to follow the failure listener");
  const fnBody = body.slice(fnStart, fnEnd);

  // Backend allowlist contract — duration/ms with elapsedMs as the value.
  assert.match(fnBody, /event_type:\s*"login_submit_failure"/);
  assert.match(fnBody, /metric:\s*"duration"/);
  assert.match(fnBody, /unit:\s*"ms"/);
  assert.match(fnBody, /value:\s*elapsedMs/);

  // Sanitized metadata: only source + failure_code (mirrors server-side
  // emitter at lib/rum-emitter.js:88).
  assert.match(fnBody, /source:\s*"client"/);
  assert.match(fnBody, /failure_code:\s*errorCode/);
  assert.match(fnBody, /failureMarker\s*!==\s*errorCode/);
  // Reuses the existing keepalive enqueue path — no separate fetch.
  assert.match(fnBody, /enqueue\(\{[\s\S]+?\},\s*\{\s*keepalive:\s*true\s*\}\)/);
  assert.doesNotMatch(fnBody, /fetch\s*\(/);

  // No PII / secret tokens may flow through the failure metadata. Use
  // word-boundary regex so legitimate appearances of "csrf" and
  // "throttled" (as failure-code allowlist VALUES) and "sessionStorage"
  // (the Web Storage API used to read the breadcrumb) are not mistaken
  // for the corresponding PII tokens.
  const piiPatterns = [
    /\busername\b/i,
    /\bpassword\b/i,
    /\bemail\b/i,
    /\bsession_?(id|token|hash|key)\b/i,
    /\bthrottle_?(key|id|token)\b/i,
    /\baccessEmail\b/i,
    /\bAuthorization\b/i,
    /\bcsrf_?token\b/i,
  ];
  for (const pattern of piiPatterns) {
    assert.doesNotMatch(
      fnBody,
      pattern,
      `failure listener must not match ${pattern}: ${fnBody.slice(0, 200)}`
    );
  }
});

test("renderRumClientScript failure listener clears the breadcrumb on every login load", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });
  const fnStart = body.indexOf("function scheduleLoginSubmitFailureListener");
  const fnEnd = body.indexOf("function bootstrap", fnStart);
  const fnBody = body.slice(fnStart, fnEnd);

  // The failure listener reads AND clears the breadcrumb on every
  // /login load — the removeItem call must precede the early-returns
  // that gate the actual emit, so stale entries can't survive a
  // /login visit that doesn't ultimately emit.
  const getIdx = fnBody.indexOf("sessionStorage.getItem(LOGIN_SUBMIT_BREADCRUMB_KEY)");
  const removeIdx = fnBody.indexOf("sessionStorage.removeItem(LOGIN_SUBMIT_BREADCRUMB_KEY)");
  assert.ok(getIdx >= 0, "expected sessionStorage.getItem call");
  assert.ok(removeIdx > getIdx, "expected sessionStorage.removeItem to follow getItem");

  // A server-set failure marker is also read and cleared on the same
  // /login load. This suppresses stale successful-submit breadcrumbs:
  // success redirects never set the marker, so a later manual
  // /login?error=... cannot emit a client failure.
  const markerReadIdx = fnBody.indexOf("readCookieValue(LOGIN_SUBMIT_FAILURE_MARKER_COOKIE)");
  const markerClearIdx = fnBody.indexOf("clearCookieValue(LOGIN_SUBMIT_FAILURE_MARKER_COOKIE)");
  assert.ok(markerReadIdx > removeIdx, "expected marker read after breadcrumb clear");
  assert.ok(markerClearIdx > markerReadIdx, "expected marker clear after marker read");
  const markerRequiredIdx = fnBody.indexOf("if (!rawStart || !failureMarker) return");
  assert.ok(markerRequiredIdx > markerClearIdx, "expected marker requirement after marker clear");

  // The error-code allowlist gate runs AFTER removeItem so unknown
  // codes still clear the breadcrumb without emitting.
  const allowedIdx = fnBody.indexOf('"csrf"');
  assert.ok(allowedIdx > removeIdx, "expected error-code allowlist check after breadcrumb clear");
  const markerMatchIdx = fnBody.indexOf("failureMarker !== errorCode");
  assert.ok(markerMatchIdx > allowedIdx, "expected marker/error match after allowlist");

  // The freshness cap also runs AFTER removeItem so stale breadcrumbs
  // are discarded rather than left to ride a future submit.
  const freshnessIdx = fnBody.indexOf("LOGIN_SUBMIT_FAILURE_FRESHNESS_MS");
  assert.ok(freshnessIdx > removeIdx, "expected freshness cap check after breadcrumb clear");
});

test("renderRumClientScript failure listener allowlist matches server-side LOGIN_RUM_FAILURE_CODES", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const { LOGIN_RUM_FAILURE_CODES } = await importFresh("../lib/rum-emitter.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });

  // Drift guard: the inline ES5 allowlist must contain every code the
  // server-side emitter recognizes. Adding a new failure code on one
  // side without the other will break this assertion.
  for (const code of Object.values(LOGIN_RUM_FAILURE_CODES)) {
    assert.ok(
      body.includes(`"${code}"`),
      `client failure allowlist missing server-side code: ${code}`
    );
  }
});

test("rum-client.js exports the login_submit_success marker constants for the portal bundle", async () => {
  const {
    LOGIN_SUBMIT_BREADCRUMB_KEY,
    LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE,
    LOGIN_SUBMIT_SUCCESS_MARKER_VALUE,
    LOGIN_SUBMIT_SUCCESS_MARKER_MAX_AGE_SECONDS,
  } = await importFresh("../lib/rum-client.js");

  // Wire-format pins. The portal bundle inlines copies of these
  // strings (see drift test below); changing either side without the
  // other is the most likely way for the success mirror to silently
  // stop firing.
  assert.equal(LOGIN_SUBMIT_BREADCRUMB_KEY, "tpLoginSubmitStartedAt");
  assert.equal(LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE, "tp_login_submit_success");
  assert.equal(LOGIN_SUBMIT_SUCCESS_MARKER_VALUE, "1");
  assert.equal(LOGIN_SUBMIT_SUCCESS_MARKER_MAX_AGE_SECONDS, 60);
});

test("portal bundle source pins the same login_submit_success constants as lib/rum-client.js", async () => {
  const {
    LOGIN_SUBMIT_BREADCRUMB_KEY,
    LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE,
    LOGIN_SUBMIT_SUCCESS_MARKER_MAX_AGE_SECONDS,
  } = await importFresh("../lib/rum-client.js");

  // Drift guard between the two surfaces:
  // - lib/rum-client.js writes the breadcrumb at submit time and the
  //   server-side login route sets the marker cookie on success.
  // - portal-src/portal.template.js reads both on /portal first load
  //   and emits login_submit_success.
  // The portal bundle deliberately inlines the literal strings (no
  // shared import) to keep the bundle self-contained; this test pins
  // both sides against the canonical lib/ exports.
  const here = path.dirname(fileURLToPath(import.meta.url));
  const portalSrc = readFileSync(
    path.resolve(here, "..", "portal-src", "portal.template.js"),
    "utf-8"
  );

  assert.match(
    portalSrc,
    new RegExp(`LOGIN_SUBMIT_BREADCRUMB_KEY\\s*=\\s*['"]${LOGIN_SUBMIT_BREADCRUMB_KEY}['"]`),
    "portal bundle must inline the same breadcrumb key as lib/rum-client.js"
  );
  assert.match(
    portalSrc,
    new RegExp(`LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE\\s*=\\s*['"]${LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE}['"]`),
    "portal bundle must inline the same success-marker cookie name as lib/rum-client.js"
  );
  // Freshness cap is expressed in milliseconds inside the bundle.
  const expectedFreshnessMs = LOGIN_SUBMIT_SUCCESS_MARKER_MAX_AGE_SECONDS * 1000;
  assert.match(
    portalSrc,
    new RegExp(`LOGIN_SUBMIT_SUCCESS_FRESHNESS_MS\\s*=\\s*${expectedFreshnessMs}\\b`),
    "portal bundle freshness cap must equal max-age in ms"
  );
  // The bundle MUST emit the login_submit_success milestone, MUST
  // route through _recordPortalRumMilestone (so the idempotent
  // emittedMilestones guard applies), and MUST carry source="client"
  // metadata so dashboards can dedupe against server-side emissions.
  assert.match(
    portalSrc,
    /_recordPortalRumMilestone\(\s*['"]login_submit_success['"]/,
    "portal bundle must emit login_submit_success via the milestone helper"
  );
  assert.match(
    portalSrc,
    /metadata:\s*\{\s*source:\s*['"]client['"]\s*\}/,
    "portal bundle must tag the success emit with metadata.source='client'"
  );
});

test("portal bundle source clears login_submit_success marker state when sessionStorage read fails", async () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const portalSrc = readFileSync(
    path.resolve(here, "..", "portal-src", "portal.template.js"),
    "utf-8"
  );
  const fnStart = portalSrc.indexOf("function _scheduleLoginSubmitSuccessRum()");
  const fnEnd = portalSrc.indexOf("function _finalizePortalRumVitals", fnStart);
  const fnBody = portalSrc.slice(fnStart, fnEnd);

  assert.ok(fnStart >= 0, "expected login_submit_success scheduler");
  assert.ok(fnEnd > fnStart, "expected scheduler body boundary");

  const getIdx = fnBody.indexOf("sessionStorage.getItem(LOGIN_SUBMIT_BREADCRUMB_KEY)");
  const removeIdx = fnBody.indexOf("sessionStorage.removeItem(LOGIN_SUBMIT_BREADCRUMB_KEY)");
  const markerReadIdx = fnBody.indexOf(
    "_readLoginSubmitMarkerCookie(LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE)"
  );
  const markerClearIdx = fnBody.indexOf(
    "_clearLoginSubmitMarkerCookie(LOGIN_SUBMIT_SUCCESS_MARKER_COOKIE)"
  );
  const markerRequiredIdx = fnBody.indexOf("if (!rawStart || !marker) return");

  assert.ok(getIdx >= 0, "expected sessionStorage getItem call");
  assert.ok(removeIdx > getIdx, "expected breadcrumb removal after read attempt");
  assert.doesNotMatch(
    fnBody.slice(getIdx, removeIdx),
    /\breturn\b/,
    "storage read failures must not skip marker cleanup"
  );
  assert.ok(markerReadIdx > removeIdx, "expected marker read after breadcrumb removal");
  assert.ok(markerClearIdx > markerReadIdx, "expected marker clear after marker read");
  assert.ok(markerRequiredIdx > markerClearIdx, "expected marker requirement after marker clear");
});

test("setRumMarkerCookie preserves explicit falsy marker values", async () => {
  const env = withRumEnvironment();

  try {
    const { setRumMarkerCookie } = await importFresh("../lib/rum-client.js");
    const marker = Object.freeze({
      name: "tp_test_marker",
      path: "/test-marker",
      maxAgeSeconds: 60,
    });
    const cases = [
      { value: 0, expected: "0" },
      { value: false, expected: "false" },
      { value: null, expected: "" },
      { value: undefined, expected: "" },
    ];

    for (const { value, expected } of cases) {
      const response = NextResponse.next();
      setRumMarkerCookie(response, marker, value);
      const cookieHeader = response.headers.get("set-cookie") || "";
      if (expected) {
        assert.match(cookieHeader, new RegExp(`\\btp_test_marker=${expected}\\b`));
      } else {
        assert.match(cookieHeader, /\btp_test_marker=;/);
      }
      assert.match(cookieHeader, /\bPath=\/test-marker\b/);
      assert.match(cookieHeader, /\bMax-Age=60\b/);
      assert.match(cookieHeader, /\bSameSite=Lax\b/i);
      assert.doesNotMatch(cookieHeader, /HttpOnly/i);
    }
  } finally {
    env.cleanup();
  }
});

test("renderRumClientScript failure listener uses Date.now() (not performance.now()) for elapsed time", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });
  const fnStart = body.indexOf("function scheduleLoginSubmitFailureListener");
  const fnEnd = body.indexOf("function bootstrap", fnStart);
  const fnBody = body.slice(fnStart, fnEnd);

  // The failure-side latency calculation uses Date.now() so it remains
  // valid across the form-submit page navigation. performance.now()
  // would reset to 0 on the new page and report a meaningless duration.
  assert.match(fnBody, /Date\.now\(\)\s*-\s*startedAt/);
  assert.doesNotMatch(fnBody, /nowMs\s*\(\s*\)/);
});

test("renderRumClientScript inline failure listener uses ES5-safe syntax only", async () => {
  const { renderRumClientScript } = await importFresh("../lib/rum-client.js");
  const body = renderRumClientScript({
    route: "/login",
    view: "login",
    traceparent: "",
  });
  const fnStart = body.indexOf("function scheduleLoginSubmitFailureListener");
  const fnEnd = body.indexOf("function bootstrap", fnStart);
  const fnBody = body.slice(fnStart, fnEnd);

  // Match the rest of the inline rum-client: var, function declarations,
  // no numeric separators, no arrow functions, no optional chaining,
  // no nullish coalescing.
  assert.doesNotMatch(fnBody, /\b\d[\d_]*_\d+\b/);
  assert.doesNotMatch(fnBody, /=>/);
  assert.doesNotMatch(fnBody, /\?\./);
  assert.doesNotMatch(fnBody, /\?\?/);
  assert.doesNotMatch(fnBody, /\blet\s/);
  assert.doesNotMatch(fnBody, /\bconst\s/);
});

test("login GET HTML embeds the failure listener payload when RUM is enabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/login/route.js");
    // The failure listener must be present in the rendered <script>
    // body regardless of whether the URL carries an ?error= param —
    // the function self-gates on URL inspection at runtime, but the
    // server-rendered script body is the same.
    const request = buildRequest("https://portal.example.com/login?error=invalid");

    const response = await GET(request);
    const html = await response.text();

    assert.equal(response.status, 200);
    assert.match(html, /scheduleLoginSubmitFailureListener\s*\(\s*\)/);
    assert.match(html, /event_type:\s*"login_submit_failure"/);
    assert.match(html, /failure_code:\s*errorCode/);
    assert.match(html, /"tpLoginSubmitStartedAt"/);
    assert.match(html, /"tp_login_submit_failure"/);
  } finally {
    env.cleanup();
  }
});

test("login GET HTML omits the failure listener when RUM is disabled", async () => {
  const env = withRumEnvironment({ rumEnabled: false });

  try {
    getDb(env.dbPath);
    const { GET } = await importFresh("../app/login/route.js");
    const request = buildRequest("https://portal.example.com/login?error=invalid");

    const response = await GET(request);
    const html = await response.text();

    assert.equal(response.status, 200);
    // No inline RUM script means no failure listener.
    assert.doesNotMatch(html, /<script\b/);
    assert.doesNotMatch(html, /scheduleLoginSubmitFailureListener/);
    assert.doesNotMatch(html, /tpLoginSubmitStartedAt/);
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

test("public RUM route noops when only front-door RUM is disabled", async () => {
  const env = withRumEnvironment({ rumEnabled: true, frontdoorRumEnabled: false });

  try {
    const route = await importFresh("../app/v1/portal/rum/route.js");
    const restoreFetch = withMockedFetch(async () => {
      assert.fail("front-door-disabled public RUM requests must not reach the backend");
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
