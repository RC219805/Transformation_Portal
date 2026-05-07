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

function withRumEnvironment({ rumEnabled = false } = {}) {
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
    process.env.TP_PORTAL_RUM_ENABLED = "true";
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
