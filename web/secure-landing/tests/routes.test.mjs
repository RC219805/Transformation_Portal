import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createSign, generateKeyPairSync } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";

import argon2 from "argon2";
import { transformSync } from "esbuild";
import { NextRequest, NextResponse } from "next/server.js";

import { getDb, resetDbCache } from "../lib/db.js";

const ENV_KEYS = [
  "NODE_ENV",
  "TP_FASTAPI_ORIGIN",
  "TP_BACKEND_API_KEY",
  "TP_FRONTDOOR_USERS_FILE",
  "TP_FRONTDOOR_USERS_JSON",
  "TP_FRONTDOOR_SESSION_DB",
  "TP_FRONTDOOR_SESSION_SCALING_MODE",
  "TP_FRONTDOOR_SESSION_STORE",
  "TP_FRONTDOOR_REDIS_URL",
  "TP_FRONTDOOR_REDIS_KEY_PREFIX",
  "TP_CF_ACCESS_TEAM_DOMAIN",
  "TP_CF_ACCESS_AUD",
  "TP_ALLOW_LOCAL_ACCESS_BYPASS",
  "TP_PORTAL_UPLOAD_STAGING_ENABLED",
  "TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT",
  "TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT",
  "TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT",
  "TP_PORTAL_RUM_ENABLED",
  "TP_PORTAL_RUM_ROLLOUT_PERCENT",
  "TP_FRONTDOOR_RUM_ENABLED",
  "TP_FRONTDOOR_RUM_ROLLOUT_PERCENT",
  "TP_PORTAL_FASTVLM_CAPTIONING_ENABLED",
  "TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT"
];

const TEST_CF_ACCESS_TEAM_DOMAIN = "https://tp-frontdoor-tests.cloudflareaccess.com";
const TEST_CF_ACCESS_AUD = "tp-frontdoor-aud";
const TEST_CF_ACCESS_KID = "tp-frontdoor-key";
const FRONTDOOR_APP_ROOT = fileURLToPath(new URL("../", import.meta.url));
const REPO_ROOT = path.resolve(FRONTDOOR_APP_ROOT, "../..");
const TEST_CF_ACCESS_KEYS = generateKeyPairSync("rsa", {
  modulusLength: 2048
});
const TEST_CF_ACCESS_ROTATED_KEYS = generateKeyPairSync("rsa", {
  modulusLength: 2048
});
const TEST_CF_ACCESS_PUBLIC_JWK = {
  ...TEST_CF_ACCESS_KEYS.publicKey.export({ format: "jwk" }),
  kid: TEST_CF_ACCESS_KID,
  alg: "RS256",
  use: "sig"
};
const TEST_CF_ACCESS_ROTATED_KID = "tp-frontdoor-key-rotated";
const TEST_CF_ACCESS_ROTATED_PUBLIC_JWK = {
  ...TEST_CF_ACCESS_ROTATED_KEYS.publicKey.export({ format: "jwk" }),
  kid: TEST_CF_ACCESS_ROTATED_KID,
  alg: "RS256",
  use: "sig"
};

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
  const dbPath = typeof overrides.TP_FRONTDOOR_SESSION_DB === "string"
    ? overrides.TP_FRONTDOOR_SESSION_DB
    : path.join(tempDir, "sessions.sqlite");
  const usersFilePath = path.join(tempDir, "frontdoor-users.json");

  process.env.NODE_ENV = overrides.NODE_ENV ?? "production";
  process.env.TP_FASTAPI_ORIGIN = overrides.TP_FASTAPI_ORIGIN ?? "http://127.0.0.1:8000";
  process.env.TP_BACKEND_API_KEY = overrides.TP_BACKEND_API_KEY ?? "backend-secret";
  process.env.TP_FRONTDOOR_SESSION_DB = dbPath;
  process.env.TP_CF_ACCESS_TEAM_DOMAIN = overrides.TP_CF_ACCESS_TEAM_DOMAIN ?? TEST_CF_ACCESS_TEAM_DOMAIN;
  process.env.TP_CF_ACCESS_AUD = overrides.TP_CF_ACCESS_AUD ?? TEST_CF_ACCESS_AUD;

  if (typeof overrides.TP_FRONTDOOR_USERS_FILE === "string") {
    process.env.TP_FRONTDOOR_USERS_FILE = overrides.TP_FRONTDOOR_USERS_FILE;
  } else if (Array.isArray(overrides.usersFileEntries)) {
    writeFileSync(usersFilePath, JSON.stringify(overrides.usersFileEntries), "utf-8");
    process.env.TP_FRONTDOOR_USERS_FILE = usersFilePath;
  } else {
    delete process.env.TP_FRONTDOOR_USERS_FILE;
  }

  process.env.TP_FRONTDOOR_USERS_JSON = overrides.TP_FRONTDOOR_USERS_JSON ?? "[]";

  if (typeof overrides.TP_FRONTDOOR_SESSION_SCALING_MODE === "string") {
    process.env.TP_FRONTDOOR_SESSION_SCALING_MODE = overrides.TP_FRONTDOOR_SESSION_SCALING_MODE;
  } else {
    delete process.env.TP_FRONTDOOR_SESSION_SCALING_MODE;
  }

  if (typeof overrides.TP_FRONTDOOR_SESSION_STORE === "string") {
    process.env.TP_FRONTDOOR_SESSION_STORE = overrides.TP_FRONTDOOR_SESSION_STORE;
  } else {
    delete process.env.TP_FRONTDOOR_SESSION_STORE;
  }

  if (typeof overrides.TP_FRONTDOOR_REDIS_URL === "string") {
    process.env.TP_FRONTDOOR_REDIS_URL = overrides.TP_FRONTDOOR_REDIS_URL;
  } else {
    delete process.env.TP_FRONTDOOR_REDIS_URL;
  }

  if (typeof overrides.TP_FRONTDOOR_REDIS_KEY_PREFIX === "string") {
    process.env.TP_FRONTDOOR_REDIS_KEY_PREFIX = overrides.TP_FRONTDOOR_REDIS_KEY_PREFIX;
  } else {
    delete process.env.TP_FRONTDOOR_REDIS_KEY_PREFIX;
  }

  if (typeof overrides.TP_ALLOW_LOCAL_ACCESS_BYPASS === "string") {
    process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS = overrides.TP_ALLOW_LOCAL_ACCESS_BYPASS;
  } else {
    delete process.env.TP_ALLOW_LOCAL_ACCESS_BYPASS;
  }

  if (typeof overrides.TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT === "string") {
    process.env.TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT =
      overrides.TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT;
  } else {
    delete process.env.TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT;
  }

  if (typeof overrides.TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT === "string") {
    process.env.TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT =
      overrides.TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT;
  } else {
    delete process.env.TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT;
  }

  if (typeof overrides.TP_PORTAL_UPLOAD_STAGING_ENABLED === "string") {
    process.env.TP_PORTAL_UPLOAD_STAGING_ENABLED = overrides.TP_PORTAL_UPLOAD_STAGING_ENABLED;
  } else {
    delete process.env.TP_PORTAL_UPLOAD_STAGING_ENABLED;
  }

  if (typeof overrides.TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT === "string") {
    process.env.TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT = overrides.TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT;
  } else {
    delete process.env.TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT;
  }

  if (typeof overrides.TP_PORTAL_RUM_ENABLED === "string") {
    process.env.TP_PORTAL_RUM_ENABLED = overrides.TP_PORTAL_RUM_ENABLED;
  } else {
    delete process.env.TP_PORTAL_RUM_ENABLED;
  }

  if (typeof overrides.TP_PORTAL_RUM_ROLLOUT_PERCENT === "string") {
    process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT = overrides.TP_PORTAL_RUM_ROLLOUT_PERCENT;
  } else {
    delete process.env.TP_PORTAL_RUM_ROLLOUT_PERCENT;
  }

  if (typeof overrides.TP_FRONTDOOR_RUM_ENABLED === "string") {
    process.env.TP_FRONTDOOR_RUM_ENABLED = overrides.TP_FRONTDOOR_RUM_ENABLED;
  } else {
    delete process.env.TP_FRONTDOOR_RUM_ENABLED;
  }

  if (typeof overrides.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT === "string") {
    process.env.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT = overrides.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT;
  } else {
    delete process.env.TP_FRONTDOOR_RUM_ROLLOUT_PERCENT;
  }

  if (typeof overrides.TP_PORTAL_FASTVLM_CAPTIONING_ENABLED === "string") {
    process.env.TP_PORTAL_FASTVLM_CAPTIONING_ENABLED = overrides.TP_PORTAL_FASTVLM_CAPTIONING_ENABLED;
  } else {
    delete process.env.TP_PORTAL_FASTVLM_CAPTIONING_ENABLED;
  }

  if (typeof overrides.TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT === "string") {
    process.env.TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT =
      overrides.TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT;
  } else {
    delete process.env.TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT;
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

test("next config honors TP_NEXT_DIST_DIR for isolated local frontdoor runs", async () => {
  const previous = process.env.TP_NEXT_DIST_DIR;
  const previousVercel = process.env.VERCEL;
  process.env.TP_NEXT_DIST_DIR = ".next-smoke-test";
  delete process.env.VERCEL;

  try {
    const configModule = await importFresh("../next.config.js");
    const productionConfig = configModule.default("phase-production-build");
    const developmentConfig = configModule.default("phase-development-server");

    assert.equal(productionConfig.distDir, ".next-smoke-test");
    assert.equal(developmentConfig.distDir, ".next-smoke-test");
    assert.equal(path.resolve(productionConfig.turbopack.root), REPO_ROOT);
    assert.equal(path.resolve(developmentConfig.turbopack.root), REPO_ROOT);
    assert.equal(developmentConfig.output, undefined);
    assert.equal(developmentConfig.outputFileTracingRoot, undefined);
    assert.equal(productionConfig.output, "standalone");
    assert.equal(
      path.resolve(productionConfig.outputFileTracingRoot),
      REPO_ROOT,
      "standalone tracing must still include repo-root files such as the portal asset manifest"
    );
  } finally {
    if (typeof previous === "string") {
      process.env.TP_NEXT_DIST_DIR = previous;
    } else {
      delete process.env.TP_NEXT_DIST_DIR;
    }
    if (typeof previousVercel === "string") {
      process.env.VERCEL = previousVercel;
    } else {
      delete process.env.VERCEL;
    }
  }
});

test("next config delegates deployment output to Vercel without disabling tracing", async () => {
  const previousVercel = process.env.VERCEL;
  process.env.VERCEL = "1";

  try {
    const configModule = await importFresh("../next.config.js");
    const productionConfig = configModule.default("phase-production-build");

    assert.equal(
      productionConfig.output,
      undefined,
      "Vercel's deployment adapter owns its build output and must not also copy standalone traces"
    );
    assert.equal(path.resolve(productionConfig.outputFileTracingRoot), REPO_ROOT);
    assert.deepEqual(productionConfig.outputFileTracingIncludes, {
      "/portal/assets/[...path]": ["../../config/portal_asset_manifest.json"]
    });
  } finally {
    if (typeof previousVercel === "string") {
      process.env.VERCEL = previousVercel;
    } else {
      delete process.env.VERCEL;
    }
  }
});

function frontdoorWebServerEntry(config) {
  // The webServer field is now an array — entry 0 is the mock FastAPI
  // origin, entry 1 is the front-door dev server. The front-door entry
  // is identified by its `npm run dev` command.
  const entries = Array.isArray(config.webServer) ? config.webServer : [config.webServer];
  return entries.find((entry) => /npm run dev/.test(entry.command));
}

function mockFastapiWebServerEntry(config) {
  const entries = Array.isArray(config.webServer) ? config.webServer : [config.webServer];
  return entries.find((entry) => /mock-fastapi-origin/.test(entry.command));
}

// Env vars that influence playwright.config.mjs's runtime shape — the
// config tests below pin defaults, so the caller's environment must be
// scrubbed during the test and restored on exit.
const PLAYWRIGHT_CONFIG_ENV_KEYS = [
  "PLAYWRIGHT_BASE_URL",
  "MOCK_FASTAPI_HOST",
  "MOCK_FASTAPI_PORT"
];

function snapshotPlaywrightConfigEnv() {
  return new Map(PLAYWRIGHT_CONFIG_ENV_KEYS.map((key) => [key, process.env[key]]));
}

function clearPlaywrightConfigEnv() {
  for (const key of PLAYWRIGHT_CONFIG_ENV_KEYS) {
    delete process.env[key];
  }
}

function restorePlaywrightConfigEnv(snapshot) {
  for (const key of PLAYWRIGHT_CONFIG_ENV_KEYS) {
    const previous = snapshot.get(key);
    if (typeof previous === "string") {
      process.env[key] = previous;
    } else {
      delete process.env[key];
    }
  }
}

test("playwright frontdoor smoke config uses local preflight-safe fixtures", async () => {
  const snapshot = snapshotPlaywrightConfigEnv();
  clearPlaywrightConfigEnv();

  try {
    const configModule = await importFresh("../playwright.config.mjs");
    const frontdoor = frontdoorWebServerEntry(configModule.default);
    assert.ok(frontdoor, "front-door webServer entry must exist");
    const env = frontdoor.env;
    const users = JSON.parse(env.TP_FRONTDOOR_USERS_JSON);

    assert.equal(frontdoor.command, "npm run dev -- --webpack --port 3000");
    assert.equal(env.NODE_ENV, "development");
    assert.equal(env.TP_ALLOW_LOCAL_ACCESS_BYPASS, "1");
    assert.equal(env.TP_FASTAPI_ORIGIN, "http://127.0.0.1:9999");
    assert.equal(env.TP_BACKEND_API_KEY, "frontdoor-browser-smoke");
    assert.equal(env.WATCHPACK_POLLING, "true");
    assert.equal(env.CHOKIDAR_USEPOLLING, "true");
    assert.equal(users.length, 1);
    assert.equal(users[0].username, "smoke-admin");
    assert.equal(users[0].access_email, "smoke-admin@local.invalid");
    assert.match(users[0].password_hash, /^\$argon2/);

    // The mock FastAPI origin must be wired alongside the front-door so
    // the @portal-browser suite can serve portal.html upstream-proxied
    // through the front-door without spawning real FastAPI.
    const mock = mockFastapiWebServerEntry(configModule.default);
    assert.ok(mock, "mock FastAPI origin webServer entry must exist");
    assert.equal(mock.url, "http://127.0.0.1:9999/healthz");
    assert.equal(mock.env.MOCK_FASTAPI_PORT, "9999");
    assert.equal(mock.env.MOCK_FASTAPI_HOST, "127.0.0.1");
  } finally {
    restorePlaywrightConfigEnv(snapshot);
  }
});

test("playwright frontdoor smoke config derives webServer port from PLAYWRIGHT_BASE_URL", async () => {
  const snapshot = snapshotPlaywrightConfigEnv();
  clearPlaywrightConfigEnv();
  process.env.PLAYWRIGHT_BASE_URL = "http://127.0.0.1:3017";

  try {
    const configModule = await importFresh("../playwright.config.mjs");
    const frontdoor = frontdoorWebServerEntry(configModule.default);

    assert.equal(configModule.default.use.baseURL, "http://127.0.0.1:3017");
    assert.equal(frontdoor.url, "http://127.0.0.1:3017");
    assert.equal(frontdoor.command, "npm run dev -- --webpack --port 3017");
  } finally {
    restorePlaywrightConfigEnv(snapshot);
  }
});

test("playwright frontdoor smoke config rejects PLAYWRIGHT_BASE_URL without an explicit port", async () => {
  const previous = process.env.PLAYWRIGHT_BASE_URL;
  process.env.PLAYWRIGHT_BASE_URL = "http://127.0.0.1";

  try {
    await assert.rejects(
      importFresh("../playwright.config.mjs"),
      /PLAYWRIGHT_BASE_URL must include an explicit port for frontdoor browser smoke tests/
    );
  } finally {
    if (typeof previous === "string") {
      process.env.PLAYWRIGHT_BASE_URL = previous;
    } else {
      delete process.env.PLAYWRIGHT_BASE_URL;
    }
  }
});

test("run_frontdoor_local launcher supports isolated port, distdir, and local user seeding defaults", async () => {
  const scriptPath = path.resolve(process.cwd(), "..", "..", "scripts", "setup", "run_frontdoor_local.sh");
  const script = readFileSync(scriptPath, "utf-8");

  assert.match(script, /TP_FRONTDOOR_PORT/);
  assert.match(script, /TP_FRONTDOOR_DIST_DIR/);
  assert.match(script, /TP_NEXT_DIST_DIR/);
  assert.match(script, /TP_FRONTDOOR_NEXT_DEV_ENGINE:-webpack/);
  assert.match(script, /TP_FRONTDOOR_WATCH_POLLING:-1/);
  assert.match(script, /WATCHPACK_POLLING:-true/);
  assert.match(script, /CHOKIDAR_USEPOLLING:-true/);
  assert.match(script, /npm run dev -- "\$\{NEXT_DEV_ARGS\[@\]\}"/);
  assert.match(script, /TP_FRONTDOOR_USERS_FILE:-\/tmp\/tp-frontdoor-users\.json/);
  assert.match(script, /TP_FRONTDOOR_USERNAME:-smoke-admin/);
  assert.match(script, /TP_FRONTDOOR_PASSWORD:-correct horse battery staple/);
  assert.match(script, /if \[\[ -z "\$\{TP_FRONTDOOR_USERS_FILE:-\}" && -z "\$\{TP_FRONTDOOR_USERS_JSON:-\}" \]\]; then/);
  assert.match(script, /seed-frontdoor-user\.mjs/);
  assert.match(script, /TP_FRONTDOOR_PRINT_PASSWORD/);
  assert.match(script, /Local operator username:/);
  assert.match(script, /Password not printed\./);
  assert.doesNotMatch(script, /Local operator credentials:/);
  assert.match(script, /Start the backend with that host trusted before launching the frontdoor\./);
  assert.doesNotMatch(script, /Appended \$\{CF_HOSTNAME\} to TP_TRUSTED_HOSTS/);
});

test("preflight requires the same FastAPI origin variable consumed by runtime proxy", async () => {
  const scriptPath = path.resolve(process.cwd(), "scripts", "preflight-backend-auth.mjs");
  const result = spawnSync(process.execPath, [scriptPath], {
    cwd: process.cwd(),
    encoding: "utf-8",
    env: {
      ...process.env,
      NODE_ENV: "development",
      TP_BACKEND_ORIGIN: "https://backend-origin.example.com",
      TP_FASTAPI_ORIGIN: "",
      TP_BACKEND_API_KEY: "secret",
      TP_FRONTDOOR_PREFLIGHT_DISABLE: "0",
      TP_FRONTDOOR_USERS_FILE: "",
      TP_FRONTDOOR_USERS_JSON: JSON.stringify([
        {
          username: "admin",
          password_hash: "hash",
          access_email: "admin@example.com"
        }
      ])
    }
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /TP_FASTAPI_ORIGIN is not set/);
  assert.match(result.stderr, /TP_BACKEND_ORIGIN is not consumed/);
});

test("preflight rejects unreadable or empty frontdoor user sources", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-preflight-users-"));
  const emptyUsersFile = path.join(tempDir, "users.json");
  writeFileSync(emptyUsersFile, "[]", "utf-8");

  try {
    const scriptPath = path.resolve(process.cwd(), "scripts", "preflight-backend-auth.mjs");
    const result = spawnSync(process.execPath, [scriptPath], {
      cwd: process.cwd(),
      encoding: "utf-8",
      env: {
        ...process.env,
        NODE_ENV: "development",
        TP_FASTAPI_ORIGIN: "http://127.0.0.1:8000",
        TP_BACKEND_API_KEY: "secret",
        TP_FRONTDOOR_PREFLIGHT_DISABLE: "0",
        TP_FRONTDOOR_USERS_FILE: emptyUsersFile,
        TP_FRONTDOOR_USERS_JSON: ""
      }
    });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /TP_FRONTDOOR_USERS_FILE did not contain any valid users/);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("seed-frontdoor-user helper encodes the canonical local smoke credential defaults and writes restrictive fixtures", async () => {
  const tmpDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-seed-"));

  try {
    const scriptPath = path.resolve(process.cwd(), "scripts", "seed-frontdoor-user.mjs");
    const script = readFileSync(scriptPath, "utf-8");

    assert.match(script, /DEFAULT_OUTPUT_PATH = "\/tmp\/tp-frontdoor-users\.json"/);
    assert.match(script, /DEFAULT_USERNAME = "smoke-admin"/);
    assert.match(script, /DEFAULT_PASSWORD = "correct horse battery staple"/);
    assert.match(script, /@local\.invalid/);
    assert.match(script, /renameSync/);
    assert.match(script, /mode: 0o600/);

    const module = await importFresh("../scripts/seed-frontdoor-user.mjs");
    const outputPath = path.join(tmpDir, "frontdoor-users.json");
    const writtenPath = await module.seedFrontdoorUser({
      outputPath,
      username: "seed-admin",
      password: "correct horse battery staple",
      accessEmail: "seed-admin@local.invalid",
      role: "admin",
    });

    assert.equal(writtenPath, path.resolve(outputPath));
    assert.equal(statSync(outputPath).mode & 0o777, 0o600);

    const payload = JSON.parse(readFileSync(outputPath, "utf-8"));
    assert.equal(payload.length, 1);
    assert.equal(payload[0].username, "seed-admin");
    assert.equal(payload[0].access_email, "seed-admin@local.invalid");
    assert.equal(payload[0].role, "admin");
    assert.match(payload[0].password_hash, /^\$argon2/);
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("app router root shell exists for framework not-found and global-error routes", () => {
  const appRoot = path.resolve(process.cwd(), "app");
  const layoutSource = readFileSync(path.join(appRoot, "layout.js"), "utf-8");
  const notFoundSource = readFileSync(path.join(appRoot, "not-found.js"), "utf-8");
  const globalErrorSource = readFileSync(path.join(appRoot, "global-error.js"), "utf-8");

  assert.match(layoutSource, /<html lang="en">/);
  assert.match(layoutSource, /<body>/);
  assert.match(layoutSource, /dynamic = "force-dynamic"/);
  assert.match(notFoundSource, /requested front door route was not found/i);
  assert.match(notFoundSource, /dynamic = "force-dynamic"/);
  assert.match(globalErrorSource, /^"use client";/);
  assert.match(globalErrorSource, /dynamic = "force-dynamic"/);
  assert.match(globalErrorSource, /reset\(\)/);
});

async function withCapturedAuditEvents(run) {
  const { clearAuditObserver, setAuditObserver } = await importFresh("../lib/audit.js");
  const events = [];
  setAuditObserver((payload) => {
    events.push(payload);
  });
  try {
    return await run(events);
  } finally {
    clearAuditObserver();
  }
}

function extractSessionCookie(response) {
  const cookieHeader = response.headers.get("set-cookie") || "";
  const match = cookieHeader.match(/(?:__Host-tp_session|tp_session)=([^;]+)/);
  return {
    raw: cookieHeader,
    value: match?.[1] || ""
  };
}

function buildRequest(url, options = {}) {
  return new NextRequest(url, options);
}

function createAccessJwt(overrides = {}) {
  const now = Math.floor(Date.now() / 1000);
  const kid = overrides.kid ?? TEST_CF_ACCESS_KID;
  const privateKey = overrides.privateKey ?? TEST_CF_ACCESS_KEYS.privateKey;
  const alg = overrides.alg ?? "RS256";
  const header = {
    alg,
    kid,
    typ: "JWT"
  };
  const payload = {
    aud: [TEST_CF_ACCESS_AUD],
    email: "admin@example.com",
    exp: now + 300,
    iat: now - 5,
    iss: TEST_CF_ACCESS_TEAM_DOMAIN,
    nbf: now - 5,
    ...overrides
  };

  const encodedHeader = Buffer.from(JSON.stringify(header)).toString("base64url");
  const encodedPayload = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signingInput = `${encodedHeader}.${encodedPayload}`;
  const encodedSignature = createSign("RSA-SHA256")
    .update(signingInput)
    .end()
    .sign(privateKey)
    .toString("base64url");

  return `${signingInput}.${encodedSignature}`;
}

function withMockedAccessCerts(fallbackFetch = null) {
  const originalFetch = global.fetch;
  global.fetch = async (url, init) => {
    if (String(url) === `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`) {
      return Response.json(
        {
          keys: [TEST_CF_ACCESS_PUBLIC_JWK]
        },
        {
          status: 200,
          headers: {
            "content-type": "application/json"
          }
        }
      );
    }

    if (fallbackFetch) {
      return fallbackFetch(url, init);
    }

    if (typeof originalFetch === "function") {
      return originalFetch(url, init);
    }

    throw new Error(`Unexpected fetch to ${String(url)}`);
  };

  return () => {
    global.fetch = originalFetch;
  };
}

test("login POST rotates the session and redirects authenticated users to /portal", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
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
          "Cf-Access-Jwt-Assertion": createAccessJwt(),
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
      assert.equal(await sessions.getSessionById(anonymousSession.id, { touch: false }), null);

      const authenticatedSession = await sessions.getSessionById(rotatedCookie.value, { touch: false });
      assert.equal(authenticatedSession?.authenticated, true);
      assert.equal(authenticatedSession?.username, "admin");
      assert.match(rotatedCookie.raw, /HttpOnly/);
      assert.match(rotatedCookie.raw, /Secure/);
      assert.match(rotatedCookie.raw, /SameSite=lax/i);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login POST redirects authenticated users to a validated returnTo route", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
      const form = new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken,
        returnTo: "/portal?view=build"
      });

      const request = buildRequest("https://portal.example.com/login", {
        method: "POST",
        headers: new Headers({
          origin: "https://portal.example.com",
          "content-type": "application/x-www-form-urlencoded",
          cookie: `__Host-tp_session=${anonymousSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt(),
          "cf-access-authenticated-user-email": "admin@example.com"
        }),
        body: form
      });

      const response = await POST(request);

      assert.equal(response.status, 303);
      assert.equal(response.headers.get("location"), "https://portal.example.com/portal?view=build");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login GET serves a minimal branded sign-in shell and boots an anonymous session", async () => {
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/login/route.js");
    const request = buildRequest("http://localhost:3000/login");

    const response = await GET(request);
    const html = await response.text();
    const sessionCookie = extractSessionCookie(response);

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
    assert.match(html, /class="hero-video"/);
    assert.match(html, /preload="metadata"/);
    assert.match(html, /\/video\/dna-loop\.mp4/);
    assert.match(html, /\/brand\/dna-lockup-dark\.svg/);
    assert.match(html, /href="#main-content">Skip to sign-in</);
    assert.match(html, /id="main-content"/);
    assert.match(html, /Transformation Portal operator console/);
    assert.match(html, /data-ui="login-title"/);
    assert.match(html, /data-ui="login-entry-state"/);
    assert.match(html, /data-ui="login-access-status"/);
    assert.match(html, /data-ui="login-credential-status"/);
    assert.match(html, /data-ui="login-capability-summary"/);
    assert.match(html, /Archive gates, staged uploads, Lux Depth, SAM2, reconstruction, RAW ingest, and FastVLM controls/);
    assert.match(html, /data-ui="login-sequence"/);
    assert.match(html, /data-ui="login-form"/);
    assert.match(html, /form method="post" action="\/login"/);
    assert.match(html, /name="username"/);
    assert.match(html, /name="password"/);
    assert.match(html, /data-ui="login-helper"/);
    assert.match(html, /data-ui="login-submit"[^>]*>Sign in</);
    assert.match(html, /(?:data-ui="login-secondary-link"[^>]*href="\/"|href="\/"[^>]*data-ui="login-secondary-link")/);
    assert.match(html, /data-access-state="verified"/);
    assert.match(html, /Local development bypass active/);
    assert.match(html, /Credential handoff ready/);
    assert.match(html, /Bypass context/);
    assert.match(html, /login-sequence-step login-sequence-step--ready/);
    assert.doesNotMatch(html, /Authorized operators only\./);
    assert.doesNotMatch(html, /Need access\?/);
    assert.doesNotMatch(html, /Secure operator access to governed orchestration\./);
    assert.doesNotMatch(html, /Local development bypass is enabled\./);
    assert.doesNotMatch(html, /TP_CF_ACCESS_TEAM_DOMAIN/);
    assert.ok(sessionCookie.value);
    assert.equal((await sessions.getSessionById(sessionCookie.value, { touch: false }))?.authenticated, false);
  } finally {
    env.cleanup();
  }
});

test("login GET preserves a validated returnTo in the rendered form", async () => {
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1"
  });

  try {
    const { GET } = await importFresh("../app/login/route.js");
    const request = buildRequest("http://localhost:3000/login?returnTo=%2Fportal%3Fview%3Dbuild");

    const response = await GET(request);
    const html = await response.text();

    assert.equal(response.status, 200);
    assert.match(html, /name="returnTo" value="\/portal\?view=build"/);
  } finally {
    env.cleanup();
  }
});

test("login GET redirects authenticated sessions to a validated returnTo route", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      const request = buildRequest("https://portal.example.com/login?returnTo=%2Fportal%3Fview%3Dreview", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt()
        })
      });

      const response = await GET(request);

      assert.equal(response.status, 302);
      assert.equal(response.headers.get("location"), "https://portal.example.com/portal?view=review");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login GET escapes verified access context exactly once in the recovery card", async () => {
  const env = withTempEnvironment();

  try {
    const { GET } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const accessEmail = "admin+ops&support@example.com";
      const request = buildRequest("https://portal.example.com/login", {
        headers: new Headers({
          "Cf-Access-Jwt-Assertion": createAccessJwt({ email: accessEmail }),
          "cf-access-authenticated-user-email": accessEmail
        })
      });

      const response = await GET(request);
      const html = await response.text();

      assert.equal(response.status, 200);
      assert.match(html, /data-access-state="verified"/);
      assert.match(html, /data-ui="login-recovery-card"/);
      assert.match(html, /Managed access has already been verified for admin\+ops&amp;support@example\.com\./);
      assert.doesNotMatch(html, /admin\+ops&amp;amp;support@example\.com/);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("homepage GET serves the public DNA landing page instead of redirecting", async () => {
  const env = withTempEnvironment();

  try {
    const db = getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const sessionCountBefore = db.prepare("SELECT COUNT(*) AS count FROM sessions").get().count;
    const response = await GET(request);
    const html = await response.text();
    const sessionCountAfter = db.prepare("SELECT COUNT(*) AS count FROM sessions").get().count;
    const cacheControl = response.headers.get("cache-control") || "";

    assert.equal(response.status, 200);
    assert.doesNotMatch(cacheControl, /no-store/i);
    assert.match(cacheControl, /\bpublic\b/i);
    assert.match(cacheControl, /\bmax-age=/i);
    assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
    assert.match(response.headers.get("content-security-policy") || "", /script-src 'none'/);
    assert.equal(response.headers.get("set-cookie"), null);
    assert.equal(sessionCountBefore, 0);
    assert.equal(sessionCountAfter, 0);
    assert.match(html, /\/video\/dna-loop\.mp4/);
    assert.match(html, /\/brand\/dna-symbol-dark\.svg/);
    assert.match(html, /\/brand\/dna-lockup-dark\.svg/);
    assert.match(html, /data-ui="homepage-hero-lockup"/);
    assert.match(html, /data-ui="homepage-hero-title"/);
    assert.match(html, /data-ui="homepage-entry-rail"/);
    assert.match(html, /Managed access opens a governed console for dispatch, queue operation, artifact review, archive gates, and optional runtimes/);
    assert.match(html, /Lux Depth, SAM2, reconstruction, RAW ingest, and FastVLM stay visible only as real controls/);
    assert.match(html, /(?:data-ui="homepage-learn-link"[^>]*href="#workflow"|href="#workflow"[^>]*data-ui="homepage-learn-link")/);
    assert.match(html, /(?:data-ui="homepage-primary-cta"[^>]*href="\/login"|href="\/login"[^>]*data-ui="homepage-primary-cta")/);
    assert.match(html, /(?:data-ui="homepage-secondary-cta"[^>]*href="#proof-report"|href="#proof-report"[^>]*data-ui="homepage-secondary-cta")/);
    assert.match(html, /(?:data-ui="homepage-utility-cta"[^>]*href="\/login"|href="\/login"[^>]*data-ui="homepage-utility-cta")/);
    assert.match(html, /(?:data-ui="homepage-final-primary-cta"[^>]*href="\/login"|href="\/login"[^>]*data-ui="homepage-final-primary-cta")/);
    assert.match(html, /Verify\. Enhance\. Enforce\. Distribute\./);
    assert.match(html, /(?:data-ui="homepage-operator-link"[^>]*href="\/login"|href="\/login"[^>]*data-ui="homepage-operator-link")/);
    assert.match(html, /tp\.meta\.verification_report\.v1/);
    assert.match(html, /when enabled/i);
    assert.match(html, /strip metadata/i);
    assert.doesNotMatch(html, /href="\/start"/);
    assert.doesNotMatch(html, /href="\/console"/);
    assert.doesNotMatch(html, /href="\/privacy"/);
    assert.doesNotMatch(html, /href="\/security"/);
    assert.doesNotMatch(html, /href="\/docs"/);
    assert.doesNotMatch(html, /href="\/contact"/);
    assert.doesNotMatch(html, /\bCompliant\b/);
    assert.doesNotMatch(html, /\bSigned\b/);
    assert.doesNotMatch(html, /\bimmutable\b/i);
    assert.doesNotMatch(html, /\bunfakeable\b/i);
  } finally {
    env.cleanup();
  }
});

test("homepage GET is stateless and ignores authenticated session hints", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const db = getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/", {
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`
      })
    });

    const before = db
      .prepare("SELECT last_seen_at, idle_expires_at FROM sessions WHERE id = ?")
      .get(authenticatedSession.id);
    const response = await GET(request);
    const html = await response.text();
    const after = db
      .prepare("SELECT last_seen_at, idle_expires_at FROM sessions WHERE id = ?")
      .get(authenticatedSession.id);

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("set-cookie"), null);
    assert.match(html, /(?:data-ui="homepage-utility-cta"[^>]*href="\/login"|href="\/login"[^>]*data-ui="homepage-utility-cta")/);
    assert.match(html, />Operator Access</);
    assert.match(html, /(?:data-ui="homepage-primary-cta"[^>]*href="\/login"|href="\/login"[^>]*data-ui="homepage-primary-cta")/);
    assert.equal(after.last_seen_at, before.last_seen_at);
    assert.equal(after.idle_expires_at, before.idle_expires_at);
    assert.doesNotMatch(html, /307|302/);
  } finally {
    env.cleanup();
  }
});

test("homepage GET succeeds without backend coupling or fetch side effects", async () => {
  const env = withTempEnvironment();
  const originalFetch = global.fetch;
  global.fetch = async () => {
    throw new Error("homepage should not fetch");
  };

  try {
    const { GET } = await importFresh("../app/route.js");
    const request = buildRequest("https://portal.example.com/");

    const response = await GET(request);
    const html = await response.text();

    assert.equal(response.status, 200);
    assert.match(html, /Dynamic Neural Access/);
  } finally {
    global.fetch = originalFetch;
    env.cleanup();
  }
});

test("homepage GET does not touch session state even when a stale cookie is present", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/route.js");
    const expiredSession = await sessions.createAnonymousSession();
    const db = getDb(env.dbPath);
    const now = Date.now();
    db.prepare("UPDATE sessions SET idle_expires_at = ?, absolute_expires_at = ? WHERE id = ?").run(
      now - 1_000,
      now - 1_000,
      expiredSession.id
    );

    const request = buildRequest("https://portal.example.com/", {
      headers: new Headers({
        cookie: `__Host-tp_session=${expiredSession.id}`
      })
    });

    const response = await GET(request);
    const html = await response.text();
    const persisted = db.prepare("SELECT id FROM sessions WHERE id = ?").get(expiredSession.id);

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("set-cookie"), null);
    assert.equal(persisted.id, expiredSession.id);
    assert.match(html, />Operator Access</);
  } finally {
    env.cleanup();
  }
});

test("login POST keeps failures generic when Access email does not match the configured account", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
      const request = buildRequest("https://portal.example.com/login", {
        method: "POST",
        headers: new Headers({
          origin: "https://portal.example.com",
          "content-type": "application/x-www-form-urlencoded",
          cookie: `__Host-tp_session=${anonymousSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt({ email: "other@example.com" }),
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
      assert.equal((await sessions.getSessionById(anonymousSession.id, { touch: false }))?.authenticated, false);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login POST preserves a validated returnTo when sign-in fails", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
      const request = buildRequest("https://portal.example.com/login", {
        method: "POST",
        headers: new Headers({
          origin: "https://portal.example.com",
          "content-type": "application/x-www-form-urlencoded",
          cookie: `__Host-tp_session=${anonymousSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt(),
          "cf-access-authenticated-user-email": "admin@example.com"
        }),
        body: new URLSearchParams({
          username: "admin",
          password: "wrong password",
          csrf_token: anonymousSession.csrfToken,
          returnTo: "/portal?view=review"
        })
      });

      const response = await POST(request);

      assert.equal(response.status, 303);
      assert.equal(
        response.headers.get("location"),
        "https://portal.example.com/login?error=invalid&returnTo=%2Fportal%3Fview%3Dreview"
      );
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login POST rejects invalid CSRF before credential verification", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
      const request = buildRequest("https://portal.example.com/login", {
        method: "POST",
        headers: new Headers({
          origin: "https://portal.example.com",
          "content-type": "application/x-www-form-urlencoded",
          cookie: `__Host-tp_session=${anonymousSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt(),
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
      assert.equal((await sessions.getSessionById(anonymousSession.id, { touch: false }))?.authenticated, false);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login POST rejects convenience headers without a verified Access JWT in production", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
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
        cookie: `__Host-tp_session=${anonymousSession.id}`,
        "cf-access-authenticated-user-email": "admin@example.com",
        "x-access-email": "admin@example.com"
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
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

test("login POST rejects Access JWTs with the wrong audience", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
      const request = buildRequest("https://portal.example.com/login", {
        method: "POST",
        headers: new Headers({
          origin: "https://portal.example.com",
          "content-type": "application/x-www-form-urlencoded",
          cookie: `__Host-tp_session=${anonymousSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt({ aud: ["wrong-audience"] })
        }),
        body: new URLSearchParams({
          username: "admin",
          password: "correct horse battery staple",
          csrf_token: anonymousSession.csrfToken
        })
      });

      const response = await POST(request);

      assert.equal(response.status, 303);
      assert.equal(response.headers.get("location"), "https://portal.example.com/login?error=access");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("login POST rejects Access JWTs with the wrong issuer", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const anonymousSession = await sessions.createAnonymousSession();
      const request = buildRequest("https://portal.example.com/login", {
        method: "POST",
        headers: new Headers({
          origin: "https://portal.example.com",
          "content-type": "application/x-www-form-urlencoded",
          cookie: `__Host-tp_session=${anonymousSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt({ iss: "https://wrong-team.cloudflareaccess.com" })
        }),
        body: new URLSearchParams({
          username: "admin",
          password: "correct horse battery staple",
          csrf_token: anonymousSession.csrfToken
        })
      });

      const response = await POST(request);

      assert.equal(response.status, 303);
      assert.equal(response.headers.get("location"), "https://portal.example.com/login?error=access");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("production ignores the local bypass flag without a verified Access JWT", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
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
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login?error=access");
  } finally {
    env.cleanup();
  }
});

test("development local bypass still allows login without Cloudflare Access", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = await sessions.createAnonymousSession();
    const request = buildRequest("http://localhost:3000/login", {
      method: "POST",
      headers: new Headers({
        origin: "http://localhost:3000",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `tp_session=${anonymousSession.id}`
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "http://localhost:3000/portal");
  } finally {
    env.cleanup();
  }
});

test("development local bypass login still works when the browser omits origin and referrer", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = await sessions.createAnonymousSession();
    const request = buildRequest("http://127.0.0.1:3000/login", {
      method: "POST",
      headers: new Headers({
        host: "127.0.0.1:3000",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `tp_session=${anonymousSession.id}`
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "http://127.0.0.1:3000/portal");
  } finally {
    env.cleanup();
  }
});

test("login POST ignores untrusted host overrides when building redirect targets", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
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
        host: "evil.example.com",
        "x-forwarded-host": "evil.example.com",
        "x-forwarded-proto": "http",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `tp_session=${anonymousSession.id}`
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken,
        returnTo: "/portal?view=build"
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "https://portal.example.com/portal?view=build");
  } finally {
    env.cleanup();
  }
});

test("login POST rejects invalid returnTo values and falls back to /portal", async () => {
  const passwordHash = await argon2.hash("correct horse battery staple");
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: passwordHash,
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/login/route.js");

    const anonymousSession = await sessions.createAnonymousSession();
    const request = buildRequest("http://127.0.0.1:3000/login", {
      method: "POST",
      headers: new Headers({
        host: "127.0.0.1:3000",
        "content-type": "application/x-www-form-urlencoded",
        cookie: `tp_session=${anonymousSession.id}`
      }),
      body: new URLSearchParams({
        username: "admin",
        password: "correct horse battery staple",
        csrf_token: anonymousSession.csrfToken,
        returnTo: "https://evil.example.com/portal"
      })
    });

    const response = await POST(request);

    assert.equal(response.status, 303);
    assert.equal(response.headers.get("location"), "http://127.0.0.1:3000/portal");
  } finally {
    env.cleanup();
  }
});

test("logout POST invalidates the authenticated session and clears the cookie", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { POST } = await importFresh("../app/logout/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
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
    assert.equal(await sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
  } finally {
    env.cleanup();
  }
});

test("expired sessions are removed for both idle and absolute timeout breaches", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const idleExpired = await sessions.createAnonymousSession();
    const absoluteExpired = await sessions.createAnonymousSession();
    const db = getDb(env.dbPath);
    const now = Date.now();

    db.prepare("UPDATE sessions SET idle_expires_at = ? WHERE id = ?").run(now - 1_000, idleExpired.id);
    db.prepare("UPDATE sessions SET absolute_expires_at = ? WHERE id = ?").run(now - 1_000, absoluteExpired.id);

    assert.equal(await sessions.getSessionById(idleExpired.id, { touch: false }), null);
    assert.equal(await sessions.getSessionById(absoluteExpired.id, { touch: false }), null);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap returns actor metadata and CSRF for authenticated sessions", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      const request = buildRequest("https://portal.example.com/portal/bootstrap", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt()
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
      assert.equal(body.features.artifactViewerModal, false);
      assert.equal(body.features.reviewSurfaceDeferred, false);
      assert.equal(body.features.stagedUploads, false);
      assert.equal(body.features.rumTelemetry, false);
      assert.equal(body.features.fastVlmCaptioning, false);
      assert.equal(body.csrfToken, authenticatedSession.csrfToken);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap enables the artifact viewer modal for rollout cohorts", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT: "100"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.artifactViewerModal, true);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap enables review surface deferral for rollout cohorts", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT: "100"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.reviewSurfaceDeferred, true);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap enables staged uploads for rollout cohorts", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_UPLOAD_STAGING_ENABLED: "1",
    TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT: "100"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.stagedUploads, true);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap enables rum telemetry for rollout cohorts", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_RUM_ENABLED: "1",
    TP_PORTAL_RUM_ROLLOUT_PERCENT: "100"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.rumTelemetry, true);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap rum telemetry ignores front-door RUM rollout controls", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_RUM_ENABLED: "1",
    TP_PORTAL_RUM_ROLLOUT_PERCENT: "100",
    TP_FRONTDOOR_RUM_ENABLED: "0",
    TP_FRONTDOOR_RUM_ROLLOUT_PERCENT: "0"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.rumTelemetry, true);
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap enables FastVLM captioning for enabled rollout cohorts", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_FASTVLM_CAPTIONING_ENABLED: "1",
    TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT: "100"
  });
  const restoreFetch = withMockedAccessCerts();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.fastVlmCaptioning, true);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("managed bootstrap keeps FastVLM captioning disabled when the master switch is off", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_FASTVLM_CAPTIONING_ENABLED: "0",
    TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT: "100"
  });
  const restoreFetch = withMockedAccessCerts();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.fastVlmCaptioning, false);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("managed bootstrap keeps FastVLM captioning disabled when rollout is zero", async () => {
  const env = withTempEnvironment({
    TP_PORTAL_FASTVLM_CAPTIONING_ENABLED: "1",
    TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT: "0"
  });
  const restoreFetch = withMockedAccessCerts();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal/bootstrap", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
      })
    });

    const response = await GET(request);
    const body = await response.json();

    assert.equal(response.status, 200);
    assert.equal(body.features.fastVlmCaptioning, false);
  } finally {
    restoreFetch();
    env.cleanup();
  }
});

test("managed bootstrap echoes traceparent and includes trace id in audit payloads", async () => {
  const env = withTempEnvironment();
  const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      await withCapturedAuditEvents(async (events) => {
        const request = buildRequest("https://portal.example.com/portal/bootstrap", {
          method: "GET",
          headers: new Headers({
            cookie: `__Host-tp_session=${authenticatedSession.id}`,
            traceparent,
            "Cf-Access-Jwt-Assertion": createAccessJwt()
          })
        });

        const response = await GET(request);
        const body = await response.json();

        assert.equal(response.status, 200);
        assert.equal(response.headers.get("traceparent"), traceparent);
        assert.equal(body.features.rumTelemetry, false);
        assert.equal(events.length, 1);
        assert.equal(events[0].event, "portal_bootstrap");
        assert.equal(events[0].traceId, "4bf92f3577b34da6a3ce929d0e0e4736");
      });
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("managed bootstrap rejects authenticated sessions without a current Access JWT", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
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

    assert.equal(response.status, 401);
    assert.equal(body.error, "authentication required");
    assert.equal(body.reason, "auth_failure");
    assert.equal(body.retryable, false);
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
    assert.equal(await sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
  } finally {
    env.cleanup();
  }
});

test("audit observer failures are swallowed so instrumentation cannot break runtime paths", async () => {
  const { audit, clearAuditObserver, setAuditObserver } = await importFresh("../lib/audit.js");
  const originalInfo = console.info;
  const originalWarn = console.warn;
  const infos = [];
  const warns = [];

  console.info = (message) => {
    infos.push(message);
  };
  console.warn = (...args) => {
    warns.push(args);
  };
  setAuditObserver(() => {
    throw new Error("observer boom");
  });

  try {
    assert.doesNotThrow(() => {
      audit("test_event", { path: "/portal/bootstrap" });
    });
    assert.equal(infos.length, 1);
    assert.equal(warns.length, 1);
    assert.equal(warns[0][0], "Audit observer error");
    assert.match(String(warns[0][1]), /observer boom/);
  } finally {
    clearAuditObserver();
    console.info = originalInfo;
    console.warn = originalWarn;
  }
});

test("managed bootstrap classifies Access verification outages as retryable access_outage and audits them", async () => {
  const outageTeamDomain = "https://tp-frontdoor-outage.cloudflareaccess.com";
  const env = withTempEnvironment({
    TP_CF_ACCESS_TEAM_DOMAIN: outageTeamDomain
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/bootstrap/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      assert.equal(String(url), `${outageTeamDomain}/cdn-cgi/access/certs`);
      assert.ok(init.signal);
      throw Object.assign(new Error("timed out"), { name: "AbortError" });
    };

    try {
      await withCapturedAuditEvents(async (events) => {
        const request = buildRequest("https://portal.example.com/portal/bootstrap", {
          method: "GET",
          headers: new Headers({
            cookie: `__Host-tp_session=${authenticatedSession.id}`,
            "Cf-Access-Jwt-Assertion": createAccessJwt({ iss: outageTeamDomain })
          })
        });

        const response = await GET(request);
        const body = await response.json();

        assert.equal(response.status, 503);
        assert.equal(body.error, "managed access unavailable");
        assert.equal(body.reason, "access_outage");
        assert.equal(body.retryable, true);
        assert.match(body.message, /temporarily unavailable/i);
        assert.equal(events.length, 1);
        assert.equal(events[0].event, "managed_surface_failure");
        assert.equal(events[0].surface, "portal_bootstrap");
        assert.equal(events[0].reason, "access_outage");
        assert.equal(events[0].retryable, true);
        assert.equal(events[0].status, 503);
      });
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal returns 503 with no-store when the FastAPI UI origin is unavailable", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const restoreFetch = withMockedAccessCerts(async (url) => {
      assert.equal(String(url), "http://127.0.0.1:8000/");
      throw new Error("connection refused");
    });

    try {
      const request = buildRequest("https://portal.example.com/portal?view=build", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt()
        })
      });

      const response = await GET(request);

      assert.equal(response.status, 503);
      assert.equal(response.headers.get("cache-control"), "no-store");
      assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
      const html = await response.text();
      assert.match(html, /data-ui="managed-recovery-shell"/);
      assert.match(html, /data-reason="upstream_unavailable"/);
      assert.match(html, /data-ui="managed-recovery-capabilities"/);
      assert.match(html, /Queue, artifact viewer, staged uploads, FastVLM sidecars, archive gates, and run-card proof controls/);
      assert.match(html, /Portal upstream unavailable/);
      assert.match(html, /href="\/login\?returnTo=%2Fportal%3Fview%3Dbuild"/);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("portal renders waiting recovery posture for Access outages", async () => {
  const outageTeamDomain = "https://tp-frontdoor-outage.cloudflareaccess.com";
  const env = withTempEnvironment({
    TP_CF_ACCESS_TEAM_DOMAIN: outageTeamDomain
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/route.js");
    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      assert.equal(String(url), `${outageTeamDomain}/cdn-cgi/access/certs`);
      assert.ok(init.signal);
      throw Object.assign(new Error("timed out"), { name: "AbortError" });
    };

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      const request = buildRequest("https://portal.example.com/portal", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt({ iss: outageTeamDomain })
        })
      });

      const response = await GET(request);
      const html = await response.text();

      assert.equal(response.status, 503);
      assert.match(html, /data-ui="managed-recovery-shell"/);
      assert.match(html, /data-reason="access_outage"/);
      assert.match(html, /class="[^"]*\blogin-status-card\b[^"]*"/);
      assert.match(html, /data-state="waiting"/);
      assert.match(html, /Retry when Access recovers/);
      assert.match(html, /href="\/login\?returnTo=%2Fportal"/);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal returns configuration guidance when managed access config is incomplete", async () => {
  const env = withTempEnvironment({
    TP_CF_ACCESS_AUD: ""
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    await withCapturedAuditEvents(async (events) => {
      const request = buildRequest("https://portal.example.com/portal", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`
        })
      });

      const response = await GET(request);

      assert.equal(response.status, 503);
      assert.equal(response.headers.get("cache-control"), "no-store");
      assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
      const html = await response.text();
      assert.match(html, /data-ui="managed-recovery-shell"/);
      assert.match(html, /data-reason="config_failure"/);
      assert.match(html, /Managed front door configuration unavailable/);
      assert.match(html, /data-ui="managed-recovery-capabilities"/);
      assert.match(html, /Managed boundary stays fail-closed/);
      assert.match(html, /href="\/login\?returnTo=%2Fportal"/);
      assert.equal(events.length, 1);
      assert.equal(events[0].surface, "portal");
      assert.equal(events[0].reason, "config_failure");
      assert.equal(events[0].status, 503);
    });
  } finally {
    env.cleanup();
  }
});

test("portal redirects to login with route context and clears the session when Access verification is missing", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/route.js");
    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest(
      "https://portal.example.com/portal?view=review&job=job_demo&artifact=synthetic%2Freview-primary.png&compare=1",
      {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`
        })
      }
    );

    const response = await GET(request);

    assert.equal(response.status, 302);
    assert.equal(
      response.headers.get("location"),
      "https://portal.example.com/login?returnTo=%2Fportal%3Fview%3Dreview%26job%3Djob_demo%26artifact%3Dsynthetic%252Freview-primary.png%26compare%3D1"
    );
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
    assert.equal(await sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
  } finally {
    env.cleanup();
  }
});

test("portal video proxy preserves cache-friendly binary delivery", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { GET } = await importFresh("../app/portal/video/[assetName]/route.js");
    const videoBytes = Uint8Array.from([0, 0, 0, 32, 102, 116, 121, 112, 105, 115, 111, 109]);

    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/portal/video/dna-portal-video-2.mp4");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      assert.equal(init.headers.get("range"), "bytes=0-31");
      assert.equal(init.headers.has("cookie"), false);
      return new Response(videoBytes, {
        status: 200,
        headers: {
          "content-type": "video/mp4",
          "cache-control": "public, max-age=86400",
          "accept-ranges": "bytes",
          "content-range": "bytes 0-31/128"
        }
      });
    };

    try {
      const request = buildRequest("https://portal.example.com/portal/video/dna-portal-video-2.mp4", {
        method: "GET",
        headers: new Headers({
          accept: "video/mp4",
          range: "bytes=0-31"
        })
      });

      const response = await GET(request, {
        params: Promise.resolve({ assetName: "dna-portal-video-2.mp4" })
      });

      assert.equal(response.status, 200);
      assert.equal(response.headers.get("content-type"), "video/mp4");
      assert.equal(response.headers.get("cache-control"), "public, max-age=86400");
      assert.equal(response.headers.get("accept-ranges"), "bytes");
      assert.equal(response.headers.get("content-range"), "bytes 0-31/128");
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), Buffer.from(videoBytes));
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal video proxy classifies upstream 404s as managed config failures", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { GET } = await importFresh("../app/portal/video/[assetName]/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(null, {
        status: 404,
        headers: {
          "cache-control": "no-store"
        }
      });

    try {
      await withCapturedAuditEvents(async (events) => {
        const request = buildRequest("https://portal.example.com/portal/video/dna-portal-video-2.mp4", {
          method: "GET"
        });

        const response = await GET(request, {
          params: Promise.resolve({ assetName: "dna-portal-video-2.mp4" })
        });

        assert.equal(response.status, 503);
        assert.equal(await response.text(), "Portal video proxy configuration unavailable");
        assert.equal(events.length, 1);
        assert.equal(events[0].surface, "portal_video");
        assert.equal(events[0].reason, "config_failure");
        assert.equal(events[0].upstreamStatus, 404);
      });
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal video proxy rejects unknown asset names with 404", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { GET } = await importFresh("../app/portal/video/[assetName]/route.js");

    const rejectedCases = [
      "evil.mp4",
      "",
      "   ",
      " dna-portal-video-2.mp4",
      "dna-portal-video-2.mp4 ",
      "../dna-portal-video-2.mp4",
      "../../etc/passwd",
      "%2e%2e%2fdna-portal-video-2.mp4",
      "dna-portal-video-2.mp4.exe"
    ];

    for (const assetName of rejectedCases) {
      const request = buildRequest(`https://portal.example.com/portal/video/${encodeURIComponent(assetName)}`, {
        method: "GET"
      });

      const response = await GET(request, {
        params: Promise.resolve({ assetName })
      });

      assert.equal(response.status, 404, `Expected 404 for assetName: ${JSON.stringify(assetName)}`);
      assert.equal(response.headers.get("Cache-Control"), "no-store", `Expected no-store for assetName: ${JSON.stringify(assetName)}`);
    }
  } finally {
    env.cleanup();
  }
});

test("portal asset proxy preserves content type and strips browser cookies", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { GET } = await importFresh("../app/portal/assets/[...path]/route.js");
    const cssBody = "body { color: #fff; }";

    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/portal/assets/portal.css?v=portal-css-v1");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      assert.equal(init.headers.get("accept"), "text/css");
      assert.equal(init.headers.has("cookie"), false);
      assert.equal(init.headers.get("x-forwarded-host"), "portal.example.com");
      assert.equal(init.headers.get("x-forwarded-proto"), "https");
      return new Response(cssBody, {
        status: 200,
        headers: {
          "content-type": "text/css; charset=utf-8",
          "cache-control": "no-store"
        }
      });
    };

    try {
      const request = buildRequest("https://portal.example.com/portal/assets/portal.css?v=portal-css-v1", {
        method: "GET",
        headers: new Headers({
          accept: "text/css",
          cookie: "__Host-tp_session=secret-session"
        })
      });

      const response = await GET(request, {
        params: Promise.resolve({ path: ["portal.css"] })
      });

      assert.equal(response.status, 200);
      assert.equal(response.headers.get("content-type"), "text/css; charset=utf-8");
      assert.equal(response.headers.get("cache-control"), "no-store");
      assert.equal(await response.text(), cssBody);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal asset proxy classifies missing backend auth as a config failure", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: ""
  });

  try {
    const { GET } = await importFresh("../app/portal/assets/[...path]/route.js");

    await withCapturedAuditEvents(async (events) => {
      const request = buildRequest("https://portal.example.com/portal/assets/portal.css", {
        method: "GET"
      });

      const response = await GET(request, {
        params: Promise.resolve({ path: ["portal.css"] })
      });

      assert.equal(response.status, 503);
      assert.equal(await response.text(), "Portal asset proxy configuration unavailable");
      assert.equal(events.length, 1);
      assert.equal(events[0].surface, "portal_asset");
      assert.equal(events[0].reason, "config_failure");
      assert.equal(events[0].assetPath, "portal.css");
    });
  } finally {
    env.cleanup();
  }
});

test("portal asset proxy supports nested asset paths and HEAD fallback", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { HEAD } = await importFresh("../app/portal/assets/[...path]/route.js");
    const fetchCalls = [];
    let canceledFallbackBody = false;

    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      fetchCalls.push({ url: String(url), method: init.method });
      assert.equal(String(url), "http://127.0.0.1:8000/portal/assets/fonts/portal-sans.woff2?v=font-v1");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      if (init.method === "HEAD") {
        return new Response(null, {
          status: 405,
          headers: {
            "cache-control": "no-store"
          }
        });
      }
      const fallbackBody = new ReadableStream({
        start(controller) {
          controller.enqueue(Uint8Array.from([119, 79, 70, 50]));
        },
        cancel() {
          canceledFallbackBody = true;
        }
      });
      return new Response(fallbackBody, {
        status: 200,
        headers: {
          "content-type": "font/woff2",
          "cache-control": "public, max-age=3600"
        }
      });
    };

    try {
      const request = buildRequest("https://portal.example.com/portal/assets/fonts/portal-sans.woff2?v=font-v1", {
        method: "HEAD"
      });

      const response = await HEAD(request, {
        params: Promise.resolve({ path: ["fonts", "portal-sans.woff2"] })
      });

      assert.equal(response.status, 200);
      assert.equal(response.headers.get("content-type"), "font/woff2");
      assert.equal(response.headers.get("cache-control"), "public, max-age=3600");
      assert.deepEqual(fetchCalls, [
        {
          url: "http://127.0.0.1:8000/portal/assets/fonts/portal-sans.woff2?v=font-v1",
          method: "HEAD"
        },
        {
          url: "http://127.0.0.1:8000/portal/assets/fonts/portal-sans.woff2?v=font-v1",
          method: "GET"
        }
      ]);
      assert.equal(canceledFallbackBody, true);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal asset proxy preserves upstream cache headers for 304 responses", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { GET } = await importFresh("../app/portal/assets/[...path]/route.js");

    const originalFetch = global.fetch;
    global.fetch = async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/portal/assets/portal.js?v=portal-js-v1");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("if-none-match"), '"portal-js-v1"');
      return new Response(null, {
        status: 304,
        headers: {
          "cache-control": "public, max-age=3600",
          etag: '"portal-js-v1"'
        }
      });
    };

    try {
      const request = buildRequest("https://portal.example.com/portal/assets/portal.js?v=portal-js-v1", {
        method: "GET",
        headers: new Headers({
          "if-none-match": '"portal-js-v1"'
        })
      });

      const response = await GET(request, {
        params: Promise.resolve({ path: ["portal.js"] })
      });

      assert.equal(response.status, 304);
      assert.equal(response.headers.get("cache-control"), "public, max-age=3600");
      assert.equal(response.headers.get("etag"), '"portal-js-v1"');
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("portal asset proxy rejects unknown and traversal-shaped asset paths with 404", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const { GET } = await importFresh("../app/portal/assets/[...path]/route.js");

    const rejectedCases = [
      [],
      ["evil.css"],
      [""],
      [" portal.css"],
      ["portal.css "],
      ["..", "portal.css"],
      ["fonts/portal-sans.woff2"],
      ["fonts", "..", "portal.css"],
      ["fonts", "portal-sans.woff2", ".."],
      ["fonts", "portal-sans.woff2.exe"]
    ];

    const originalFetch = global.fetch;
    let fetchCalled = false;
    global.fetch = async () => {
      fetchCalled = true;
      throw new Error("should not reach upstream");
    };

    try {
      for (const pathParts of rejectedCases) {
        const request = buildRequest("https://portal.example.com/portal/assets/rejected", {
          method: "GET"
        });

        const response = await GET(request, {
          params: Promise.resolve({ path: pathParts })
        });

        assert.equal(response.status, 404, `Expected 404 for path parts: ${JSON.stringify(pathParts)}`);
        assert.equal(response.headers.get("Cache-Control"), "no-store");
      }
      assert.equal(fetchCalled, false);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("shared portal asset manifest pins the managed asset proxy allowlist", async () => {
  const { PORTAL_ASSET_PATHS } = await importFresh("../lib/portal-asset-manifest.js");

  assert.deepEqual(PORTAL_ASSET_PATHS, [
    "portal.css",
    "shared-ui-tokens.css",
    "portal.js",
    "portal-review.js",
    "portal-review.css",
    "portal-operate.js",
    "portal-build.js",
    "portal-overview.js",
    "fonts/portal-sans.woff2",
    "fonts/portal-mono.woff2",
    "brand/dna-symbol-dark.svg",
    "brand/dna-symbol-light.svg"
  ]);
});

test("shared UI tokens stay synced to the canonical source", () => {
  const repoRoot = path.resolve(process.cwd(), "..", "..");
  const canonicalTokenPath = path.join(repoRoot, "web", "shared", "shared-ui-tokens.css");
  const frontdoorTokenPath = path.join(process.cwd(), "public", "shared-ui-tokens.css");
  const portalTokenPath = path.join(repoRoot, "public", "portal-assets", "shared-ui-tokens.css");
  const portalCssPath = path.join(repoRoot, "public", "portal-assets", "portal.css");
  const portalCssIndexPath = path.join(process.cwd(), "portal-src", "styles", "index.css");
  const buildScriptPath = path.join(process.cwd(), "scripts", "build-portal-bundle.mjs");
  const cssContractScriptPath = path.join(process.cwd(), "scripts", "check-portal-css-contract.mjs");
  const canonicalTokens = readFileSync(canonicalTokenPath, "utf-8");
  const normalizedCanonicalTokens = `${transformSync(canonicalTokens, {
    loader: "css",
    legalComments: "none",
    minify: true
  }).code.trim()}\n`;
  const frontdoorTokens = readFileSync(frontdoorTokenPath, "utf-8");
  const portalTokens = readFileSync(portalTokenPath, "utf-8");
  const portalCss = readFileSync(portalCssPath, "utf-8");
  const portalCssIndex = readFileSync(portalCssIndexPath, "utf-8");
  const buildScript = readFileSync(buildScriptPath, "utf-8");
  const cssContractScript = readFileSync(cssContractScriptPath, "utf-8");

  assert.match(canonicalTokens, /Canonical shared UI tokens/);
  assert.equal(frontdoorTokens, normalizedCanonicalTokens);
  assert.equal(portalTokens, normalizedCanonicalTokens);
  assert.notEqual(frontdoorTokens, canonicalTokens);
  assert.notEqual(portalTokens, canonicalTokens);
  assert.match(portalCss, /--ux-target-min-size\s*:/);
  assert.match(portalCss, /@layer\s+tokens\s*,\s*base\s*,\s*components\s*,\s*utilities\s*,\s*overrides\s*;/);
  assert.doesNotMatch(portalCss, /@import\b/);
  assert.doesNotMatch(portalCss, /__PORTAL_SHARED_TOKENS_URL__/);
  assert.match(portalCss, /__PORTAL_FONT_SANS_URL__/);
  assert.match(portalCss, /__PORTAL_FONT_MONO_URL__/);
  assert.match(portalCssIndex, /@layer tokens, base, components, utilities, overrides;/);
  assert.match(portalCssIndex, /@import "\.\/tokens\.css" layer\(tokens\);/);
  assert.match(portalCssIndex, /@import "\.\/components\/workspace-surfaces\.css" layer\(components\);/);
  assert.match(portalCssIndex, /@import "\.\/components\/operator-console\.css" layer\(components\);/);
  assert.match(portalCssIndex, /@import "\.\/components\/surface-normalization\.css" layer\(components\);/);
  assert.match(portalCssIndex, /@import "\.\/utilities\.required\.css" layer\(utilities\);/);
  assert.match(portalCssIndex, /@import "\.\/utilities\.dynamic\.css" layer\(utilities\);/);
  assert.match(portalCssIndex, /@import "\.\/utilities\.compat-hold\.css" layer\(utilities\);/);
  assert.doesNotMatch(portalCssIndex, /@import "\.\/overrides\.compat\.css" layer\(overrides\);/);
  assert.doesNotMatch(portalCssIndex, /overrides\.operator-console-reset\.css/);
  assert.doesNotMatch(portalCssIndex, /workspace-performance\.css" layer\(utilities\)/);
  assert.doesNotMatch(portalCssIndex, /utilities\.deprecated\.css/);
  assert.match(portalCssIndex, /@import "\.\/overrides\.performance\.css" layer\(overrides\);/);
  assert.match(portalCssIndex, /@import "\.\/overrides\.accessibility\.css" layer\(overrides\);/);
  const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const cssClassLiteral = (value) => value.replace(/[^A-Za-z0-9_-]/g, (character) => `\\${character}`);
  for (const classToken of [
    "lg:col-span-12",
    "lg:mb-0",
    "sm:grid-cols-4",
    "hover:border-indigo-400",
    "hover:border-slate-300",
    "dark:hover:border-slate-600",
    "dark:bg-slate-950/40",
    "bg-slate-200",
    "w-0",
    "bg-indigo-500",
    "transition-[width]",
    "bg-slate-950/72",
    "dark:bg-black/80",
    "max-w-6xl"
  ]) {
    assert.match(portalCss, new RegExp(`\\.${escapeRegex(cssClassLiteral(classToken))}`));
  }
  assert.match(buildScript, /const SHARED_TOKEN_SOURCE_PATH = path\.resolve\(REPO_ROOT, "web", "shared", "shared-ui-tokens\.css"\);/);
  assert.match(buildScript, /const PORTAL_CSS_ASSET_PATH = path\.resolve\(REPO_ROOT, "public", "portal-assets", "portal\.css"\);/);
  assert.match(buildScript, /const PORTAL_CSS_INDEX_PATH = path\.resolve\(PORTAL_CSS_SOURCE_DIR, "index\.css"\);/);
  assert.match(buildScript, /async function bundleCssEntry\(entryPoint\)/);
  assert.match(buildScript, /async function buildPortalCssAsset\(\)/);
  assert.match(buildScript, /async function writeMinifiedCssCopy\(sourcePath,\s*targetPath\)/);
  assert.match(buildScript, /loader: "css"/);
  assert.match(buildScript, /minify: true/);
  assert.match(buildScript, /Object\.prototype\.hasOwnProperty\.call\(options,\s*"minifyIdentifiers"\)/);
  assert.match(buildScript, /Object\.prototype\.hasOwnProperty\.call\(options,\s*"minifySyntax"\)/);
  assert.match(buildScript, /Object\.prototype\.hasOwnProperty\.call\(options,\s*"minifyWhitespace"\)/);
  assert.doesNotMatch(buildScript, /minifyIdentifiers:\s*Boolean\(options\.minifyIdentifiers\)/);
  assert.doesNotMatch(buildScript, /minifySyntax:\s*Boolean\(options\.minifySyntax\)/);
  assert.doesNotMatch(buildScript, /minifyWhitespace:\s*Boolean\(options\.minifyWhitespace\)/);
  assert.match(buildScript, /Generated portal\.css must not contain runtime @import rules/);
  assert.match(buildScript, /Generated portal\.css missing required font placeholder/);
  assert.match(buildScript, /writeMinifiedCssCopy\(SHARED_TOKEN_SOURCE_PATH,\s*PORTAL_SHARED_TOKEN_TARGET\)/);
  assert.match(buildScript, /writeMinifiedCssCopy\(SHARED_TOKEN_SOURCE_PATH,\s*FRONTDOOR_SHARED_TOKEN_TARGET\)/);
  assert.match(cssContractScript, /PORTAL_HTML_PATH/);
  assert.match(cssContractScript, /PORTAL_TEMPLATE_SOURCE_PATH/);
  assert.match(cssContractScript, /missing utility compatibility coverage/);
});

test("v1 POST rejects requests missing valid same-origin CSRF protections", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/v1/jobs", {
      method: "POST",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`,
        "Cf-Access-Jwt-Assertion": createAccessJwt()
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

test("v1 rejects authenticated sessions without a current Access JWT", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/v1/jobs", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`
      })
    });

      const response = await route.GET(request, {
        params: { path: ["jobs"] }
      });
      const body = await response.json();

      assert.equal(response.status, 401);
      assert.equal(body.error.code, "UNAUTHORIZED");
      assert.equal(body.error.details.reason, "auth_failure");
      assert.equal(body.error.details.retryable, false);
      assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
      assert.equal(await sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
    } finally {
      env.cleanup();
  }
});

test("v1 returns forbidden when the current Access identity does not match the authenticated session", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      const request = buildRequest("https://portal.example.com/v1/jobs", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt({ email: "other@example.com" })
        })
      });

      const response = await route.GET(request, {
        params: { path: ["jobs"] }
      });
      const body = await response.json();

      assert.equal(response.status, 403);
      assert.equal(body.error.code, "FORBIDDEN");
      assert.equal(body.error.message, "forbidden");
      assert.equal(body.error.details.reason, "auth_failure");
      assert.equal(body.error.details.retryable, false);
      assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
      assert.equal(await sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("v1 reports managed proxy config failures when the backend API key is missing", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: ""
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");
    const restoreFetch = withMockedAccessCerts();

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      await withCapturedAuditEvents(async (events) => {
        const request = buildRequest("https://portal.example.com/v1/jobs", {
          method: "GET",
          headers: new Headers({
            cookie: `__Host-tp_session=${authenticatedSession.id}`,
            "Cf-Access-Jwt-Assertion": createAccessJwt()
          })
        });

        const response = await route.GET(request, {
          params: { path: ["jobs"] }
        });
        const body = await response.json();

        assert.equal(response.status, 503);
        assert.equal(body.error.code, "AUTH_CONFIGURATION_ERROR");
        assert.equal(body.error.details.reason, "config_failure");
        assert.equal(body.error.details.retryable, false);
        assert.equal(events.length, 1);
        assert.equal(events[0].surface, "v1_proxy");
        assert.equal(events[0].reason, "config_failure");
      });
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("v1 normalizes upstream outages into a structured retryable error envelope", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");
    const restoreFetch = withMockedAccessCerts(async (url) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs");
      return Response.json(
        {
          detail: "backend temporarily unavailable"
        },
        {
          status: 503
        }
      );
    });

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      await withCapturedAuditEvents(async (events) => {
        const request = buildRequest("https://portal.example.com/v1/jobs", {
          method: "GET",
          headers: new Headers({
            cookie: `__Host-tp_session=${authenticatedSession.id}`,
            "Cf-Access-Jwt-Assertion": createAccessJwt()
          })
        });

        const response = await route.GET(request, {
          params: { path: ["jobs"] }
        });
        const body = await response.json();

        assert.equal(response.status, 502);
        assert.equal(body.error.code, "UPSTREAM_UNAVAILABLE");
        assert.equal(body.error.details.reason, "upstream_unavailable");
        assert.equal(body.error.details.retryable, true);
        assert.equal(body.error.details.upstreamStatus, 503);
        assert.equal(events.length, 1);
        assert.equal(events[0].surface, "v1_proxy");
        assert.equal(events[0].reason, "upstream_unavailable");
        assert.equal(events[0].upstreamStatus, 503);
      });
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("v1 preserves rate-limit retry headers and envelope on upstream 429", async () => {
  // Backend contract (Retry-After + X-RateLimit-Limit/Remaining/Reset) must
  // reach the browser unchanged through the managed proxy. 429 is *not*
  // classified as an upstream failure, so the proxy returns the upstream
  // body and headers verbatim (modulo the hop-by-hop denylist).
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");
    const restoreFetch = withMockedAccessCerts(async (url) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs");
      return new Response(
        JSON.stringify({
          schema: "tp.orchestrator.error.v1",
          success: false,
          data: null,
          error: {
            code: "RATE_LIMITED",
            message: "rate limit exceeded",
            details: { client_ip: "203.0.113.5" }
          }
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "Retry-After": "12",
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1700000060"
          }
        }
      );
    });

    try {
      const authenticatedSession = await sessions.rotateAuthenticatedSession(
        await sessions.createAnonymousSession(),
        {
          username: "admin",
          accessEmail: "admin@example.com",
          role: "admin"
        }
      );

      const request = buildRequest("https://portal.example.com/v1/jobs", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt()
        })
      });

      const response = await route.GET(request, {
        params: { path: ["jobs"] }
      });

      // Status forwarded.
      assert.equal(response.status, 429);

      // The four contract headers reach the browser intact. Header lookup
      // is case-insensitive by Headers semantics; assert lower-case to
      // avoid coupling the test to a specific casing convention.
      assert.equal(response.headers.get("retry-after"), "12");
      assert.equal(response.headers.get("x-ratelimit-limit"), "60");
      assert.equal(response.headers.get("x-ratelimit-remaining"), "0");
      assert.equal(response.headers.get("x-ratelimit-reset"), "1700000060");

      // Sensitive request-side headers must not be reflected on responses.
      assert.equal(response.headers.get("x-api-key"), null);
      assert.equal(response.headers.get("authorization"), null);

      // Envelope body untouched on 429.
      const body = await response.json();
      assert.equal(body.schema, "tp.orchestrator.error.v1");
      assert.equal(body.success, false);
      assert.equal(body.data, null);
      assert.equal(body.error.code, "RATE_LIMITED");
      assert.equal(body.error.message, "rate limit exceeded");
      assert.equal(body.error.details.client_ip, "203.0.113.5");

      // Existing v1-proxy invariant: responses are not cacheable.
      assert.equal(response.headers.get("cache-control"), "no-store");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("v1 normalizes upstream auth responses into managed config failures", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");
    const statuses = [401, 403];

    for (const upstreamStatus of statuses) {
      const restoreFetch = withMockedAccessCerts(async (url) => {
        assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs");
        return new Response(null, {
          status: upstreamStatus
        });
      });

      try {
        const authenticatedSession = await sessions.rotateAuthenticatedSession(
          await sessions.createAnonymousSession(),
          {
            username: "admin",
            accessEmail: "admin@example.com",
            role: "admin"
          }
        );

        await withCapturedAuditEvents(async (events) => {
          const request = buildRequest("https://portal.example.com/v1/jobs", {
            method: "GET",
            headers: new Headers({
              cookie: `__Host-tp_session=${authenticatedSession.id}`,
              "Cf-Access-Jwt-Assertion": createAccessJwt()
            })
          });

          const response = await route.GET(request, {
            params: { path: ["jobs"] }
          });
          const body = await response.json();

          assert.equal(response.status, 503);
          assert.equal(body.error.code, "AUTH_CONFIGURATION_ERROR");
          assert.equal(body.error.message, "managed proxy misconfigured");
          assert.equal(body.error.details.reason, "config_failure");
          assert.equal(body.error.details.retryable, false);
          assert.equal(body.error.details.upstreamStatus, upstreamStatus);
          assert.equal(events.length, 1);
          assert.equal(events[0].surface, "v1_proxy");
          assert.equal(events[0].reason, "config_failure");
          assert.equal(events[0].status, 503);
          assert.equal(events[0].upstreamStatus, upstreamStatus);
        });
      } finally {
        restoreFetch();
      }
    }
  } finally {
    env.cleanup();
  }
});

test("v1 forwards browser traceparent upstream and includes trace id in queue-action audits", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });
  const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const restoreFetch = withMockedAccessCerts(async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs");
      assert.equal(init.method, "POST");
      assert.equal(init.headers.get("traceparent"), traceparent);
      return Response.json(
        {
          ok: true
        },
        {
          status: 200,
          headers: {
            traceparent
          }
        }
      );
    });

    try {
      await withCapturedAuditEvents(async (events) => {
        const request = buildRequest("https://portal.example.com/v1/jobs", {
          method: "POST",
          headers: new Headers({
            cookie: `__Host-tp_session=${authenticatedSession.id}`,
            traceparent,
            origin: "https://portal.example.com",
            referer: "https://portal.example.com/portal?view=build",
            "x-csrf-token": authenticatedSession.csrfToken,
            "content-type": "application/json",
            "Cf-Access-Jwt-Assertion": createAccessJwt()
          }),
          body: JSON.stringify({ pipeline: "lux-depth-v3" })
        });

        const response = await route.POST(request, {
          params: { path: ["jobs"] }
        });
        const body = await response.json();

        assert.equal(response.status, 200);
        assert.equal(response.headers.get("traceparent"), traceparent);
        assert.deepEqual(body, { ok: true });
        assert.equal(events.length, 1);
        assert.equal(events[0].event, "job_submit");
        assert.equal(events[0].traceId, "4bf92f3577b34da6a3ce929d0e0e4736");
      });
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("v1 SSE proxy preserves event-stream framing while injecting backend auth server-side", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });
  const requestTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
  const upstreamTraceparent = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01";

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
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

    const restoreFetch = withMockedAccessCerts(async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs/job-123/events");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("Forwarded"), 'for="203.0.113.10";host="portal.example.com";proto="https"');
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      assert.equal(init.headers.get("traceparent"), requestTraceparent);
      assert.equal(init.headers.get("Accept-Encoding"), "identity");
      assert.equal(init.headers.has("cookie"), false);
      assert.equal(init.headers.has("x-csrf-token"), false);
      assert.equal(init.headers.get("x-forwarded-for"), "203.0.113.10");
      assert.equal(init.headers.get("x-forwarded-host"), "portal.example.com");
      assert.equal(init.headers.get("x-forwarded-proto"), "https");
      assert.equal(init.headers.get("x-real-ip"), "203.0.113.10");

      return new Response(upstreamEvents, {
        status: 200,
        headers: {
          "content-type": "text/event-stream; charset=utf-8",
          "cache-control": "private",
          traceparent: upstreamTraceparent
        }
      });
    });

    try {
      const request = buildRequest("https://portal.example.com/v1/jobs/job-123/events", {
        method: "GET",
        headers: new Headers({
          "cf-connecting-ip": "203.0.113.10",
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          accept: "text/event-stream",
          traceparent: requestTraceparent,
          "Cf-Access-Jwt-Assertion": createAccessJwt()
        })
      });

      const response = await route.GET(request, {
        params: { path: ["jobs", "job-123", "events"] }
      });

      assert.equal(response.status, 200);
      assert.match(response.headers.get("content-type") || "", /text\/event-stream/);
      assert.equal(response.headers.get("cache-control"), "no-store, no-transform");
      assert.equal(response.headers.get("traceparent"), upstreamTraceparent);
      assert.equal(await response.text(), upstreamEvents);
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("v1 artifact proxy passes binary previews through the managed front door with async params", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "backend-secret"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");

    const authenticatedSession = await sessions.rotateAuthenticatedSession(
      await sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const previewBytes = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 112, 114, 101, 118, 105, 101, 119]);
    const restoreFetch = withMockedAccessCerts(async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs/job-123/artifacts/renders/hero.png");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
      assert.equal(init.headers.has("cookie"), false);
      return new Response(previewBytes, {
        status: 200,
        headers: {
          "content-type": "image/png",
          "cache-control": "private"
        }
      });
    });

    try {
      const request = buildRequest("https://portal.example.com/v1/jobs/job-123/artifacts/renders/hero.png", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          accept: "image/png",
          "Cf-Access-Jwt-Assertion": createAccessJwt()
        })
      });

      const response = await route.GET(request, {
        params: Promise.resolve({ path: ["jobs", "job-123", "artifacts", "renders", "hero.png"] })
      });

      assert.equal(response.status, 200);
      assert.equal(response.headers.get("content-type"), "image/png");
      assert.equal(response.headers.get("cache-control"), "no-store");
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), Buffer.from(previewBytes));
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("Access JWT verification refreshes certs on unknown kid after rotation", async () => {
  const { verifyAccessJwt } = await importFresh("../lib/access-jwt.js");
  const originalFetch = global.fetch;
  let fetchCount = 0;

  global.fetch = async (url) => {
    assert.equal(String(url), `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`);
    fetchCount += 1;
    const keys = fetchCount === 1 ? [TEST_CF_ACCESS_PUBLIC_JWK] : [TEST_CF_ACCESS_ROTATED_PUBLIC_JWK];
    return Response.json({ keys }, { status: 200 });
  };

  try {
    const initial = await verifyAccessJwt(createAccessJwt(), {
      teamDomain: TEST_CF_ACCESS_TEAM_DOMAIN,
      audience: TEST_CF_ACCESS_AUD
    });
    const rotated = await verifyAccessJwt(
      createAccessJwt({
        kid: TEST_CF_ACCESS_ROTATED_KID,
        privateKey: TEST_CF_ACCESS_ROTATED_KEYS.privateKey
      }),
      {
        teamDomain: TEST_CF_ACCESS_TEAM_DOMAIN,
        audience: TEST_CF_ACCESS_AUD
      }
    );

    assert.equal(initial.accessEmail, "admin@example.com");
    assert.equal(rotated.accessEmail, "admin@example.com");
    assert.equal(fetchCount, 2);
  } finally {
    global.fetch = originalFetch;
  }
});

test("Access team domain normalization rejects insecure HTTP origins", async () => {
  const { normalizeAccessTeamDomain } = await importFresh("../lib/config.js");

  assert.equal(normalizeAccessTeamDomain(TEST_CF_ACCESS_TEAM_DOMAIN), TEST_CF_ACCESS_TEAM_DOMAIN);
  assert.equal(normalizeAccessTeamDomain("tp-frontdoor-tests.cloudflareaccess.com"), TEST_CF_ACCESS_TEAM_DOMAIN);
  assert.equal(normalizeAccessTeamDomain("http://tp-frontdoor-tests.cloudflareaccess.com"), "");
});

test("Access JWT verification returns only minimal verified identity fields", async () => {
  const { verifyAccessJwt } = await importFresh("../lib/access-jwt.js");
  const originalFetch = global.fetch;

  global.fetch = async (url, init) => {
    assert.equal(String(url), `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`);
    assert.ok(init.signal);
    return Response.json({ keys: [TEST_CF_ACCESS_PUBLIC_JWK] }, { status: 200 });
  };

  try {
    const verified = await verifyAccessJwt(createAccessJwt(), {
      teamDomain: TEST_CF_ACCESS_TEAM_DOMAIN,
      audience: TEST_CF_ACCESS_AUD
    });

    assert.deepEqual(Object.keys(verified).sort(), ["accessEmail", "audience", "issuer"]);
    assert.equal(verified.accessEmail, "admin@example.com");
    assert.equal(verified.issuer, TEST_CF_ACCESS_TEAM_DOMAIN);
    assert.equal(verified.audience, TEST_CF_ACCESS_AUD);
  } finally {
    global.fetch = originalFetch;
  }
});

test("Access JWT verification preserves unsupported algorithm failures", async () => {
  const { verifyAccessJwt } = await importFresh("../lib/access-jwt.js");
  const originalFetch = global.fetch;

  global.fetch = async (url, init) => {
    assert.equal(String(url), `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`);
    assert.ok(init.signal);
    return Response.json({ keys: [TEST_CF_ACCESS_PUBLIC_JWK] }, { status: 200 });
  };

  try {
    await assert.rejects(
      () =>
        verifyAccessJwt(createAccessJwt({ alg: "HS256" }), {
          teamDomain: TEST_CF_ACCESS_TEAM_DOMAIN,
          audience: TEST_CF_ACCESS_AUD
        }),
      (error) => error?.code === "unsupported_algorithm"
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test("Access JWT verification fails closed when cert fetch times out", async () => {
  const { verifyAccessJwt } = await importFresh("../lib/access-jwt.js");
  const originalFetch = global.fetch;

  global.fetch = async (url, init) => {
    assert.equal(String(url), `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`);
    assert.ok(init.signal);
    throw Object.assign(new Error("timed out"), { name: "AbortError" });
  };

  try {
    await assert.rejects(
      () =>
        verifyAccessJwt(createAccessJwt(), {
          teamDomain: TEST_CF_ACCESS_TEAM_DOMAIN,
          audience: TEST_CF_ACCESS_AUD
        }),
      (error) => error?.code === "jwks_unreachable"
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test("healthz reports structured readiness checks without leaking the backend origin", async () => {
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 200);
      assert.equal(body.ok, true);
      assert.equal(body.frontend, "ready");
      assert.equal(body.backend.ok, true);
      assert.equal(body.backend.status, 200);
      assert.equal("origin" in body.backend, false);
      assert.equal(body.checks.backend.ok, true);
      assert.equal(body.checks.backend.required, true);
      assert.equal(body.checks.backend.configured, true);
      assert.equal(body.checks.access_config.ok, true);
      assert.equal(body.checks.access_config.required, true);
      assert.equal(body.checks.user_source.ok, true);
      assert.equal(body.checks.user_source.source, "file");
      assert.equal(body.checks.user_source.userCount, 1);
      assert.equal(body.checks.session_store.ok, true);
      assert.equal(body.checks.session_scaling.ok, true);
      assert.equal(body.checks.session_scaling.backend, "sqlite");
      assert.equal(body.checks.session_scaling.mode, "single_instance");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz fails closed when backend auth is not configured", async () => {
  const env = withTempEnvironment({
    TP_BACKEND_API_KEY: "",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const response = await GET();
    const body = await response.json();

    assert.equal(response.status, 503);
    assert.equal(body.ok, false);
    assert.equal(body.checks.backend.ok, false);
    assert.equal(body.checks.backend.reason, "missing_backend_api_key");
  } finally {
    env.cleanup();
  }
});

test("healthz marks Access config as required in production when bypass is disabled", async () => {
  const env = withTempEnvironment({
    TP_CF_ACCESS_AUD: "",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.access_config.ok, false);
      assert.equal(body.checks.access_config.required, true);
      assert.equal(body.checks.access_config.reason, "missing_access_audience");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz rejects an empty user source", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_USERS_JSON: "[]"
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.user_source.ok, false);
      assert.equal(body.checks.user_source.userCount, 0);
      assert.equal(body.checks.user_source.reason, "no_configured_users");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz rejects an unavailable session store", async () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-session-store-"));
  const blockedParent = path.join(tempDir, "blocked-parent");
  writeFileSync(blockedParent, "occupied", "utf-8");

  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_DB: path.join(blockedParent, "sessions.sqlite"),
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.session_store.ok, false);
      assert.equal(body.checks.session_store.reason, "session_store_unavailable");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("healthz rejects multi-instance session scaling until an external session store exists", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_SCALING_MODE: "multi_instance",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.session_scaling.ok, false);
      assert.equal(body.checks.session_scaling.backend, "sqlite");
      assert.equal(body.checks.session_scaling.mode, "multi_instance");
      assert.equal(
        body.checks.session_scaling.reason,
        "multi_instance_requires_external_session_store"
      );
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz accepts Redis-backed multi-instance session readiness", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_SCALING_MODE: "multi_instance",
    TP_FRONTDOOR_SESSION_STORE: "redis",
    TP_FRONTDOOR_REDIS_URL: "rediss://redis.example.com:6380/0",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });
  const sessionStoreModule = await import("../lib/session-store/index.js");

  try {
    sessionStoreModule.__setSessionStoreForTesting(
      {
        backend: "redis",
        ping: async () => "PONG"
      },
      "redis"
    );
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 200);
      assert.equal(body.ok, true);
      assert.equal(body.checks.session_store.ok, true);
      assert.equal(body.checks.session_store.backend, "redis");
      assert.equal(body.checks.session_scaling.ok, true);
      assert.equal(body.checks.session_scaling.backend, "redis");
      assert.equal(body.checks.session_scaling.mode, "multi_instance");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    sessionStoreModule.resetSessionStoreSingleton();
    env.cleanup();
  }
});

test("healthz bounds Redis session readiness probes", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_SCALING_MODE: "multi_instance",
    TP_FRONTDOOR_SESSION_STORE: "redis",
    TP_FRONTDOOR_REDIS_URL: "rediss://redis.example.com:6380/0",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });
  const sessionStoreModule = await import("../lib/session-store/index.js");

  try {
    sessionStoreModule.__setSessionStoreForTesting(
      {
        backend: "redis",
        ping: async () => new Promise(() => {})
      },
      "redis"
    );
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const startedAt = Date.now();
      const response = await GET();
      const elapsedMs = Date.now() - startedAt;
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.session_store.ok, false);
      assert.equal(body.checks.session_store.backend, "redis");
      assert.equal(body.checks.session_store.reason, "session_store_unavailable");
      assert.ok(elapsedMs < 1800, `Redis health probe should be timeout-bounded, took ${elapsedMs}ms`);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    sessionStoreModule.resetSessionStoreSingleton();
    env.cleanup();
  }
});

test("healthz fails closed when Redis session store lacks a URL", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_SCALING_MODE: "multi_instance",
    TP_FRONTDOOR_SESSION_STORE: "redis",
    TP_FRONTDOOR_REDIS_URL: "",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.session_store.ok, false);
      assert.equal(body.checks.session_store.backend, "redis");
      assert.equal(body.checks.session_store.reason, "missing_frontdoor_redis_url");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz fails closed for unsupported session store backends", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_STORE: "memcached",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.session_store.ok, false);
      assert.equal(body.checks.session_store.backend, "memcached");
      assert.equal(body.checks.session_store.reason, "invalid_session_store_backend");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz rejects invalid session scaling mode declarations", async () => {
  const env = withTempEnvironment({
    TP_FRONTDOOR_SESSION_SCALING_MODE: "planet-scale",
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "content-type": "application/json"
        }
      });

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.checks.session_scaling.ok, false);
      assert.equal(body.checks.session_scaling.mode, "planet_scale");
      assert.equal(body.checks.session_scaling.reason, "invalid_session_scaling_mode");
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz reports backend outages as degraded readiness", async () => {
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async () => {
      throw new Error("connection refused");
    };

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.frontend, "degraded");
      assert.equal(body.checks.backend.ok, false);
      assert.equal(body.checks.backend.reason, "backend_unreachable");
      assert.equal(body.backend.status, 0);
      assert.equal(body.checks.session_scaling.ok, true);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz reports backend_auth_mismatch when the protected probe returns 401", async () => {
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async (input) => {
      const url = String(input?.url || input);
      if (url.includes("/v1/config-metadata")) {
        return new Response(JSON.stringify({ error: "unauthorized" }), {
          status: 401,
          headers: { "content-type": "application/json" }
        });
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    };

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.frontend, "degraded");
      assert.equal(body.checks.backend.ok, false);
      assert.equal(body.checks.backend.reason, "backend_auth_mismatch");
      assert.equal(body.checks.backend.auth_status, 401);
      assert.equal(body.checks.backend.status, 200);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("healthz fails closed when protected backend probe returns an unexpected non-OK status", async () => {
  const env = withTempEnvironment({
    usersFileEntries: [
      {
        username: "admin",
        password_hash: "hash",
        access_email: "admin@example.com",
        role: "admin"
      }
    ]
  });

  try {
    const { GET } = await importFresh("../app/healthz/route.js");
    const originalFetch = global.fetch;
    global.fetch = async (input) => {
      const url = String(input?.url || input);
      if (url.includes("/v1/config-metadata")) {
        return new Response(JSON.stringify({ error: "not found" }), {
          status: 404,
          headers: { "content-type": "application/json" }
        });
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    };

    try {
      const response = await GET();
      const body = await response.json();

      assert.equal(response.status, 503);
      assert.equal(body.frontend, "degraded");
      assert.equal(body.checks.backend.ok, false);
      assert.equal(body.checks.backend.reason, "backend_protected_probe_unexpected_status");
      assert.equal(body.checks.backend.auth_status, 404);
      assert.equal(body.checks.backend.status, 200);
    } finally {
      global.fetch = originalFetch;
    }
  } finally {
    env.cleanup();
  }
});

test("proxy does not redirect /login based only on cookie presence", async () => {
  const { proxy } = await importFresh("../proxy.js");
  const response = proxy(
    buildRequest("https://portal.example.com/login", {
      headers: new Headers({
        cookie: "__Host-tp_session=stale-cookie"
      })
    })
  );

  assert.equal(response.headers.get("location"), null);
});

test("development cookies relax __Host/Secure requirements for local HTTP", async () => {
  const env = withTempEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1"
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const response = NextResponse.json({ ok: true });
    const anonymousSession = await sessions.createAnonymousSession();

    sessions.setSessionCookie(response, anonymousSession.id);

    const cookieHeader = response.headers.get("set-cookie") || "";
    assert.match(cookieHeader, /^tp_session=/);
    assert.doesNotMatch(cookieHeader, /__Host-tp_session/);
    assert.doesNotMatch(cookieHeader, /Secure/i);
  } finally {
    env.cleanup();
  }
});
