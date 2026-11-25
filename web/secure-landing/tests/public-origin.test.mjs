// TP_FRONTDOOR_PUBLIC_ORIGIN contract tests.
//
// Next derives the URL of a request from the runtime, not from the browser:
// a self-hosted start uses the listener's own host and port and takes the
// scheme from x-forwarded-proto, and a Vercel build uses the Host header of
// the deployment. Behind an edge proxy none of those is the origin the browser
// is on. These tests model each runtime shape explicitly and prove that a
// configured public origin gives one answer whatever the runtime URL and the
// host or forwarded headers say.
import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createSign, generateKeyPairSync } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";

import argon2 from "argon2";
import { NextRequest } from "next/server.js";

import worker from "../../../cloudflare/transformationportal-worker/src/index.ts";
import {
  isCanonicalPublicOrigin,
  PUBLIC_ORIGIN_BYPASS_CONFLICT_REASON,
  PUBLIC_ORIGIN_INVALID_REASON,
  resolvePublicOrigin
} from "../lib/config.js";
import { resetDbCache } from "../lib/db.js";
import { buildRequestUrl } from "../lib/http.js";
import { validateOriginAndReferrer } from "../lib/request-security.js";

const ENV_KEYS = [
  "NODE_ENV",
  "TP_FASTAPI_ORIGIN",
  "TP_BACKEND_API_KEY",
  "TP_PILOT_CONTROL_PLANE_ENABLED",
  "TP_FRONTDOOR_IDENTITY_SECRET",
  "TP_FRONTDOOR_USERS_FILE",
  "TP_FRONTDOOR_USERS_JSON",
  "TP_FRONTDOOR_SESSION_DB",
  "TP_FRONTDOOR_SESSION_SCALING_MODE",
  "TP_FRONTDOOR_SESSION_STORE",
  "TP_FRONTDOOR_REDIS_URL",
  "TP_CF_ACCESS_TEAM_DOMAIN",
  "TP_CF_ACCESS_AUD",
  "TP_ALLOW_LOCAL_ACCESS_BYPASS",
  "TP_FRONTDOOR_PUBLIC_ORIGIN",
  "TP_PORTAL_RUM_ENABLED",
  "TP_FRONTDOOR_RUM_ENABLED"
];

const FRONTDOOR_APP_ROOT = fileURLToPath(new URL("../", import.meta.url));
const VECTORS = JSON.parse(
  readFileSync(path.join(FRONTDOOR_APP_ROOT, "tests", "fixtures", "public-origin-vectors.json"), "utf-8")
);

const PUBLIC_ORIGIN = "https://portal.example.com";
const PUBLIC_HOST = "portal.example.com";
const INTERNAL_ORIGIN = "https://frontdoor.internal.example";
const INTERNAL_HOST = "frontdoor.internal.example";
const FASTAPI_ORIGIN = "http://127.0.0.1:8000";
const OPERATOR_PASSWORD = "correct horse battery staple";

const TEST_CF_ACCESS_TEAM_DOMAIN = "https://tp-frontdoor-tests.cloudflareaccess.com";
const TEST_CF_ACCESS_AUD = "tp-frontdoor-aud";
const TEST_CF_ACCESS_KID = "tp-frontdoor-key";
const TEST_CF_ACCESS_CERTS_URL = `${TEST_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`;
const TEST_CF_ACCESS_KEYS = generateKeyPairSync("rsa", { modulusLength: 2048 });
const TEST_CF_ACCESS_PUBLIC_JWK = {
  ...TEST_CF_ACCESS_KEYS.publicKey.export({ format: "jwk" }),
  kid: TEST_CF_ACCESS_KID,
  alg: "RS256",
  use: "sig"
};

// What a request looks like once it reaches a route handler, by runtime.
const RUNTIME_SHAPES = [
  { name: "loopback listener over plain http", base: "http://localhost:3000" },
  { name: "loopback listener behind TLS termination", base: "https://localhost:3000" },
  { name: "wildcard listener", base: "http://0.0.0.0:3000" },
  { name: "container hostname", base: "https://frontdoor-7d9c:3000" },
  { name: "Vercel deployment host", base: "https://tp-frontdoor.vercel.app" }
];

// The headers the Worker edge proxy sends upstream (see its index.ts).
const WORKER_HEADERS = {
  host: INTERNAL_HOST,
  "x-forwarded-host": PUBLIC_HOST,
  "x-forwarded-proto": "https"
};

const HEADER_SETS = [
  { name: "Worker edge proxy", headers: WORKER_HEADERS },
  { name: "public Host with no forwarded headers", headers: { host: PUBLIC_HOST } },
  { name: "no host headers", headers: {} },
  {
    name: "hostile overrides",
    headers: {
      host: "evil.example",
      "x-forwarded-host": "evil.example",
      "x-forwarded-proto": "http",
      forwarded: "host=evil.example;proto=http"
    }
  },
  {
    name: "forwarded host rewritten to the internal host",
    headers: { host: INTERNAL_HOST, "x-forwarded-host": INTERNAL_HOST, "x-forwarded-proto": "https" }
  }
];

function originVerdicts(runtimeBase) {
  return [
    { name: "public Origin", headers: { origin: PUBLIC_ORIGIN }, accepted: true },
    {
      name: "public Referer without Origin",
      headers: { referer: `${PUBLIC_ORIGIN}/portal?view=build` },
      accepted: true
    },
    { name: "attacker Origin", headers: { origin: "https://evil.example" }, accepted: false },
    {
      name: "attacker Origin with a public Referer",
      headers: { origin: "https://evil.example", referer: `${PUBLIC_ORIGIN}/portal` },
      accepted: false
    },
    { name: "attacker Referer without Origin", headers: { referer: "https://evil.example/portal" }, accepted: false },
    { name: "internal Origin", headers: { origin: INTERNAL_ORIGIN }, accepted: false },
    { name: "the runtime's own Origin", headers: { origin: new URL(runtimeBase).origin }, accepted: false },
    { name: "http downgrade of the public origin", headers: { origin: `http://${PUBLIC_HOST}` }, accepted: false },
    { name: "subdomain of the public host", headers: { origin: `https://evil.${PUBLIC_HOST}` }, accepted: false },
    { name: "public host as a prefix", headers: { origin: `${PUBLIC_ORIGIN}.evil.example` }, accepted: false },
    { name: "opaque Origin", headers: { origin: "null" }, accepted: false },
    { name: "neither Origin nor Referer", headers: {}, accepted: false }
  ];
}

function withEnvironment(overrides = {}) {
  const snapshot = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));
  const tempDir = mkdtempSync(path.join(os.tmpdir(), "tp-frontdoor-public-origin-"));
  const { users, ...env } = overrides;
  const baseline = {
    NODE_ENV: "production",
    TP_FASTAPI_ORIGIN: FASTAPI_ORIGIN,
    TP_BACKEND_API_KEY: "backend-secret",
    TP_PILOT_CONTROL_PLANE_ENABLED: "0",
    TP_FRONTDOOR_USERS_JSON: "[]",
    TP_FRONTDOOR_SESSION_DB: path.join(tempDir, "sessions.sqlite"),
    TP_CF_ACCESS_TEAM_DOMAIN: TEST_CF_ACCESS_TEAM_DOMAIN,
    TP_CF_ACCESS_AUD: TEST_CF_ACCESS_AUD
  };
  if (Array.isArray(users)) {
    const usersFilePath = path.join(tempDir, "frontdoor-users.json");
    writeFileSync(usersFilePath, JSON.stringify(users), "utf-8");
    baseline.TP_FRONTDOOR_USERS_FILE = usersFilePath;
  }

  // Every listed key is set or deleted, so nothing leaks in from the shell.
  for (const key of ENV_KEYS) {
    const value = Object.hasOwn(env, key) ? env[key] : baseline[key];
    if (typeof value === "string") {
      process.env[key] = value;
    } else {
      delete process.env[key];
    }
  }
  resetDbCache();

  return {
    restore() {
      resetDbCache();
      for (const key of ENV_KEYS) {
        const previous = snapshot.get(key);
        if (typeof previous === "string") {
          process.env[key] = previous;
        } else {
          delete process.env[key];
        }
      }
      rmSync(tempDir, { recursive: true, force: true });
    }
  };
}

async function importFresh(relativePath) {
  return import(`${relativePath}?case=${Date.now()}-${Math.random()}`);
}

function runtimeRequest(base, pathname, { method = "GET", headers = {}, body } = {}) {
  const init = { method, headers: new Headers(headers) };
  if (body !== undefined) init.body = body;
  return new NextRequest(new URL(pathname, base), init);
}

function createAccessJwt() {
  const now = Math.floor(Date.now() / 1000);
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
  const signingInput = [
    encode({ alg: "RS256", kid: TEST_CF_ACCESS_KID, typ: "JWT" }),
    encode({
      aud: [TEST_CF_ACCESS_AUD],
      email: "admin@example.com",
      exp: now + 300,
      iat: now - 5,
      iss: TEST_CF_ACCESS_TEAM_DOMAIN,
      nbf: now - 5
    })
  ].join(".");
  const signature = createSign("RSA-SHA256")
    .update(signingInput)
    .end()
    .sign(TEST_CF_ACCESS_KEYS.privateKey)
    .toString("base64url");
  return `${signingInput}.${signature}`;
}

function accessCertsResponse() {
  return Response.json({ keys: [TEST_CF_ACCESS_PUBLIC_JWK] });
}

// Replace fetch for one test. Access certificate requests are answered here;
// everything else goes to `fallback`, which fails the test by default.
function withFetch(fallback = null) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = String(input instanceof Request ? input.url : input);
    if (url === TEST_CF_ACCESS_CERTS_URL) return accessCertsResponse();
    if (fallback) return fallback(input, init);
    throw new Error(`Unexpected fetch to ${url}`);
  };
  return () => {
    globalThis.fetch = originalFetch;
  };
}

async function operatorUsers() {
  return [
    {
      username: "admin",
      password_hash: await argon2.hash(OPERATOR_PASSWORD),
      access_email: "admin@example.com",
      role: "admin"
    }
  ];
}

async function signedInSession(sessions) {
  return sessions.rotateAuthenticatedSession(await sessions.createAnonymousSession(), {
    username: "admin",
    accessEmail: "admin@example.com",
    role: "admin"
  });
}

function sessionCookieValue(response) {
  const match = (response.headers.get("set-cookie") || "").match(/__Host-tp_session=([^;]*)/);
  return match ? match[1] : null;
}

test("the validator accepts exactly the canonical https origins in the shared vectors", () => {
  assert.equal(VECTORS.schema, "tp.frontdoor.public_origin_vectors.v1");
  assert.ok(VECTORS.accepted.length > 0 && VECTORS.rejected.length > 0 && VECTORS.trimmed.length > 0);

  for (const value of VECTORS.accepted) {
    assert.equal(isCanonicalPublicOrigin(value), true, value);
    assert.deepEqual(
      resolvePublicOrigin({ NODE_ENV: "production", TP_FRONTDOOR_PUBLIC_ORIGIN: value }),
      { configured: true, origin: value, reason: null },
      value
    );
  }

  for (const { value, why } of VECTORS.rejected) {
    assert.equal(isCanonicalPublicOrigin(value), false, why);
    assert.deepEqual(
      resolvePublicOrigin({ NODE_ENV: "production", TP_FRONTDOOR_PUBLIC_ORIGIN: value }),
      { configured: true, origin: "", reason: PUBLIC_ORIGIN_INVALID_REASON },
      why
    );
  }

  for (const { value, origin } of VECTORS.trimmed) {
    assert.deepEqual(
      resolvePublicOrigin({ NODE_ENV: "production", TP_FRONTDOOR_PUBLIC_ORIGIN: value }),
      { configured: true, origin, reason: null },
      JSON.stringify(value)
    );
  }
});

test("the validator bounds host length and rejects non-string input", () => {
  const label = "a".repeat(63);
  const longestHost = `${label}.${label}.${label}.${"a".repeat(57)}.com`;
  assert.equal(longestHost.length, 253);
  assert.equal(isCanonicalPublicOrigin(`https://${longestHost}`), true);
  assert.equal(isCanonicalPublicOrigin(`https://${label}.${label}.${label}.${"a".repeat(58)}.com`), false);

  for (const value of [undefined, null, 443, {}, ["https://portal.example.com"]]) {
    assert.equal(isCanonicalPublicOrigin(value), false);
  }
});

test("the resolver reports unset, malformed and bypass-conflicting values without throwing", () => {
  for (const env of [{}, { TP_FRONTDOOR_PUBLIC_ORIGIN: "" }, { TP_FRONTDOOR_PUBLIC_ORIGIN: " \t\r\n" }]) {
    assert.deepEqual(resolvePublicOrigin(env), { configured: false, origin: "", reason: null });
  }

  // The bypass is effective only in development, so only then does it conflict.
  assert.deepEqual(
    resolvePublicOrigin({
      NODE_ENV: "development",
      TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
      TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN
    }),
    { configured: true, origin: "", reason: PUBLIC_ORIGIN_BYPASS_CONFLICT_REASON }
  );
  assert.deepEqual(
    resolvePublicOrigin({ TP_ALLOW_LOCAL_ACCESS_BYPASS: "1", TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN }),
    { configured: true, origin: "", reason: PUBLIC_ORIGIN_BYPASS_CONFLICT_REASON }
  );
  assert.deepEqual(
    resolvePublicOrigin({
      NODE_ENV: "production",
      TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
      TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN
    }),
    { configured: true, origin: PUBLIC_ORIGIN, reason: null }
  );

  // A malformed value is reported as malformed even when the bypass is on.
  assert.equal(
    resolvePublicOrigin({
      NODE_ENV: "development",
      TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
      TP_FRONTDOOR_PUBLIC_ORIGIN: "portal.example.com"
    }).reason,
    PUBLIC_ORIGIN_INVALID_REASON
  );
});

test("with a public origin, the resolved origin ignores the runtime URL and every host header", () => {
  const env = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN });

  try {
    for (const shape of RUNTIME_SHAPES) {
      for (const headerSet of HEADER_SETS) {
        const label = `${shape.name} / ${headerSet.name}`;
        const request = runtimeRequest(shape.base, "/login", { headers: headerSet.headers });

        assert.equal(buildRequestUrl(request, "/login?error=csrf").href, `${PUBLIC_ORIGIN}/login?error=csrf`, label);

        for (const verdict of originVerdicts(shape.base)) {
          const unsafe = runtimeRequest(shape.base, "/logout", {
            method: "POST",
            headers: { ...headerSet.headers, ...verdict.headers }
          });
          assert.equal(validateOriginAndReferrer(unsafe), verdict.accepted, `${label} / ${verdict.name}`);
        }

        // Safe methods are never subject to the same-origin check.
        assert.equal(validateOriginAndReferrer(request), true, label);
      }
    }
  } finally {
    env.restore();
  }
});

test("without a public origin, the equivalence rule is unchanged", () => {
  const env = withEnvironment();

  try {
    // The shape the Worker produces in a self-hosted runtime. The forwarded
    // public host does not match the runtime URL, so it is ignored: this is
    // the behaviour a deployment behind a proxy needs the setting for.
    const proxied = runtimeRequest("https://localhost:3000", "/login", { headers: WORKER_HEADERS });
    assert.equal(buildRequestUrl(proxied, "/login").href, "https://localhost:3000/login");
    assert.equal(
      validateOriginAndReferrer(
        runtimeRequest("https://localhost:3000", "/login", {
          method: "POST",
          headers: { ...WORKER_HEADERS, origin: PUBLIC_ORIGIN }
        })
      ),
      false
    );

    // A platform that presents the public URL natively keeps working.
    const native = runtimeRequest(PUBLIC_ORIGIN, "/login", {
      method: "POST",
      headers: { host: PUBLIC_HOST, origin: PUBLIC_ORIGIN }
    });
    assert.equal(buildRequestUrl(native, "/portal").href, `${PUBLIC_ORIGIN}/portal`);
    assert.equal(validateOriginAndReferrer(native), true);

    // Loopback aliases remain equivalent, and hostile overrides are ignored.
    const loopback = runtimeRequest("http://localhost:3000", "/login", {
      method: "POST",
      headers: { host: "127.0.0.1:3000", origin: "http://127.0.0.1:3000" }
    });
    assert.equal(buildRequestUrl(loopback, "/portal").href, "http://127.0.0.1:3000/portal");
    assert.equal(validateOriginAndReferrer(loopback), true);

    const hostile = runtimeRequest(PUBLIC_ORIGIN, "/login", {
      headers: { host: "evil.example", "x-forwarded-host": "evil.example", "x-forwarded-proto": "http" }
    });
    assert.equal(buildRequestUrl(hostile, "/portal").href, `${PUBLIC_ORIGIN}/portal`);
  } finally {
    env.restore();
  }
});

test("a public origin keeps the request path, query and fragment and nothing else", () => {
  const env = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN });

  try {
    const request = runtimeRequest("https://localhost:3000", "/login?from=request", { headers: WORKER_HEADERS });

    assert.equal(buildRequestUrl(request, "/portal?view=build#top").href, `${PUBLIC_ORIGIN}/portal?view=build#top`);

    // The listener's port must not leak into the public URL.
    const resolved = buildRequestUrl(request, "/login");
    assert.equal(resolved.host, PUBLIC_HOST);
    assert.equal(resolved.port, "");

    // A path can never become a new authority, however it is spelled.
    for (const hostilePath of [
      "//attacker.example/capture",
      "///attacker.example/capture",
      "/.//attacker.example/capture",
      "/\\attacker.example/capture",
      "https://attacker.example/capture",
      "//attacker.example\\@portal.example.com/capture"
    ]) {
      const target = buildRequestUrl(request, hostilePath);
      assert.equal(target.origin, PUBLIC_ORIGIN, hostilePath);
      assert.equal(target.username, "", hostilePath);
      assert.equal(new URL(target.href).origin, PUBLIC_ORIGIN, hostilePath);
    }

    // Callers mutate the returned URL, so each call must return a fresh one.
    const first = buildRequestUrl(request, "/login");
    first.searchParams.set("error", "csrf");
    assert.equal(buildRequestUrl(request, "/login").href, `${PUBLIC_ORIGIN}/login`);
  } finally {
    env.restore();
  }

  const withPort = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: `${PUBLIC_ORIGIN}:8443` });
  try {
    const request = runtimeRequest("http://0.0.0.0:3000", "/login", { headers: WORKER_HEADERS });
    assert.equal(buildRequestUrl(request, "/login").href, `${PUBLIC_ORIGIN}:8443/login`);
  } finally {
    withPort.restore();
  }
});

test("a malformed value or one set with the local Access bypass is not honoured", () => {
  const malformed = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: `${PUBLIC_ORIGIN}/` });
  try {
    const request = runtimeRequest("https://localhost:3000", "/login", {
      method: "POST",
      headers: { ...WORKER_HEADERS, origin: PUBLIC_ORIGIN }
    });
    assert.equal(buildRequestUrl(request, "/login").href, "https://localhost:3000/login");
    assert.equal(validateOriginAndReferrer(request), false);
  } finally {
    malformed.restore();
  }

  const bypass = withEnvironment({
    NODE_ENV: "development",
    TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN
  });
  try {
    // A bypass-mode front door must stay unusable from a public origin.
    const proxied = runtimeRequest("https://localhost:3000", "/login", {
      method: "POST",
      headers: { ...WORKER_HEADERS, origin: PUBLIC_ORIGIN }
    });
    assert.equal(buildRequestUrl(proxied, "/login").href, "https://localhost:3000/login");
    assert.equal(validateOriginAndReferrer(proxied), false);

    // The loopback allowance for header-less local tools is unchanged, and it
    // does not extend to a request that arrives with any other host.
    const localTool = runtimeRequest("http://localhost:3000", "/logout", {
      method: "POST",
      headers: { host: "localhost:3000" }
    });
    assert.equal(validateOriginAndReferrer(localTool), true);
  } finally {
    bypass.restore();
  }
});

test("login through the edge proxy stays on the public origin and signs in", async () => {
  const users = await operatorUsers();

  for (const scenario of [
    { publicOrigin: PUBLIC_ORIGIN, location: `${PUBLIC_ORIGIN}/portal`, signedIn: true },
    { publicOrigin: undefined, location: "https://localhost:3000/login?error=csrf", signedIn: false }
  ]) {
    const env = withEnvironment({ users, TP_FRONTDOOR_PUBLIC_ORIGIN: scenario.publicOrigin });
    const restoreFetch = withFetch();

    try {
      const sessions = await importFresh("../lib/sessions.js");
      const { POST } = await importFresh("../app/login/route.js");
      const anonymous = await sessions.createAnonymousSession();

      const response = await POST(
        runtimeRequest("https://localhost:3000", "/login", {
          method: "POST",
          headers: {
            ...WORKER_HEADERS,
            origin: PUBLIC_ORIGIN,
            "content-type": "application/x-www-form-urlencoded",
            cookie: `__Host-tp_session=${anonymous.id}`,
            "Cf-Access-Jwt-Assertion": createAccessJwt()
          },
          body: new URLSearchParams({
            username: "admin",
            password: OPERATOR_PASSWORD,
            csrf_token: anonymous.csrfToken
          })
        })
      );

      assert.equal(response.status, 303);
      assert.equal(response.headers.get("location"), scenario.location);

      const cookie = sessionCookieValue(response);
      const session = cookie ? await sessions.getSessionById(cookie, { touch: false }) : null;
      assert.equal(Boolean(session?.authenticated), scenario.signedIn);
    } finally {
      restoreFetch();
      env.restore();
    }
  }
});

test("portal and logout redirects use the public origin", async () => {
  const env = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN });
  const restoreFetch = withFetch();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const portal = await importFresh("../app/portal/route.js");
    const logout = await importFresh("../app/logout/route.js");

    const unauthenticated = await portal.GET(
      runtimeRequest("https://localhost:3000", "/portal?view=build", { headers: WORKER_HEADERS })
    );
    assert.equal(unauthenticated.status, 302);
    const loginUrl = new URL(unauthenticated.headers.get("location"));
    assert.equal(loginUrl.origin, PUBLIC_ORIGIN);
    assert.equal(loginUrl.pathname, "/login");
    assert.equal(loginUrl.searchParams.get("returnTo"), "/portal?view=build");

    const session = await signedInSession(sessions);
    const signedOut = await logout.POST(
      runtimeRequest("https://localhost:3000", "/logout", {
        method: "POST",
        headers: {
          ...WORKER_HEADERS,
          origin: PUBLIC_ORIGIN,
          cookie: `__Host-tp_session=${session.id}`,
          "x-csrf-token": session.csrfToken
        }
      })
    );
    assert.equal(signedOut.status, 303);
    assert.equal(signedOut.headers.get("location"), `${PUBLIC_ORIGIN}/login`);
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);

    // A cross-site logout is refused, is sent to the public login page and
    // leaves the server-side session alive.
    const victim = await signedInSession(sessions);
    const forged = await logout.POST(
      runtimeRequest("https://localhost:3000", "/logout", {
        method: "POST",
        headers: {
          ...WORKER_HEADERS,
          origin: "https://evil.example",
          cookie: `__Host-tp_session=${victim.id}`,
          "x-csrf-token": victim.csrfToken
        }
      })
    );
    assert.equal(forged.status, 303);
    assert.equal(forged.headers.get("location"), `${PUBLIC_ORIGIN}/login?error=csrf`);
    assert.notEqual(await sessions.getSessionById(victim.id, { touch: false }), null);
  } finally {
    restoreFetch();
    env.restore();
  }
});

test("a /v1 mutation through the edge proxy passes the origin check and still forwards the internal host", async () => {
  const env = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN });
  const backendCalls = [];
  const restoreFetch = withFetch(async (input, init) => {
    backendCalls.push({ url: String(input), headers: new Headers(init?.headers) });
    return Response.json({ ok: true }, { status: 200 });
  });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const route = await importFresh("../app/v1/[...path]/route.js");
    const session = await signedInSession(sessions);

    const mutation = (origin) =>
      route.POST(
        runtimeRequest("https://localhost:3000", "/v1/jobs", {
          method: "POST",
          headers: {
            ...WORKER_HEADERS,
            origin,
            "content-type": "application/json",
            cookie: `__Host-tp_session=${session.id}`,
            "x-csrf-token": session.csrfToken,
            "Cf-Access-Jwt-Assertion": createAccessJwt()
          },
          body: JSON.stringify({ pipeline: "lux-depth-v3" })
        }),
        { params: { path: ["jobs"] } }
      );

    const refused = await mutation("https://evil.example");
    assert.equal(refused.status, 403);
    assert.equal((await refused.json()).error.code, "INVALID_CSRF");
    assert.equal(backendCalls.length, 0);

    const accepted = await mutation(PUBLIC_ORIGIN);
    assert.equal(accepted.status, 200);
    assert.equal(backendCalls.length, 1);
    assert.equal(backendCalls[0].url, `${FASTAPI_ORIGIN}/v1/jobs`);
    // What FastAPI is told about the hop is deliberately unchanged: it still
    // receives the host this front door was addressed as, not the public one.
    assert.equal(backendCalls[0].headers.get("x-forwarded-host"), INTERNAL_HOST);
  } finally {
    restoreFetch();
    env.restore();
  }
});

// Turn the Worker's upstream request into what a self-hosted front door sees:
// the Host of the configured FRONTDOOR_ORIGIN, the Worker's headers untouched,
// and a request URL built from the listener's own host and port with the
// scheme taken from x-forwarded-proto.
function asSelfHostedRequest(upstream) {
  const target = new URL(upstream.url);
  const headers = new Headers(upstream.headers);
  headers.set("host", target.host);
  const scheme = (headers.get("x-forwarded-proto") || "").includes("https") ? "https" : "http";
  return new NextRequest(`${scheme}://localhost:3000${target.pathname}${target.search}`, {
    method: upstream.method,
    headers
  });
}

test("the real Worker in front of the front door keeps the browser on the public origin", async (t) => {
  const env = withEnvironment({ TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN });

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const portal = await importFresh("../app/portal/route.js");
    const logout = await importFresh("../app/logout/route.js");
    const upstreamRequests = [];

    t.mock.method(globalThis, "fetch", async (input, init) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const url = new URL(request.url);
      if (request.url === TEST_CF_ACCESS_CERTS_URL) return accessCertsResponse();
      if (url.origin !== INTERNAL_ORIGIN) throw new Error(`Unexpected fetch to ${request.url}`);

      upstreamRequests.push(request);
      const frontdoorRequest = asSelfHostedRequest(request);
      if (request.method === "GET" && url.pathname === "/portal") return portal.GET(frontdoorRequest);
      if (request.method === "POST" && url.pathname === "/logout") return logout.POST(frontdoorRequest);
      throw new Error(`Unexpected front-door request ${request.method} ${url.pathname}`);
    });

    const throughWorker = (pathname, init) =>
      worker.fetch(new Request(`${PUBLIC_ORIGIN}${pathname}`, init), { FRONTDOOR_ORIGIN: INTERNAL_ORIGIN });
    const logoutThroughWorker = (session) =>
      throughWorker("/logout", {
        method: "POST",
        headers: {
          origin: PUBLIC_ORIGIN,
          cookie: `__Host-tp_session=${session.id}`,
          "x-csrf-token": session.csrfToken
        }
      });

    const redirected = await throughWorker("/portal");
    assert.equal(redirected.status, 302);
    assert.equal(redirected.headers.get("location"), `${PUBLIC_ORIGIN}/login?returnTo=%2Fportal`);

    const session = await signedInSession(sessions);
    const signedOut = await logoutThroughWorker(session);
    assert.equal(signedOut.status, 303);
    assert.equal(signedOut.headers.get("location"), `${PUBLIC_ORIGIN}/login`);
    assert.equal(await sessions.getSessionById(session.id, { touch: false }), null);

    // The Worker really did hide the public host from the Host header.
    assert.equal(upstreamRequests.length, 2);
    for (const upstream of upstreamRequests) {
      assert.equal(new URL(upstream.url).origin, INTERNAL_ORIGIN);
      assert.equal(upstream.headers.get("x-forwarded-host"), PUBLIC_HOST);
    }

    // Control: the same two requests without the setting leave the public
    // origin and fail the same-origin check.
    delete process.env.TP_FRONTDOOR_PUBLIC_ORIGIN;
    const stranded = await throughWorker("/portal");
    assert.equal(stranded.headers.get("location"), "https://localhost:3000/login?returnTo=%2Fportal");

    const stillSignedIn = await signedInSession(sessions);
    const refused = await logoutThroughWorker(stillSignedIn);
    assert.equal(refused.headers.get("location"), "https://localhost:3000/login?error=csrf");
    assert.notEqual(await sessions.getSessionById(stillSignedIn.id, { touch: false }), null);
  } finally {
    env.restore();
  }
});

test("healthz reports the public origin check without echoing the value", async () => {
  const users = [{ username: "admin", password_hash: "hash", access_email: "admin@example.com", role: "admin" }];
  const scenarios = [
    { name: "unset", env: {}, status: 200, check: { ok: true, required: true, configured: false, reason: null } },
    {
      name: "valid",
      env: { TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN },
      status: 200,
      check: { ok: true, required: true, configured: true, reason: null }
    },
    {
      name: "malformed",
      env: { TP_FRONTDOOR_PUBLIC_ORIGIN: `${PUBLIC_ORIGIN}/portal` },
      status: 503,
      check: { ok: false, required: true, configured: true, reason: PUBLIC_ORIGIN_INVALID_REASON }
    },
    {
      name: "set with the local Access bypass",
      env: { NODE_ENV: "development", TP_ALLOW_LOCAL_ACCESS_BYPASS: "1", TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN },
      status: 503,
      check: { ok: false, required: true, configured: true, reason: PUBLIC_ORIGIN_BYPASS_CONFLICT_REASON }
    }
  ];

  for (const scenario of scenarios) {
    const env = withEnvironment({ users, ...scenario.env });
    const restoreFetch = withFetch(async () => Response.json({ ok: true }, { status: 200 }));

    try {
      const { GET } = await importFresh("../app/healthz/route.js");
      const response = await GET();
      const text = await response.text();
      const body = JSON.parse(text);

      assert.equal(response.status, scenario.status, scenario.name);
      assert.deepEqual(body.checks.public_origin, scenario.check, scenario.name);
      assert.equal(body.ok, scenario.status === 200, scenario.name);
      assert.equal(text.includes(PUBLIC_HOST), false, scenario.name);
    } finally {
      restoreFetch();
      env.restore();
    }
  }
});

test("the startup preflight fails closed on a public origin that is not honoured", () => {
  const scriptPath = path.join(FRONTDOOR_APP_ROOT, "scripts", "preflight-backend-auth.mjs");
  const preflight = (overrides) =>
    spawnSync(process.execPath, [scriptPath], {
      cwd: FRONTDOOR_APP_ROOT,
      encoding: "utf-8",
      env: {
        ...process.env,
        NODE_ENV: "development",
        // Nothing listens here; in development an unreachable backend is a warning.
        TP_FASTAPI_ORIGIN: "http://127.0.0.1:9",
        TP_BACKEND_API_KEY: "secret",
        TP_PILOT_CONTROL_PLANE_ENABLED: "0",
        TP_FRONTDOOR_PREFLIGHT_DISABLE: "0",
        TP_FRONTDOOR_USERS_FILE: "",
        TP_FRONTDOOR_USERS_JSON: JSON.stringify([
          { username: "admin", password_hash: "hash", access_email: "admin@example.com" }
        ]),
        TP_ALLOW_LOCAL_ACCESS_BYPASS: "0",
        TP_FRONTDOOR_PUBLIC_ORIGIN: "",
        ...overrides
      }
    });

  const malformed = preflight({ TP_FRONTDOOR_PUBLIC_ORIGIN: "portal.example.com" });
  assert.notEqual(malformed.status, 0);
  assert.match(malformed.stderr, /TP_FRONTDOOR_PUBLIC_ORIGIN is not a canonical https origin/);

  const conflict = preflight({ TP_ALLOW_LOCAL_ACCESS_BYPASS: "1", TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN });
  assert.notEqual(conflict.status, 0);
  assert.match(conflict.stderr, /TP_FRONTDOOR_PUBLIC_ORIGIN is set while the local Access bypass is active/);

  for (const accepted of [{}, { TP_FRONTDOOR_PUBLIC_ORIGIN: PUBLIC_ORIGIN }]) {
    const result = preflight(accepted);
    assert.equal(result.status, 0, result.stderr);
    assert.doesNotMatch(result.stderr, /TP_FRONTDOOR_PUBLIC_ORIGIN/);
  }
});
