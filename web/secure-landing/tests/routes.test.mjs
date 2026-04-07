import test from "node:test";
import assert from "node:assert/strict";
import { createSign, generateKeyPairSync } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";

import argon2 from "argon2";
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
  "TP_ALLOW_LOCAL_ACCESS_BYPASS"
];

const TEST_CF_ACCESS_TEAM_DOMAIN = "https://tp-frontdoor-tests.cloudflareaccess.com";
const TEST_CF_ACCESS_AUD = "tp-frontdoor-aud";
const TEST_CF_ACCESS_KID = "tp-frontdoor-key";
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
  const dbPath = path.join(tempDir, "sessions.sqlite");
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
      assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false }), null);

      const authenticatedSession = sessions.getSessionById(rotatedCookie.value, { touch: false });
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
    assert.match(html, /\/brand\/dna-mark-dark\.svg/);
    assert.match(html, /Transformation Portal operator console/);
    assert.match(html, /form method="post" action="\/login"/);
    assert.match(html, /name="username"/);
    assert.match(html, /name="password"/);
    assert.match(html, />Sign in</);
    assert.doesNotMatch(html, /Authorized operators only\./);
    assert.doesNotMatch(html, /Need access\?/);
    assert.doesNotMatch(html, /Secure operator access to governed orchestration\./);
    assert.doesNotMatch(html, /Local development bypass is enabled\./);
    assert.doesNotMatch(html, /TP_CF_ACCESS_TEAM_DOMAIN/);
    assert.ok(sessionCookie.value);
    assert.equal(sessions.getSessionById(sessionCookie.value, { touch: false })?.authenticated, false);
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

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.match(response.headers.get("content-security-policy") || "", /default-src 'self'/);
    assert.match(response.headers.get("content-security-policy") || "", /script-src 'none'/);
    assert.equal(response.headers.get("set-cookie"), null);
    assert.equal(sessionCountBefore, 0);
    assert.equal(sessionCountAfter, 0);
    assert.match(html, /\/video\/dna-loop\.mp4/);
    assert.match(html, /\/brand\/dna-mark-dark\.svg/);
    assert.match(html, /Make premium media verifiable before it ships\./);
    assert.match(html, /Start Certification/);
    assert.match(html, /View Verification Report/);
    assert.match(html, /Verify\. Enhance\. Enforce\. Distribute\./);
    assert.match(html, /Operator Login/);
    assert.match(html, />Secure Access</);
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

test("homepage GET keeps authenticated operators on the public landing page while surfacing console entry", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const db = getDb(env.dbPath);
    const { GET } = await importFresh("../app/route.js");
    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
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
    assert.match(html, />Open Console</);
    assert.match(html, /href="\/login"[^>]*>Start Certification</);
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

test("homepage GET may prune expired sessions without setting cookies", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/route.js");
    const expiredSession = sessions.createAnonymousSession();
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

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("set-cookie"), null);
    assert.equal(sessions.getSessionById(expiredSession.id, { touch: false }), null);
    assert.match(html, />Secure Access</);
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
      const anonymousSession = sessions.createAnonymousSession();
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
      assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false })?.authenticated, false);
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
      const anonymousSession = sessions.createAnonymousSession();
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
      assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false })?.authenticated, false);
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

    const anonymousSession = sessions.createAnonymousSession();
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
    assert.equal(sessions.getSessionById(anonymousSession.id, { touch: false })?.authenticated, false);
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
      const anonymousSession = sessions.createAnonymousSession();
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
      const anonymousSession = sessions.createAnonymousSession();
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

    const anonymousSession = sessions.createAnonymousSession();
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

    const anonymousSession = sessions.createAnonymousSession();
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
    const restoreFetch = withMockedAccessCerts();

    try {
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
      assert.equal(body.csrfToken, authenticatedSession.csrfToken);
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

    assert.equal(response.status, 401);
    assert.equal(body.error, "authentication required");
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
    assert.equal(sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
  } finally {
    env.cleanup();
  }
});

test("portal returns 503 with no-store when the FastAPI UI origin is unavailable", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/route.js");
    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
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
      const request = buildRequest("https://portal.example.com/portal", {
        method: "GET",
        headers: new Headers({
          cookie: `__Host-tp_session=${authenticatedSession.id}`,
          "Cf-Access-Jwt-Assertion": createAccessJwt()
        })
      });

      const response = await GET(request);

      assert.equal(response.status, 503);
      assert.equal(response.headers.get("cache-control"), "no-store");
      assert.equal(await response.text(), "Upstream service unavailable");
    } finally {
      restoreFetch();
    }
  } finally {
    env.cleanup();
  }
});

test("portal redirects to login and clears the session when Access verification is missing", async () => {
  const env = withTempEnvironment();

  try {
    const sessions = await importFresh("../lib/sessions.js");
    const { GET } = await importFresh("../app/portal/route.js");
    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
      {
        username: "admin",
        accessEmail: "admin@example.com",
        role: "admin"
      }
    );

    const request = buildRequest("https://portal.example.com/portal", {
      method: "GET",
      headers: new Headers({
        cookie: `__Host-tp_session=${authenticatedSession.id}`
      })
    });

    const response = await GET(request);

    assert.equal(response.status, 302);
    assert.equal(response.headers.get("location"), "https://portal.example.com/login");
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
    assert.equal(sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
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

    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
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
    assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
    assert.equal(sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
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
      const authenticatedSession = sessions.rotateAuthenticatedSession(
        sessions.createAnonymousSession(),
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
      assert.match(response.headers.get("set-cookie") || "", /__Host-tp_session=/);
      assert.equal(sessions.getSessionById(authenticatedSession.id, { touch: false }), null);
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

    const restoreFetch = withMockedAccessCerts(async (url, init) => {
      assert.equal(String(url), "http://127.0.0.1:8000/v1/jobs/job-123/events");
      assert.equal(init.method, "GET");
      assert.equal(init.headers.get("Authorization"), "Bearer backend-secret");
      assert.equal(init.headers.get("Forwarded"), 'for="203.0.113.10";host="portal.example.com";proto="https"');
      assert.equal(init.headers.get("x-api-key"), "backend-secret");
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
          "cache-control": "private"
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
          "Cf-Access-Jwt-Assertion": createAccessJwt()
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

    const authenticatedSession = sessions.rotateAuthenticatedSession(
      sessions.createAnonymousSession(),
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

test("healthz reports backend status without leaking the backend origin", async () => {
  const env = withTempEnvironment();

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
      assert.equal(body.backend.ok, true);
      assert.equal(body.backend.status, 200);
      assert.equal("origin" in body.backend, false);
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
    const anonymousSession = sessions.createAnonymousSession();

    sessions.setSessionCookie(response, anonymousSession.id);

    const cookieHeader = response.headers.get("set-cookie") || "";
    assert.match(cookieHeader, /^tp_session=/);
    assert.doesNotMatch(cookieHeader, /__Host-tp_session/);
    assert.doesNotMatch(cookieHeader, /Secure/i);
  } finally {
    env.cleanup();
  }
});
