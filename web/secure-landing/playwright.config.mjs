// Playwright configuration for the front-door + portal browser smoke
// suites.
//
// Scope:
//   - Chromium only, headless only.
//   - @frontdoor-browser specs covering /, /login, and the
//     unauthenticated /portal redirect to /login (#1690).
//   - @portal-browser specs covering the signed-in /portal shell
//     served via the front-door upstream proxy (this config). The
//     proxy fetches from a Node mock FastAPI origin started by the
//     second webServer entry below — no Python in CI.
//   - Supplemental coverage that runs alongside the existing node:test
//     unit suite. The CDP-based scripts/validation/validate_portal_browser_smoke.py
//     and validate_frontdoor_browser_smoke.py remain the governed Make
//     lanes; this Playwright config does NOT replace them.
//
// The two webServer entries boot:
//   1. The Node mock FastAPI origin on MOCK_FASTAPI_PORT (default 9999).
//      Serves portal.html with placeholder URLs substituted to inert
//      mock-asset stubs.
//   2. The front-door Next.js dev server on PLAYWRIGHT_BASE_URL's port.
//      Configured via TP_FASTAPI_ORIGIN to proxy to the mock above.
//
// Authentication for @portal-browser is handled by the portal-setup
// project, which performs a real form-POST login against the seeded
// smoke-admin user (the same credentials already pre-baked here for
// #1690). The login flow uses production code paths verbatim — no
// test-only auth shortcut.

import { defineConfig, devices } from "@playwright/test";

const isCi = Boolean(process.env.CI);
const baseURL = process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:3000";
const webServerPort = getWebServerPort(baseURL);
const mockFastapiHost = process.env.MOCK_FASTAPI_HOST || "127.0.0.1";
const mockFastapiPort = process.env.MOCK_FASTAPI_PORT || "9999";
const mockFastapiOrigin = `http://${mockFastapiHost}:${mockFastapiPort}`;
const frontdoorSmokeUsersJson = JSON.stringify([
  {
    username: "smoke-admin",
    password_hash:
      "$argon2id$v=19$m=65536,t=3,p=4$PGOdFy1HPgK0hvRb9JmVfQ$/9WdFZwFgUs0IY0Pv0EKNixg/LB3oQH5w9y9lLwqYZQ",
    access_email: "smoke-admin@local.invalid",
    role: "admin",
  },
]);
const portalAuthStateFile = "tests/e2e/.auth/portal-state.json";

function getWebServerPort(playwrightBaseURL) {
  const url = new URL(playwrightBaseURL);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Unsupported PLAYWRIGHT_BASE_URL protocol: ${url.protocol}`);
  }
  if (!url.port) {
    throw new Error("PLAYWRIGHT_BASE_URL must include an explicit port for frontdoor browser smoke tests.");
  }
  return url.port;
}

export default defineConfig({
  testDir: "tests/e2e",
  testMatch: "**/*.spec.mjs",
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: !isCi,
  retries: isCi ? 2 : 0,
  workers: isCi ? 1 : undefined,
  reporter: isCi
    ? [
        ["list"],
        ["html", { open: "never", outputFolder: "playwright-report" }],
      ]
    : [["list"]],
  use: {
    baseURL,
    headless: true,
    trace: "on-first-retry",
    video: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      // Default project: @frontdoor-browser specs (landing, login,
      // unauthenticated portal redirect). Excludes the portal-setup
      // bootstrap and the .portal.spec.mjs files which require an
      // authenticated context.
      testIgnore: [
        "**/portal-setup.spec.mjs",
        "**/*.portal.spec.mjs",
      ],
      use: { ...devices["Desktop Chrome"] },
    },
    {
      // Setup project: performs the real form-POST login against the
      // pre-seeded smoke-admin user and persists storageState for the
      // portal-browser project. Runs once per session.
      name: "portal-setup",
      testMatch: /portal-setup\.spec\.mjs/,
      use: { ...devices["Desktop Chrome"] },
    },
    {
      // @portal-browser specs run against the authenticated context
      // captured by portal-setup. Project dependency makes Playwright
      // run portal-setup first and skip portal-browser if it fails.
      name: "portal-browser",
      testMatch: /\.portal\.spec\.mjs$/,
      dependencies: ["portal-setup"],
      use: {
        ...devices["Desktop Chrome"],
        storageState: portalAuthStateFile,
      },
    },
  ],
  webServer: [
    {
      // Node mock FastAPI origin — serves portal.html (with placeholders
      // substituted) for the front-door's upstream proxy. No Python in CI.
      command: `node tests/e2e/fixtures/mock-fastapi-origin.mjs`,
      url: `${mockFastapiOrigin}/healthz`,
      reuseExistingServer: !isCi,
      timeout: 30_000,
      env: {
        MOCK_FASTAPI_HOST: mockFastapiHost,
        MOCK_FASTAPI_PORT: mockFastapiPort,
      },
    },
    {
      // Next.js front-door dev server. Proxies /portal upstream to the
      // mock FastAPI origin above. Auth bypass for Cloudflare Access is
      // enabled via TP_ALLOW_LOCAL_ACCESS_BYPASS=1 +
      // NODE_ENV=development (lib/config.js:10-12).
      command: `npm run dev -- --port ${webServerPort}`,
      url: baseURL,
      reuseExistingServer: !isCi,
      timeout: 120_000,
      env: {
        NODE_ENV: "development",
        TP_FASTAPI_ORIGIN: mockFastapiOrigin,
        TP_BACKEND_API_KEY: "frontdoor-browser-smoke",
        TP_FRONTDOOR_USERS_JSON: frontdoorSmokeUsersJson,
        TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
        WATCHPACK_POLLING: "true",
      },
    },
  ],
});
