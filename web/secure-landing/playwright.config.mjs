// Playwright configuration for the front-door browser smoke suite.
//
// Scope (Phase-1 closeout):
//   - Chromium only.
//   - Three @frontdoor-browser specs covering /, /login, and the
//     unauthenticated /portal redirect to /login.
//   - Supplemental coverage that runs alongside the existing node:test
//     unit suite. The CDP-based scripts/validation/validate_portal_browser_smoke.py
//     and validate_frontdoor_browser_smoke.py remain the governed Make
//     lanes; this Playwright config does NOT replace them.
//
// The webServer block boots the Next.js dev server with a minimal env so
// the smoke tests can render pages without depending on a live FastAPI
// origin.

import { defineConfig, devices } from "@playwright/test";

const isCi = Boolean(process.env.CI);
const baseURL = process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:3000";

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
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev -- --port 3000",
    url: baseURL,
    reuseExistingServer: !isCi,
    timeout: 120_000,
    env: {
      NODE_ENV: "production",
      // Stub backend so the dev server boots; smoke tests do not call
      // protected /v1/* routes that would require a real upstream.
      TP_FASTAPI_ORIGIN: "http://127.0.0.1:9999",
      TP_BACKEND_API_KEY: "frontdoor-browser-smoke",
      TP_FRONTDOOR_USERS_JSON: "[]",
      TP_ALLOW_LOCAL_ACCESS_BYPASS: "1",
    },
  },
});
