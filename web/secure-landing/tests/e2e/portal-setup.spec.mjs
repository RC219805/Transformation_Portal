// Playwright "setup" project for the @portal-browser suite.
//
// Runs once before the @portal-browser specs and persists an
// authenticated browser context (storageState) for them to load via
// project dependencies. The login flow is the production form-POST at
// /login — no test-only auth path or session-mint shortcut.
//
// Auth fixture cost analysis:
//   - The smoke-admin user (username/password/access_email) is already
//     pre-seeded in playwright.config.mjs's TP_FRONTDOOR_USERS_JSON.
//   - TP_ALLOW_LOCAL_ACCESS_BYPASS=1 + NODE_ENV=development is already
//     set on the front-door webServer, so resolveAccessContext returns
//     {bypass: true} (lib/access.js:13-21, lib/config.js:10-12).
//   - The form login POST flow (app/login/route.js:316-444) calls
//     rotateAuthenticatedSession + setSessionCookie verbatim — same
//     code path the front-door uses in production.
// No new env vars, no new auth code, no new password hashes.

import { mkdirSync } from "node:fs";
import path from "node:path";

import { test as setup, expect } from "@playwright/test";

const STORAGE_STATE_PATH = "tests/e2e/.auth/portal-state.json";
const SMOKE_USERNAME = "smoke-admin";
const SMOKE_PASSWORD = "correct horse battery staple";

setup("authenticate the smoke-admin operator", async ({ page }) => {
  // The login form lives at /login. Form selectors mirror those pinned
  // by the @frontdoor-browser login spec (login.spec.mjs:46-50).
  await page.goto("/login");
  await expect(page.locator('[data-ui="login-form"]')).toBeVisible();

  await page
    .locator('[data-ui="login-username-field"] input[name="username"]')
    .fill(SMOKE_USERNAME);
  await page
    .locator('[data-ui="login-password-field"] input[name="password"]')
    .fill(SMOKE_PASSWORD);

  // Submit and wait for the 303 redirect chain to land at /portal.
  await Promise.all([
    page.waitForURL((url) => url.pathname === "/portal"),
    page.locator('[data-ui="login-submit"]').click()
  ]);

  // The portal shell is served upstream-proxied from the mock FastAPI
  // origin; assert one stable anchor before persisting state to catch
  // any auth-side regressions before the @portal-browser specs run.
  await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();

  // Ensure the parent directory exists before storageState() writes —
  // a clean checkout has no tests/e2e/.auth/ and node:fs writes do not
  // mkdir-p, so the persist would otherwise fail with ENOENT.
  mkdirSync(path.dirname(STORAGE_STATE_PATH), { recursive: true });
  await page.context().storageState({ path: STORAGE_STATE_PATH });
});
