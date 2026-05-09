// @portal-browser portal-auth/session smoke.
//
// Verifies the authenticated session captured by the portal-setup
// project survives across reloads and deep-link navigations, and
// that an unauthenticated context redirects to /login (the negative
// path covered by portal-shell.spec.mjs's @frontdoor-browser tag is
// inverted here against an authenticated context).

import { test, expect } from "@playwright/test";

test.describe(
  "portal auth boundary (authenticated)",
  { tag: "@portal-browser" },
  () => {
    test("session survives reload and deep-link navigation", async ({ page }) => {
      await page.goto("/portal");
      expect(page.url()).toMatch(/\/portal(\?|$)/);
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();

      // Reload — session cookie must keep the user on /portal, not
      // redirect to /login.
      await page.reload();
      expect(page.url()).toMatch(/\/portal(\?|$)/);
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();

      // Deep link to a non-default view stays inside /portal.
      const response = await page.goto("/portal?view=review");
      expect(response?.status()).toBe(200);
      expect(page.url()).toContain("/portal");
      expect(page.url()).not.toContain("/login");
      await expect(page.locator('[data-ui="view-switcher"]')).toBeVisible();
    });

    test("session is scoped — clearing cookies forces a redirect to /login", async ({ page, context }) => {
      // Pre-condition: authenticated visit succeeds.
      await page.goto("/portal");
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();

      // Drop the session cookie. The next /portal hit must 302 to /login.
      await context.clearCookies();

      const response = await page.goto("/portal");
      expect(response?.status()).toBe(200);
      expect(page.url()).toContain("/login");
      // Login form is rendered — same anchor #1690's @frontdoor-browser
      // suite pins.
      await expect(page.locator('[data-ui="login-form"]')).toBeVisible();
    });
  }
);
