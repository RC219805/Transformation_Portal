// @frontdoor-browser portal auth-boundary smoke.
//
// Verifies that an unauthenticated visit to /portal redirects to /login
// and lands on a renderable login surface. Exercises the managed auth
// boundary end-to-end without exercising the signed-in portal flow
// itself — that's deferred to a follow-up PR with portal workflow
// fixtures and SSE/config-preview mocking.

import { test, expect } from "@playwright/test";

test.describe(
  "portal auth boundary",
  { tag: "@frontdoor-browser" },
  () => {
    test("redirects unauthenticated visitors from /portal to /login", async ({ page }) => {
      // Default Playwright follows redirects; the response is the final
      // page after the 302 from /portal.
      const response = await page.goto("/portal");
      expect(response?.status()).toBe(200);

      // Final URL is the login surface, not the portal shell.
      expect(page.url()).toContain("/login");

      // The login form is visible — the redirect target is fully
      // renderable, not a bare redirect to a broken page.
      await expect(page.locator('[data-ui="login-form"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-submit"]')).toBeVisible();
    });
  }
);
