// @frontdoor-browser landing-page smoke.
//
// Asserts the public DNA landing page renders end-to-end in Chromium
// without console errors and exposes the stable data-ui hooks that
// the source-string tests in tests/routes.test.mjs already pin at
// the rendered-HTML layer. Catches regressions where the hooks
// disappear under real DOM rendering even when the source still
// includes them.

import { test, expect } from "@playwright/test";

test.describe(
  "landing page",
  { tag: "@frontdoor-browser" },
  () => {
    test("renders the public DNA landing without console errors", async ({ page }) => {
      const consoleErrors = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(msg.text());
      });
      page.on("pageerror", (err) => consoleErrors.push(String(err)));

      const response = await page.goto("/");
      expect(response?.status()).toBe(200);

      // Stable hooks pinned by lib/homepage.js renderHomepage().
      await expect(page.locator('[data-ui="homepage-shell"]')).toBeVisible();
      await expect(page.locator('[data-ui="homepage-hero-title"]')).toBeVisible();
      await expect(page.locator('[data-ui="homepage-final-primary-cta"]')).toBeVisible();

      // Final CTA points operators at the managed login surface.
      await expect(
        page.locator('[data-ui="homepage-final-primary-cta"]')
      ).toHaveAttribute("href", "/login");

      expect(
        consoleErrors,
        `unexpected console errors: ${consoleErrors.join("\n")}`
      ).toEqual([]);
    });
  }
);
