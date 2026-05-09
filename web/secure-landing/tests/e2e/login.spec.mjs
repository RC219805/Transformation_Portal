// @frontdoor-browser login-page smoke.
//
// Verifies the managed sign-in form renders with the stable data-ui
// hooks the source-string tests in routes.test.mjs and rum-client.test.mjs
// pin at the source level. The browser layer catches hooks that
// disappear under real DOM rendering even when the rendered HTML
// string still contains them (e.g., a parent block whose CSS
// `display: none` would hide the form).

import { test, expect } from "@playwright/test";

test.describe(
  "login page",
  { tag: "@frontdoor-browser" },
  () => {
    test("renders the managed login form with stable data-ui hooks", async ({ page }) => {
      const consoleErrors = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(msg.text());
      });
      page.on("pageerror", (err) => consoleErrors.push(String(err)));

      const response = await page.goto("/login");
      expect(response?.status()).toBe(200);

      // Stable login-template hooks (app/login/route.js renderLoginPage).
      await expect(page.locator('[data-ui="login-shell"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-card"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-title"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-form"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-username-field"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-password-field"]')).toBeVisible();
      await expect(page.locator('[data-ui="login-submit"]')).toBeVisible();

      // Form submits POST to the managed entry route — pinning the
      // contract end-to-end protects against accidental re-targeting.
      const form = page.locator('[data-ui="login-form"]');
      await expect(form).toHaveAttribute("method", "post");
      await expect(form).toHaveAttribute("action", "/login");

      // Required fields drive native HTML5 validation, which is
      // important for the client-side login_submit_attempt RUM contract
      // shipped in #1689 (the submit listener only fires after native
      // validation passes).
      await expect(
        page.locator('[data-ui="login-username-field"] input[name="username"]')
      ).toHaveAttribute("required", "");
      await expect(
        page.locator('[data-ui="login-password-field"] input[name="password"]')
      ).toHaveAttribute("required", "");

      expect(
        consoleErrors,
        `unexpected console errors: ${consoleErrors.join("\n")}`
      ).toEqual([]);
    });
  }
);
