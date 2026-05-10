// @portal-browser portal-logout smoke.
//
// Verifies the topbar logout button:
//   - is visible on the authenticated /portal shell (managed mode)
//   - POSTs to /logout via the portal bundle's existing fetch path
//   - lands the operator on /login with the session cookie cleared,
//     completing the same boundary that portal-auth.portal.spec.mjs
//     exercises from the cookie-clear side.
//
// Suite-coverage map: this is the only Playwright spec that drives a
// click on [data-ui="logout-button"]; the portal-shell, portal-views,
// and portal-auth specs assert presence of the topbar and view switcher
// but do not invoke logout themselves.

import { test, expect } from "@playwright/test";

test.describe(
  "portal logout (authenticated)",
  { tag: "@portal-browser" },
  () => {
    test("logout button posts to /logout and redirects to /login", async ({ page }) => {
      const consoleErrors = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(msg.text());
      });
      page.on("pageerror", (err) => consoleErrors.push(String(err)));

      // Pre-condition: authenticated /portal renders the topbar.
      await page.goto("/portal");
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();

      // The logout button is rendered with `hidden` and only un-hidden in
      // managed mode after bootstrap; this assertion catches the case
      // where bootstrap silently fails to flip the flag, which would
      // otherwise pass an HTML-only presence test.
      const logoutButton = page.locator('[data-ui="logout-button"]');
      await expect(logoutButton).toBeVisible();
      await expect(logoutButton).toHaveAccessibleName(
        /sign out of the operator console/i
      );

      // Watch for the POST /logout request so we can pin both the method
      // and the X-CSRF-Token header before the navigation completes.
      const logoutRequestPromise = page.waitForRequest((request) => {
        return request.method() === "POST" && request.url().endsWith("/logout");
      });

      await logoutButton.click();

      const logoutRequest = await logoutRequestPromise;
      const csrfHeader = await logoutRequest.headerValue("x-csrf-token");
      expect(csrfHeader, "POST /logout must carry a non-empty X-CSRF-Token").toBeTruthy();

      // Wait for the explicit window.location.assign('/login') to land
      // the operator on the login page.
      await page.waitForURL(/\/login(\?|$)/);
      expect(page.url()).toContain("/login");
      await expect(page.locator('[data-ui="login-form"]')).toBeVisible();

      expect(
        consoleErrors,
        `unexpected console errors: ${consoleErrors.join("\n")}`
      ).toEqual([]);
    });
  }
);
