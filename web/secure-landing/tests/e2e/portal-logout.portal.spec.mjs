// @portal-browser portal-logout structural smoke.
//
// Verifies that the logout button is server-rendered into the topbar
// of the live /portal HTML stream with all the structural contracts a
// managed-mode operator depends on: stable selector hook, accessible
// label, `hidden`-by-default discipline, placement inside the topbar
// actions row. The full click → POST /logout → /login redirect flow
// is intentionally NOT exercised here.
//
// Architectural split (preserved from PR #1692):
//   - The @portal-browser fixture (mock-fastapi-origin.mjs) substitutes
//     __PORTAL_JS_URL__ to /__mock-portal/portal.js, an inert stub. The
//     real portal bundle never executes in this lane, so the
//     bootstrap-ready code that un-hides the logout button never runs
//     and a click handler is never attached. Asserting visibility or
//     POST-fire from this suite would always fail.
//   - The end-to-end click flow is pinned by two complementary suites:
//     (1) tests/test_app_orchestrator_runtime.py::
//         test_portal_bundle_logout_handler_posts_to_logout_and_navigates
//         greps the minified bundle for the six behavioral invariants
//         (state.auth.logoutPending guard, fetchWithTimeout to /logout,
//         redirect:"manual", always-navigate to /login, exactly-one
//         click listener, managed-mode-only un-hide); and
//     (2) scripts/validation/validate_portal_browser_smoke.py is the
//         governed CDP suite that runs the real backend + real bundle
//         and is the natural home for any future live-click flow.
//
// Suite-coverage map: this is the only Playwright spec that asserts
// the logout button's server-rendered HTML structure. The portal-shell,
// portal-views, and portal-auth specs assert presence of the topbar
// and view switcher but do not assert the logout button explicitly.

import { test, expect } from "@playwright/test";

test.describe(
  "portal logout (server-rendered structure)",
  { tag: "@portal-browser" },
  () => {
    test("logout button is server-rendered into the topbar with the expected contract", async ({ page }) => {
      const consoleErrors = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(msg.text());
      });
      page.on("pageerror", (err) => consoleErrors.push(String(err)));

      await page.goto("/portal");
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeAttached();

      // The logout button lives in the live HTML response — proves the
      // FastAPI portal HTML template substituted the topbar markup
      // correctly under the @portal-browser fixture.
      const logoutButton = page.locator('[data-ui="logout-button"]');
      await expect(logoutButton).toBeAttached();

      // The button must default to ``hidden`` in the server-rendered
      // HTML so direct-debug operators never see a control that would
      // silently bounce them to /login. The portal bundle un-hides it
      // only in managed mode after bootstrap-ready; the stubbed bundle
      // in this fixture cannot un-hide it, so it must remain not
      // visible here.
      await expect(logoutButton).toBeHidden();
      await expect(logoutButton).toHaveAttribute("aria-label", /sign out of the operator console/i);
      await expect(logoutButton).toHaveAttribute("type", "button");

      // The button lives inside the portal-topbar__actions row, not in
      // any deferred surface; a managed-mode operator always sees it
      // without any further bundle load (when the real bundle runs).
      const topbarActions = page.locator('[data-ui="portal-topbar-actions"]');
      const logoutInsideActions = topbarActions.locator('[data-ui="logout-button"]');
      await expect(logoutInsideActions).toBeAttached();

      expect(
        consoleErrors,
        `unexpected console errors: ${consoleErrors.join("\n")}`
      ).toEqual([]);
    });
  }
);
