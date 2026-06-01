// @portal-browser portal-shell smoke.
//
// Verifies the signed-in /portal shell renders with the stable
// data-ui hooks and view-switcher links the source-string tests
// already pin (e.g., portal-css-architecture.test.mjs). This spec
// runs through the full request chain:
//
//   browser → front-door /portal route → fetch upstream
//           → mock FastAPI origin /  (serves portal.html with the
//             __PORTAL_*_URL__ placeholders substituted)
//
// The mock origin keeps the live portal.html template as the source
// of truth, so accidental removal of a stable anchor in the real
// markup will fail this spec — same regression signal the
// @frontdoor-browser specs provide for the login surface.

import { test, expect } from "@playwright/test";

test.describe(
  "portal shell",
  { tag: "@portal-browser" },
  () => {
    test("authenticated /portal renders the shell with stable view-switcher hooks", async ({ page }) => {
      const consoleErrors = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(msg.text());
      });
      page.on("pageerror", (err) => consoleErrors.push(String(err)));

      const response = await page.goto("/portal");
      expect(response?.status()).toBe(200);

      // Top-level shell anchors served via the front-door's upstream proxy.
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();
      await expect(page.locator('[data-ui="view-switcher"]')).toBeVisible();
      await expect(page.locator('[data-ui="capability-matrix"]')).toBeAttached();
      await expect(page.locator('[data-ui="staged-upload-shell"]')).toBeAttached();
      await expect(page.locator('[data-ui="captioning-controls"]')).toBeAttached();

      // All four workspace sections are rendered with stable
      // data-view-link hooks and ?view=… hrefs. The portal JS is not
      // executed in this mock environment, so we assert the static
      // markup contract only.
      for (const view of ["overview", "build", "operate", "review"]) {
        const link = page.locator(`[data-view-link="${view}"]`);
        await expect(link).toBeVisible();
        await expect(link).toHaveAttribute("href", `?view=${view}`);
      }

      // Overview is the default view — aria-current pinned in markup.
      await expect(
        page.locator('[data-view-link="overview"]')
      ).toHaveAttribute("aria-current", "page");

      expect(
        consoleErrors,
        `unexpected console errors: ${consoleErrors.join("\n")}`
      ).toEqual([]);
    });
  }
);
