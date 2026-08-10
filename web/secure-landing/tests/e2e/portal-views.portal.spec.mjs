// @portal-browser portal-views smoke.
//
// Verifies the per-view affordances remain in the static portal markup
// served via the front-door upstream proxy, and that the universally
// visible shell anchors remain rendered regardless of the ?view= deep
// link. The portal interactivity bundle is stubbed in this mock
// environment, so route-owned surfaces are structural assertions — the
// hydrated runtime suite governs their JS-driven visibility.

import { test, expect } from "@playwright/test";

const VISIBLE_SHELL_ANCHORS = [
  '[data-ui="portal-topbar"]',
  '[data-ui="view-switcher"]'
];

const STRUCTURAL_SHELL_ANCHORS = [
  '[data-ui="console-context-shell"]'
];

const VIEW_AFFORDANCES = {
  build: ['[data-ui="connection-card"]', '[data-ui="build-stepper"]'],
  operate: ['[data-ui="operate-grid"]'],
  review: ['[data-ui="review-surface"]']
};

test.describe(
  "portal view shells",
  { tag: "@portal-browser" },
  () => {
    for (const [view, affordances] of Object.entries(VIEW_AFFORDANCES)) {
      test(`/portal?view=${view} preserves the shared shell and ${view} affordance hooks`, async ({ page }) => {
        const consoleErrors = [];
        page.on("console", (msg) => {
          if (msg.type() === "error") consoleErrors.push(msg.text());
        });
        page.on("pageerror", (err) => consoleErrors.push(String(err)));

        const response = await page.goto(`/portal?view=${view}`);
        expect(response?.status()).toBe(200);

        for (const visibleAnchor of VISIBLE_SHELL_ANCHORS) {
          await expect(page.locator(visibleAnchor)).toBeVisible();
        }

        for (const structuralAnchor of STRUCTURAL_SHELL_ANCHORS) {
          await expect(page.locator(structuralAnchor)).toHaveCount(1);
        }

        for (const viewAnchor of affordances) {
          await expect(page.locator(viewAnchor)).toHaveCount(1);
        }

        expect(
          consoleErrors,
          `unexpected console errors: ${consoleErrors.join("\n")}`
        ).toEqual([]);
      });
    }
  }
);
