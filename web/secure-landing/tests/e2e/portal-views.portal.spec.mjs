// @portal-browser portal-views smoke.
//
// Verifies the per-view affordances are present in the static portal
// markup served via the front-door upstream proxy, and that the
// shared shell anchors remain visible regardless of the ?view= deep
// link. The portal interactivity bundle is stubbed in this mock
// environment, so we assert the markup contract — not the JS-driven
// view-toggling behavior, which is governed by
// validate_portal_browser_smoke.py's live CDP suite.

import { test, expect } from "@playwright/test";

const SHARED_SHELL_ANCHORS = [
  '[data-ui="portal-topbar"]',
  '[data-ui="view-switcher"]',
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
      test(`/portal?view=${view} keeps the shared shell and reaches the ${view} affordances`, async ({ page }) => {
        const response = await page.goto(`/portal?view=${view}`);
        expect(response?.status()).toBe(200);

        for (const sharedAnchor of SHARED_SHELL_ANCHORS) {
          await expect(page.locator(sharedAnchor)).toBeVisible();
        }

        for (const viewAnchor of affordances) {
          await expect(page.locator(viewAnchor).first()).toBeVisible();
        }
      });
    }
  }
);
