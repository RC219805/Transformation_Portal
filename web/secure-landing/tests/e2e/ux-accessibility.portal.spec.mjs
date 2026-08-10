import { test, expect } from "@playwright/test";

import {
  expectNoHorizontalOverflow,
  expectNoWcagViolations,
  getMaximumMotionDurationMs,
} from "./helpers/accessibility.mjs";

test.describe(
  "authenticated Portal UX accessibility",
  { tag: "@portal-browser" },
  () => {
    test("overview has no automated WCAG A/AA violations", async ({ page }) => {
      await page.goto("/portal", { waitUntil: "domcontentloaded" });
      await expect(page.locator('[data-ui="portal-topbar"]')).toBeVisible();
      await expectNoWcagViolations(page, "Portal overview");
    });

    test("phone layouts keep health, navigation, and content in the viewport", async ({ page }) => {
      for (const width of [320, 375, 390]) {
        await page.setViewportSize({ width, height: 844 });
        await page.goto("/portal", { waitUntil: "domcontentloaded" });
        await expectNoHorizontalOverflow(page, `Portal at ${width}px`);

        const topbar = await page.locator('[data-ui="portal-topbar"]').boundingBox();
        const switcher = await page.locator('[data-ui="view-switcher"]').boundingBox();
        expect(topbar, "Portal topbar should render").not.toBeNull();
        expect(switcher, "Portal navigation should render").not.toBeNull();
        expect(topbar.height).toBeLessThanOrEqual(220);
        expect(switcher.y).toBeLessThan(844);
        await expect(page.locator("#healthText")).toBeVisible();

        const targetSizes = await page.locator('[data-ui="view-link"]').evaluateAll((links) =>
          links.map((link) => {
            const rect = link.getBoundingClientRect();
            return { width: rect.width, height: rect.height };
          })
        );
        for (const size of targetSizes) {
          expect(size.width).toBeGreaterThanOrEqual(44);
          expect(size.height).toBeGreaterThanOrEqual(44);
        }
      }
    });

    test("workspace and Build-step semantics expose coherent navigation models", async ({ page }) => {
      await page.goto("/portal", { waitUntil: "domcontentloaded" });

      const navBeforeOverview = await page.evaluate(() => {
        const nav = document.querySelector('[data-ui="view-switcher"]');
        const overview = document.querySelector('[data-ui="overview-grid"]');
        return Boolean(nav && overview && (nav.compareDocumentPosition(overview) & Node.DOCUMENT_POSITION_FOLLOWING));
      });
      expect(navBeforeOverview).toBe(true);
      await expect(page.locator('[data-ui="view-link"][aria-selected]')).toHaveCount(0);
      await expect(page.locator('[data-ui="view-link"][aria-current="page"]')).toHaveCount(1);

      const brokenStepReferences = await page.evaluate(() =>
        Array.from(document.querySelectorAll('[data-ui="build-step-tab"][aria-controls]'))
          .filter((step) => step.getAttribute("aria-controls").split(/\s+/).some((id) => !document.getElementById(id)))
          .map((step) => step.id || step.textContent.trim())
      );
      expect(brokenStepReferences).toEqual([]);
      await expect(page.locator('#buildStepTabs[role="toolbar"]')).toHaveCount(1);
      await expect(page.locator('[data-ui="build-step-tab"][aria-current="step"][aria-pressed="true"]')).toHaveCount(1);
      await expect(page.locator('[data-build-step-panel][role="tabpanel"]')).toHaveCount(0);
      await expect(page.locator('[data-modal-inert-target]')).toHaveCount(2);
    });

    test("reduced motion and forced colors preserve Portal controls", async ({ page }) => {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.goto("/portal", { waitUntil: "domcontentloaded" });
      expect(await getMaximumMotionDurationMs(page)).toBeLessThanOrEqual(20);

      await page.emulateMedia({ reducedMotion: "reduce", forcedColors: "active" });
      await page.reload({ waitUntil: "domcontentloaded" });
      await expect(page.locator('[data-ui="view-switcher"]')).toBeVisible();
      await expect(page.locator('[data-ui="view-link"]')).toHaveCount(4);
    });
  }
);
