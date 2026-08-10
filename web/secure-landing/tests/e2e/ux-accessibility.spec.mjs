import { test, expect } from "@playwright/test";

import {
  expectNoHorizontalOverflow,
  expectNoWcagViolations,
  getContrastRatio,
  getMaximumMotionDurationMs,
} from "./helpers/accessibility.mjs";

test.describe(
  "public UX accessibility",
  { tag: "@frontdoor-browser" },
  () => {
    for (const surface of [
      { label: "homepage", path: "/", shell: '[data-ui="homepage-shell"]' },
      { label: "login", path: "/login", shell: '[data-ui="login-shell"]' },
    ]) {
      test(`${surface.label} has no automated WCAG A/AA violations`, async ({ page }) => {
        await page.goto(surface.path, { waitUntil: "domcontentloaded" });
        await expect(page.locator(surface.shell)).toBeVisible();
        await expectNoWcagViolations(page, surface.label);
      });
    }

    test("homepage contrast and phone reflow remain readable", async ({ page }) => {
      for (const width of [320, 375, 390]) {
        await page.setViewportSize({ width, height: 844 });
        await page.goto("/", { waitUntil: "domcontentloaded" });
        await expect(page.locator("body.frontdoor-homepage")).toHaveCSS(
          "background-color",
          "rgb(2, 6, 23)"
        );
        await expectNoHorizontalOverflow(page, `homepage at ${width}px`);

        const contrast = await getContrastRatio(page, '[data-ui="homepage-hero-title"]');
        expect(
          contrast.ratio,
          `hero contrast was ${contrast.ratio.toFixed(2)}:1 at ${width}px (${JSON.stringify(contrast)})`
        ).toBeGreaterThanOrEqual(4.5);
      }
    });

    test("homepage keeps one managed-access action per navigation group and defers proof detail", async ({ page }) => {
      await page.goto("/", { waitUntil: "domcontentloaded" });
      await expect(page.locator('[data-ui="homepage-nav"] a[href="/login"]')).toHaveCount(0);
      await expect(page.locator(".site-actions a[href=\"/login\"]")).toHaveCount(1);
      await expect(page.locator(".site-mobile-menu__actions a[href=\"/login\"]")).toHaveCount(1);

      const proofFollowsHero = await page.evaluate(() => {
        const hero = document.querySelector('[data-ui="homepage-hero"]');
        const proof = document.getElementById("proof-report");
        return Boolean(hero && proof && (hero.compareDocumentPosition(proof) & Node.DOCUMENT_POSITION_FOLLOWING));
      });
      expect(proofFollowsHero).toBe(true);
      await expect(page.locator("#proof-report")).not.toHaveAttribute("open", "");
      await page.locator('[data-ui="homepage-secondary-cta"]').click();
      await expect(page).toHaveURL(/#proof$/);
      await expect(page.locator("#proof-title")).toBeVisible();
      await expect(page.locator("#proof-report")).not.toHaveAttribute("open", "");
    });

    test("the credential form starts above the fold on desktop and phone", async ({ page }) => {
      for (const viewport of [
        { width: 1280, height: 720 },
        { width: 390, height: 844 },
      ]) {
        await page.setViewportSize(viewport);
        await page.goto("/login", { waitUntil: "domcontentloaded" });
        await expectNoHorizontalOverflow(page, `login at ${viewport.width}px`);

        const username = page.locator(
          '[data-ui="login-username-field"] input[name="username"]'
        );
        const box = await username.boundingBox();
        expect(box, "username field should have a rendered box").not.toBeNull();
        expect(
          box.y,
          `username field begins below the ${viewport.height}px viewport`
        ).toBeLessThan(viewport.height - 44);
      }
    });

    test("reduced motion and forced-colors modes preserve the public entry path", async ({ page }) => {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.goto("/", { waitUntil: "domcontentloaded" });
      expect(await getMaximumMotionDurationMs(page)).toBeLessThanOrEqual(20);

      await page.emulateMedia({ reducedMotion: "reduce", forcedColors: "active" });
      await page.reload({ waitUntil: "domcontentloaded" });
      await expect(page.locator('[data-ui="homepage-primary-cta"]')).toBeVisible();
      await page.keyboard.press("Tab");
      await expect(page.locator(".skip-link")).toBeFocused();
      await page.keyboard.press("Enter");
      await expect(page.locator("#main-content")).toBeFocused();
    });
  }
);
