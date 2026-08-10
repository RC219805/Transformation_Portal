import AxeBuilder from "@axe-core/playwright";
import { expect } from "@playwright/test";

const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

async function waitForStylesheets(page) {
  await page.waitForFunction(() =>
    Array.from(document.querySelectorAll('link[rel~="stylesheet"]')).every(
      (link) => Boolean(link.sheet)
    )
  );
}

function formatViolations(violations) {
  return violations
    .map((violation) => {
      const targets = violation.nodes
        .slice(0, 4)
        .map((node) => node.target.join(" "))
        .join(", ");
      return `${violation.id} (${violation.impact || "unknown"}): ${targets}`;
    })
    .join("\n");
}

export async function expectNoWcagViolations(page, label) {
  await waitForStylesheets(page);
  const results = await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze();
  expect(
    results.violations,
    `${label} WCAG violations:\n${formatViolations(results.violations)}`
  ).toEqual([]);
}

export async function expectNoHorizontalOverflow(page, label) {
  await waitForStylesheets(page);
  const overflow = await page.evaluate(() => {
    const root = document.documentElement;
    const viewportWidth = window.innerWidth || root.clientWidth;
    const delta = root.scrollWidth - root.clientWidth;
    const isInactive = (element) => Boolean(element.closest('[hidden], [aria-hidden="true"], [inert]'));
    const isInsideBoundedScroller = (element) => {
      let ancestor = element.parentElement;
      while (ancestor && ancestor !== document.body) {
        const style = getComputedStyle(ancestor);
        if (style.overflowX === "auto" || style.overflowX === "scroll") {
          const rect = ancestor.getBoundingClientRect();
          return rect.left >= -1 && rect.right <= viewportWidth + 1;
        }
        ancestor = ancestor.parentElement;
      }
      return false;
    };
    const offenders = Array.from(document.querySelectorAll("body *"))
      .filter((element) => {
        const style = getComputedStyle(element);
        if (
          style.display === "none"
          || style.visibility === "hidden"
          || isInactive(element)
          || isInsideBoundedScroller(element)
        ) return false;
        const rect = element.getBoundingClientRect();
        if (rect.width <= 1 || rect.height <= 1) return false;
        return rect.right > viewportWidth + 1 || rect.left < -1;
      })
      .slice(0, 8)
      .map((element) => ({
        tag: element.tagName.toLowerCase(),
        id: element.id,
        className: String(element.className || "").slice(0, 100),
        rect: element.getBoundingClientRect().toJSON(),
      }));
    return { delta, clientWidth: root.clientWidth, scrollWidth: root.scrollWidth, offenders };
  });

  expect(
    overflow.delta,
    `${label} overflows by ${overflow.delta}px: ${JSON.stringify(overflow.offenders)}`
  ).toBeLessThanOrEqual(1);
  expect(
    overflow.offenders,
    `${label} clips visible elements outside the viewport: ${JSON.stringify(overflow.offenders)}`
  ).toEqual([]);
}

export async function getContrastRatio(page, selector) {
  await waitForStylesheets(page);
  return page.locator(selector).evaluate((element) => {
    function parseColor(value) {
      const input = String(value || "").trim().toLowerCase();
      if (input === "transparent") {
        return { r: 0, g: 0, b: 0, a: 0 };
      }

      const numericTokens = input.match(/[+-]?(?:\d*\.)?\d+(?:e[+-]?\d+)?%?/g) || [];
      const numericValue = (token, percentScale) => {
        if (!token) return 0;
        if (token.endsWith("%")) return (Number.parseFloat(token) / 100) * percentScale;
        return Number.parseFloat(token);
      };

      if (input.startsWith("color(srgb ")) {
        return {
          r: numericValue(numericTokens[0], 1) * 255,
          g: numericValue(numericTokens[1], 1) * 255,
          b: numericValue(numericTokens[2], 1) * 255,
          a: numericTokens.length > 3 ? numericValue(numericTokens[3], 1) : 1,
        };
      }

      const channels = numericTokens.map((token, index) =>
        numericValue(token, index < 3 ? 255 : 1)
      );
      return {
        r: channels[0] || 0,
        g: channels[1] || 0,
        b: channels[2] || 0,
        a: channels.length > 3 ? channels[3] : 1,
      };
    }

    function luminance({ r, g, b }) {
      const linear = [r, g, b].map((channel) => {
        const normalized = channel / 255;
        return normalized <= 0.04045
          ? normalized / 12.92
          : ((normalized + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    }

    const foregroundCss = getComputedStyle(element).color;
    const foreground = parseColor(foregroundCss);
    let background = null;
    let backgroundCss = "";
    let current = element;
    while (current) {
      const candidateCss = getComputedStyle(current).backgroundColor;
      const candidate = parseColor(candidateCss);
      if (candidate.a >= 0.99) {
        background = candidate;
        backgroundCss = candidateCss;
        break;
      }
      current = current.parentElement;
    }
    background ||= { r: 255, g: 255, b: 255, a: 1 };

    const lighter = Math.max(luminance(foreground), luminance(background));
    const darker = Math.min(luminance(foreground), luminance(background));
    return {
      foreground,
      background,
      foregroundCss,
      backgroundCss,
      ratio: (lighter + 0.05) / (darker + 0.05),
    };
  });
}

export async function getMaximumMotionDurationMs(page) {
  return page.evaluate(() => {
    function milliseconds(value) {
      const parsed = Number.parseFloat(value);
      if (!Number.isFinite(parsed)) return 0;
      return value.trim().endsWith("ms") ? parsed : parsed * 1000;
    }

    let maximum = 0;
    for (const element of document.querySelectorAll("*")) {
      const style = getComputedStyle(element);
      for (const value of `${style.animationDuration},${style.transitionDuration}`.split(",")) {
        maximum = Math.max(maximum, milliseconds(value));
      }
    }
    return maximum;
  });
}
