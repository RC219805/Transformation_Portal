import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { renderHomepage } from "../lib/homepage.js";

function readFrontdoorFile(relativePath) {
  return readFileSync(new URL(`../${relativePath}`, import.meta.url), "utf8");
}

function hexToRgb(hex) {
  const value = String(hex).replace(/^#/, "");
  assert.match(value, /^[0-9a-f]{6}$/i);
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16) / 255);
}

function relativeLuminance(hex) {
  const [red, green, blue] = hexToRgb(hex).map((channel) => (
    channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4
  ));
  return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue);
}

function contrastRatio(first, second) {
  const firstLuminance = relativeLuminance(first);
  const secondLuminance = relativeLuminance(second);
  const lighter = Math.max(firstLuminance, secondLuminance);
  const darker = Math.min(firstLuminance, secondLuminance);
  return (lighter + 0.05) / (darker + 0.05);
}

function customProperty(block, name) {
  const match = block.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6})`, "i"));
  assert.ok(match, `missing ${name} in dark token block`);
  return match[1];
}

test("frontdoor pages explicitly activate the accessible dark token set", () => {
  const homepageHtml = renderHomepage();
  const homepageCss = readFrontdoorFile("public/frontdoor-homepage.css");
  const loginCss = readFrontdoorFile("public/login.css");
  const sharedTokens = readFrontdoorFile("public/shared-ui-tokens.css");
  const darkTokenBlock = sharedTokens.match(/:root\.dark,\.dark:root\{([^}]*)\}/)?.[1] || "";

  assert.match(homepageHtml, /<html lang="en" class="dark" data-theme="dark">/);
  assert.match(homepageHtml, /<meta name="color-scheme" content="dark" \/>/);
  assert.match(homepageCss, /color-scheme:\s*dark/);
  assert.match(homepageCss, /--fd-text:\s*var\(--ux-text-primary\)/);
  assert.match(homepageCss, /--fd-text-strong:\s*var\(--ux-text-strong\)/);
  assert.match(loginCss, /:root\s*\{[\s\S]*?color-scheme:\s*dark/);

  const canvas = customProperty(darkTokenBlock, "--ux-surface-canvas");
  const primaryText = customProperty(darkTokenBlock, "--ux-text-primary");
  const mutedText = customProperty(darkTokenBlock, "--ux-text-muted");
  assert.ok(contrastRatio(primaryText, canvas) >= 4.5);
  assert.ok(contrastRatio(mutedText, canvas) >= 4.5);
  assert.ok(contrastRatio("#f8fafc", "#0f766e") >= 4.5);
  assert.ok(contrastRatio("#f8fafc", "#0e7490") >= 4.5);
});

test("homepage keeps one managed-entry destination per navigation cluster", () => {
  const html = renderHomepage();
  const desktopActions = html.match(/<div class="site-actions">([\s\S]*?)<\/div>/)?.[1] || "";
  const mobileActions = html.match(/<div class="site-mobile-menu__actions">([\s\S]*?)<\/div>/)?.[1] || "";

  assert.equal((desktopActions.match(/href="\/login"/g) || []).length, 1);
  assert.equal((mobileActions.match(/href="\/login"/g) || []).length, 1);
  assert.match(desktopActions, /data-ui="homepage-operator-link"/);
  assert.match(desktopActions, /href="#proof" data-ui="homepage-utility-cta"/);
  assert.match(mobileActions, /data-ui="homepage-mobile-operator-link"/);
  assert.match(mobileActions, /href="#proof" data-ui="homepage-mobile-utility-cta"/);
  assert.match(html, /href="\/login" data-ui="homepage-primary-cta"/);
  assert.match(html, /href="#proof" data-ui="homepage-secondary-cta"/);
});

test("homepage moves detailed proof below the proof overview and exposes a focusable skip target", () => {
  const html = renderHomepage();
  const heroIndex = html.indexOf('data-ui="homepage-hero"');
  const proofOverviewIndex = html.indexOf('id="proof"');
  const proofDetailsIndex = html.indexOf('id="proof-report"');
  const reportExcerptIndex = html.indexOf('data-ui="homepage-report-excerpt"');

  assert.match(html, /<main id="main-content" class="homepage-main" tabindex="-1" data-ui="homepage-main">/);
  assert.ok(heroIndex >= 0);
  assert.ok(proofOverviewIndex > heroIndex);
  assert.ok(proofDetailsIndex > proofOverviewIndex);
  assert.ok(reportExcerptIndex > proofDetailsIndex);
  assert.match(html, /<details id="proof-report"[^>]*data-ui="homepage-proof-details"/);
});

test("accessibility browser gate rebuilds Portal assets before serving them", () => {
  const packageJson = JSON.parse(readFrontdoorFile("package.json"));
  const command = packageJson.scripts?.["test:browser:a11y"] || "";
  const portalFixture = readFrontdoorFile("tests/e2e/fixtures/mock-fastapi-origin.mjs");

  assert.match(command, /^npm run build:portal && /);
  assert.match(command, /playwright test ux-accessibility\.spec\.mjs ux-accessibility\.portal\.spec\.mjs ux-runtime\.portal\.spec\.mjs$/);
  assert.doesNotMatch(portalFixture, /cachedPortalHtml/);
  assert.match(portalFixture, /function getPortalHtml\(\) \{\s*return renderPortalHtml\(\);\s*\}/);
});
