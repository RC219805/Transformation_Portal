import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const FRONTDOOR_ROOT = path.resolve(path.dirname(__filename), "..");
const PACKAGE_JSON_PATH = path.join(FRONTDOOR_ROOT, "package.json");
const ARCHITECTURE_BASELINE_PATH = path.join(
  FRONTDOOR_ROOT,
  "portal-src",
  "styles",
  "architecture-baseline.json"
);
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_ASSET_PATH = path.join(REPO_ROOT, "public", "portal-assets", "portal.css");
const LAYER_PARITY_SCRIPT_PATH = path.join(FRONTDOOR_ROOT, "scripts", "check-portal-css-layer-parity.mjs");
const LAYER_PARITY_CONTRACT_PATH = path.join(
  REPO_ROOT,
  "tests",
  "fixtures",
  "portal-css",
  "layer-parity-contract.json"
);
const LAYER_PARITY_BASELINE_PATH = path.join(
  REPO_ROOT,
  "tests",
  "fixtures",
  "portal-css",
  "layer-parity-baseline.json"
);
const HOTSPOT_SELECTORS = [
  ".shell-bg",
  ".workspace-rail",
  ".workspace-link",
  ".hero-action",
  ".operator-action-btn",
  ".stepper-nav-btn",
  ".dispatch-tool-btn",
  ".console-context-ribbon",
  ".topbar-status"
];

function runNodeScript(scriptPath, ...args) {
  return execFileSync(process.execPath, [scriptPath, ...args], {
    cwd: FRONTDOOR_ROOT,
    env: process.env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
}

function runNpmScript(...args) {
  return execFileSync("npm", args, {
    cwd: FRONTDOOR_ROOT,
    env: process.env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
}

test("portal CSS lint script checks generated artifact freshness and architecture gates", () => {
  const packageJson = JSON.parse(readFileSync(PACKAGE_JSON_PATH, "utf8"));
  const lintCss = String(packageJson.scripts["lint:css"] || "");

  assert.match(lintCss, /build-portal-bundle\.mjs --check-css/);
  assert.match(lintCss, /check-portal-css-contract\.mjs/);
  assert.match(lintCss, /check-portal-css-architecture\.mjs/);
  assert.match(String(packageJson.scripts["check:css-layer-parity"] || ""), /check-portal-css-layer-parity\.mjs/);
  assert.match(String(packageJson.scripts["check:css-layer-dry-run"] || ""), /check:css-layer-parity --/);

  assert.match(runNodeScript("scripts/build-portal-bundle.mjs", "--check-css"), /generated artifact is fresh/);
  assert.match(runNodeScript("scripts/check-portal-css-contract.mjs"), /portal css contract: OK/);
  assert.match(runNodeScript("scripts/check-portal-css-architecture.mjs"), /portal css architecture: OK/);
});

test("portal CSS architecture baseline keeps hotspot selectors consolidated", () => {
  const baseline = JSON.parse(readFileSync(ARCHITECTURE_BASELINE_PATH, "utf8"));
  const duplicateKeys = baseline.duplicateKeys || [];

  for (const selector of HOTSPOT_SELECTORS) {
    const entry = duplicateKeys.find((candidate) => candidate.key === `${selector}|||`);
    assert.equal(entry, undefined, `unexpected duplicate baseline entry for ${selector}`);
  }
});

test("portal CSS layer parity validates the production layered graph", () => {
  const contract = JSON.parse(readFileSync(LAYER_PARITY_CONTRACT_PATH, "utf8"));
  const baseline = JSON.parse(readFileSync(LAYER_PARITY_BASELINE_PATH, "utf8"));
  const output = runNodeScript("scripts/check-portal-css-layer-parity.mjs");

  assert.match(output, /portal css layer parity: OK/);
  assert.match(output, /representative selectors/);
  assert.match(output, /style properties tracked for browser parity/);
  assert.deepEqual(baseline.representativeStyleSelectors, contract.representativeStyleSelectors);
  assert.deepEqual(baseline.representativeStyleProperties, contract.representativeStyleProperties);
  assert.ok(contract.representativeStyleProperties.includes("content-visibility"));
  assert.ok(contract.representativeStyleSelectors.includes("[data-ui=\"staged-upload-shell\"]"));
});

test("portal CSS layer parity checks nested generated keyframes", () => {
  const parityScript = readFileSync(LAYER_PARITY_SCRIPT_PATH, "utf8");

  assert.match(parityScript, /root\.walkAtRules\(["']keyframes["']/);
});

test("portal CSS layer dry-run compatibility writes the validated CSS artifact", () => {
  const tempDir = mkdtempSync(path.join(tmpdir(), "portal-layer-css-"));
  const outputPath = path.join(tempDir, "portal.css");

  try {
    const output = runNpmScript("run", "check:css-layer-dry-run", "--", "--write-css", outputPath);

    assert.match(output, /portal css layer parity: OK/);
    assert.equal(readFileSync(outputPath, "utf8"), readFileSync(PORTAL_CSS_ASSET_PATH, "utf8"));
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});
