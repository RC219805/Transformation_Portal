import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const FRONTDOOR_ROOT = path.resolve(path.dirname(__filename), "..");
const MAKEFILE_PATH = path.resolve(FRONTDOOR_ROOT, "..", "..", "Makefile");
const PACKAGE_JSON_PATH = path.join(FRONTDOOR_ROOT, "package.json");
const ARCHITECTURE_BASELINE_PATH = path.join(
  FRONTDOOR_ROOT,
  "portal-src",
  "styles",
  "architecture-baseline.json"
);
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_INDEX_PATH = path.join(FRONTDOOR_ROOT, "portal-src", "styles", "index.css");
const PORTAL_CSS_ASSET_PATH = path.join(REPO_ROOT, "public", "portal-assets", "portal.css");
const COMPAT_HOLD_CSS_PATH = path.join(FRONTDOOR_ROOT, "portal-src", "styles", "utilities.compat-hold.css");
const OVERRIDES_COMPAT_CSS_PATH = path.join(FRONTDOOR_ROOT, "portal-src", "styles", "overrides.compat.css");
const OWNERSHIP_DRAIN_REPORT_PATH = path.join(FRONTDOOR_ROOT, "reports", "portal-css-ownership-drain.json");
const SENTINEL_FIXTURE_DIR = path.join(FRONTDOOR_ROOT, "tests", "fixtures", "sentinel");
const LAYER_PARITY_SCRIPT_PATH = path.join(FRONTDOOR_ROOT, "scripts", "check-portal-css-layer-parity.mjs");
const PYTHON_LAYER_PARITY_VALIDATOR_PATH = path.join(
  REPO_ROOT,
  "scripts",
  "validation",
  "validate_portal_css_layer_parity.py"
);
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

function runNodeScriptWithEnv(env, scriptPath, ...args) {
  return execFileSync(process.execPath, [scriptPath, ...args], {
    cwd: FRONTDOOR_ROOT,
    env: { ...process.env, ...env },
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
}

function runNodeScriptFailure(scriptPath, ...args) {
  try {
    execFileSync(process.execPath, [scriptPath, ...args], {
      cwd: FRONTDOOR_ROOT,
      env: process.env,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"]
    });
  } catch (error) {
    return `${error.stdout || ""}${error.stderr || ""}`;
  }
  assert.fail(`${scriptPath} ${args.join(" ")} unexpectedly passed`);
}

function runNpmScript(...args) {
  return execFileSync("npm", args, {
    cwd: FRONTDOOR_ROOT,
    env: {
      ...process.env,
      PATH: [path.dirname(process.execPath), process.env.PATH].filter(Boolean).join(path.delimiter)
    },
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
  assert.match(lintCss, /check-portal-utility-ownership\.mjs/);
  assert.match(String(packageJson.scripts["check:utility-ownership"] || ""), /check-portal-utility-ownership\.mjs/);
  assert.match(String(packageJson.scripts["check:css-layer-parity"] || ""), /check-portal-css-layer-parity\.mjs/);
  assert.match(String(packageJson.scripts["check:css-layer-dry-run"] || ""), /check:css-layer-parity --/);

  assert.match(runNodeScript("scripts/build-portal-bundle.mjs", "--check-css"), /generated artifact is fresh/);
  assert.match(
    runNodeScriptWithEnv(
      { PORTAL_CSS_DISABLE_COMPAT_OVERRIDES: "1" },
      "scripts/build-portal-bundle.mjs",
      "--check-css"
    ),
    /generated artifact is fresh/
  );
  assert.match(runNodeScript("scripts/check-portal-css-contract.mjs"), /portal css contract: OK/);
  assert.match(runNodeScript("scripts/check-portal-css-architecture.mjs"), /portal css architecture: OK/);
  assert.match(runNodeScript("scripts/check-portal-utility-ownership.mjs"), /portal utility ownership: OK/);
});

test("portal CSS sentinel fixtures enforce comments-only architecture", () => {
  for (const fixtureName of ["sentinel-comments-only.css", "sentinel-empty.css"]) {
    const fixturePath = path.relative(FRONTDOOR_ROOT, path.join(SENTINEL_FIXTURE_DIR, fixtureName));
    assert.match(
      runNodeScript("scripts/check-portal-css-architecture.mjs", "--check-sentinel-fixture", fixturePath),
      /portal css sentinel fixture: OK/
    );
  }

  for (const [fixtureName, expectedNode] of [
    ["sentinel-with-rule.css", /Found rule "\.foo"/],
    ["sentinel-with-empty-rule.css", /Found rule "\.foo"/],
    ["sentinel-with-import.css", /Found at-rule @import "\.\/owned\.css"/],
    ["sentinel-with-media.css", /Found at-rule @media/],
    ["sentinel-with-keyframes.css", /Found at-rule @keyframes/],
    ["sentinel-with-root-token.css", /Found rule ":root"/]
  ]) {
    const fixturePath = path.relative(FRONTDOOR_ROOT, path.join(SENTINEL_FIXTURE_DIR, fixtureName));
    const output = runNodeScriptFailure(
      "scripts/check-portal-css-architecture.mjs",
      "--check-sentinel-fixture",
      fixturePath
    );
    assert.match(output, /must remain sentinel-only/);
    assert.match(output, expectedNode);
  }
});

test("portal CSS layer parity make target checks generated artifact freshness", () => {
  const makefile = readFileSync(MAKEFILE_PATH, "utf8");
  const start = makefile.indexOf("validate-portal-css-layer-parity:");
  const end = makefile.indexOf("\nvalidate-portal-browser:", start);

  assert.notEqual(start, -1, "missing validate-portal-css-layer-parity target");
  assert.notEqual(end, -1, "missing validate-portal-browser target after CSS layer parity target");

  const target = makefile.slice(start, end);
  const freshnessCheckIndex = target.indexOf("build-portal-bundle.mjs --check-css");
  const parityCheckIndex = target.indexOf("npm run check:css-layer-parity");

  assert.notEqual(freshnessCheckIndex, -1, "CSS layer parity target must check generated artifact freshness");
  assert.notEqual(parityCheckIndex, -1, "CSS layer parity target must run the layer parity check");
  assert.ok(freshnessCheckIndex < parityCheckIndex, "freshness must be checked before parity validators");
});

test("portal CSS architecture baseline keeps hotspot selectors consolidated", () => {
  const baseline = JSON.parse(readFileSync(ARCHITECTURE_BASELINE_PATH, "utf8"));
  const duplicateKeys = baseline.duplicateKeys || [];

  for (const selector of HOTSPOT_SELECTORS) {
    const entry = duplicateKeys.find((candidate) => candidate.key === `${selector}|||`);
    assert.equal(entry, undefined, `unexpected duplicate baseline entry for ${selector}`);
  }
});

test("portal CSS ownership drain keeps utilities layer honest", () => {
  const portalCssIndex = readFileSync(PORTAL_CSS_INDEX_PATH, "utf8");
  const ownershipDrain = JSON.parse(readFileSync(OWNERSHIP_DRAIN_REPORT_PATH, "utf8"));

  assert.match(portalCssIndex, /@import "\.\/utilities\.required\.css" layer\(utilities\);/);
  assert.match(portalCssIndex, /@import "\.\/utilities\.dynamic\.css" layer\(utilities\);/);
  assert.match(portalCssIndex, /@import "\.\/utilities\.compat-hold\.css" layer\(utilities\);/);
  assert.doesNotMatch(portalCssIndex, /@import "\.\/overrides\.compat\.css" layer\(overrides\);/);
  assert.doesNotMatch(portalCssIndex, /@import "\.\/overrides\.[^"]+" layer\(utilities\);/);
  assert.doesNotMatch(portalCssIndex, /@import "\.\/components\/[^"]+" layer\(utilities\);/);
  assert.doesNotMatch(portalCssIndex, /operator-console-reset/);
  assert.equal(ownershipDrain.summary.utilityLayerImportsAfter, 3);
  assert.equal(ownershipDrain.summary.compatHoldCount, 0);
  assert.equal(ownershipDrain.summary.overridesCompatRuleCount, 0);
  assert.equal(ownershipDrain.summary.overridesCompatBytes, 0);
  assert.match(readFileSync(COMPAT_HOLD_CSS_PATH, "utf8"), /phase-8-governance-sentinel/);
  assert.match(readFileSync(OVERRIDES_COMPAT_CSS_PATH, "utf8"), /phase-8-governance-sentinel/);
  assert.deepEqual(ownershipDrain.phase8SentinelState.utilitiesCompatHold, {
    path: "web/secure-landing/portal-src/styles/utilities.compat-hold.css",
    imported: true,
    layer: "utilities",
    sourceRuleCount: 0,
    shippedRuleCount: 0,
    shippedByteDebt: 0,
    sentinelOnly: true
  });
  assert.deepEqual(ownershipDrain.phase8SentinelState.overridesCompat, {
    path: "web/secure-landing/portal-src/styles/overrides.compat.css",
    imported: false,
    layer: null,
    sourceRuleCount: 0,
    shippedRuleCount: 0,
    shippedByteDebt: 0,
    sentinelOnly: true
  });
  assert.ok(
    ownershipDrain.moves.every((move) => move.parity === "green"),
    "all ownership drain moves must be parity green"
  );
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

test("portal CSS parity census forces feature states and canonical theme hooks", () => {
  const validator = readFileSync(PYTHON_LAYER_PARITY_VALIDATOR_PATH, "utf8");

  assert.match(validator, /TP_PORTAL_UPLOAD_STAGING_ENABLED/);
  assert.match(validator, /TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT/);
  assert.match(validator, /TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT/);
  assert.match(validator, /TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT/);
  assert.match(validator, /localStorage\.setItem\('tp_theme'/);
  assert.doesNotMatch(validator, /portal-theme/);
  assert.match(validator, /root = document\.documentElement/);
  assert.match(validator, /root\.classList\.toggle\('performance-lite'/);
  assert.doesNotMatch(validator, /body\.classList\.toggle\('performance-lite'/);
  assert.match(validator, /portalState\.auth\.features\.stagedUploads = true/);
  assert.match(validator, /portalState\.auth\.features\.artifactViewerModal = true/);
  assert.match(validator, /portalState\.auth\.features\.reviewSurfaceDeferred = true/);
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
