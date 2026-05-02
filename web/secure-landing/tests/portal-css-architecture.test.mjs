import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import postcss from "postcss";

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
const ARCHITECTURE_SCRIPT_PATH = path.join(FRONTDOOR_ROOT, "scripts", "check-portal-css-architecture.mjs");
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

function duplicateFixture() {
  return {
    key: ".owned|||components|||",
    selector: ".owned",
    layer: "components",
    context: [],
    stateContext: [],
    category: "additive",
    hotspot: false,
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 1,
        column: 1,
        layer: "components",
        declarationSignature: "left",
        properties: ["display"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 8,
        column: 1,
        layer: "components",
        declarationSignature: "right",
        properties: ["gap"]
      }
    ]
  };
}

function baselineEntryFor(duplicate, overrides = {}) {
  return {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context,
    stateContext: duplicate.stateContext,
    category: duplicate.category,
    hotspot: duplicate.hotspot,
    records: duplicate.records,
    owners: ["portal-css-architecture"],
    ownerReason:
      duplicate.category === "conflicting"
        ? "Intentional cascade: source order preserves the final declaration set."
        : "Additive ownership: split declarations preserve computed-style parity.",
    phase: "phase-9-duplicate-ownership-closure",
    contextType: "component-family-shared-state",
    disposition: "report-only",
    removalStatus: "removable-later",
    declarationConflict: duplicate.category,
    parity: "green",
    ...overrides
  };
}

function runDuplicateBaselineFixture(baseline, duplicates) {
  const tempDir = mkdtempSync(path.join(tmpdir(), "portal-duplicate-baseline-"));
  const baselinePath = path.join(tempDir, "baseline.json");
  const duplicatesPath = path.join(tempDir, "duplicates.json");
  writeFileSync(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`, "utf8");
  writeFileSync(duplicatesPath, `${JSON.stringify({ duplicates }, null, 2)}\n`, "utf8");
  try {
    return runNodeScript("scripts/check-portal-css-architecture.mjs", "--check-duplicate-baseline-fixture", baselinePath, duplicatesPath);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runDuplicateBaselineFixtureFailure(baseline, duplicates) {
  try {
    runDuplicateBaselineFixture(baseline, duplicates);
  } catch (error) {
    return `${error.stdout || ""}${error.stderr || ""}`;
  }
  assert.fail("duplicate baseline fixture unexpectedly passed");
}

function runPhase10AdditiveFixture(fixture) {
  const tempDir = mkdtempSync(path.join(tmpdir(), "portal-phase10-additive-"));
  const fixturePath = path.join(tempDir, "phase10.json");
  writeFileSync(fixturePath, `${JSON.stringify(fixture, null, 2)}\n`, "utf8");
  try {
    return runNodeScript("scripts/check-portal-css-architecture.mjs", "--check-phase10-additive-fixture", fixturePath);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runPhase10AdditiveFixtureFailure(fixture) {
  try {
    runPhase10AdditiveFixture(fixture);
  } catch (error) {
    return `${error.stdout || ""}${error.stderr || ""}`;
  }
  assert.fail("Phase 10 additive fixture unexpectedly passed");
}

function runPhase11SurfaceFixture(fixture) {
  const tempDir = mkdtempSync(path.join(tmpdir(), "portal-phase11-surface-"));
  const fixturePath = path.join(tempDir, "phase11.json");
  writeFileSync(fixturePath, `${JSON.stringify(fixture, null, 2)}\n`, "utf8");
  try {
    return runNodeScript("scripts/check-portal-css-architecture.mjs", "--check-phase11-surface-fixture", fixturePath);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runPhase11SurfaceFixtureFailure(fixture) {
  try {
    runPhase11SurfaceFixture(fixture);
  } catch (error) {
    return `${error.stdout || ""}${error.stderr || ""}`;
  }
  assert.fail("Phase 11 surface fixture unexpectedly passed");
}

function runPhase12ComponentFixture(fixture) {
  const tempDir = mkdtempSync(path.join(tmpdir(), "portal-phase12-component-"));
  const fixturePath = path.join(tempDir, "phase12.json");
  writeFileSync(fixturePath, `${JSON.stringify(fixture, null, 2)}\n`, "utf8");
  try {
    return runNodeScript("scripts/check-portal-css-architecture.mjs", "--check-phase12-component-fixture", fixturePath);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runPhase12ComponentFixtureFailure(fixture) {
  try {
    runPhase12ComponentFixture(fixture);
  } catch (error) {
    return `${error.stdout || ""}${error.stderr || ""}`;
  }
  assert.fail("Phase 12 component fixture unexpectedly passed");
}

function runPhase13InteractionFixture(fixture) {
  const tempDir = mkdtempSync(path.join(tmpdir(), "portal-phase13-interaction-"));
  const fixturePath = path.join(tempDir, "phase13.json");
  writeFileSync(fixturePath, `${JSON.stringify(fixture, null, 2)}\n`, "utf8");
  try {
    return runNodeScript("scripts/check-portal-css-architecture.mjs", "--check-phase13-interaction-fixture", fixturePath);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function runPhase13InteractionFixtureFailure(fixture) {
  try {
    runPhase13InteractionFixture(fixture);
  } catch (error) {
    return `${error.stdout || ""}${error.stderr || ""}`;
  }
  assert.fail("Phase 13 interaction fixture unexpectedly passed");
}

function phase10AdditiveDuplicate(overrides = {}) {
  const selector = overrides.selector || ".owned";
  return {
    key: `${selector}|||components|||`,
    selector,
    layer: "components",
    context: [],
    stateContext: [],
    category: "additive",
    hotspot: false,
    removalStatus: "removable-later",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 1,
        column: 1,
        layer: "components",
        selectorList: [selector],
        declarations: [["min-height", "2rem", false]],
        declarationSignature: "singleton",
        properties: ["min-height"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 4,
        column: 1,
        layer: "components",
        selectorList: [".secondary", selector],
        declarations: [["background", "var(--surface)", false]],
        declarationSignature: "list",
        properties: ["background"]
      }
    ],
    ...overrides
  };
}

function phase11SurfaceDuplicate(overrides = {}) {
  const selector = overrides.selector || ".review-compare-summary";
  return {
    key: overrides.key || `${selector}|||components|||`,
    selector,
    layer: "components",
    context: [],
    stateContext: [],
    category: "additive",
    hotspot: false,
    removalStatus: "removable-later",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 146,
        column: 1,
        layer: "components",
        selectorList: [selector],
        declarations: [["border-radius", "var(--ux-radius-lg)", false]],
        declarationSignature: "radius",
        properties: ["border-radius"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 176,
        column: 1,
        layer: "components",
        selectorList: [".other-surface", selector],
        declarations: [["border-color", "var(--ux-border-subtle)", false]],
        declarationSignature: "chrome",
        properties: ["border-color"]
      }
    ],
    ...overrides
  };
}

function phase12ComponentDuplicate(overrides = {}) {
  const selector = overrides.selector || ".ambient-orb";
  return {
    key: overrides.key || `${selector}|||components|||`,
    selector,
    layer: "components",
    context: overrides.context || [],
    stateContext: [],
    category: "additive",
    hotspot: false,
    removalStatus: "removable-later",
    records: overrides.records || [
      {
        source: "web/secure-landing/portal-src/styles/components/shell-foundation.css",
        line: 322,
        column: 1,
        layer: "components",
        selectorList: [".ambient-veil", selector],
        declarations: [["position", "absolute", false], ["inset", "auto", false]],
        declarationSignature: "ambient-base",
        properties: ["inset", "position"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/shell-foundation.css",
        line: 417,
        column: 1,
        layer: "components",
        selectorList: [selector],
        declarations: [["border-radius", "999px", false], ["filter", "blur(34px)", false]],
        declarationSignature: "ambient-singleton",
        properties: ["border-radius", "filter"]
      }
    ],
    ...overrides
  };
}

function phase13InteractionDuplicate(overrides = {}) {
  const selector = overrides.selector || ".build-step-tab:hover";
  const focusSelector = selector.replace(/:hover$/, ":focus-visible");
  return {
    key: overrides.key || `${selector}|||components|||`,
    selector,
    layer: "components",
    context: [],
    stateContext: [],
    category: "additive",
    hotspot: false,
    removalStatus: "removable-later",
    records: overrides.records || [
      {
        source: overrides.source || "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 308,
        column: 1,
        layer: "components",
        selectorList: [selector, focusSelector],
        declarations: [["transform", "translateY(-1px)", false], ["border-color", "rgba(8, 145, 178, 0.28)", false]],
        declarationSignature: "interaction-shared",
        properties: ["border-color", "transform"]
      },
      {
        source: overrides.source || "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 313,
        column: 1,
        layer: "components",
        selectorList: [selector],
        declarations: [["outline", "none", false]],
        declarationSignature: "hover-outline",
        properties: ["outline"]
      }
    ],
    ...overrides
  };
}

function normalizedSelectorList(selectorText) {
  return selectorText.split(",").map((selector) => selector.trim().replace(/\s+/g, " ")).sort();
}

function declarationsForRule(rule) {
  const declarations = new Map();
  rule.walkDecls((decl) => {
    declarations.set(decl.prop, decl.value);
  });
  return declarations;
}

function findRuleBySelectors(root, selectors) {
  const expected = [...selectors].sort();
  let match = null;
  root.walkRules((rule) => {
    if (JSON.stringify(normalizedSelectorList(rule.selector)) === JSON.stringify(expected)) {
      match = rule;
    }
  });
  return match;
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

test("portal CSS duplicate baseline fixtures enforce Phase 9 ownership", () => {
  const duplicate = duplicateFixture();
  const ownedBaseline = { version: 1, duplicateKeys: [baselineEntryFor(duplicate)] };

  assert.match(
    runDuplicateBaselineFixture(ownedBaseline, [duplicate]),
    /portal css duplicate baseline fixture: OK/
  );

  for (const [name, baseline, expected] of [
    [
      "unowned duplicate baseline entry",
      { version: 1, duplicateKeys: [baselineEntryFor(duplicate, { owners: undefined })] },
      /missing owners/
    ],
    [
      "empty owners",
      { version: 1, duplicateKeys: [baselineEntryFor(duplicate, { owners: [] })] },
      /missing owners/
    ],
    [
      "missing ownerReason",
      { version: 1, duplicateKeys: [baselineEntryFor(duplicate, { ownerReason: "" })] },
      /missing ownerReason/
    ],
    [
      "wrong phase",
      { version: 1, duplicateKeys: [baselineEntryFor(duplicate, { phase: "phase-8-governance-sentinel" })] },
      /must declare phase phase-9-duplicate-ownership-closure/
    ],
    [
      "invalid disposition",
      { version: 1, duplicateKeys: [baselineEntryFor(duplicate, { disposition: "consolidated-safe" })] },
      /invalid disposition consolidated-safe/
    ],
    ["new duplicate absent from baseline", { version: 1, duplicateKeys: [] }, /new unclassified duplicate selector/]
  ]) {
    assert.match(runDuplicateBaselineFixtureFailure(baseline, [duplicate]), expected, name);
  }

  const hotspotDuplicate = { ...duplicate, key: ".shell-bg|||components|||", selector: ".shell-bg", hotspot: true };
  assert.match(
    runDuplicateBaselineFixtureFailure(
      { version: 1, duplicateKeys: [baselineEntryFor(hotspotDuplicate)] },
      [hotspotDuplicate]
    ),
    /hotspot duplicate context .* is forbidden/
  );

  const conflictingDuplicate = { ...duplicate, category: "conflicting" };
  assert.match(
    runDuplicateBaselineFixtureFailure(
      {
        version: 1,
        duplicateKeys: [
          baselineEntryFor(conflictingDuplicate, {
            ownerReason: "Shared styling is intentionally retained.",
            declarationConflict: "conflicting",
            disposition: "keep-owned",
            removalStatus: "permanent"
          })
        ]
      },
      [conflictingDuplicate]
    ),
    /must explain intended cascade\/source-order behavior/
  );
});

test("portal CSS Phase 10 additive fixtures classify safe and deferred candidates", () => {
  const safeDuplicate = phase10AdditiveDuplicate();
  const shorthandDuplicate = phase10AdditiveDuplicate({
    key: ".shorthand|||components|||",
    selector: ".shorthand",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 1,
        column: 1,
        layer: "components",
        selectorList: [".shorthand"],
        declarations: [["background-color", "white", false]],
        declarationSignature: "singleton",
        properties: ["background-color"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 4,
        column: 1,
        layer: "components",
        selectorList: [".secondary", ".shorthand"],
        declarations: [["background", "black", false]],
        declarationSignature: "list",
        properties: ["background"]
      }
    ]
  });
  const customPropertyDuplicate = phase10AdditiveDuplicate({
    key: ".custom-prop|||components|||",
    selector: ".custom-prop",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 1,
        column: 1,
        layer: "components",
        selectorList: [".custom-prop"],
        declarations: [["--portal-offset", "1rem", false]],
        declarationSignature: "singleton",
        properties: ["--portal-offset"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 4,
        column: 1,
        layer: "components",
        selectorList: [".secondary", ".custom-prop"],
        declarations: [["transform", "translateX(var(--portal-offset))", false]],
        declarationSignature: "list",
        properties: ["transform"]
      }
    ]
  });
  const sourceOrderDuplicate = phase10AdditiveDuplicate({
    key: ".source-order|||components|||",
    selector: ".source-order",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 10,
        column: 1,
        layer: "components",
        selectorList: [".source-order"],
        declarations: [["outline", "none", false]],
        declarationSignature: "singleton",
        properties: ["outline"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 4,
        column: 1,
        layer: "components",
        selectorList: [".source-order", ".source-order:focus-visible"],
        declarations: [["color", "red", false]],
        declarationSignature: "list",
        properties: ["color"]
      }
    ]
  });
  const selectorListDuplicate = phase10AdditiveDuplicate({
    key: ".ambiguous|||components|||",
    selector: ".ambiguous",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 1,
        column: 1,
        layer: "components",
        selectorList: [".ambiguous"],
        declarations: [["outline", "none", false]],
        declarationSignature: "left",
        properties: ["outline"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/example.css",
        line: 4,
        column: 1,
        layer: "components",
        selectorList: [".ambiguous"],
        declarations: [["color", "red", false]],
        declarationSignature: "right",
        properties: ["color"]
      }
    ]
  });
  const hotspotDuplicate = phase10AdditiveDuplicate({
    key: ".shell-bg|||components|||",
    selector: ".shell-bg",
    hotspot: true
  });

  assert.match(
    runPhase10AdditiveFixture({
      duplicates: [
        safeDuplicate,
        shorthandDuplicate,
        customPropertyDuplicate,
        sourceOrderDuplicate,
        selectorListDuplicate,
        hotspotDuplicate
      ],
      expectedCandidates: [
        { key: safeDuplicate.key, candidateStatus: "safe" },
        { key: shorthandDuplicate.key, candidateStatus: "deferred", unsafeReason: "shorthand-longhand-overlap" },
        { key: customPropertyDuplicate.key, candidateStatus: "deferred", unsafeReason: "custom-property-order-sensitive" },
        { key: sourceOrderDuplicate.key, candidateStatus: "deferred", unsafeReason: "source-order-sensitive" },
        { key: selectorListDuplicate.key, candidateStatus: "deferred", unsafeReason: "selector-list-ambiguous" },
        { key: hotspotDuplicate.key, candidateStatus: "deferred", unsafeReason: "hotspot" }
      ]
    }),
    /portal css phase10 additive fixture: OK/
  );

  assert.match(
    runPhase10AdditiveFixtureFailure({
      duplicates: [safeDuplicate],
      expectedState: { phase: "phase-10-css-additive-duplicate-consolidation" },
      phase10AdditiveConsolidationState: { phase: "stale" }
    }),
    /phase10AdditiveConsolidationState is stale/
  );
});

test("portal CSS Phase 11 surface fixtures constrain selector-list consolidation", () => {
  const surfaceChrome = phase11SurfaceDuplicate();
  const reviewTone = phase11SurfaceDuplicate({
    key: ".review-status-banner[data-tone=\"ready\"]|||components|||",
    selector: ".review-status-banner[data-tone=\"ready\"]",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 262,
        column: 1,
        layer: "components",
        selectorList: [
          ".console-action-rail[data-tone=\"ready\"]",
          ".review-status-banner[data-tone=\"ready\"]"
        ],
        declarations: [["background", "var(--ux-surface-overlay)", false]],
        declarationSignature: "tone-bg",
        properties: ["background"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 270,
        column: 1,
        layer: "components",
        selectorList: [
          ".console-action-rail[data-tone=\"ready\"]",
          ".review-status-banner[data-tone=\"ready\"]"
        ],
        declarations: [["border-color", "rgba(15, 118, 110, 0.24)", false]],
        declarationSignature: "tone-border",
        properties: ["border-color"]
      }
    ]
  });
  const outOfScope = phase11SurfaceDuplicate({
    key: ".workspace-shell|||components|||",
    selector: ".workspace-shell"
  });
  const backgroundOverlap = phase11SurfaceDuplicate({
    selector: "#artifactMetadataBar",
    key: "#artifactMetadataBar|||components|||",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 146,
        column: 1,
        layer: "components",
        selectorList: ["#artifactMetadataBar"],
        declarations: [["background-color", "var(--ux-surface-elevated)", false]],
        declarationSignature: "background-color",
        properties: ["background-color"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 176,
        column: 1,
        layer: "components",
        selectorList: [".other-surface", "#artifactMetadataBar"],
        declarations: [["background", "var(--ux-surface-elevated)", false]],
        declarationSignature: "background",
        properties: ["background"]
      }
    ]
  });
  const interveningBackground = phase11SurfaceDuplicate({
    selector: "#artifactPreviewStage",
    key: "#artifactPreviewStage|||components|||",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 146,
        column: 1,
        layer: "components",
        selectorList: ["#artifactPreviewStage"],
        declarations: [["border-radius", "var(--ux-radius-lg)", false]],
        declarationSignature: "radius",
        properties: ["border-radius"],
        interveningBackgroundWrite: true
      },
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 176,
        column: 1,
        layer: "components",
        selectorList: [".other-surface", "#artifactPreviewStage"],
        declarations: [["background", "var(--ux-surface-elevated)", false]],
        declarationSignature: "background",
        properties: ["background"]
      }
    ]
  });
  const missingDarkPair = phase11SurfaceDuplicate({
    selector: "#artifactMetadataCard",
    key: "#artifactMetadataCard|||components|||",
    missingDarkPair: true
  });
  const specificityChanging = phase11SurfaceDuplicate({
    selector: "#reconstructionRuntimeSummary",
    key: "#reconstructionRuntimeSummary|||components|||",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 146,
        column: 1,
        layer: "components",
        ruleSelector: ":where(#reconstructionRuntimeSummary)",
        selectorList: ["#reconstructionRuntimeSummary"],
        declarations: [["border-radius", "var(--ux-radius-lg)", false]],
        declarationSignature: "radius",
        properties: ["border-radius"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/surface-normalization.css",
        line: 176,
        column: 1,
        layer: "components",
        selectorList: [".other-surface", "#reconstructionRuntimeSummary"],
        declarations: [["border-color", "var(--ux-border-subtle)", false]],
        declarationSignature: "chrome",
        properties: ["border-color"]
      }
    ]
  });
  const conflictingPermanent = phase11SurfaceDuplicate({
    key: ".review-status-banner[data-tone=\"error\"]|||components|||",
    selector: ".review-status-banner[data-tone=\"error\"]",
    category: "conflicting",
    removalStatus: "permanent"
  });

  assert.match(
    runPhase11SurfaceFixture({
      duplicates: [
        surfaceChrome,
        reviewTone,
        outOfScope,
        backgroundOverlap,
        interveningBackground,
        missingDarkPair,
        specificityChanging,
        conflictingPermanent
      ],
      expectedCandidates: [
        { key: surfaceChrome.key, candidateStatus: "safe" },
        { key: reviewTone.key, candidateStatus: "safe" },
        { key: outOfScope.key, candidateStatus: "deferred", unsafeReason: "selector-not-phase11-target" },
        {
          key: backgroundOverlap.key,
          candidateStatus: "deferred",
          unsafeReason: "background-shorthand-order-sensitive"
        },
        {
          key: interveningBackground.key,
          candidateStatus: "deferred",
          unsafeReason: "background-shorthand-order-sensitive"
        },
        { key: missingDarkPair.key, candidateStatus: "deferred", unsafeReason: "dark-pair-missing" },
        {
          key: specificityChanging.key,
          candidateStatus: "deferred",
          unsafeReason: "specificity-changing-grouping"
        },
        { key: conflictingPermanent.key, candidateStatus: "deferred", unsafeReason: "conflicting-permanent" }
      ]
    }),
    /portal css phase11 surface fixture: OK/
  );

  assert.match(
    runPhase11SurfaceFixtureFailure({
      duplicates: [surfaceChrome],
      expectedState: { phase: "phase-11-css-surface-list-consolidation" },
      phase11SurfaceListConsolidationState: { phase: "stale" }
    }),
    /phase11SurfaceListConsolidationState is stale/
  );
});

test("portal CSS Phase 12 component fixtures constrain singleton consolidation", () => {
  const ambientOrb = phase12ComponentDuplicate();
  const ambientRing = phase12ComponentDuplicate({
    key: ".ambient-ring|||components|||",
    selector: ".ambient-ring",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/shell-foundation.css",
        line: 322,
        column: 1,
        layer: "components",
        selectorList: [".ambient-veil", ".ambient-ring"],
        declarations: [["position", "absolute", false], ["inset", "auto", false]],
        declarationSignature: "ambient-base",
        properties: ["inset", "position"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/shell-foundation.css",
        line: 460,
        column: 1,
        layer: "components",
        selectorList: [".ambient-ring"],
        declarations: [["border", "1px solid rgba(255, 255, 255, 0.22)", false]],
        declarationSignature: "ambient-ring-singleton",
        properties: ["border"]
      }
    ]
  });
  const overviewActions = phase12ComponentDuplicate({
    key: ".overview-actions|||components|||@media (max-width: 767px)",
    selector: ".overview-actions",
    context: ["@media (max-width: 767px)"],
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 523,
        column: 5,
        layer: "components",
        selectorList: [".overview-actions", ".build-stepper-actions-inline"],
        declarations: [["justify-content", "stretch", false]],
        declarationSignature: "mobile-justify",
        properties: ["justify-content"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 528,
        column: 5,
        layer: "components",
        selectorList: [".overview-actions"],
        declarations: [["grid-template-columns", "1fr", false], ["width", "100%", false]],
        declarationSignature: "mobile-overview-grid",
        properties: ["grid-template-columns", "width"]
      }
    ]
  });
  const outOfScope = phase12ComponentDuplicate({
    key: ".review-status-banner|||components|||",
    selector: ".review-status-banner"
  });
  const conflictingPermanent = phase12ComponentDuplicate({
    key: ".ambient-sweep|||components|||",
    selector: ".ambient-sweep",
    category: "conflicting",
    removalStatus: "permanent"
  });
  const hotspot = phase12ComponentDuplicate({
    key: ".hero-action|||components|||",
    selector: ".hero-action",
    hotspot: true
  });
  const specificityChanging = phase12ComponentDuplicate({
    key: ".ambient-orb|||components|||",
    selector: ".ambient-orb",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/shell-foundation.css",
        line: 322,
        column: 1,
        layer: "components",
        ruleSelector: ":where(.ambient-orb)",
        selectorList: [".ambient-orb"],
        declarations: [["position", "absolute", false]],
        declarationSignature: "ambient-base",
        properties: ["position"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/shell-foundation.css",
        line: 417,
        column: 1,
        layer: "components",
        selectorList: [".ambient-orb"],
        declarations: [["filter", "blur(34px)", false]],
        declarationSignature: "ambient-singleton",
        properties: ["filter"]
      }
    ]
  });

  assert.match(
    runPhase12ComponentFixture({
      duplicates: [ambientOrb, ambientRing, overviewActions, outOfScope, conflictingPermanent, hotspot],
      expectedCandidates: [
        { key: ambientOrb.key, candidateStatus: "safe" },
        { key: ambientRing.key, candidateStatus: "safe" },
        { key: overviewActions.key, candidateStatus: "safe" },
        { key: outOfScope.key, candidateStatus: "deferred", unsafeReason: "selector-not-phase12-target" },
        { key: conflictingPermanent.key, candidateStatus: "deferred", unsafeReason: "conflicting-permanent" },
        { key: hotspot.key, candidateStatus: "deferred", unsafeReason: "hotspot" }
      ]
    }),
    /portal css phase12 component fixture: OK/
  );

  assert.match(
    runPhase12ComponentFixture({
      duplicates: [specificityChanging],
      expectedCandidates: [
        {
          key: specificityChanging.key,
          candidateStatus: "deferred",
          unsafeReason: "specificity-changing-grouping"
        }
      ]
    }),
    /portal css phase12 component fixture: OK/
  );

  assert.match(
    runPhase12ComponentFixtureFailure({
      duplicates: [ambientOrb],
      expectedState: { phase: "phase-12-component-singleton-consolidation" },
      phase12ComponentSingletonConsolidationState: { phase: "stale" }
    }),
    /phase12ComponentSingletonConsolidationState is stale/
  );
});

test("portal CSS Phase 13 interaction fixtures constrain hover outline consolidation", () => {
  const buildStepTab = phase13InteractionDuplicate();
  const dispatchToolButton = phase13InteractionDuplicate({
    key: ".dispatch-tool-btn:hover|||components|||",
    selector: ".dispatch-tool-btn:hover",
    source: "web/secure-landing/portal-src/styles/components/dispatch-surfaces.css",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/dispatch-surfaces.css",
        line: 64,
        column: 1,
        layer: "components",
        selectorList: [".dispatch-tool-btn:hover", ".dispatch-tool-btn:focus-visible"],
        declarations: [["border-color", "rgba(8, 145, 178, 0.32)", false], ["background", "rgba(255, 255, 255, 0.92)", false], ["color", "var(--shell-ink)", false], ["transform", "translateY(-1px)", false]],
        declarationSignature: "dispatch-hover-shared",
        properties: ["background", "border-color", "color", "transform"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/dispatch-surfaces.css",
        line: 71,
        column: 1,
        layer: "components",
        selectorList: [".dispatch-tool-btn:hover"],
        declarations: [["outline", "none", false]],
        declarationSignature: "hover-outline",
        properties: ["outline"]
      }
    ]
  });
  const workspaceLink = phase13InteractionDuplicate({
    key: ".workspace-link:hover|||components|||",
    selector: ".workspace-link:hover",
    source: "web/secure-landing/portal-src/styles/components/workspace-surfaces.css",
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/workspace-surfaces.css",
        line: 32,
        column: 1,
        layer: "components",
        selectorList: [".workspace-link:hover", ".workspace-link:focus-visible"],
        declarations: [["transform", "translateY(-1px)", false], ["border-color", "rgba(8, 145, 178, 0.28)", false], ["background", "rgba(255, 255, 255, 0.84)", false], ["box-shadow", "0 14px 28px rgba(14, 116, 144, 0.08)", false]],
        declarationSignature: "workspace-hover-shared",
        properties: ["background", "border-color", "box-shadow", "transform"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/workspace-surfaces.css",
        line: 39,
        column: 1,
        layer: "components",
        selectorList: [".workspace-link:hover"],
        declarations: [["outline", "none", false]],
        declarationSignature: "hover-outline",
        properties: ["outline"]
      }
    ]
  });
  const outOfScope = phase13InteractionDuplicate({
    key: ".review-status-banner|||components|||",
    selector: ".review-status-banner"
  });
  const missingFocusVisible = phase13InteractionDuplicate({
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 308,
        column: 1,
        layer: "components",
        selectorList: [".build-step-tab:hover"],
        declarations: [["transform", "translateY(-1px)", false]],
        declarationSignature: "hover-only",
        properties: ["transform"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 313,
        column: 1,
        layer: "components",
        selectorList: [".build-step-tab:hover"],
        declarations: [["outline", "none", false]],
        declarationSignature: "hover-outline",
        properties: ["outline"]
      }
    ]
  });
  const missingOutline = phase13InteractionDuplicate({
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 308,
        column: 1,
        layer: "components",
        selectorList: [".build-step-tab:hover", ".build-step-tab:focus-visible"],
        declarations: [["transform", "translateY(-1px)", false]],
        declarationSignature: "shared",
        properties: ["transform"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 313,
        column: 1,
        layer: "components",
        selectorList: [".build-step-tab:hover"],
        declarations: [["color", "red", false]],
        declarationSignature: "not-outline",
        properties: ["color"]
      }
    ]
  });
  const specificityChanging = phase13InteractionDuplicate({
    records: [
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 308,
        column: 1,
        layer: "components",
        ruleSelector: ":where(.build-step-tab:hover, .build-step-tab:focus-visible)",
        selectorList: [".build-step-tab:hover", ".build-step-tab:focus-visible"],
        declarations: [["transform", "translateY(-1px)", false]],
        declarationSignature: "shared",
        properties: ["transform"]
      },
      {
        source: "web/secure-landing/portal-src/styles/components/operator-console.css",
        line: 313,
        column: 1,
        layer: "components",
        selectorList: [".build-step-tab:hover"],
        declarations: [["outline", "none", false]],
        declarationSignature: "hover-outline",
        properties: ["outline"]
      }
    ]
  });
  const conflictingPermanent = phase13InteractionDuplicate({
    key: ".build-step-tab:hover|||components|||",
    category: "conflicting",
    removalStatus: "permanent"
  });
  const hotspot = phase13InteractionDuplicate({
    key: ".shell-bg:hover|||components|||",
    selector: ".shell-bg:hover",
    hotspot: true
  });

  assert.match(
    runPhase13InteractionFixture({
      duplicates: [
        buildStepTab,
        dispatchToolButton,
        workspaceLink,
        outOfScope,
        hotspot
      ],
      baselineEntries: [baselineEntryFor(outOfScope)],
      expectedCandidates: [
        { key: buildStepTab.key, candidateStatus: "safe" },
        { key: dispatchToolButton.key, candidateStatus: "safe" },
        { key: workspaceLink.key, candidateStatus: "safe" },
        { key: outOfScope.key, candidateStatus: "deferred", unsafeReason: "selector-not-phase13-target" },
        { key: hotspot.key, candidateStatus: "deferred", unsafeReason: "hotspot" }
      ]
    }),
    /portal css phase13 interaction fixture: OK/
  );

  for (const [duplicate, unsafeReason] of [
    [missingFocusVisible, "focus-visible-coverage-missing"],
    [missingOutline, "missing-hover-outline"],
    [specificityChanging, "specificity-changing-grouping"],
    [conflictingPermanent, "conflicting-permanent"]
  ]) {
    assert.match(
      runPhase13InteractionFixture({
        duplicates: [duplicate],
        expectedCandidates: [{ key: duplicate.key, candidateStatus: "deferred", unsafeReason }]
      }),
      /portal css phase13 interaction fixture: OK/
    );
  }

  assert.match(
    runPhase13InteractionFixtureFailure({
      duplicates: [buildStepTab],
      expectedState: { phase: "phase-13-interaction-outline-consolidation" },
      phase13InteractionOutlineConsolidationState: { phase: "stale" }
    }),
    /phase13InteractionOutlineConsolidationState is stale/
  );

  assert.match(
    runPhase13InteractionFixtureFailure({
      duplicates: [buildStepTab],
      expectedPhase12State: { phase: "phase-12-component-singleton-consolidation" },
      phase12ComponentSingletonConsolidationState: { phase: "stale" }
    }),
    /phase12ComponentSingletonConsolidationState immutable historical evidence drifted/
  );

  assert.match(
    runPhase13InteractionFixtureFailure({
      duplicates: [outOfScope],
      baselineEntries: [
        baselineEntryFor(outOfScope, {
          ownerReason: "selector-not-phase13-target"
        })
      ]
    }),
    /selector-not-phase13-target must not overwrite live baseline ownerReason/
  );
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
    const entry = duplicateKeys.find((candidate) => candidate.selector === selector);
    assert.equal(entry, undefined, `unexpected duplicate baseline entry for ${selector}`);
  }
});

test("portal CSS Phase 11 consolidation preserves explicit surface boundaries", () => {
  const surfaceNormalization = readFileSync(
    path.join(FRONTDOOR_ROOT, "portal-src", "styles", "components", "surface-normalization.css"),
    "utf8"
  );
  const root = postcss.parse(surfaceNormalization);

  const topbarRule = findRuleBySelectors(root, [".portal-topbar", ".portal-context-shell"]);
  assert.ok(topbarRule, "portal topbar/context shell radius rule must exist");
  const topbarDeclarations = declarationsForRule(topbarRule);
  assert.equal(topbarDeclarations.get("border-radius"), "var(--ux-radius-lg)");
  assert.equal(topbarDeclarations.has("background"), false);
  assert.equal(topbarDeclarations.has("border-color"), false);
  assert.doesNotMatch(surfaceNormalization, /:(?:is|where)\(/);

  for (const [tone, borderColor] of [
    ["ready", "rgba(15, 118, 110, 0.24)"],
    ["warning", "rgba(180, 83, 9, 0.26)"],
    ["error", "rgba(185, 28, 28, 0.26)"],
    ["info", "rgba(8, 145, 178, 0.24)"]
  ]) {
    const toneRule = findRuleBySelectors(root, [`.review-status-banner[data-tone="${tone}"]`]);
    assert.ok(toneRule, `review status ${tone} singleton rule must exist`);
    const toneDeclarations = declarationsForRule(toneRule);
    assert.equal(toneDeclarations.get("background"), "var(--ux-surface-overlay)");
    assert.equal(toneDeclarations.get("border-color"), borderColor);
  }

  const consoleToneRule = findRuleBySelectors(root, [
    ".console-action-rail[data-tone=\"ready\"]",
    ".console-action-rail[data-tone=\"warning\"]",
    ".console-action-rail[data-tone=\"blocked\"]"
  ]);
  assert.ok(consoleToneRule, "console action rail shared tone background rule must exist");
  assert.equal(declarationsForRule(consoleToneRule).get("background"), "var(--ux-surface-overlay)");
});

test("portal CSS Phase 12 consolidation preserves ambient and mobile boundaries", () => {
  const shellFoundation = readFileSync(
    path.join(FRONTDOOR_ROOT, "portal-src", "styles", "components", "shell-foundation.css"),
    "utf8"
  );
  const shellRoot = postcss.parse(shellFoundation);
  const ambientBaseRule = findRuleBySelectors(shellRoot, [
    ".ambient-veil",
    ".ambient-grid",
    ".ambient-sweep",
    ".ambient-focus"
  ]);
  assert.ok(ambientBaseRule, "shared ambient positioning rule must remain for non-singleton ambient layers");
  assert.equal(declarationsForRule(ambientBaseRule).get("position"), "absolute");
  assert.equal(declarationsForRule(ambientBaseRule).get("inset"), "auto");
  assert.equal(findRuleBySelectors(shellRoot, [".ambient-veil", ".ambient-grid", ".ambient-sweep", ".ambient-focus", ".ambient-orb", ".ambient-ring"]), null);

  const orbRule = findRuleBySelectors(shellRoot, [".ambient-orb"]);
  const ringRule = findRuleBySelectors(shellRoot, [".ambient-ring"]);
  assert.ok(orbRule, "ambient orb singleton rule must exist");
  assert.ok(ringRule, "ambient ring singleton rule must exist");
  for (const rule of [orbRule, ringRule]) {
    const declarations = declarationsForRule(rule);
    assert.equal(declarations.get("position"), "absolute");
    assert.equal(declarations.get("inset"), "auto");
  }
  const willChangeRule = findRuleBySelectors(shellRoot, [
    ".is-animating-ambient .ambient-veil",
    ".is-animating-ambient .ambient-grid",
    ".is-animating-ambient .ambient-sweep",
    ".is-animating-ambient .ambient-focus",
    ".is-animating-ambient .ambient-orb",
    ".is-animating-ambient .ambient-ring"
  ]);
  assert.ok(willChangeRule, "ambient animation hint rule must keep orb and ring coverage");
  assert.equal(declarationsForRule(willChangeRule).get("will-change"), "transform, opacity");
  assert.ok((orbRule.source?.start?.line || 0) < (findRuleBySelectors(shellRoot, [".ambient-orb--a"])?.source?.start?.line || 0));
  assert.ok((ringRule.source?.start?.line || 0) < (findRuleBySelectors(shellRoot, [".ambient-ring--a"])?.source?.start?.line || 0));

  const operatorConsole = readFileSync(
    path.join(FRONTDOOR_ROOT, "portal-src", "styles", "components", "operator-console.css"),
    "utf8"
  );
  const operatorRoot = postcss.parse(operatorConsole);
  let mobileRoot = null;
  operatorRoot.walkAtRules("media", (atRule) => {
    if (atRule.params === "(max-width: 767px)") {
      mobileRoot = atRule;
    }
  });
  assert.ok(mobileRoot, "mobile operator-console media rule must exist");
  const buildStepperRule = findRuleBySelectors(mobileRoot, [".build-stepper-actions-inline"]);
  const overviewRule = findRuleBySelectors(mobileRoot, [".overview-actions"]);
  assert.ok(buildStepperRule, "build-stepper mobile rule must stay isolated");
  assert.ok(overviewRule, "overview mobile singleton rule must exist");
  assert.equal(findRuleBySelectors(mobileRoot, [".overview-actions", ".build-stepper-actions-inline"]), null);
  assert.equal(declarationsForRule(buildStepperRule).get("justify-content"), "stretch");
  const overviewDeclarations = declarationsForRule(overviewRule);
  assert.equal(overviewDeclarations.get("justify-content"), "stretch");
  assert.equal(overviewDeclarations.get("grid-template-columns"), "1fr");
  assert.equal(overviewDeclarations.get("width"), "100%");
});

test("portal CSS Phase 13 consolidation preserves interaction outline boundaries", () => {
  for (const [fileName, selector, sharedDeclarations] of [
    [
      "operator-console.css",
      ".build-step-tab",
      {
        transform: "translateY(-1px)",
        "border-color": "rgba(8, 145, 178, 0.28)"
      }
    ],
    [
      "dispatch-surfaces.css",
      ".dispatch-tool-btn",
      {
        "border-color": "rgba(8, 145, 178, 0.32)",
        background: "rgba(255, 255, 255, 0.92)",
        color: "var(--shell-ink)",
        transform: "translateY(-1px)"
      }
    ],
    [
      "workspace-surfaces.css",
      ".workspace-link",
      {
        transform: "translateY(-1px)",
        "border-color": "rgba(8, 145, 178, 0.28)",
        background: "rgba(255, 255, 255, 0.84)",
        "box-shadow": "0 14px 28px rgba(14, 116, 144, 0.08)"
      }
    ]
  ]) {
    const source = readFileSync(
      path.join(FRONTDOOR_ROOT, "portal-src", "styles", "components", fileName),
      "utf8"
    );
    const root = postcss.parse(source);
    const hoverRule = findRuleBySelectors(root, [`${selector}:hover`]);
    const focusVisibleRule = findRuleBySelectors(root, [`${selector}:focus-visible`]);
    assert.ok(hoverRule, `${selector}:hover singleton rule must exist`);
    assert.ok(focusVisibleRule, `${selector}:focus-visible singleton rule must exist`);
    assert.equal(findRuleBySelectors(root, [`${selector}:hover`, `${selector}:focus-visible`]), null);
    assert.doesNotMatch(source, /:(?:is|where)\(/);

    const hoverDeclarations = declarationsForRule(hoverRule);
    const focusVisibleDeclarations = declarationsForRule(focusVisibleRule);
    for (const [property, value] of Object.entries(sharedDeclarations)) {
      assert.equal(hoverDeclarations.get(property), value);
      assert.equal(focusVisibleDeclarations.get(property), value);
    }
    assert.equal(hoverDeclarations.get("outline"), "none");
    assert.equal(focusVisibleDeclarations.has("outline"), false);
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

  const duplicateBaseline = JSON.parse(readFileSync(ARCHITECTURE_BASELINE_PATH, "utf8"));
  assert.equal(ownershipDrain.phase9DuplicateState.phase, "phase-9-duplicate-ownership-closure");
  assert.equal(ownershipDrain.phase9DuplicateState.baselinePath, "web/secure-landing/portal-src/styles/architecture-baseline.json");
  assert.equal(ownershipDrain.phase9DuplicateState.duplicateContextCountBefore, 87);
  assert.equal(ownershipDrain.phase9DuplicateState.duplicateContextCountAfter, 85);
  assert.equal(ownershipDrain.phase9DuplicateState.ownedDuplicateContextCount, 85);
  assert.equal(ownershipDrain.phase9DuplicateState.unownedDuplicateContextCount, 0);
  assert.equal(ownershipDrain.phase9DuplicateState.hotspotDuplicateContextCount, 0);
  assert.equal(ownershipDrain.phase9DuplicateState.consolidatedDuplicateContextCount, 0);
  assert.equal(ownershipDrain.phase9DuplicateState.reclassifiedDuplicateContextCount, 2);
  assert.equal(ownershipDrain.phase9DuplicateState.removedRawBytes, 0);
  assert.equal(ownershipDrain.phase9DuplicateState.removedGzipBytes, 0);
  assert.equal(ownershipDrain.phase9DuplicateState.sentinelStatePreserved, true);
  assert.equal(ownershipDrain.phase9DuplicateState.parityBaselineChanged, false);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.phase, "phase-10-css-additive-duplicate-consolidation");
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.additiveDuplicateContextCountBefore, 30);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.additiveDuplicateContextCountAfter, 29);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.safeCandidateCountBefore, 1);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.safeCandidateCountAfter, 0);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.consolidatedCandidateCount, 1);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.deferredCandidateCount, 29);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.unownedDuplicateContextCount, 0);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.hotspotDuplicateContextCount, 0);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.sentinelStatePreserved, true);
  assert.equal(ownershipDrain.phase10AdditiveConsolidationState.parityBaselineChanged, false);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.phase, "phase-11-css-surface-list-consolidation");
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.targetFile, "web/secure-landing/portal-src/styles/components/surface-normalization.css");
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.baselineSha256, "ab9f4a92eb699d811a85d737ef3f5cb2b285e6c809afcf34bbf179e1e229d17a");
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.duplicateContextCountBefore, 85);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.duplicateContextCountAfter, 76);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.additiveDuplicateContextCountBefore, 29);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.additiveDuplicateContextCountAfter, 20);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.conflictingPermanentContextCountBefore, 56);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.conflictingPermanentContextCountAfter, 56);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.unownedDuplicateContextCountAfter, 0);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.hotspotDuplicateContextCountAfter, 0);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.consolidatedContextCount, 9);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.expectedConsolidatedContextCount, 9);
  assert.deepEqual(ownershipDrain.phase11SurfaceListConsolidationState.consolidatedContexts, [
    "#artifactMetadataBar|||components|||",
    "#artifactMetadataCard|||components|||",
    "#artifactPreviewStage|||components|||",
    "#reconstructionRuntimeSummary|||components|||",
    ".review-compare-summary|||components|||",
    ".review-status-banner[data-tone=\"error\"]|||components|||",
    ".review-status-banner[data-tone=\"info\"]|||components|||",
    ".review-status-banner[data-tone=\"ready\"]|||components|||",
    ".review-status-banner[data-tone=\"warning\"]|||components|||"
  ]);
  assert.deepEqual(ownershipDrain.phase11SurfaceListConsolidationState.remainingTargetContexts, []);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.unexpectedResolvedContextCount, 0);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.deferredOutOfScopeCandidateCount, 20);
  assert.deepEqual(ownershipDrain.phase11SurfaceListConsolidationState.deferredReasonCounts, {
    "selector-not-phase11-target": 20
  });
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.removedRawBytes, 246);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.removedGzipBytes, 9);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.generatedRawBytesBefore, 80599);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.generatedRawBytesAfter, 80353);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.generatedRawByteDelta, -246);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.generatedGzipBytesBefore, 15721);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.generatedGzipBytesAfter, 15712);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.generatedGzipByteDelta, -9);
  assert.equal(
    ownershipDrain.phase11SurfaceListConsolidationState.generatedPortalCssHashBefore,
    "fce12e29f1800375b5c34e1f0e1ebc9d3981ab1a6f731bea6a3e0e0d2212151e"
  );
  assert.equal(
    ownershipDrain.phase11SurfaceListConsolidationState.generatedPortalCssHashAfter,
    "67a600aafdfc066a8adf373e1323b287939977ebc306d4e1c0635069fddbbd87"
  );
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.renderedPortalCssFingerprintBefore, "d72696ab972c");
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.renderedPortalCssFingerprintAfter, "8f492bf35cbe");
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.sentinelStatePreserved, true);
  assert.equal(ownershipDrain.phase11SurfaceListConsolidationState.parityBaselineChanged, false);
  assert.match(
    readFileSync(ARCHITECTURE_SCRIPT_PATH, "utf8"),
    /report\.phase11SurfaceListConsolidationState = expectedPhase11State;/
  );
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.phase, "phase-12-component-singleton-consolidation");
  assert.deepEqual(ownershipDrain.phase12ComponentSingletonConsolidationState.targetSelectors, [
    ".ambient-orb",
    ".ambient-ring",
    ".overview-actions @media (max-width: 767px)"
  ]);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.duplicateContextCountBefore, 76);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.duplicateContextCountAfter, 73);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.additiveDuplicateContextCountBefore, 20);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.additiveDuplicateContextCountAfter, 17);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.conflictingPermanentContextCountBefore, 56);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.conflictingPermanentContextCountAfter, 56);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.unownedDuplicateContextCountAfter, 0);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.hotspotDuplicateContextCountAfter, 0);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.consolidatedContextCount, 3);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.expectedConsolidatedContextCount, 3);
  assert.deepEqual(ownershipDrain.phase12ComponentSingletonConsolidationState.remainingTargetContexts, []);
  assert.equal(
    ownershipDrain.phase12ComponentSingletonConsolidationState.nonTargetAdditiveCandidatesDeferredReason,
    "selector-not-phase12-target"
  );
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.sentinelStatePreserved, true);
  assert.equal(ownershipDrain.phase12ComponentSingletonConsolidationState.parityBaselineChanged, false);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.phase, "phase-13-interaction-outline-consolidation");
  assert.deepEqual(ownershipDrain.phase13InteractionOutlineConsolidationState.targetSelectors, [
    ".build-step-tab:hover",
    ".dispatch-tool-btn:hover",
    ".workspace-link:hover"
  ]);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.duplicateContextCountBefore, 73);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.duplicateContextCountAfter, 70);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.additiveDuplicateContextCountBefore, 17);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.additiveDuplicateContextCountAfter, 14);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.conflictingPermanentContextCountBefore, 56);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.conflictingPermanentContextCountAfter, 56);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.unownedDuplicateContextCountAfter, 0);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.hotspotDuplicateContextCountAfter, 0);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.consolidatedContextCount, 3);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.expectedConsolidatedContextCount, 3);
  assert.deepEqual(ownershipDrain.phase13InteractionOutlineConsolidationState.consolidatedContexts, [
    ".build-step-tab:hover|||components|||",
    ".dispatch-tool-btn:hover|||components|||",
    ".workspace-link:hover|||components|||"
  ]);
  assert.deepEqual(ownershipDrain.phase13InteractionOutlineConsolidationState.remainingTargetContexts, []);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.unexpectedResolvedContextCount, 0);
  assert.equal(
    ownershipDrain.phase13InteractionOutlineConsolidationState.nonTargetAdditiveCandidatesDeferredReason,
    "selector-not-phase13-target"
  );
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.deferredOutOfScopeCandidateCount, 14);
  assert.deepEqual(ownershipDrain.phase13InteractionOutlineConsolidationState.deferredReasonCounts, {
    "selector-not-phase13-target": 14
  });
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.phase12HistoricalEvidencePreserved, true);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.generatedRawBytesBefore, 80390);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.generatedRawBytesAfter, 80569);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.generatedRawByteDelta, 179);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.generatedGzipBytesBefore, 15713);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.generatedGzipBytesAfter, 15743);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.generatedGzipByteDelta, 30);
  assert.equal(
    ownershipDrain.phase13InteractionOutlineConsolidationState.generatedPortalCssHashBefore,
    "5c82d054b928ac320c932cd9eb133fb2f39cfdc9e865c648cc9987de68c0e37b"
  );
  assert.equal(
    ownershipDrain.phase13InteractionOutlineConsolidationState.generatedPortalCssHashAfter,
    "3e6719511b1536ebe2e4c067b693930aa77ab242115368ae4ba3aa8e2aa6f3b3"
  );
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.renderedPortalCssFingerprintBefore, "61c134a0012d");
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.renderedPortalCssFingerprintAfter, "c06fbcdf1f37");
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.sentinelStatePreserved, true);
  assert.equal(ownershipDrain.phase13InteractionOutlineConsolidationState.parityBaselineChanged, false);
  assert.equal(duplicateBaseline.duplicateKeys.length, 70);
  assert.ok(
    duplicateBaseline.duplicateKeys.every((entry) => entry.phase === "phase-9-duplicate-ownership-closure"),
    "all duplicate baseline entries must be Phase 9-owned"
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
  assert.match(validator, /REVIEW_STATUS_TONES = \("ready", "warning", "error", "info"\)/);
  assert.match(validator, /document\.getElementById\('reviewStatusBanner'\)/);
  assert.match(validator, /rootClassSnapshot/);
  assert.match(validator, /storageSnapshot/);
  assert.match(validator, /bannerSnapshot/);
  assert.match(validator, /current\.classList\.remove\('hidden'\)/);
  assert.match(validator, /banner\.dataset\.tone = tone/);
  assert.match(validator, /finally \{\{\n    restore\(\);/);
  assert.match(validator, /_validate_review_status_tone_states\(connection\)/);
  assert.match(validator, /_validate_overview_mobile_states\(connection\)/);
  assert.match(validator, /_validate_interaction_outline_states\(connection\)/);
  assert.match(validator, /CSS\.forcePseudoState/);
  assert.match(validator, /':hover'/);
  assert.match(validator, /':focus-visible'/);
  assert.match(validator, /dispatchTools\.open = true/);
  assert.match(validator, /"width": 767/);
  assert.match(validator, /"width": 375/);
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
