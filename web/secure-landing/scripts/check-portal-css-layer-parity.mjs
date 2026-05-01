import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import postcss from "postcss";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "index.css");
const PORTAL_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.css");
const LAYER_PARITY_CONTRACT_PATH = path.resolve(REPO_ROOT, "tests", "fixtures", "portal-css", "layer-parity-contract.json");
const EXPECTED_LAYER_ORDER = ["tokens", "base", "components", "utilities", "overrides"];
const EXPECTED_LAYER_IMPORTS = [
  ["../../../../web/shared/shared-ui-tokens.css", "tokens"],
  ["./tokens.css", "tokens"],
  ["./base.css", "base"],
  ["./components/status-feedback.css", "components"],
  ["./components/shell-foundation.css", "components"],
  ["./components/console-context.css", "components"],
  ["./components/dispatch-surfaces.css", "components"],
  ["./components/responsive-layout.css", "components"],
  ["./components/workspace-surfaces.css", "components"],
  ["./components/operator-console.css", "components"],
  ["./components/surface-normalization.css", "components"],
  ["./utilities.required.css", "utilities"],
  ["./utilities.dynamic.css", "utilities"],
  ["./utilities.compat-hold.css", "utilities"],
  ["./overrides.compat.css", "overrides"],
  ["./overrides.performance.css", "overrides"],
  ["./overrides.accessibility.css", "overrides"]
];
const FORBIDDEN_TRANSITIONAL_IMPORTS = [
  "./components.current.css",
  "./components/operator-console-reset.css",
  "./overrides.current.css",
  "./overrides.operator-console-reset.css",
  "./utilities.compat.css",
  "./utilities.deprecated.css",
  "./utilities.operator-console-reset.css",
  "./components/workspace-performance.css"
];
const FONT_PLACEHOLDERS = ["__PORTAL_FONT_SANS_URL__", "__PORTAL_FONT_MONO_URL__"];
const layerParityContract = JSON.parse(readFileSync(LAYER_PARITY_CONTRACT_PATH, "utf-8"));
const REPRESENTATIVE_STYLE_SELECTORS = layerParityContract.representativeStyleSelectors;
const REPRESENTATIVE_PROPERTIES = layerParityContract.representativeStyleProperties;
const writeCssIndex = process.argv.indexOf("--write-css");
const writeCssPath = writeCssIndex >= 0 ? process.argv[writeCssIndex + 1] : "";

if (writeCssIndex >= 0 && !writeCssPath) {
  console.error("ERROR: --write-css requires an output path");
  process.exit(1);
}

function relativePath(filePath) {
  return path.relative(REPO_ROOT, filePath);
}

function parseCss(label, content) {
  try {
    return postcss.parse(content, { from: label });
  } catch (error) {
    throw new Error(`${label} failed to parse as CSS: ${error.message}`);
  }
}

function atRuleContext(node) {
  const contexts = [];
  let current = node.parent;
  while (current) {
    if (current.type === "atrule") {
      contexts.unshift(`@${current.name} ${current.params || ""}`.trim());
    }
    current = current.parent;
  }
  return contexts;
}

function layerDeclarationNames(params) {
  return params.split(",").map((entry) => entry.trim()).filter(Boolean);
}

function parseImport(params) {
  const match = params.match(/^url\((["']?)(.+?)\1\)|^(["'])(.+?)\3/);
  const source = match ? (match[2] || match[4] || "").trim() : "";
  const layerMatch = params.match(/\blayer\(\s*([^)]+?)\s*\)/);
  return [source, layerMatch ? layerMatch[1].trim() : ""];
}

function validateSourceIndex(failures) {
  const content = readFileSync(PORTAL_CSS_INDEX_PATH, "utf-8");
  for (const forbidden of FORBIDDEN_TRANSITIONAL_IMPORTS) {
    if (content.includes(forbidden)) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} still imports transitional source ${forbidden}`);
    }
  }

  const root = parseCss(relativePath(PORTAL_CSS_INDEX_PATH), content);
  const imports = [];
  let declaredLayerOrder = null;
  root.each((node) => {
    if (node.type === "comment") {
      return;
    }
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      declaredLayerOrder = layerDeclarationNames(node.params);
      return;
    }
    if (node.type === "atrule" && node.name === "import") {
      imports.push(parseImport(node.params));
      return;
    }
    failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} contains non-import production CSS outside the layer graph`);
  });

  if (JSON.stringify(declaredLayerOrder) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
    failures.push(
      `${relativePath(PORTAL_CSS_INDEX_PATH)} declares unexpected layer order ${(declaredLayerOrder || []).join(", ")}`
    );
  }
  if (JSON.stringify(imports) !== JSON.stringify(EXPECTED_LAYER_IMPORTS)) {
    failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} layer import order drifted`);
  }
}

function validateGeneratedCss(failures) {
  const css = readFileSync(PORTAL_CSS_ASSET_PATH, "utf-8");
  if (css.includes("@import")) {
    failures.push(`${relativePath(PORTAL_CSS_ASSET_PATH)} contains a runtime @import`);
  }
  if (css.includes("__PORTAL_SHARED_TOKENS_URL__")) {
    failures.push(`${relativePath(PORTAL_CSS_ASSET_PATH)} contains the shared token URL placeholder`);
  }
  for (const placeholder of FONT_PLACEHOLDERS) {
    if (!css.includes(placeholder)) {
      failures.push(`${relativePath(PORTAL_CSS_ASSET_PATH)} is missing ${placeholder}`);
    }
  }
  if (!css.includes("--ux-target-min-size")) {
    failures.push(`${relativePath(PORTAL_CSS_ASSET_PATH)} does not contain inlined shared --ux-* tokens`);
  }

  const root = parseCss(relativePath(PORTAL_CSS_ASSET_PATH), css);
  let declaredLayerOrder = null;
  const keyframeNames = new Set();
  root.each((node) => {
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      const names = layerDeclarationNames(node.params);
      if (names.length > 1) {
        declaredLayerOrder = names;
      }
    }
  });
  root.walkAtRules("keyframes", (node) => {
    if (keyframeNames.has(node.params)) {
      failures.push(`${relativePath(PORTAL_CSS_ASSET_PATH)} duplicates @keyframes ${node.params}`);
    }
    keyframeNames.add(node.params);
  });
  if (JSON.stringify(declaredLayerOrder) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
    failures.push(
      `${relativePath(PORTAL_CSS_ASSET_PATH)} declares unexpected layer order ${(declaredLayerOrder || []).join(", ")}`
    );
  }

  root.walkRules((rule) => {
    const contexts = atRuleContext(rule);
    if (contexts.some((context) => context.startsWith("@keyframes "))) {
      return;
    }
    if (!contexts.some((context) => context.startsWith("@layer "))) {
      failures.push(`${relativePath(PORTAL_CSS_ASSET_PATH)} has unlayered ordinary selector rule ${rule.selector}`);
    }
  });

  return css;
}

const failures = [];
validateSourceIndex(failures);
const generatedCss = validateGeneratedCss(failures);

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`ERROR: ${failure}`);
  }
  process.exit(1);
}

if (writeCssPath) {
  writeFileSync(writeCssPath, generatedCss, "utf-8");
}

console.log(
  `portal css layer parity: OK (${REPRESENTATIVE_STYLE_SELECTORS.length} representative selectors, ${REPRESENTATIVE_PROPERTIES.length} style properties tracked for browser parity)`
);
