import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build, transform } from "esbuild";
import postcss from "postcss";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_FONT_TEMPLATE_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "fonts.template.css");
const LAYER_PARITY_CONTRACT_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "layer-parity-contract.json");
const EXPECTED_LAYER_ORDER = ["tokens", "base", "components", "utilities", "overrides"];
const COMPONENT_IMPORTS = [
  "./components/status-feedback.css",
  "./components/shell-foundation.css",
  "./components/console-context.css",
  "./components/dispatch-surfaces.css",
  "./components/responsive-layout.css"
];
// These files still encode current compatibility cascade behavior against
// utility classes. Keep them in the utilities layer during the dry run until a
// later PR splits true component ownership from final-order compatibility.
const TRANSITIONAL_UTILITY_IMPORTS = [
  "./components/workspace-performance.css",
  "./components/operator-console-reset.css",
  "./utilities.compat.css",
  "./overrides.current.css"
];
const FONT_PLACEHOLDERS = ["__PORTAL_FONT_SANS_URL__", "__PORTAL_FONT_MONO_URL__"];
const layerParityContract = JSON.parse(readFileSync(LAYER_PARITY_CONTRACT_PATH, "utf-8"));
const REPRESENTATIVE_STYLE_SELECTORS = layerParityContract.representativeStyleSelectors;
const REPRESENTATIVE_PROPERTIES = layerParityContract.representativeStyleProperties;

const writeCssIndex = process.argv.indexOf("--write-css");
const writeCssPath = writeCssIndex >= 0 ? process.argv[writeCssIndex + 1] : "";
if (writeCssIndex >= 0 && !writeCssPath) {
  throw new Error("--write-css requires an output path");
}

async function minifyCssText(content) {
  return (await transform(content, {
    loader: "css",
    legalComments: "none",
    minify: true
  })).code.trim() + "\n";
}

async function bundleLayeredCssBody() {
  const layeredIndex = [
    `@layer ${EXPECTED_LAYER_ORDER.join(", ")};`,
    `@import "../../../../web/shared/shared-ui-tokens.css" layer(tokens);`,
    `@import "./tokens.css" layer(tokens);`,
    `@import "./base.css" layer(base);`,
    ...COMPONENT_IMPORTS.map((sourcePath) => `@import "${sourcePath}" layer(components);`),
    ...TRANSITIONAL_UTILITY_IMPORTS.map((sourcePath) => `@import "${sourcePath}" layer(utilities);`)
  ].join("\n");

  const result = await build({
    absWorkingDir: REPO_ROOT,
    bundle: true,
    legalComments: "none",
    minify: true,
    stdin: {
      contents: layeredIndex,
      loader: "css",
      resolveDir: PORTAL_CSS_SOURCE_DIR,
      sourcefile: "portal-layer-dry-run.css"
    },
    write: false
  });
  const outputText = result.outputFiles?.[0]?.text;
  if (!outputText) {
    throw new Error("esbuild did not emit layered portal CSS dry-run output");
  }
  return outputText.trim() + "\n";
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

function validateLayeredCss(css) {
  const failures = [];
  if (css.includes("@import")) {
    failures.push("layered dry-run output contains a runtime @import");
  }
  if (css.includes("__PORTAL_SHARED_TOKENS_URL__")) {
    failures.push("layered dry-run output contains the shared token URL placeholder");
  }
  for (const placeholder of FONT_PLACEHOLDERS) {
    if (!css.includes(placeholder)) {
      failures.push(`layered dry-run output is missing ${placeholder}`);
    }
  }
  if (!css.includes("--ux-target-min-size")) {
    failures.push("layered dry-run output does not contain inlined shared --ux-* tokens");
  }

  const root = parseCss("portal-layer-dry-run.css", css);
  let declaredLayerOrder = null;
  root.each((node) => {
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      const names = layerDeclarationNames(node.params);
      if (names.length > 1) {
        declaredLayerOrder = names;
      }
    }
  });
  if (JSON.stringify(declaredLayerOrder) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
    failures.push(`layered dry-run output declares unexpected layer order: ${(declaredLayerOrder || []).join(", ")}`);
  }
  root.walkRules((rule) => {
    const contexts = atRuleContext(rule);
    if (contexts.some((context) => context.startsWith("@keyframes "))) {
      return;
    }
    if (!contexts.some((context) => context.startsWith("@layer "))) {
      failures.push(`layered dry-run output has unlayered ordinary selector rule ${rule.selector}`);
    }
  });

  if (failures.length > 0) {
    for (const failure of failures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
}

const fontTemplate = await minifyCssText(readFileSync(PORTAL_CSS_FONT_TEMPLATE_PATH, "utf-8"));
const layeredCss = `${fontTemplate}${await bundleLayeredCssBody()}`;
validateLayeredCss(layeredCss);
if (writeCssPath) {
  writeFileSync(writeCssPath, layeredCss, "utf-8");
}

console.log(
  `portal css layer dry-run: OK (${layeredCss.length} bytes, ${REPRESENTATIVE_STYLE_SELECTORS.length} representative selectors, ${REPRESENTATIVE_PROPERTIES.length} style properties tracked for browser parity)`
);
