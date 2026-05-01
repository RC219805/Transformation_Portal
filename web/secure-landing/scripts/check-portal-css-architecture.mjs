import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";

import postcss from "postcss";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "index.css");
const PORTAL_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.css");
const PORTAL_REVIEW_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.css");
const PORTAL_HTML_PATH = path.resolve(REPO_ROOT, "portal.html");
const PORTAL_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src");
const PORTAL_ASSET_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.js");
const PORTAL_REVIEW_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.js");
const BASELINE_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "architecture-baseline.json");
const OWNERSHIP_DRAIN_REPORT_PATH = path.resolve(FRONTDOOR_ROOT, "reports", "portal-css-ownership-drain.json");

const args = new Set(process.argv.slice(2));
const WRITE_BASELINE = args.has("--write-baseline");
const REPORT = args.has("--report") || WRITE_BASELINE;

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
const HOTSPOT_SELECTORS = new Set([
  ".shell-bg",
  ".workspace-rail",
  ".workspace-link",
  ".hero-action",
  ".operator-action-btn",
  ".stepper-nav-btn",
  ".dispatch-tool-btn",
  ".console-context-ribbon",
  ".topbar-status"
]);

const UTILITY_EXACT_CLASSES = new Set([
  "absolute",
  "antialiased",
  "block",
  "border",
  "flex",
  "fixed",
  "grid",
  "group",
  "hidden",
  "inline",
  "inline-block",
  "inline-flex",
  "peer",
  "relative",
  "rounded",
  "sr-only",
  "sticky",
  "transform",
  "truncate",
  "uppercase"
]);
const UTILITY_OWNER_ALLOWLIST = new Map([
  ["dark", "theme owner class"],
  ["light", "theme owner class"],
  ["performance-lite", "performance mode owner class"]
]);
const UTILITY_PREFIX_PATTERN =
  /^(?:-?m[trblxy]?-.+|p[trblxy]?-.+|space-[xy]-.+|gap-.+|grid-cols-.+|col-span-.+|row-span-.+|flex-.+|items-.+|justify-.+|self-.+|place-.+|w-.+|h-.+|min-w-.+|min-h-.+|max-w-.+|max-h-.+|rounded-.+|border-.+|bg-.+|from-.+|via-.+|to-.+|text-.+|font-.+|tracking-.+|leading-.+|shadow-.+|ring-.+|opacity-.+|overflow-.+|object-.+|inset-.+|top-.+|right-.+|bottom-.+|left-.+|-?z-.+|cursor-.+|pointer-events-.+|select-.+|resize-.+|whitespace-.+|break-.+|duration-.+|ease-.+|transition-.+|translate-.+|scale-.+|backdrop-.+|animate-.+|outline-.+|fill-.+|stroke-.+|order-.+|basis-.+|shrink-.+|grow-.+|underline|no-underline)$/;
const UTILITY_VARIANTS = new Set([
  "active",
  "dark",
  "disabled",
  "focus",
  "focus-visible",
  "group-hover",
  "hover",
  "lg",
  "md",
  "peer-checked",
  "peer-focus-visible",
  "selection",
  "sm",
  "xl"
]);

function relativePath(filePath) {
  return path.relative(REPO_ROOT, filePath);
}

function readText(filePath) {
  return readFileSync(filePath, "utf-8");
}

function listFiles(directory, predicate) {
  const results = [];
  for (const entry of readdirSync(directory)) {
    const entryPath = path.join(directory, entry);
    const stats = statSync(entryPath);
    if (stats.isDirectory()) {
      results.push(...listFiles(entryPath, predicate));
    } else if (predicate(entryPath)) {
      results.push(entryPath);
    }
  }
  return results.sort();
}

function parseCss(filePath, content = readText(filePath)) {
  try {
    return postcss.parse(content, { from: filePath });
  } catch (error) {
    throw new Error(`${relativePath(filePath)} failed to parse as CSS: ${error.message}`);
  }
}

function splitSelectorList(selectorText) {
  const selectors = [];
  let current = "";
  let bracketDepth = 0;
  let parenDepth = 0;
  let quote = "";
  for (let index = 0; index < selectorText.length; index += 1) {
    const character = selectorText[index];
    const previous = selectorText[index - 1] || "";
    if (quote) {
      current += character;
      if (character === quote && previous !== "\\") {
        quote = "";
      }
      continue;
    }
    if (character === "\"" || character === "'") {
      quote = character;
      current += character;
      continue;
    }
    if (character === "[") {
      bracketDepth += 1;
    } else if (character === "]") {
      bracketDepth = Math.max(0, bracketDepth - 1);
    } else if (character === "(") {
      parenDepth += 1;
    } else if (character === ")") {
      parenDepth = Math.max(0, parenDepth - 1);
    } else if (character === "," && bracketDepth === 0 && parenDepth === 0) {
      const selector = current.trim().replace(/\s+/g, " ");
      if (selector) {
        selectors.push(selector);
      }
      current = "";
      continue;
    }
    current += character;
  }
  const trailing = current.trim().replace(/\s+/g, " ");
  if (trailing) {
    selectors.push(trailing);
  }
  return selectors;
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

function layerContext(node) {
  return atRuleContext(node).filter((context) => context.startsWith("@layer "));
}

function selectorStateContext(selector) {
  const states = [];
  if (selector.includes(".dark")) {
    states.push("dark");
  }
  for (const state of [":hover", ":focus", ":focus-visible", ":disabled", ".is-active", ".hidden"]) {
    if (selector.includes(state)) {
      states.push(state.replace(/^[.:]/, ""));
    }
  }
  if (selector.includes("[data-")) {
    states.push("data-state");
  }
  return states.sort();
}

function declarationEntries(rule) {
  const entries = [];
  rule.walkDecls((decl) => {
    entries.push([decl.prop.trim(), decl.value.trim(), Boolean(decl.important)]);
  });
  return entries.sort((left, right) => left[0].localeCompare(right[0]) || left[1].localeCompare(right[1]));
}

function declarationSignature(entries) {
  return crypto.createHash("sha1").update(JSON.stringify(entries)).digest("hex");
}

function declarationMap(entries) {
  const map = new Map();
  for (const [property, value, important] of entries) {
    const next = `${important ? "!" : ""}${value}`;
    const values = map.get(property) || new Set();
    values.add(next);
    map.set(property, values);
  }
  return map;
}

function classifyDuplicate(records) {
  const signatures = new Set(records.map((record) => record.declarationSignature));
  if (signatures.size === 1) {
    return "identical";
  }
  const seen = new Map();
  let conflict = false;
  for (const record of records) {
    const map = declarationMap(record.declarations);
    for (const [property, values] of map) {
      const current = seen.get(property) || new Set();
      for (const value of values) {
        current.add(value);
      }
      if (current.size > 1) {
        conflict = true;
      }
      seen.set(property, current);
    }
  }
  return conflict ? "conflicting" : "additive";
}

function collectRuleRecords(cssFiles) {
  const records = [];
  for (const filePath of cssFiles) {
    const root = parseCss(filePath);
    root.walkRules((rule) => {
      const context = atRuleContext(rule);
      for (const selector of splitSelectorList(rule.selector)) {
        const declarations = declarationEntries(rule);
        records.push({
          selector,
          context,
          layerContext: layerContext(rule),
          stateContext: selectorStateContext(selector),
          source: relativePath(filePath),
          line: rule.source?.start?.line || 0,
          declarations,
          declarationSignature: declarationSignature(declarations)
        });
      }
    });
  }
  return records;
}

function duplicateKey(record) {
  return `${record.selector}|||${record.context.join(" > ")}`;
}

function collectDuplicateReport(records) {
  const byKey = new Map();
  for (const record of records) {
    const key = duplicateKey(record);
    const existing = byKey.get(key) || [];
    existing.push(record);
    byKey.set(key, existing);
  }
  return Array.from(byKey.entries())
    .filter(([, recordsForKey]) => recordsForKey.length > 1)
    .map(([key, recordsForKey]) => ({
      key,
      selector: recordsForKey[0].selector,
      context: recordsForKey[0].context,
      stateContext: Array.from(new Set(recordsForKey.flatMap((record) => record.stateContext))).sort(),
      category: classifyDuplicate(recordsForKey),
      hotspot: HOTSPOT_SELECTORS.has(recordsForKey[0].selector),
      records: recordsForKey.map((record) => ({
        source: record.source,
        line: record.line,
        declarationSignature: record.declarationSignature,
        properties: Array.from(new Set(record.declarations.map(([property]) => property))).sort()
      }))
    }))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function loadBaseline() {
  if (!existsSync(BASELINE_PATH)) {
    return null;
  }
  return JSON.parse(readText(BASELINE_PATH));
}

function baselineFromReport(duplicates) {
  return {
    version: 1,
    duplicateKeys: duplicates.map((duplicate) => ({
      key: duplicate.key,
      category: duplicate.category,
      hotspot: duplicate.hotspot,
      owners: duplicate.records.some((record) =>
        record.source === "web/secure-landing/portal-src/styles/overrides.compat.css"
      )
        ? ["compatibility-final-order"]
        : duplicate.hotspot
          ? ["compatibility-final-order"]
          : []
    }))
  };
}

function checkDuplicateBaseline(duplicates, failures) {
  if (WRITE_BASELINE) {
    writeFileSync(BASELINE_PATH, `${JSON.stringify(baselineFromReport(duplicates), null, 2)}\n`, "utf-8");
    return;
  }
  const baseline = loadBaseline();
  if (!baseline) {
    failures.push(`${relativePath(BASELINE_PATH)} is missing. Run node ./scripts/check-portal-css-architecture.mjs --write-baseline after reviewing duplicate ownership.`);
    return;
  }
  const knownKeys = new Set((baseline.duplicateKeys || []).map((entry) => entry.key));
  const currentKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  for (const duplicate of duplicates) {
    if (!knownKeys.has(duplicate.key)) {
      failures.push(`new unclassified duplicate selector ${duplicate.key}`);
    }
  }
  for (const entry of baseline.duplicateKeys || []) {
    if (!currentKeys.has(entry.key)) {
      failures.push(`stale duplicate baseline entry ${entry.key}; refresh ${relativePath(BASELINE_PATH)}`);
    }
    if (entry.hotspot && (!Array.isArray(entry.owners) || entry.owners.length === 0)) {
      failures.push(`hotspot duplicate lacks owner classification ${entry.key}`);
    }
  }
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeCssClassToken(token) {
  return token.replace(/[^A-Za-z0-9_-]/g, (character) => `\\${character}`);
}

function classTokenBase(token) {
  const parts = token.split(":");
  while (parts.length > 1 && UTILITY_VARIANTS.has(parts[0])) {
    parts.shift();
  }
  return parts.join(":");
}

function isUtilityLikeClassToken(token) {
  if (!token || UTILITY_OWNER_ALLOWLIST.has(token)) {
    return false;
  }
  if (token.startsWith(":") || token.endsWith(":") || /[$"'`;=<>?{}()]/.test(token)) {
    return false;
  }
  const base = classTokenBase(token);
  return UTILITY_EXACT_CLASSES.has(base) || UTILITY_PREFIX_PATTERN.test(base);
}

function unescapeCssClassToken(token) {
  return token
    .replace(/\\([:/.[\]()%#,])/g, "$1")
    .replace(/\\([0-9a-fA-F]{1,6})\s?/g, (_match, value) => String.fromCodePoint(parseInt(value, 16)))
    .replace(/\\/g, "");
}

function classTokensFromSelector(selector) {
  const tokens = [];
  const pattern = /\.((?:\\.|[A-Za-z0-9_-])+)/g;
  for (const match of selector.matchAll(pattern)) {
    tokens.push(unescapeCssClassToken(match[1]));
  }
  return Array.from(new Set(tokens));
}

function recordClassToken(token, sourceLabel, classTokens) {
  const normalized = token.trim();
  if (!isUtilityLikeClassToken(normalized)) {
    return;
  }
  const sources = classTokens.get(normalized) || new Set();
  sources.add(sourceLabel);
  classTokens.set(normalized, sources);
}

function recordClassTokenList(rawClassText, sourceLabel, classTokens) {
  for (const token of rawClassText.split(/\s+/)) {
    recordClassToken(token, sourceLabel, classTokens);
  }
}

function collectUtilityClassTokens() {
  const classTokens = new Map();
  const sourceFiles = [
    PORTAL_HTML_PATH,
    ...listFiles(PORTAL_SOURCE_DIR, (entryPath) => entryPath.endsWith(".js")),
    PORTAL_ASSET_JS_PATH,
    PORTAL_REVIEW_JS_PATH
  ].filter((entryPath) => existsSync(entryPath));

  for (const sourcePath of sourceFiles) {
    const sourceLabel = relativePath(sourcePath);
    const content = readText(sourcePath);
    for (const match of content.matchAll(/\bclass=(["'])(.*?)\1/gs)) {
      recordClassTokenList(match[2], sourceLabel, classTokens);
    }
    for (const match of content.matchAll(/(["'`])((?:\\.|(?!\1)[\s\S])*?)\1/g)) {
      recordClassTokenList(match[2], sourceLabel, classTokens);
    }
  }

  return classTokens;
}

function cssContainsClassToken(css, token) {
  const escapedClass = escapeCssClassToken(token);
  const pattern = new RegExp(`\\.${escapeRegExp(escapedClass)}(?=[\\s.#:{,>+~\\[]|$)`);
  return pattern.test(css);
}

function checkUtilityCoverage(failures) {
  const searchableCss = `${readText(PORTAL_CSS_ASSET_PATH)}\n${readText(PORTAL_REVIEW_CSS_ASSET_PATH)}`;
  const missing = [];
  for (const [token, sources] of collectUtilityClassTokens()) {
    if (!cssContainsClassToken(searchableCss, token)) {
      missing.push(`${token} (${Array.from(sources).sort().join(", ")})`);
    }
  }
  if (missing.length > 0) {
    failures.push(`missing utility compatibility coverage: ${missing.sort().join("; ")}`);
  }
}

function parseLayeredImport(params) {
  const match = params.match(/^url\((["']?)(.+?)\1\)|^(["'])(.+?)\3/);
  const source = match ? (match[2] || match[4] || "").trim() : "";
  const layerMatch = params.match(/\blayer\(\s*([^)]+?)\s*\)/);
  return [source, layerMatch ? layerMatch[1].trim() : ""];
}

function checkProductionLayerImports(failures) {
  const root = parseCss(PORTAL_CSS_INDEX_PATH);
  const imports = [];
  let declaredLayerOrder = null;
  root.each((node) => {
    if (node.type === "comment") {
      return;
    }
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      declaredLayerOrder = layerDeclarationNames(node.params);
    } else if (node.type === "atrule" && node.name === "import") {
      imports.push(parseLayeredImport(node.params));
    } else if (node.type !== "comment") {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} should only contain the layer declaration and layered imports`);
    }
  });
  if (JSON.stringify(declaredLayerOrder) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
    failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} declares unexpected layer order ${(declaredLayerOrder || []).join(", ")}`);
  }
  if (JSON.stringify(imports) !== JSON.stringify(EXPECTED_LAYER_IMPORTS)) {
    failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} production layer import order drifted`);
  }
  for (const [source, layerName] of imports) {
    if (layerName === "utilities" && source.startsWith("./overrides.")) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports override source ${source} into layer(utilities)`);
    }
    if (layerName === "utilities" && source.startsWith("./components/")) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports component source ${source} into layer(utilities)`);
    }
    if (layerName !== "utilities" && /^\.\/utilities\./.test(source)) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports utility source ${source} into layer(${layerName})`);
    }
    if (
      layerName === "utilities" &&
      /(?:reset|compat)/.test(source) &&
      source !== "./utilities.compat-hold.css"
    ) {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports transitional compatibility source ${source} into layer(utilities)`);
    }
    if (source === "./components/workspace-performance.css" && layerName === "utilities") {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} imports workspace performance source through layer(utilities)`);
    }
    if (source === "./overrides.operator-console-reset.css") {
      failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} still imports transitional operator console reset source`);
    }
  }
}

function layerDeclarationNames(params) {
  return params.split(",").map((entry) => entry.trim()).filter(Boolean);
}

function checkSourceCssGovernance(cssFiles, failures) {
  for (const filePath of cssFiles) {
    const root = parseCss(filePath);
    const source = relativePath(filePath);
    root.walkAtRules((atRule) => {
      if (atRule.name === "import" && filePath !== PORTAL_CSS_INDEX_PATH) {
        failures.push(`${source} must not use @import; production imports belong in index.css`);
      }
      if (atRule.name === "layer" && filePath !== PORTAL_CSS_INDEX_PATH) {
        failures.push(`${source} must not declare @layer; production layer ownership belongs in index.css`);
      }
    });
  }
}

function previousOwnerComment(node) {
  let current = node.prev();
  while (current && current.type === "comment") {
    const text = String(current.text || "");
    if (text.includes("portal-override-owner:")) {
      return text;
    }
    current = current.prev();
  }
  return "";
}

function ownerCommentForNode(node) {
  let current = node;
  while (current) {
    const owner = previousOwnerComment(current);
    if (owner) {
      return owner;
    }
    current = current.parent;
  }
  return "";
}

function fileOwnerComment(content) {
  const match = content.match(/\/\*\s*portal-override-owner:\s*([\s\S]*?)\*\//);
  return match ? match[0] : "";
}

function importantOwnerFor(decl, sourceContent) {
  const contexts = atRuleContext(decl);
  if (contexts.some((context) => context.includes("prefers-reduced-motion"))) {
    return "reduced-motion";
  }
  const ruleOwner = previousOwnerComment(decl.parent);
  if (ruleOwner) {
    return ruleOwner;
  }
  return fileOwnerComment(sourceContent);
}

function checkImportantGovernance(cssFiles, failures) {
  for (const filePath of cssFiles) {
    const content = readText(filePath);
    const root = parseCss(filePath, content);
    const source = relativePath(filePath);
    root.walkDecls((decl) => {
      if (!decl.important) {
        return;
      }
      const owner = importantOwnerFor(decl, content);
      if (
        !owner.includes("portal-override-owner:") &&
        !owner.includes("reduced-motion") &&
        !owner.includes("performance-lite") &&
        !owner.includes("forced-colors") &&
        !owner.includes("compatibility-final-order")
      ) {
        failures.push(`${source}:${decl.source?.start?.line || 0} has unowned !important declaration ${decl.prop}`);
      }
    });
  }
}

function checkOverrideOwners(cssFiles, failures) {
  for (const filePath of cssFiles.filter((entryPath) => path.basename(entryPath).startsWith("overrides."))) {
    const content = readText(filePath);
    const source = relativePath(filePath);
    if (!content.includes("portal-override-owner:")) {
      failures.push(`${source} missing portal-override-owner comment`);
    }
    for (const comment of content.matchAll(/\/\*\s*portal-override-owner:\s*compatibility-final-order[\s\S]*?\*\//g)) {
      if (!comment[0].includes("reason:")) {
        failures.push(`${source} compatibility-final-order owner comment missing reason`);
      }
      if (!comment[0].includes("removal-phase:")) {
        failures.push(`${source} compatibility-final-order owner comment missing removal-phase`);
      }
    }
    if (path.basename(filePath) === "overrides.compat.css") {
      const root = parseCss(filePath, content);
      root.walkRules((rule) => {
        const owner = ownerCommentForNode(rule);
        if (!owner.includes("portal-override-owner: compatibility-final-order")) {
          failures.push(`${source}:${rule.source?.start?.line || 0} compatibility rule missing compatibility-final-order owner`);
        }
        if (!owner.includes("reason:")) {
          failures.push(`${source}:${rule.source?.start?.line || 0} compatibility rule missing reason`);
        }
        if (!owner.includes("removal-phase:")) {
          failures.push(`${source}:${rule.source?.start?.line || 0} compatibility rule missing removal-phase`);
        }
      });
    }
  }
}

function checkSelectorFileBoundaries(cssFiles, failures) {
  for (const filePath of cssFiles) {
    const source = relativePath(filePath);
    const basename = path.basename(filePath);
    const inUtilityFile = basename.startsWith("utilities.");
    const inComponentFile = source.includes("portal-src/styles/components/");
    if (!inUtilityFile && !inComponentFile) {
      continue;
    }

    const root = parseCss(filePath);
    root.walkRules((rule) => {
      for (const selector of splitSelectorList(rule.selector)) {
        const classTokens = classTokensFromSelector(selector)
          .filter((token) => !UTILITY_OWNER_ALLOWLIST.has(token));
        if (classTokens.length === 0) {
          continue;
        }
        const utilityTokens = classTokens.filter(isUtilityLikeClassToken);
        const componentTokens = classTokens.filter((token) => !isUtilityLikeClassToken(token));

        if (inUtilityFile && componentTokens.length > 0) {
          failures.push(`${source}:${rule.source?.start?.line || 0} has component-shaped selector in utilities file ${selector}`);
        }
        if (
          inComponentFile &&
          utilityTokens.length > 0 &&
          componentTokens.length === 0 &&
          !/[#[]/.test(selector)
        ) {
          failures.push(`${source}:${rule.source?.start?.line || 0} has utility-shaped selector in component file ${selector}`);
        }
      }
    });
  }
}

function checkLayerContract(failures) {
  const sourceIndex = parseCss(PORTAL_CSS_INDEX_PATH);
  let sourceDeclaresLayers = false;
  sourceIndex.each((node) => {
    if (node.type === "atrule" && node.name === "layer" && !node.nodes) {
      sourceDeclaresLayers = true;
      const names = layerDeclarationNames(node.params);
      if (JSON.stringify(names) !== JSON.stringify(EXPECTED_LAYER_ORDER)) {
        failures.push(`${relativePath(PORTAL_CSS_INDEX_PATH)} declares unexpected layer order ${names.join(", ")}`);
      }
    }
  });

  if (!sourceDeclaresLayers) {
    return;
  }

  const generated = parseCss(PORTAL_CSS_ASSET_PATH);
  generated.walkRules((rule) => {
    const contexts = atRuleContext(rule);
    if (contexts.some((context) => context.startsWith("@keyframes "))) {
      return;
    }
    if (contexts.filter((context) => context.startsWith("@layer ")).length === 0) {
      failures.push(`generated CSS has unlayered ordinary selector rule ${rule.selector}`);
    }
  });
}

function selectorSetForFile(filePath) {
  const selectors = new Set();
  const root = parseCss(filePath);
  root.walkRules((rule) => {
    for (const selector of splitSelectorList(rule.selector)) {
      selectors.add(selector);
    }
  });
  return selectors;
}

function ruleCountForFile(filePath) {
  let count = 0;
  parseCss(filePath).walkRules(() => {
    count += 1;
  });
  return count;
}

function checkOwnershipDrainReport(failures) {
  if (!existsSync(OWNERSHIP_DRAIN_REPORT_PATH)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} is missing`);
    return;
  }
  const report = JSON.parse(readText(OWNERSHIP_DRAIN_REPORT_PATH));
  const moves = Array.isArray(report.moves) ? report.moves : [];
  const summary = report.summary || {};
  if (summary.utilityLayerImportsAfter !== 3) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} must record 3 utility-layer imports after Phase 5`);
  }
  if (summary.compatHoldCount !== 0) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} must record 0 compat-hold utilities`);
  }
  for (const [index, move] of moves.entries()) {
    for (const field of ["selector", "from", "fromLayer", "to", "toLayer", "classification", "parity"]) {
      if (!move[field]) {
        failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} move ${index} missing ${field}`);
      }
    }
    if (move.parity !== "green") {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} move ${index} is not parity green`);
    }
    if (move.classification === "compatibility-final-order" && !move.reason) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} compatibility move ${index} missing reason`);
    }
  }

  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  const compatSelectors = selectorSetForFile(compatPath);
  const reportedCompatSelectors = new Set(
    moves
      .filter((move) => move.to === "overrides.compat.css")
      .map((move) => move.selector)
  );
  for (const selector of compatSelectors) {
    if (!reportedCompatSelectors.has(selector)) {
      failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} missing overrides.compat.css selector ${selector}`);
    }
  }
  if (summary.overridesCompatRuleCount !== ruleCountForFile(compatPath)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} overrides.compat.css rule count is stale`);
  }
  const compatBytes = Buffer.byteLength(readText(compatPath), "utf8");
  if (summary.overridesCompatBytes !== compatBytes) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} overrides.compat.css byte count is stale`);
  }
}

const failures = [];
const cssFiles = listFiles(PORTAL_CSS_SOURCE_DIR, (entryPath) => entryPath.endsWith(".css"));
const records = collectRuleRecords(cssFiles);
const duplicates = collectDuplicateReport(records);

checkProductionLayerImports(failures);
checkSourceCssGovernance(cssFiles, failures);
checkImportantGovernance(cssFiles, failures);
checkOverrideOwners(cssFiles, failures);
checkSelectorFileBoundaries(cssFiles, failures);
checkDuplicateBaseline(duplicates, failures);
checkUtilityCoverage(failures);
checkLayerContract(failures);
checkOwnershipDrainReport(failures);

if (REPORT) {
  const hotspotCount = duplicates.filter((duplicate) => duplicate.hotspot).length;
  console.log(
    `portal css architecture report: ${records.length} selector records, ${duplicates.length} duplicate contexts, ${hotspotCount} hotspot duplicate contexts`
  );
  for (const duplicate of duplicates.filter((entry) => entry.hotspot).slice(0, 20)) {
    console.log(`hotspot ${duplicate.category}: ${duplicate.key}`);
  }
}

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`ERROR: ${failure}`);
  }
  process.exit(1);
}

console.log("portal css architecture: OK");
