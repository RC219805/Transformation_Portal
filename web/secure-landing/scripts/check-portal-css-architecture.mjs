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
const PORTAL_COMPONENT_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "components.current.css");
const PORTAL_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.css");
const PORTAL_REVIEW_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.css");
const PORTAL_HTML_PATH = path.resolve(REPO_ROOT, "portal.html");
const PORTAL_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src");
const PORTAL_ASSET_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.js");
const PORTAL_REVIEW_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.js");
const BASELINE_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "architecture-baseline.json");

const args = new Set(process.argv.slice(2));
const WRITE_BASELINE = args.has("--write-baseline");
const REPORT = args.has("--report") || WRITE_BASELINE;

const EXPECTED_COMPONENT_IMPORTS = [
  "./components/status-feedback.css",
  "./components/shell-foundation.css",
  "./components/console-context.css",
  "./components/dispatch-surfaces.css",
  "./components/workspace-performance.css",
  "./components/responsive-layout.css",
  "./components/operator-console-reset.css"
];

const EXPECTED_LAYER_ORDER = ["tokens", "base", "components", "utilities", "overrides"];
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
  "block",
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
  /^(?:-?m[trblxy]?-.+|p[trblxy]?-.+|space-[xy]-.+|gap-.+|grid-cols-.+|col-span-.+|row-span-.+|flex-.+|items-.+|justify-.+|self-.+|place-.+|w-.+|h-.+|min-w-.+|min-h-.+|max-w-.+|max-h-.+|rounded-.+|border-.+|bg-.+|from-.+|via-.+|to-.+|text-.+|font-.+|tracking-.+|leading-.+|shadow-.+|ring-.+|opacity-.+|overflow-.+|object-.+|inset-.+|top-.+|right-.+|bottom-.+|left-.+|z-.+|cursor-.+|pointer-events-.+|select-.+|resize-.+|whitespace-.+|break-.+|duration-.+|ease-.+|transition-.+|scale-.+|backdrop-.+|animate-.+|outline-.+|fill-.+|stroke-.+|order-.+|basis-.+|shrink-.+|grow-.+|underline|no-underline)$/;
const UTILITY_VARIANTS = new Set([
  "dark",
  "disabled",
  "focus",
  "focus-visible",
  "group-hover",
  "hover",
  "lg",
  "md",
  "peer-checked",
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
      owners: duplicate.hotspot ? ["compatibility-final-order"] : []
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

function checkComponentSplit(failures) {
  const root = parseCss(PORTAL_COMPONENT_INDEX_PATH);
  const imports = [];
  root.each((node) => {
    if (node.type === "atrule" && node.name === "import") {
      const match = node.params.match(/^["'](.+)["']$/);
      imports.push(match ? match[1] : node.params);
    } else if (node.type !== "comment") {
      failures.push(`${relativePath(PORTAL_COMPONENT_INDEX_PATH)} should only contain component imports during the mechanical split phase`);
    }
  });
  if (JSON.stringify(imports) !== JSON.stringify(EXPECTED_COMPONENT_IMPORTS)) {
    failures.push(`${relativePath(PORTAL_COMPONENT_INDEX_PATH)} component import order drifted`);
  }
}

function layerDeclarationNames(params) {
  return params.split(",").map((entry) => entry.trim()).filter(Boolean);
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

const failures = [];
const cssFiles = listFiles(PORTAL_CSS_SOURCE_DIR, (entryPath) => entryPath.endsWith(".css"));
const records = collectRuleRecords(cssFiles);
const duplicates = collectDuplicateReport(records);

checkComponentSplit(failures);
checkDuplicateBaseline(duplicates, failures);
checkUtilityCoverage(failures);
checkLayerContract(failures);

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
