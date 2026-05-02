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
const COMPAT_HOLD_UTILITIES_SOURCE = "./utilities.compat-hold.css";
const COMPAT_HOLD_UTILITIES_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utilities.compat-hold.css");
const OVERRIDES_COMPAT_SOURCE = "./overrides.compat.css";
const OVERRIDES_COMPAT_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
const PHASE9_DUPLICATE_PHASE = "phase-9-duplicate-ownership-closure";
const PHASE9_DUPLICATE_CONTEXT_COUNT_BEFORE = 87;
const VALID_PHASE9_CONTEXT_TYPES = new Set([
  "responsive-context",
  "theme-context",
  "accessibility-context",
  "runtime-state",
  "component-family-shared-state",
  "utility-owned",
  "document-root-normalization",
  "surface-normalization-final-pass"
]);
const VALID_PHASE9_DISPOSITIONS = new Set(["keep-owned", "defer-owned", "report-only"]);
const VALID_PHASE9_REMOVAL_STATUSES = new Set(["permanent", "removable-later"]);
const VALID_PHASE9_DECLARATION_CONFLICTS = new Set(["identical", "additive", "conflicting", "contextual"]);

const args = new Set(process.argv.slice(2));
const WRITE_BASELINE = args.has("--write-baseline");
const REPORT = args.has("--report") || WRITE_BASELINE;
const SENTINEL_FIXTURE_ARG = "--check-sentinel-fixture";
const sentinelFixtureIndex = process.argv.indexOf(SENTINEL_FIXTURE_ARG);
const SENTINEL_FIXTURE_PATH = sentinelFixtureIndex >= 0 ? process.argv[sentinelFixtureIndex + 1] : "";
const DUPLICATE_BASELINE_FIXTURE_ARG = "--check-duplicate-baseline-fixture";
const duplicateBaselineFixtureIndex = process.argv.indexOf(DUPLICATE_BASELINE_FIXTURE_ARG);
const DUPLICATE_BASELINE_FIXTURE_PATH =
  duplicateBaselineFixtureIndex >= 0 ? process.argv[duplicateBaselineFixtureIndex + 1] : "";
const DUPLICATE_BASELINE_DUPLICATES_PATH =
  duplicateBaselineFixtureIndex >= 0 ? process.argv[duplicateBaselineFixtureIndex + 2] : "";

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
  [COMPAT_HOLD_UTILITIES_SOURCE, "utilities"],
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

function collectRuleRecords(cssFiles, layerByPath = new Map()) {
  const records = [];
  for (const filePath of cssFiles) {
    const root = parseCss(filePath);
    const layer = layerByPath.get(path.resolve(filePath)) || "unlayered";
    root.walkRules((rule) => {
      const context = atRuleContext(rule);
      for (const selector of splitSelectorList(rule.selector)) {
        const declarations = declarationEntries(rule);
        records.push({
          selector,
          layer,
          context,
          layerContext: layerContext(rule),
          stateContext: selectorStateContext(selector),
          source: relativePath(filePath),
          line: rule.source?.start?.line || 0,
          column: rule.source?.start?.column || 0,
          declarations,
          declarationSignature: declarationSignature(declarations)
        });
      }
    });
  }
  return records;
}

function duplicateKey(record) {
  return `${record.selector}|||${record.layer}|||${record.context.join(" > ")}`;
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
      layer: recordsForKey[0].layer,
      context: recordsForKey[0].context,
      stateContext: Array.from(new Set(recordsForKey.flatMap((record) => record.stateContext))).sort(),
      category: classifyDuplicate(recordsForKey),
      hotspot: HOTSPOT_SELECTORS.has(recordsForKey[0].selector),
      records: recordsForKey.map((record) => ({
        source: record.source,
        line: record.line,
        column: record.column,
        layer: record.layer,
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

function productionLayerBySourcePath() {
  const imports = collectCssImportGraph(PORTAL_CSS_INDEX_PATH, []);
  const layerByPath = new Map();
  for (const record of imports) {
    if (record.resolvedPath) {
      layerByPath.set(path.resolve(record.resolvedPath), record.layerName || "unlayered");
    }
  }
  return layerByPath;
}

function uniqueSorted(values) {
  return Array.from(new Set(values)).sort();
}

function duplicateSources(duplicate) {
  return uniqueSorted(duplicate.records.map((record) => path.basename(record.source)));
}

function contextTypeForDuplicate(duplicate) {
  const selector = duplicate.selector || "";
  const contextText = (duplicate.context || []).join(" ");
  if (/prefers-reduced-motion|forced-colors|prefers-contrast/.test(contextText)) {
    return "accessibility-context";
  }
  if (contextText.includes("@media")) {
    return "responsive-context";
  }
  if (selector.startsWith(".dark ") || selector.includes("html:not(.light)")) {
    return "theme-context";
  }
  if (selector === "html" || selector.startsWith("#")) {
    return "document-root-normalization";
  }
  if (/\[data-|:hover|:focus|:focus-visible|:disabled|\.is-active/.test(selector)) {
    return "runtime-state";
  }
  if (duplicate.records.some((record) => path.basename(record.source).startsWith("utilities."))) {
    return "utility-owned";
  }
  if (duplicate.records.some((record) => record.source.endsWith("components/surface-normalization.css"))) {
    return "surface-normalization-final-pass";
  }
  return "component-family-shared-state";
}

function ownerReasonForDuplicate(duplicate) {
  const sourceList = duplicateSources(duplicate).join(", ");
  if (duplicate.category === "conflicting") {
    return `Intentional cascade: ${duplicate.selector} keeps the reviewed source order across ${sourceList}; the later production rule provides the final declaration set.`;
  }
  if (duplicate.category === "additive") {
    return `Additive ownership: ${duplicate.selector} splits non-overlapping declarations across ${sourceList} while preserving the current computed-style parity.`;
  }
  if (duplicate.category === "identical") {
    return `Identical duplicate retained as report-only evidence until a mechanically safe consolidation removes one source without selector-list coverage loss.`;
  }
  return `Contextual duplicate retained with explicit ownership until a later parity-backed consolidation proves removal safe.`;
}

function duplicateBaselineMetadata(duplicate) {
  return {
    key: duplicate.key,
    selector: duplicate.selector,
    layer: duplicate.layer,
    atRuleContext: duplicate.context,
    stateContext: duplicate.stateContext,
    category: duplicate.category,
    hotspot: duplicate.hotspot,
    records: duplicate.records.map((record) => ({
      source: record.source,
      line: record.line,
      column: record.column,
      layer: record.layer,
      declarationSignature: record.declarationSignature,
      properties: record.properties
    }))
  };
}

function baselineFromReport(duplicates) {
  return {
    version: 1,
    phase: PHASE9_DUPLICATE_PHASE,
    duplicateKeys: duplicates.map((duplicate) => ({
      ...duplicateBaselineMetadata(duplicate),
      owners: ["portal-css-architecture"],
      ownerReason: ownerReasonForDuplicate(duplicate),
      phase: PHASE9_DUPLICATE_PHASE,
      contextType: contextTypeForDuplicate(duplicate),
      disposition: duplicate.category === "additive" ? "report-only" : "keep-owned",
      removalStatus: duplicate.category === "additive" ? "removable-later" : "permanent",
      declarationConflict: duplicate.category,
      parity: "green"
    }))
  };
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function entryHasOwners(entry) {
  return Array.isArray(entry.owners) && entry.owners.some((owner) => isNonEmptyString(owner));
}

function checkDuplicateBaselineMetadata(entry, duplicate, failures, label) {
  const expected = duplicateBaselineMetadata(duplicate);
  for (const field of ["selector", "layer", "category", "hotspot"]) {
    if (JSON.stringify(entry[field]) !== JSON.stringify(expected[field])) {
      failures.push(`${label} duplicate ${duplicate.key} has stale ${field}`);
    }
  }
  for (const field of ["atRuleContext", "stateContext", "records"]) {
    if (JSON.stringify(entry[field]) !== JSON.stringify(expected[field])) {
      failures.push(`${label} duplicate ${duplicate.key} has stale ${field}`);
    }
  }
}

function checkDuplicateBaselineEntry(entry, duplicate, failures, label) {
  checkDuplicateBaselineMetadata(entry, duplicate, failures, label);
  if (!entryHasOwners(entry)) {
    failures.push(`${label} duplicate ${duplicate.key} missing owners`);
  }
  if (!isNonEmptyString(entry.ownerReason)) {
    failures.push(`${label} duplicate ${duplicate.key} missing ownerReason`);
  }
  if (entry.phase !== PHASE9_DUPLICATE_PHASE) {
    failures.push(`${label} duplicate ${duplicate.key} must declare phase ${PHASE9_DUPLICATE_PHASE}`);
  }
  if (!VALID_PHASE9_CONTEXT_TYPES.has(entry.contextType)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid contextType ${entry.contextType || "missing"}`);
  }
  if (!VALID_PHASE9_DISPOSITIONS.has(entry.disposition)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid disposition ${entry.disposition || "missing"}`);
  }
  if (!VALID_PHASE9_REMOVAL_STATUSES.has(entry.removalStatus)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid removalStatus ${entry.removalStatus || "missing"}`);
  }
  if (!VALID_PHASE9_DECLARATION_CONFLICTS.has(entry.declarationConflict)) {
    failures.push(`${label} duplicate ${duplicate.key} has invalid declarationConflict ${entry.declarationConflict || "missing"}`);
  } else if (entry.declarationConflict !== duplicate.category) {
    failures.push(`${label} duplicate ${duplicate.key} declarationConflict must match current ${duplicate.category} classification`);
  }
  if (entry.parity !== "green") {
    failures.push(`${label} duplicate ${duplicate.key} must record parity green`);
  }
  if (duplicate.hotspot || entry.hotspot) {
    failures.push(`${label} hotspot duplicate context ${duplicate.key} is forbidden even when owned`);
  }
  if (
    duplicate.category === "conflicting" &&
    isNonEmptyString(entry.ownerReason) &&
    !/(cascade|source order|final declaration|final-order|winning|later production rule)/i.test(entry.ownerReason)
  ) {
    failures.push(`${label} conflicting duplicate ${duplicate.key} ownerReason must explain intended cascade/source-order behavior`);
  }
}

function checkDuplicateBaselineEntries(duplicates, baseline, failures, label = relativePath(BASELINE_PATH)) {
  if (!baseline || !Array.isArray(baseline.duplicateKeys)) {
    failures.push(`${label} is missing duplicateKeys`);
    return;
  }
  const knownByKey = new Map((baseline.duplicateKeys || []).map((entry) => [entry.key, entry]));
  const currentKeys = new Set(duplicates.map((duplicate) => duplicate.key));
  for (const duplicate of duplicates) {
    const entry = knownByKey.get(duplicate.key);
    if (!entry) {
      failures.push(`new unclassified duplicate selector ${duplicate.key}`);
      continue;
    }
    checkDuplicateBaselineEntry(entry, duplicate, failures, label);
  }
  for (const entry of baseline.duplicateKeys || []) {
    if (!currentKeys.has(entry.key)) {
      failures.push(`stale duplicate baseline entry ${entry.key}; refresh ${label}`);
    }
  }
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
  checkDuplicateBaselineEntries(duplicates, baseline, failures);
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

function resolveImportPath(fromPath, source) {
  if (!source.startsWith(".")) {
    return null;
  }
  return path.resolve(path.dirname(fromPath), source);
}

function collectCssImportGraph(entryPath, failures, seen = new Set(), records = []) {
  const resolvedEntry = path.resolve(entryPath);
  if (seen.has(resolvedEntry)) {
    return records;
  }
  seen.add(resolvedEntry);

  const root = parseCss(resolvedEntry);
  root.walkAtRules("import", (atRule) => {
    const [source, layerName] = parseLayeredImport(atRule.params);
    const resolvedPath = resolveImportPath(resolvedEntry, source);
    const record = {
      fromPath: resolvedEntry,
      source,
      layerName,
      line: atRule.source?.start?.line || 0,
      resolvedPath
    };
    records.push(record);

    if (!resolvedPath) {
      return;
    }
    if (!existsSync(resolvedPath)) {
      failures.push(`${relativePath(resolvedEntry)}:${record.line} imports missing CSS source ${source}`);
      return;
    }
    collectCssImportGraph(resolvedPath, failures, seen, records);
  });

  return records;
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

function describeCssNode(node) {
  if (node.type === "rule") {
    return `rule "${node.selector}"`;
  }
  if (node.type === "atrule") {
    const params = node.params ? ` ${node.params}` : "";
    return `at-rule @${node.name}${params}`;
  }
  if (node.type === "decl") {
    return `declaration "${node.prop}"`;
  }
  return `${node.type} node`;
}

function sentinelGuidance(filePath) {
  const basename = path.basename(filePath);
  if (basename === "utilities.compat-hold.css") {
    return "Move real utility rules to utilities.required.css or utilities.dynamic.css; do not add rules to the sentinel.";
  }
  if (basename === "overrides.compat.css") {
    return "Move real override rules to overrides.performance.css, overrides.accessibility.css, or an owned component stylesheet; do not add rules to the sentinel.";
  }
  return "Move real CSS to the appropriate owned source file; do not add rules to the sentinel.";
}

function sentinelCssViolations(filePath) {
  const root = parseCss(filePath);
  return (root.nodes || [])
    .filter((node) => node.type !== "comment")
    .map((node) => ({
      description: describeCssNode(node),
      line: node.source?.start?.line || 0
    }));
}

function checkSentinelOnlyFile(filePath, failures, label = relativePath(filePath)) {
  if (!existsSync(filePath)) {
    failures.push(`${label} sentinel file is missing`);
    return;
  }
  for (const violation of sentinelCssViolations(filePath)) {
    failures.push(
      `${label} must remain sentinel-only. Found ${violation.description} at line ${violation.line}. ${sentinelGuidance(filePath)}`
    );
  }
}

function checkPhase8SentinelImportGraph(failures) {
  const imports = collectCssImportGraph(PORTAL_CSS_INDEX_PATH, failures);
  const compatHoldImports = imports.filter((record) => record.resolvedPath === COMPAT_HOLD_UTILITIES_PATH);
  if (compatHoldImports.length !== 1) {
    failures.push(
      `${relativePath(PORTAL_CSS_INDEX_PATH)} import graph must import ${COMPAT_HOLD_UTILITIES_SOURCE} exactly once in layer(utilities); found ${compatHoldImports.length}`
    );
  } else if (compatHoldImports[0].layerName !== "utilities") {
    failures.push(
      `${relativePath(compatHoldImports[0].fromPath)}:${compatHoldImports[0].line} imports ${COMPAT_HOLD_UTILITIES_SOURCE} into layer(${compatHoldImports[0].layerName || "none"})`
    );
  }

  const overridesCompatImports = imports.filter((record) => record.resolvedPath === OVERRIDES_COMPAT_PATH);
  for (const record of overridesCompatImports) {
    failures.push(
      `${relativePath(record.fromPath)}:${record.line} must not import ${OVERRIDES_COMPAT_SOURCE}; overrides.compat.css is a Phase 8 unshipped sentinel`
    );
  }
}

function checkPhase8Sentinels(failures) {
  checkSentinelOnlyFile(COMPAT_HOLD_UTILITIES_PATH, failures);
  checkSentinelOnlyFile(OVERRIDES_COMPAT_PATH, failures);
  checkPhase8SentinelImportGraph(failures);
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

function selectorContainsIdToken(selector) {
  // Strip attribute selector contents (`[href^="#"]`) and quoted string
  // bodies. PostCSS already strips comments before surfacing rule.selector,
  // so once those two sources of `#` are removed every remaining `#` in
  // the selector starts an ID token (CSS selector grammar allows ID tokens
  // adjacent to type/class selectors with no separator, e.g. `a#anchor`).
  let stripped = selector;
  let previous;
  do {
    previous = stripped;
    stripped = stripped.replace(/\[[^\[\]]*\]/g, "");
  } while (stripped !== previous);
  stripped = stripped.replace(/"[^"]*"/g, "").replace(/'[^']*'/g, "");
  return /#(?:\\.|[A-Za-z_-])/.test(stripped);
}

function checkOverridesCompatNoNewIds(failures) {
  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  if (!existsSync(compatPath)) {
    return;
  }
  const source = relativePath(compatPath);
  const root = parseCss(compatPath);
  root.walkRules((rule) => {
    for (const selector of splitSelectorList(rule.selector)) {
      if (selectorContainsIdToken(selector)) {
        failures.push(
          `${source}:${rule.source?.start?.line || 0} ID-specific selector ${selector} is forbidden in overrides.compat.css after Phase 6 semantic-ownership migration`
        );
      }
    }
  });
}

function checkOverridesCompatNoImportant(failures) {
  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  if (!existsSync(compatPath)) {
    return;
  }
  const source = relativePath(compatPath);
  const root = parseCss(compatPath);
  root.walkDecls((decl) => {
    if (decl.important) {
      failures.push(
        `${source}:${decl.source?.start?.line || 0} !important on ${decl.prop} is forbidden in overrides.compat.css; raise the issue in the Phase 7 markup migration instead`
      );
    }
  });
}

function checkPhase6SemanticHooksPresent(failures) {
  if (!existsSync(PORTAL_HTML_PATH)) {
    failures.push(`portal.html missing at ${relativePath(PORTAL_HTML_PATH)}; cannot validate Phase 6 semantic hooks`);
    return;
  }
  const compatPath = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");
  if (!existsSync(compatPath)) {
    return;
  }
  const compatRoot = parseCss(compatPath);
  const semanticTokens = new Set();
  compatRoot.walkRules((rule) => {
    for (const selector of splitSelectorList(rule.selector)) {
      for (const token of classTokensFromSelector(selector)) {
        if (token.startsWith("portal-") || token.startsWith("artifact-")) {
          semanticTokens.add(token);
        }
      }
    }
  });
  if (semanticTokens.size === 0) {
    return;
  }
  const html = readText(PORTAL_HTML_PATH);
  for (const token of semanticTokens) {
    const wordBoundary = new RegExp(`(^|[\\s"'])${escapeRegExp(token)}([\\s"']|$)`, "m");
    if (!wordBoundary.test(html)) {
      failures.push(
        `Phase 6 semantic hook .${token} used in overrides.compat.css does not match any class on portal.html; fix the typo or restore the markup hook`
      );
    }
  }
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(readText(filePath)).digest("hex");
}

function phase9EntryIsOwned(entry, duplicate) {
  if (!entry) {
    return false;
  }
  const failures = [];
  checkDuplicateBaselineEntry(entry, duplicate, failures, relativePath(BASELINE_PATH));
  return failures.length === 0;
}

function countDuplicatesByCategory(duplicates, category) {
  return duplicates.filter((duplicate) => duplicate.category === category).length;
}

function buildPhase9DuplicateState(duplicates, baseline, phase8State) {
  const baselineEntries = new Map(((baseline && baseline.duplicateKeys) || []).map((entry) => [entry.key, entry]));
  const ownedDuplicateContextCount = duplicates.filter((duplicate) =>
    phase9EntryIsOwned(baselineEntries.get(duplicate.key), duplicate)
  ).length;
  const duplicateContextCountAfter = duplicates.length;
  const reclassifiedDuplicateContextCount = Math.max(0, PHASE9_DUPLICATE_CONTEXT_COUNT_BEFORE - duplicateContextCountAfter);
  return {
    phase: PHASE9_DUPLICATE_PHASE,
    baselinePath: relativePath(BASELINE_PATH),
    baselineSha256: existsSync(BASELINE_PATH) ? sha256File(BASELINE_PATH) : null,
    duplicateContextCountBefore: PHASE9_DUPLICATE_CONTEXT_COUNT_BEFORE,
    duplicateContextCountAfter,
    duplicateContextCount: duplicateContextCountAfter,
    ownedDuplicateContextCount,
    unownedDuplicateContextCount: duplicateContextCountAfter - ownedDuplicateContextCount,
    hotspotDuplicateContextCount: duplicates.filter((duplicate) => duplicate.hotspot).length,
    consolidatedDuplicateContextCount: 0,
    reclassifiedDuplicateContextCount,
    conflictingDuplicateContextCount: countDuplicatesByCategory(duplicates, "conflicting"),
    additiveDuplicateContextCount: countDuplicatesByCategory(duplicates, "additive"),
    identicalDuplicateContextCount: countDuplicatesByCategory(duplicates, "identical"),
    contextualDuplicateContextCount: countDuplicatesByCategory(duplicates, "contextual"),
    removedRawBytes: 0,
    removedGzipBytes: 0,
    sentinelStatePreserved:
      Boolean(phase8State?.utilitiesCompatHold?.imported) &&
      phase8State.utilitiesCompatHold.layer === "utilities" &&
      Boolean(phase8State.utilitiesCompatHold.sentinelOnly) &&
      phase8State.utilitiesCompatHold.sourceRuleCount === 0 &&
      !phase8State?.overridesCompat?.imported &&
      Boolean(phase8State?.overridesCompat?.sentinelOnly) &&
      phase8State.overridesCompat.sourceRuleCount === 0,
    parityBaselineChanged: false
  };
}

function checkOwnershipDrainReport(failures, duplicates) {
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
  if (summary.overridesCompatBytes !== 0) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} overrides.compat.css shipped byte debt must be 0`);
  }

  const expectedPhase8State = buildPhase8SentinelState(collectCssImportGraph(PORTAL_CSS_INDEX_PATH, []));
  if (JSON.stringify(report.phase8SentinelState || null) !== JSON.stringify(expectedPhase8State)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase8SentinelState is stale`);
  }

  const expectedPhase9State = buildPhase9DuplicateState(duplicates, loadBaseline(), expectedPhase8State);
  if (JSON.stringify(report.phase9DuplicateState || null) !== JSON.stringify(expectedPhase9State)) {
    failures.push(`${relativePath(OWNERSHIP_DRAIN_REPORT_PATH)} phase9DuplicateState is stale`);
  }
}

function sentinelSourceRuleCount(filePath) {
  return existsSync(filePath) ? ruleCountForFile(filePath) : 0;
}

function sentinelStateFor(filePath, imports) {
  const importRecords = imports.filter((record) => record.resolvedPath === filePath);
  const sourceRuleCount = sentinelSourceRuleCount(filePath);
  const imported = importRecords.length > 0;
  const shippedRuleCount = imported ? sourceRuleCount : 0;
  return {
    path: relativePath(filePath),
    imported,
    layer: imported ? importRecords[0].layerName : null,
    sourceRuleCount,
    shippedRuleCount,
    shippedByteDebt: shippedRuleCount > 0 ? Buffer.byteLength(readText(filePath), "utf8") : 0,
    sentinelOnly: existsSync(filePath) && sentinelCssViolations(filePath).length === 0
  };
}

function buildPhase8SentinelState(imports) {
  return {
    utilitiesCompatHold: sentinelStateFor(COMPAT_HOLD_UTILITIES_PATH, imports),
    overridesCompat: sentinelStateFor(OVERRIDES_COMPAT_PATH, imports)
  };
}

if (sentinelFixtureIndex >= 0) {
  if (!SENTINEL_FIXTURE_PATH) {
    console.error(`ERROR: ${SENTINEL_FIXTURE_ARG} requires a CSS fixture path`);
    process.exit(1);
  }
  const fixtureFailures = [];
  checkSentinelOnlyFile(path.resolve(FRONTDOOR_ROOT, SENTINEL_FIXTURE_PATH), fixtureFailures, SENTINEL_FIXTURE_PATH);
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log("portal css sentinel fixture: OK");
  process.exit(0);
}

function resolveFixturePath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.resolve(FRONTDOOR_ROOT, filePath);
}

if (duplicateBaselineFixtureIndex >= 0) {
  if (!DUPLICATE_BASELINE_FIXTURE_PATH || !DUPLICATE_BASELINE_DUPLICATES_PATH) {
    console.error(`ERROR: ${DUPLICATE_BASELINE_FIXTURE_ARG} requires baseline and duplicate JSON fixture paths`);
    process.exit(1);
  }
  const fixtureFailures = [];
  const baseline = JSON.parse(readText(resolveFixturePath(DUPLICATE_BASELINE_FIXTURE_PATH)));
  const duplicateFixture = JSON.parse(readText(resolveFixturePath(DUPLICATE_BASELINE_DUPLICATES_PATH)));
  const fixtureDuplicates = Array.isArray(duplicateFixture) ? duplicateFixture : duplicateFixture.duplicates || [];
  checkDuplicateBaselineEntries(
    fixtureDuplicates,
    baseline,
    fixtureFailures,
    path.basename(DUPLICATE_BASELINE_FIXTURE_PATH)
  );
  if (fixtureFailures.length > 0) {
    for (const failure of fixtureFailures) {
      console.error(`ERROR: ${failure}`);
    }
    process.exit(1);
  }
  console.log("portal css duplicate baseline fixture: OK");
  process.exit(0);
}

const failures = [];
const cssFiles = listFiles(PORTAL_CSS_SOURCE_DIR, (entryPath) => entryPath.endsWith(".css"));
const records = collectRuleRecords(cssFiles, productionLayerBySourcePath());
const duplicates = collectDuplicateReport(records);

checkProductionLayerImports(failures);
checkPhase8Sentinels(failures);
checkSourceCssGovernance(cssFiles, failures);
checkImportantGovernance(cssFiles, failures);
checkOverrideOwners(cssFiles, failures);
checkSelectorFileBoundaries(cssFiles, failures);
checkDuplicateBaseline(duplicates, failures);
checkUtilityCoverage(failures);
checkLayerContract(failures);
checkOwnershipDrainReport(failures, duplicates);
checkOverridesCompatNoNewIds(failures);
checkOverridesCompatNoImportant(failures);
checkPhase6SemanticHooksPresent(failures);

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
