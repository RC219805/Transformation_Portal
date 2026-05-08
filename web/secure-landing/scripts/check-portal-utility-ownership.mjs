import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import postcss from "postcss";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "index.css");
const PORTAL_HTML_PATH = path.resolve(REPO_ROOT, "portal.html");
const PORTAL_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src");
const PORTAL_ASSET_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.js");
const PORTAL_REVIEW_JS_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.js");
const PORTAL_REVIEW_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.css");
const OWNERSHIP_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utility-ownership.json");
const REPORT_PATH = path.resolve(FRONTDOOR_ROOT, "reports", "portal-utility-usage.generated.json");
const LEGACY_UTILITIES_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utilities.compat.css");
const REQUIRED_UTILITIES_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utilities.required.css");
const DYNAMIC_UTILITIES_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utilities.dynamic.css");
const COMPAT_HOLD_UTILITIES_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "utilities.compat-hold.css");

const args = new Set(process.argv.slice(2));
const WRITE_OWNERSHIP = args.has("--write-ownership");
const WRITE_REPORT = args.has("--write-report");
const WRITE_SPLIT = args.has("--write-split");

const UTILITY_EXACT_CLASSES = new Set([
  "absolute",
  "antialiased",
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
const UTILITY_OWNER_ALLOWLIST = new Set(["dark", "light", "performance-lite"]);
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
const DYNAMIC_UTILITY_TOKENS = new Set([
  "group",
  "hidden",
  "peer",
  "pointer-events-auto",
  "pointer-events-none",
  "sr-only"
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
  return postcss.parse(content, { from: filePath });
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

function collectStringLiteralBodies(content) {
  const bodies = [];
  let index = 0;
  let state = "code";
  let quote = "";
  let body = "";

  while (index < content.length) {
    const char = content[index];
    const next = content[index + 1];

    if (state === "code") {
      if (char === "/" && next === "/") {
        state = "line-comment";
        index += 2;
        continue;
      }
      if (char === "/" && next === "*") {
        state = "block-comment";
        index += 2;
        continue;
      }
      if (char === "\"" || char === "'" || char === "`") {
        state = "string";
        quote = char;
        body = "";
        index += 1;
        continue;
      }
      index += 1;
      continue;
    }

    if (state === "line-comment") {
      if (char === "\n" || char === "\r") {
        state = "code";
      }
      index += 1;
      continue;
    }

    if (state === "block-comment") {
      if (char === "*" && next === "/") {
        state = "code";
        index += 2;
        continue;
      }
      index += 1;
      continue;
    }

    if (char === "\\") {
      body += content.slice(index, index + 2);
      index += 2;
      continue;
    }
    if (char === quote) {
      bodies.push(body);
      state = "code";
      quote = "";
      body = "";
      index += 1;
      continue;
    }
    body += char;
    index += 1;
  }

  return bodies;
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
    for (const literalBody of collectStringLiteralBodies(content)) {
      recordClassTokenList(literalBody, sourceLabel, classTokens);
    }
  }

  return classTokens;
}

function unescapeCssClassToken(token) {
  return token
    .replace(/\\([:/.[\]()%#,])/g, "$1")
    .replace(/\\([0-9a-fA-F]{1,6})\s?/g, (_match, value) => String.fromCodePoint(parseInt(value, 16)))
    .replace(/\\/g, "");
}

function extractClassTokens(selector) {
  const tokens = [];
  const pattern = /\.((?:\\.|[A-Za-z0-9_-])+)/g;
  for (const match of selector.matchAll(pattern)) {
    const token = unescapeCssClassToken(match[1]);
    if (isUtilityLikeClassToken(token)) {
      tokens.push(token);
    }
  }
  return Array.from(new Set(tokens));
}

function collectCssUtilityOwners() {
  const owners = new Map();
  const cssFiles = [
    ...listFiles(PORTAL_CSS_SOURCE_DIR, (entryPath) => entryPath.endsWith(".css")),
    PORTAL_REVIEW_CSS_ASSET_PATH
  ].filter((entryPath) => existsSync(entryPath));

  for (const cssFile of cssFiles) {
    const root = parseCss(cssFile);
    root.walkRules((rule) => {
      for (const token of extractClassTokens(rule.selector)) {
        const sources = owners.get(token) || new Set();
        sources.add(relativePath(cssFile));
        owners.set(token, sources);
      }
    });
  }
  return owners;
}

function utilityOwnerFor(token, sources) {
  const sourceList = Array.from(sources);
  const observedInStaticMarkup = sourceList.some(
    (source) => source === "portal.html" || source.endsWith("portal-src/portal.template.js")
  );
  if (observedInStaticMarkup) {
    return "static";
  }
  if (sourceList.some((source) => source.endsWith("portal-review.js") || source.includes("review-surface"))) {
    return "review-only";
  }
  if (DYNAMIC_UTILITY_TOKENS.has(classTokenBase(token)) || token.startsWith("peer-")) {
    return "shared-state";
  }
  if (sourceList.some((source) => source.includes("portal-src/internal/"))) {
    return "js-state";
  }
  return "static";
}

function reasonFor(owner) {
  return {
    "js-state": "observed in portal JavaScript class strings",
    "review-only": "observed in deferred review surface JavaScript",
    "shared-state": "shared state utility used by markup or classList toggles",
    static: "observed in portal static markup or generated template output"
  }[owner] || "observed utility class";
}

function primaryCssOwner(cssOwners) {
  const owners = Array.from(cssOwners || []).sort();
  const utilityOwner = owners.find((owner) => owner.includes("/utilities."));
  return utilityOwner ? path.basename(utilityOwner) : owners[0] || "";
}

function buildOwnership() {
  const usage = collectUtilityClassTokens();
  const cssOwners = collectCssUtilityOwners();
  const utilities = {};

  for (const [token, sources] of Array.from(usage.entries()).sort(([left], [right]) => left.localeCompare(right))) {
    const owner = utilityOwnerFor(token, sources);
    utilities[token] = {
      owner,
      sources: Array.from(sources).sort(),
      cssFile: primaryCssOwner(cssOwners.get(token)),
      reason: reasonFor(owner)
    };
  }

  const compatHoldTokens = Array.from(cssOwners.entries())
    .filter(([, owners]) => Array.from(owners).some((owner) => owner.endsWith("utilities.compat-hold.css")))
    .map(([token]) => token)
    .filter((token) => !Object.prototype.hasOwnProperty.call(utilities, token))
    .sort();
  for (const token of compatHoldTokens) {
    utilities[token] = {
      owner: "compat-hold",
      sources: [],
      cssFile: "utilities.compat-hold.css",
      reason: "retained until a later runtime census or semantic migration proves removal is safe"
    };
  }

  const previous = existsSync(OWNERSHIP_PATH) ? JSON.parse(readText(OWNERSHIP_PATH)) : {};
  const compatHoldCount = Object.values(utilities).filter((entry) => entry.owner === "compat-hold").length;
  return {
    version: 1,
    maxCompatHold: Number.isInteger(previous.maxCompatHold) ? previous.maxCompatHold : compatHoldCount,
    utilities
  };
}

function buildReport(ownership) {
  const cssOwners = collectCssUtilityOwners();
  const entries = Object.entries(ownership.utilities).map(([className, entry]) => ({
    className,
    owner: entry.owner,
    sources: entry.sources,
    cssFile: entry.cssFile,
    cssOwners: Array.from(cssOwners.get(className) || []).sort(),
    status: entry.cssFile ? "covered" : "missing-css"
  }));
  const compatHoldCount = entries.filter((entry) => entry.owner === "compat-hold").length;
  return {
    version: 1,
    summary: {
      utilityCount: entries.length,
      compatHoldCount,
      deprecatedImported: readText(PORTAL_CSS_INDEX_PATH).includes("utilities.deprecated.css")
    },
    utilities: entries.sort((left, right) => left.className.localeCompare(right.className))
  };
}

function stableJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function writeJson(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, stableJson(value), "utf-8");
}

function ruleCategory(rule, usage) {
  const tokens = extractClassTokens(rule.selector);
  if (tokens.length === 0) {
    return "required";
  }
  const observedTokens = tokens.filter((token) => usage.has(token));
  if (observedTokens.length === 0) {
    return "drop";
  }
  const dynamic = observedTokens.some((token) => {
    const owner = utilityOwnerFor(token, usage.get(token) || new Set());
    return owner !== "static";
  });
  return dynamic ? "dynamic" : "required";
}

function appendNodeByCategory(targets, node, usage, counters) {
  if (node.type === "rule") {
    const category = ruleCategory(node, usage);
    counters[category] += 1;
    if (category !== "drop") {
      targets[category].append(node.clone());
    }
    return;
  }
  if (node.type === "atrule" && node.nodes) {
    const clones = {
      required: node.clone({ nodes: [] }),
      dynamic: node.clone({ nodes: [] }),
      "compat-hold": node.clone({ nodes: [] })
    };
    for (const child of node.nodes) {
      if (child.type !== "rule") {
        continue;
      }
      const category = ruleCategory(child, usage);
      counters[category] += 1;
      if (category !== "drop") {
        clones[category].append(child.clone());
      }
    }
    for (const category of ["required", "dynamic", "compat-hold"]) {
      if (clones[category].nodes.length > 0) {
        targets[category].append(clones[category]);
      }
    }
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

function ruleDeclarationSignature(rule) {
  const declarations = [];
  rule.walkDecls((decl) => {
    declarations.push(`${decl.prop.trim()}:${decl.value.trim()}${decl.important ? " !important" : ""}`);
  });
  return declarations.join(";");
}

function ruleSignature(rule) {
  return `${atRuleContext(rule).join("|")}||${rule.selector.trim()}||${ruleDeclarationSignature(rule)}`;
}

function pruneDuplicateUtilityRules(root) {
  const seen = new Set();
  let removed = 0;
  root.walkRules((rule) => {
    const signature = ruleSignature(rule);
    if (seen.has(signature)) {
      rule.remove();
      removed += 1;
      return;
    }
    seen.add(signature);
  });
  root.walkAtRules((atRule) => {
    if (atRule.nodes && atRule.nodes.length === 0) {
      atRule.remove();
    }
  });
  return removed;
}

function buildSplitCss() {
  const usage = collectUtilityClassTokens();
  const splitSourcePaths = existsSync(LEGACY_UTILITIES_PATH)
    ? [LEGACY_UTILITIES_PATH]
    : [REQUIRED_UTILITIES_PATH, DYNAMIC_UTILITIES_PATH, COMPAT_HOLD_UTILITIES_PATH].filter((filePath) =>
        existsSync(filePath)
      );
  if (splitSourcePaths.length === 0) {
    throw new Error(`${relativePath(LEGACY_UTILITIES_PATH)} or existing split utility files are required for --write-split`);
  }
  const targets = {
    required: postcss.root(),
    dynamic: postcss.root(),
    "compat-hold": postcss.root()
  };
  targets.required.append({ text: "portal-utility-owner: static\n   reason: required utility classes observed in portal markup or generated templates." });
  targets.dynamic.append({ text: "portal-utility-owner: dynamic-state\n   reason: utility classes observed in JS, peer/state controls, or deferred review surfaces." });
  targets["compat-hold"].append({ text: "portal-utility-owner: phase-8-governance-sentinel\n   reason: empty guard rail for drained utility compatibility debt; rules must not be added here." });
  const counters = { required: 0, dynamic: 0, "compat-hold": 0, drop: 0 };

  for (const sourcePath of splitSourcePaths) {
    const source = parseCss(sourcePath);
    for (const node of source.nodes || []) {
      appendNodeByCategory(targets, node, usage, counters);
    }
  }
  counters.dedupe = Object.values(targets)
    .map((target) => pruneDuplicateUtilityRules(target))
    .reduce((total, count) => total + count, 0);

  return {
    counters,
    files: {
      [REQUIRED_UTILITIES_PATH]: targets.required.toString().trim() + "\n",
      [DYNAMIC_UTILITIES_PATH]: targets.dynamic.toString().trim() + "\n",
      [COMPAT_HOLD_UTILITIES_PATH]: targets["compat-hold"].toString().trim() + "\n"
    }
  };
}

function compareJsonFile(filePath, expected, failures) {
  const expectedJson = stableJson(expected);
  if (!existsSync(filePath)) {
    failures.push(`${relativePath(filePath)} is missing`);
    return;
  }
  const actual = readText(filePath);
  if (actual !== expectedJson) {
    failures.push(`${relativePath(filePath)} is stale; run npm run check:utility-ownership -- --write-ownership --write-report`);
  }
}

const failures = [];

if (WRITE_SPLIT) {
  const { counters, files } = buildSplitCss();
  for (const [filePath, content] of Object.entries(files)) {
    writeFileSync(filePath, content, "utf-8");
  }
  console.log(
    `portal utility split: ${counters.required} required, ${counters.dynamic} dynamic, ${counters["compat-hold"]} compat-hold, ${counters.drop} pruned rules, ${counters.dedupe || 0} duplicate rules removed`
  );
}

const ownership = buildOwnership();
const report = buildReport(ownership);

if (WRITE_OWNERSHIP) {
  writeJson(OWNERSHIP_PATH, ownership);
}
if (WRITE_REPORT) {
  writeJson(REPORT_PATH, report);
}

if (!WRITE_OWNERSHIP) {
  compareJsonFile(OWNERSHIP_PATH, ownership, failures);
}
if (!WRITE_REPORT) {
  compareJsonFile(REPORT_PATH, report, failures);
}

if (readText(PORTAL_CSS_INDEX_PATH).includes("utilities.deprecated.css")) {
  failures.push("utilities.deprecated.css must not be imported into production");
}
if (report.summary.compatHoldCount > ownership.maxCompatHold) {
  failures.push(`compat-hold utility count grew from ${ownership.maxCompatHold} to ${report.summary.compatHoldCount}`);
}
for (const entry of report.utilities) {
  if (entry.status !== "covered") {
    failures.push(`utility-like class ${entry.className} has no CSS owner`);
  }
}

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`ERROR: ${failure}`);
  }
  process.exit(1);
}

console.log(
  `portal utility ownership: OK (${report.summary.utilityCount} utilities, ${report.summary.compatHoldCount} compat-hold)`
);
