import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.css");
const PORTAL_REVIEW_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.css");
const PORTAL_HTML_PATH = path.resolve(REPO_ROOT, "portal.html");
const PORTAL_TEMPLATE_SOURCE_PATH = path.resolve(FRONTDOOR_ROOT, "portal-src", "portal.template.js");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "index.css");

const GENERATED_REQUIRED_PATTERNS = [
  ["font sans placeholder", /__PORTAL_FONT_SANS_URL__/],
  ["font mono placeholder", /__PORTAL_FONT_MONO_URL__/],
  ["inlined shared target token", /--ux-target-min-size\s*:/],
  ["standard scrollbar support", /scrollbar-width\s*:/],
  ["prefers contrast support", /prefers-contrast\s*:\s*more/],
  ["forced colors support", /forced-colors\s*:\s*active/],
  ["deferred content visibility", /content-visibility\s*:\s*auto/],
  ["dark group hover selector", /\.dark\s+\.group:hover\s+\.dark\\:group-hover\\:text-white/]
];

const DISALLOWED_PATTERNS = [
  ["runtime @import", /@import\b/],
  ["substring class selector", /\[class\*=/],
  ["transition all shorthand", /transition\s*:\s*all\b/],
  ["transition-property all", /transition-property\s*:\s*all\b/],
  ["broken dark group hover selector", /\.group:hover\s+\.dark\s+\.dark\\:group-hover\\:text-white/],
  ["shared token URL placeholder", /__PORTAL_SHARED_TOKENS_URL__/]
];
const SOURCE_DISALLOWED_PATTERNS = DISALLOWED_PATTERNS.filter(([name]) => name !== "runtime @import");
const INDEX_REQUIRED_PATTERNS = [
  ["layer order", /@layer\s+tokens\s*,\s*base\s*,\s*components\s*,\s*utilities\s*,\s*overrides\s*;/],
  ["shared tokens import", /@import\s+"(?:\.\.\/){4}web\/shared\/shared-ui-tokens\.css"\s+layer\(tokens\)\s*;/],
  ["tokens import", /@import\s+"\.\/tokens\.css"\s+layer\(tokens\)\s*;/],
  ["base import", /@import\s+"\.\/base\.css"\s+layer\(base\)\s*;/],
  ["workspace surfaces component import", /@import\s+"\.\/components\/workspace-surfaces\.css"\s+layer\(components\)\s*;/],
  ["operator console component import", /@import\s+"\.\/components\/operator-console\.css"\s+layer\(components\)\s*;/],
  ["surface normalization component import", /@import\s+"\.\/components\/surface-normalization\.css"\s+layer\(components\)\s*;/],
  ["required utilities import", /@import\s+"\.\/utilities\.required\.css"\s+layer\(utilities\)\s*;/],
  ["dynamic utilities import", /@import\s+"\.\/utilities\.dynamic\.css"\s+layer\(utilities\)\s*;/],
  ["compat-hold utilities import", /@import\s+"\.\/utilities\.compat-hold\.css"\s+layer\(utilities\)\s*;/],
  ["performance overrides import", /@import\s+"\.\/overrides\.performance\.css"\s+layer\(overrides\)\s*;/],
  ["accessibility overrides import", /@import\s+"\.\/overrides\.accessibility\.css"\s+layer\(overrides\)\s*;/]
];
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
const UTILITY_OWNER_ALLOWLIST = new Set([
  "dark",
  "light",
  "performance-lite"
]);
const UTILITY_PREFIX_PATTERN =
  /^(?:-?m[trblxy]?-.+|p[trblxy]?-.+|space-[xy]-.+|gap-.+|grid-cols-.+|col-span-.+|row-span-.+|flex-.+|items-.+|justify-.+|self-.+|place-.+|w-.+|h-.+|min-w-.+|min-h-.+|max-w-.+|max-h-.+|rounded-.+|border-.+|bg-.+|from-.+|via-.+|to-.+|text-.+|font-.+|tracking-.+|leading-.+|shadow-.+|ring-.+|opacity-.+|overflow-.+|object-.+|inset-.+|top-.+|right-.+|bottom-.+|left-.+|-?z-.+|cursor-.+|pointer-events-.+|select-.+|resize-.+|whitespace-.+|break-.+|duration-.+|ease-.+|transition-.+|translate-.+|scale-.+|backdrop-.+|animate-.+|outline-.+|fill-.+|stroke-.+|order-.+|basis-.+|shrink-.+|grow-.+|underline|no-underline)$/;
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

function listCssFiles(directory) {
  const cssFiles = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      cssFiles.push(...listCssFiles(entryPath));
    } else if (entry.isFile() && entry.name.endsWith(".css")) {
      cssFiles.push(entryPath);
    }
  }
  return cssFiles.sort();
}

function checkDisallowed(label, content, failures, patterns = DISALLOWED_PATTERNS) {
  for (const [name, pattern] of patterns) {
    if (pattern.test(content)) {
      failures.push(`${label}: disallowed ${name}`);
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
  return (
    UTILITY_EXACT_CLASSES.has(base) ||
    UTILITY_PREFIX_PATTERN.test(base)
  );
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
  const html = readFileSync(PORTAL_HTML_PATH, "utf-8");
  const portalTemplate = readFileSync(PORTAL_TEMPLATE_SOURCE_PATH, "utf-8");

  for (const match of html.matchAll(/\bclass=(["'])(.*?)\1/gs)) {
    recordClassTokenList(match[2], path.relative(REPO_ROOT, PORTAL_HTML_PATH), classTokens);
  }

  for (const match of portalTemplate.matchAll(/\bclass=(["'])(.*?)\1/gs)) {
    recordClassTokenList(match[2], path.relative(REPO_ROOT, PORTAL_TEMPLATE_SOURCE_PATH), classTokens);
  }

  for (const match of portalTemplate.matchAll(/(["'`])((?:\\.|(?!\1)[\s\S])*?)\1/g)) {
    recordClassTokenList(match[2], path.relative(REPO_ROOT, PORTAL_TEMPLATE_SOURCE_PATH), classTokens);
  }

  return classTokens;
}

function cssContainsClassToken(css, token) {
  const escapedClass = escapeCssClassToken(token);
  const pattern = new RegExp(`\\.${escapeRegExp(escapedClass)}(?=[\\s.#:{,>+~\\[]|$)`);
  return pattern.test(css);
}

function checkUtilityCoverage(generatedCss, failures) {
  const reviewCss = readFileSync(PORTAL_REVIEW_CSS_ASSET_PATH, "utf-8");
  const searchableCss = `${generatedCss}\n${reviewCss}`;
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

const failures = [];
const generatedCss = readFileSync(PORTAL_CSS_ASSET_PATH, "utf-8");
checkDisallowed(path.relative(REPO_ROOT, PORTAL_CSS_ASSET_PATH), generatedCss, failures);

for (const [name, pattern] of GENERATED_REQUIRED_PATTERNS) {
  if (!pattern.test(generatedCss)) {
    failures.push(`${path.relative(REPO_ROOT, PORTAL_CSS_ASSET_PATH)}: missing ${name}`);
  }
}

const indexCss = readFileSync(PORTAL_CSS_INDEX_PATH, "utf-8");
for (const [name, pattern] of INDEX_REQUIRED_PATTERNS) {
  if (!pattern.test(indexCss)) {
    failures.push(`${path.relative(REPO_ROOT, PORTAL_CSS_INDEX_PATH)}: missing ${name}`);
  }
}

for (const sourcePath of listCssFiles(PORTAL_CSS_SOURCE_DIR)) {
  const sourceLabel = path.relative(REPO_ROOT, sourcePath);
  const sourceContent = readFileSync(sourcePath, "utf-8");
  const disallowedPatterns =
    sourcePath === PORTAL_CSS_INDEX_PATH
      ? SOURCE_DISALLOWED_PATTERNS
      : DISALLOWED_PATTERNS;
  checkDisallowed(sourceLabel, sourceContent, failures, disallowedPatterns);
}

checkUtilityCoverage(generatedCss, failures);

if (failures.length > 0) {
  for (const failure of failures) {
    console.error(`ERROR: ${failure}`);
  }
  process.exit(1);
}

console.log("portal css contract: OK");
