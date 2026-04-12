import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

import { ensureSupportedRuntime } from "../lib/runtime-guard.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_TEMPLATE_PATH = path.resolve(FRONTDOOR_ROOT, "portal-src", "portal.template.js");
const PORTAL_INTERNAL_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "internal", "index.js");
const PORTAL_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.js");
const SHARED_TOKEN_SOURCE_PATH = path.resolve(REPO_ROOT, "web", "shared", "shared-ui-tokens.css");
const PORTAL_SHARED_TOKEN_TARGET = path.resolve(REPO_ROOT, "public", "portal-assets", "shared-ui-tokens.css");
const FRONTDOOR_SHARED_TOKEN_TARGET = path.resolve(FRONTDOOR_ROOT, "public", "shared-ui-tokens.css");
const PORTAL_INTERNALS_PLACEHOLDER = "/* __PORTAL_INTERNALS__ */";

function writeIfChanged(targetPath, content) {
  const nextContent = typeof content === "string" ? content : String(content);
  const currentContent = existsSync(targetPath) ? readFileSync(targetPath, "utf-8") : null;
  if (currentContent === nextContent) {
    return false;
  }
  mkdirSync(path.dirname(targetPath), { recursive: true });
  writeFileSync(targetPath, nextContent, "utf-8");
  return true;
}

function copyIfChanged(sourcePath, targetPath) {
  const sourceContent = readFileSync(sourcePath, "utf-8");
  return writeIfChanged(targetPath, sourceContent);
}

await ensureSupportedRuntime();

const portalTemplate = readFileSync(PORTAL_TEMPLATE_PATH, "utf-8");
if (!portalTemplate.includes(PORTAL_INTERNALS_PLACEHOLDER)) {
  throw new Error(`Portal template missing internal bundle placeholder: ${PORTAL_TEMPLATE_PATH}`);
}

const bundleResult = await build({
  absWorkingDir: REPO_ROOT,
  bundle: true,
  entryPoints: [PORTAL_INTERNAL_ENTRY],
  format: "iife",
  globalName: "__PortalInternal",
  minify: false,
  platform: "browser",
  target: ["es2022"],
  write: false
});

const internalBundle = bundleResult.outputFiles?.[0]?.text;
if (!internalBundle) {
  throw new Error("esbuild did not emit the internal portal bundle");
}

const nextPortalBundle = portalTemplate.replace(PORTAL_INTERNALS_PLACEHOLDER, internalBundle.trim());
const portalChanged = writeIfChanged(PORTAL_ASSET_PATH, nextPortalBundle);
const portalTokenChanged = copyIfChanged(SHARED_TOKEN_SOURCE_PATH, PORTAL_SHARED_TOKEN_TARGET);
const frontdoorTokenChanged = copyIfChanged(SHARED_TOKEN_SOURCE_PATH, FRONTDOOR_SHARED_TOKEN_TARGET);

const portalStats = statSync(PORTAL_ASSET_PATH);
console.log(
  `portal bundle ${portalChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_ASSET_PATH)} (${portalStats.size} bytes)`
);
console.log(
  `shared tokens ${portalTokenChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_SHARED_TOKEN_TARGET)}`
);
console.log(
  `frontdoor tokens ${frontdoorTokenChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, FRONTDOOR_SHARED_TOKEN_TARGET)}`
);
