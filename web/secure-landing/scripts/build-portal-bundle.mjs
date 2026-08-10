import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build, transform } from "esbuild";

import { ensureSupportedRuntime } from "../lib/runtime-guard.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTDOOR_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTDOOR_ROOT, "..", "..");
const PORTAL_TEMPLATE_PATH = path.resolve(FRONTDOOR_ROOT, "portal-src", "portal.template.js");
const PORTAL_INTERNAL_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "internal", "index.js");
const PORTAL_REVIEW_SURFACE_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "review-surface-deferred.js");
const PORTAL_OPERATE_SURFACE_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "operate-surface-deferred.js");
const PORTAL_BUILD_SURFACE_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "build-surface-deferred.js");
const PORTAL_PROFILE_SURFACE_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "profile-surface-deferred.js");
const PORTAL_OVERVIEW_SURFACE_ENTRY = path.resolve(FRONTDOOR_ROOT, "portal-src", "overview-surface-deferred.js");
const PORTAL_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.js");
const PORTAL_REVIEW_SURFACE_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-review.js");
const PORTAL_OPERATE_SURFACE_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-operate.js");
const PORTAL_BUILD_SURFACE_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-build.js");
const PORTAL_PROFILE_SURFACE_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-profile.js");
const PORTAL_OVERVIEW_SURFACE_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal-overview.js");
const PORTAL_CSS_ASSET_PATH = path.resolve(REPO_ROOT, "public", "portal-assets", "portal.css");
const PORTAL_CSS_SOURCE_DIR = path.resolve(FRONTDOOR_ROOT, "portal-src", "styles");
const PORTAL_CSS_INDEX_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "index.css");
const PORTAL_CSS_FONT_TEMPLATE_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "fonts.template.css");
const SHARED_TOKEN_SOURCE_PATH = path.resolve(REPO_ROOT, "web", "shared", "shared-ui-tokens.css");
const PORTAL_SHARED_TOKEN_TARGET = path.resolve(REPO_ROOT, "public", "portal-assets", "shared-ui-tokens.css");
const FRONTDOOR_SHARED_TOKEN_TARGET = path.resolve(FRONTDOOR_ROOT, "public", "shared-ui-tokens.css");
const PORTAL_INTERNALS_PLACEHOLDER = "/* __PORTAL_INTERNALS__ */";
const PORTAL_FONT_PLACEHOLDERS = ["__PORTAL_FONT_SANS_URL__", "__PORTAL_FONT_MONO_URL__"];
const PORTAL_COMPAT_OVERRIDE_DISABLE_FLAG = "PORTAL_CSS_DISABLE_COMPAT_OVERRIDES";
const PORTAL_COMPAT_OVERRIDE_PATH = path.resolve(PORTAL_CSS_SOURCE_DIR, "overrides.compat.css");

function compatOverridesDisabled() {
  return process.env[PORTAL_COMPAT_OVERRIDE_DISABLE_FLAG] === "1";
}

function sourceImportsCompatOverrides() {
  return readFileSync(PORTAL_CSS_INDEX_PATH, "utf-8").includes(
    '@import "./overrides.compat.css" layer(overrides);'
  );
}

function emptyCompatOverridePlugin() {
  return {
    name: "portal-compat-overrides-empty",
    setup(buildApi) {
      buildApi.onLoad({ filter: /overrides\.compat\.css$/ }, (args) => {
        if (path.resolve(args.path) !== PORTAL_COMPAT_OVERRIDE_PATH) {
          return null;
        }
        return {
          contents: "/* PORTAL_CSS_DISABLE_COMPAT_OVERRIDES=1: overrides.compat.css emitted as empty for parity probe */\n",
          loader: "css"
        };
      });
    }
  };
}

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

async function minifyCssText(content) {
  return (await transform(content, {
    loader: "css",
    legalComments: "none",
    minify: true
  })).code.trim() + "\n";
}

async function writeMinifiedCssCopy(sourcePath, targetPath) {
  const sourceContent = readFileSync(sourcePath, "utf-8");
  return writeIfChanged(targetPath, await minifyCssText(sourceContent));
}

async function bundleCssEntry(entryPoint) {
  const plugins = [];
  if (compatOverridesDisabled()) {
    plugins.push(emptyCompatOverridePlugin());
  }
  const bundleResult = await build({
    absWorkingDir: REPO_ROOT,
    bundle: true,
    entryPoints: [entryPoint],
    legalComments: "none",
    minify: true,
    outfile: "portal.css",
    plugins,
    write: false
  });
  const outputText = bundleResult.outputFiles?.[0]?.text;
  if (!outputText) {
    throw new Error(`esbuild did not emit CSS for ${path.relative(REPO_ROOT, entryPoint)}`);
  }
  return outputText.trim() + "\n";
}

async function renderPortalCssAsset() {
  const fontTemplate = readFileSync(PORTAL_CSS_FONT_TEMPLATE_PATH, "utf-8");
  const bundledPortalCss = await bundleCssEntry(PORTAL_CSS_INDEX_PATH);
  const renderedCss = `${await minifyCssText(fontTemplate)}${bundledPortalCss}`;

  if (renderedCss.includes("@import")) {
    throw new Error("Generated portal.css must not contain runtime @import rules");
  }
  if (renderedCss.includes("__PORTAL_SHARED_TOKENS_URL__")) {
    throw new Error("Generated portal.css must not contain the shared token URL placeholder");
  }
  for (const placeholder of PORTAL_FONT_PLACEHOLDERS) {
    if (!renderedCss.includes(placeholder)) {
      throw new Error(`Generated portal.css missing required font placeholder: ${placeholder}`);
    }
  }

  return renderedCss;
}

async function buildPortalCssAsset() {
  if (compatOverridesDisabled()) {
    console.log(
      `portal css probe mode: ${PORTAL_COMPAT_OVERRIDE_DISABLE_FLAG}=1 – overrides.compat.css emitted as empty; do not commit the resulting portal.css.`
    );
  }
  return writeIfChanged(PORTAL_CSS_ASSET_PATH, await renderPortalCssAsset());
}

function stripStandaloneLineComments(content) {
  return content.replace(/^\s*\/\/.*$/gm, "").replace(/\n\s*\n+/g, "\n").trimEnd() + "\n";
}

async function bundleText(entryPoint, options = {}) {
  const buildOptions = {
    absWorkingDir: REPO_ROOT,
    bundle: true,
    entryPoints: [entryPoint],
    format: options.format || "iife",
    globalName: options.globalName,
    legalComments: "none",
    metafile: Boolean(options.metafile),
    minify: Boolean(options.minify),
    platform: "browser",
    target: ["es2022"],
    write: false
  };
  if (Object.prototype.hasOwnProperty.call(options, "minifyIdentifiers")) {
    buildOptions.minifyIdentifiers = Boolean(options.minifyIdentifiers);
  }
  if (Object.prototype.hasOwnProperty.call(options, "keepNames")) {
    buildOptions.keepNames = Boolean(options.keepNames);
  }
  if (Object.prototype.hasOwnProperty.call(options, "minifySyntax")) {
    buildOptions.minifySyntax = Boolean(options.minifySyntax);
  }
  if (Object.prototype.hasOwnProperty.call(options, "minifyWhitespace")) {
    buildOptions.minifyWhitespace = Boolean(options.minifyWhitespace);
  }
  const bundleResult = await build(buildOptions);
  const outputText = bundleResult.outputFiles?.[0]?.text;
  if (!outputText) {
    throw new Error(`esbuild did not emit a bundle for ${path.relative(REPO_ROOT, entryPoint)}`);
  }
  return { text: outputText, metafile: bundleResult.metafile || null };
}

await ensureSupportedRuntime();

const cssOnly = process.argv.includes("--css-only");
const emitMetafile = process.argv.includes("--emit-metafile");
const METAFILE_DIR = path.resolve(FRONTDOOR_ROOT, ".metafiles");
const PORTAL_METAFILE_PATH = path.resolve(METAFILE_DIR, "portal-bundle.json");
const REVIEW_SURFACE_METAFILE_PATH = path.resolve(METAFILE_DIR, "review-surface-bundle.json");
const OPERATE_SURFACE_METAFILE_PATH = path.resolve(METAFILE_DIR, "operate-surface-bundle.json");
const BUILD_SURFACE_METAFILE_PATH = path.resolve(METAFILE_DIR, "build-surface-bundle.json");
const PROFILE_SURFACE_METAFILE_PATH = path.resolve(METAFILE_DIR, "profile-surface-bundle.json");
const OVERVIEW_SURFACE_METAFILE_PATH = path.resolve(METAFILE_DIR, "overview-surface-bundle.json");

if (process.argv.includes("--check-css")) {
  if (compatOverridesDisabled() && sourceImportsCompatOverrides()) {
    throw new Error(
      `${PORTAL_COMPAT_OVERRIDE_DISABLE_FLAG}=1 is a parity-probe build flag and must not be combined with --check-css.`
    );
  }
  if (cssOnly) {
    throw new Error("--css-only cannot be combined with --check-css; --check-css already runs only the CSS render path.");
  }
  const expectedCss = await renderPortalCssAsset();
  const currentCss = existsSync(PORTAL_CSS_ASSET_PATH) ? readFileSync(PORTAL_CSS_ASSET_PATH, "utf-8") : "";
  if (currentCss !== expectedCss) {
    throw new Error(
      `${path.relative(REPO_ROOT, PORTAL_CSS_ASSET_PATH)} is stale. Run npm run build:portal from ${path.relative(REPO_ROOT, FRONTDOOR_ROOT)}.`
    );
  }
  console.log(`portal css generated artifact is fresh: ${path.relative(REPO_ROOT, PORTAL_CSS_ASSET_PATH)}`);
  process.exit(0);
}

if (cssOnly) {
  const portalCssChanged = await buildPortalCssAsset();
  const portalCssStats = statSync(PORTAL_CSS_ASSET_PATH);
  console.log(
    `portal css ${portalCssChanged ? "updated" : "unchanged"} (--css-only): ${path.relative(REPO_ROOT, PORTAL_CSS_ASSET_PATH)} (${portalCssStats.size} bytes)`
  );
  process.exit(0);
}

const portalTemplate = readFileSync(PORTAL_TEMPLATE_PATH, "utf-8");
if (!portalTemplate.includes(PORTAL_INTERNALS_PLACEHOLDER)) {
  throw new Error(`Portal template missing internal bundle placeholder: ${PORTAL_TEMPLATE_PATH}`);
}

const internalBuild = await bundleText(PORTAL_INTERNAL_ENTRY, {
  format: "iife",
  globalName: "__PortalInternal",
  metafile: emitMetafile,
  minify: true
});
const deferredReviewSurfaceBuild = await bundleText(PORTAL_REVIEW_SURFACE_ENTRY, {
  format: "esm",
  metafile: emitMetafile,
  minifySyntax: true,
  minifyWhitespace: true
});
const deferredOperateSurfaceBuild = await bundleText(PORTAL_OPERATE_SURFACE_ENTRY, {
  format: "esm",
  keepNames: true,
  metafile: emitMetafile,
  minifyIdentifiers: true,
  minifySyntax: true,
  minifyWhitespace: true
});
const deferredBuildSurfaceBuild = await bundleText(PORTAL_BUILD_SURFACE_ENTRY, {
  format: "esm",
  metafile: emitMetafile,
  minifySyntax: true,
  minifyWhitespace: true
});
const deferredProfileSurfaceBuild = await bundleText(PORTAL_PROFILE_SURFACE_ENTRY, {
  format: "esm",
  metafile: emitMetafile,
  minifySyntax: true,
  minifyWhitespace: true
});
const deferredOverviewSurfaceBuild = await bundleText(PORTAL_OVERVIEW_SURFACE_ENTRY, {
  format: "esm",
  metafile: emitMetafile,
  minifySyntax: true,
  minifyWhitespace: true
});

if (emitMetafile) {
  mkdirSync(METAFILE_DIR, { recursive: true });
  if (internalBuild.metafile) {
    writeFileSync(PORTAL_METAFILE_PATH, JSON.stringify(internalBuild.metafile, null, 2), "utf-8");
  }
  if (deferredReviewSurfaceBuild.metafile) {
    writeFileSync(REVIEW_SURFACE_METAFILE_PATH, JSON.stringify(deferredReviewSurfaceBuild.metafile, null, 2), "utf-8");
  }
  if (deferredOperateSurfaceBuild.metafile) {
    writeFileSync(OPERATE_SURFACE_METAFILE_PATH, JSON.stringify(deferredOperateSurfaceBuild.metafile, null, 2), "utf-8");
  }
  if (deferredBuildSurfaceBuild.metafile) {
    writeFileSync(BUILD_SURFACE_METAFILE_PATH, JSON.stringify(deferredBuildSurfaceBuild.metafile, null, 2), "utf-8");
  }
  if (deferredProfileSurfaceBuild.metafile) {
    writeFileSync(PROFILE_SURFACE_METAFILE_PATH, JSON.stringify(deferredProfileSurfaceBuild.metafile, null, 2), "utf-8");
  }
  if (deferredOverviewSurfaceBuild.metafile) {
    writeFileSync(OVERVIEW_SURFACE_METAFILE_PATH, JSON.stringify(deferredOverviewSurfaceBuild.metafile, null, 2), "utf-8");
  }
  console.log(`portal metafile: ${path.relative(REPO_ROOT, PORTAL_METAFILE_PATH)}`);
  console.log(`review surface metafile: ${path.relative(REPO_ROOT, REVIEW_SURFACE_METAFILE_PATH)}`);
  console.log(`operate surface metafile: ${path.relative(REPO_ROOT, OPERATE_SURFACE_METAFILE_PATH)}`);
  console.log(`build surface metafile: ${path.relative(REPO_ROOT, BUILD_SURFACE_METAFILE_PATH)}`);
  console.log(`profile surface metafile: ${path.relative(REPO_ROOT, PROFILE_SURFACE_METAFILE_PATH)}`);
  console.log(`overview surface metafile: ${path.relative(REPO_ROOT, OVERVIEW_SURFACE_METAFILE_PATH)}`);
}

const nextPortalBundle = stripStandaloneLineComments(
  portalTemplate.replace(PORTAL_INTERNALS_PLACEHOLDER, internalBuild.text.trim())
);
// Mangle lexical identifiers only. keepNames preserves function/class names
// used in diagnostics, and esbuild leaves object keys and explicit exports intact.
const compactPortalBundle = (await transform(nextPortalBundle, {
  loader: "js",
  keepNames: true,
  legalComments: "none",
  minifyIdentifiers: true,
  minifySyntax: true,
  minifyWhitespace: true,
  target: ["es2022"]
})).code.trim();
const portalChanged = writeIfChanged(PORTAL_ASSET_PATH, `${compactPortalBundle}\n`);
const reviewSurfaceChanged = writeIfChanged(PORTAL_REVIEW_SURFACE_ASSET_PATH, `${deferredReviewSurfaceBuild.text.trim()}\n`);
const operateSurfaceChanged = writeIfChanged(PORTAL_OPERATE_SURFACE_ASSET_PATH, `${deferredOperateSurfaceBuild.text.trim()}\n`);
const buildSurfaceChanged = writeIfChanged(PORTAL_BUILD_SURFACE_ASSET_PATH, `${deferredBuildSurfaceBuild.text.trim()}\n`);
const profileSurfaceChanged = writeIfChanged(PORTAL_PROFILE_SURFACE_ASSET_PATH, `${deferredProfileSurfaceBuild.text.trim()}\n`);
const overviewSurfaceChanged = writeIfChanged(PORTAL_OVERVIEW_SURFACE_ASSET_PATH, `${deferredOverviewSurfaceBuild.text.trim()}\n`);
const portalCssChanged = await buildPortalCssAsset();
const portalTokenChanged = await writeMinifiedCssCopy(SHARED_TOKEN_SOURCE_PATH, PORTAL_SHARED_TOKEN_TARGET);
const frontdoorTokenChanged = await writeMinifiedCssCopy(SHARED_TOKEN_SOURCE_PATH, FRONTDOOR_SHARED_TOKEN_TARGET);

const portalStats = statSync(PORTAL_ASSET_PATH);
const reviewSurfaceStats = statSync(PORTAL_REVIEW_SURFACE_ASSET_PATH);
const operateSurfaceStats = statSync(PORTAL_OPERATE_SURFACE_ASSET_PATH);
const buildSurfaceStats = statSync(PORTAL_BUILD_SURFACE_ASSET_PATH);
const profileSurfaceStats = statSync(PORTAL_PROFILE_SURFACE_ASSET_PATH);
const overviewSurfaceStats = statSync(PORTAL_OVERVIEW_SURFACE_ASSET_PATH);
const portalCssStats = statSync(PORTAL_CSS_ASSET_PATH);
console.log(
  `portal bundle ${portalChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_ASSET_PATH)} (${portalStats.size} bytes)`
);
console.log(
  `review surface bundle ${reviewSurfaceChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_REVIEW_SURFACE_ASSET_PATH)} (${reviewSurfaceStats.size} bytes)`
);
console.log(
  `operate surface bundle ${operateSurfaceChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_OPERATE_SURFACE_ASSET_PATH)} (${operateSurfaceStats.size} bytes)`
);
console.log(
  `build surface bundle ${buildSurfaceChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_BUILD_SURFACE_ASSET_PATH)} (${buildSurfaceStats.size} bytes)`
);
console.log(
  `profile surface bundle ${profileSurfaceChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_PROFILE_SURFACE_ASSET_PATH)} (${profileSurfaceStats.size} bytes)`
);
console.log(
  `overview surface bundle ${overviewSurfaceChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_OVERVIEW_SURFACE_ASSET_PATH)} (${overviewSurfaceStats.size} bytes)`
);
console.log(
  `portal css ${portalCssChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_CSS_ASSET_PATH)} (${portalCssStats.size} bytes)`
);
console.log(
  `shared tokens ${portalTokenChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, PORTAL_SHARED_TOKEN_TARGET)}`
);
console.log(
  `frontdoor tokens ${frontdoorTokenChanged ? "updated" : "unchanged"}: ${path.relative(REPO_ROOT, FRONTDOOR_SHARED_TOKEN_TARGET)}`
);
