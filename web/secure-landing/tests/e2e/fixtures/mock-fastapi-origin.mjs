// Mock FastAPI origin for the @portal-browser Playwright suite.
//
// The front-door's /portal route fetches GET / from the configured
// TP_FASTAPI_ORIGIN and proxies the response body through to the
// browser (web/secure-landing/app/portal/route.js:174). That fetch is
// server-side from the Next.js process, so browser-level page.route()
// cannot intercept it. This tiny Node origin stands in for FastAPI
// and serves the real portal.html template with placeholder URLs
// substituted to mock-served assets. CSS, fonts, and brand assets come
// from the production bundle; JavaScript stays inert by default so
// structural specs are deterministic. Hydrated specs can intercept the
// distinct script URLs and fulfill them with the production bundles.
//
// Lifecycle is owned by Playwright's webServer config (no
// globalSetup/globalTeardown). The script starts a node:http server
// and stays alive until Playwright SIGTERMs it.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HOST = process.env.MOCK_FASTAPI_HOST || "127.0.0.1";
const PORT = Number(process.env.MOCK_FASTAPI_PORT || 9999);
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..", "..");
const PORTAL_HTML_PATH = process.env.MOCK_FASTAPI_PORTAL_HTML
  ? path.resolve(process.env.MOCK_FASTAPI_PORTAL_HTML)
  : path.join(REPO_ROOT, "portal.html");
const PORTAL_ASSETS_DIR = path.join(REPO_ROOT, "public", "portal-assets");
const PUBLIC_ORIGIN = `http://${HOST}:${PORT}`;

// Mock-served asset paths the placeholder substitutions point at.
// Asset URLs in the substituted HTML are *fully qualified* back to the
// mock origin (PUBLIC_ORIGIN below), because the front-door serves the
// proxied portal HTML at port 3000 — root-relative paths would resolve
// to the front-door origin and 404 there. Cross-origin GETs to port
// 9999 work without CORS preflight (no fetch credentials, no custom
// headers), and the /portal route doesn't apply CSP (lib/http.js:138),
// so cross-origin <link>/<script>/<img> loads have no policy hurdle.
//
// Production visual assets keep layout/a11y checks representative. Script
// stubs keep older structural specs deterministic; hydrated specs intercept
// their distinct paths with production bundles.
const SCRIPT_STUB_ROUTES = new Map([
  "/__mock-portal/portal.js",
  "/__mock-portal/portal-build.js",
  "/__mock-portal/portal-profile.js",
  "/__mock-portal/portal-operate.js",
  "/__mock-portal/portal-overview.js",
  "/__mock-portal/portal-review.js",
].map((pathname) => [pathname, { type: "text/javascript; charset=utf-8", body: "/* mock portal js */\n" }]));

const FILE_ROUTES = new Map([
  ["/__mock-portal/portal-review.css", { type: "text/css; charset=utf-8", file: "portal-review.css" }],
  ["/__mock-portal/portal-sans.woff2", { type: "font/woff2", file: "fonts/portal-sans.woff2" }],
  ["/__mock-portal/portal-mono.woff2", { type: "font/woff2", file: "fonts/portal-mono.woff2" }],
  ["/__mock-portal/dna-symbol-light.svg", { type: "image/svg+xml", file: "brand/dna-symbol-light.svg" }],
  ["/__mock-portal/dna-symbol-dark.svg", { type: "image/svg+xml", file: "brand/dna-symbol-dark.svg" }],
]);

function readPortalCss() {
  return readFileSync(path.join(PORTAL_ASSETS_DIR, "portal.css"), "utf-8")
    .replaceAll("__PORTAL_FONT_SANS_URL__", `${PUBLIC_ORIGIN}/__mock-portal/portal-sans.woff2`)
    .replaceAll("__PORTAL_FONT_MONO_URL__", `${PUBLIC_ORIGIN}/__mock-portal/portal-mono.woff2`);
}

// Substitution map for the __PORTAL_*_URL__ placeholders defined in
// src/transformation_portal/portal/asset_bundle.py. All placeholders
// resolve to fully-qualified URLs at the mock origin so the browser
// loads them from port 9999 (where the stubs are served), not from the
// front-door at port 3000 where they would 404.
const PLACEHOLDER_SUBSTITUTIONS = Object.freeze({
  __PORTAL_CSS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal.css`,
  __PORTAL_REVIEW_CSS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-review.css`,
  __PORTAL_JS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal.js`,
  __PORTAL_BUILD_JS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-build.js`,
  __PORTAL_PROFILE_JS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-profile.js`,
  __PORTAL_OPERATE_JS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-operate.js`,
  __PORTAL_OVERVIEW_JS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-overview.js`,
  __PORTAL_REVIEW_JS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-review.js`,
  __PORTAL_BRAND_LIGHT_URL__: `${PUBLIC_ORIGIN}/__mock-portal/dna-symbol-light.svg`,
  __PORTAL_BRAND_DARK_URL__: `${PUBLIC_ORIGIN}/__mock-portal/dna-symbol-dark.svg`,
  __PORTAL_FONT_SANS_URL__: `${PUBLIC_ORIGIN}/__mock-portal/portal-sans.woff2`
});

function renderPortalHtml() {
  const template = readFileSync(PORTAL_HTML_PATH, "utf-8");
  let rendered = template;
  for (const [placeholder, replacement] of Object.entries(PLACEHOLDER_SUBSTITUTIONS)) {
    rendered = rendered.split(placeholder).join(replacement);
  }
  const unresolved = rendered.match(/__PORTAL_[A-Z0-9_]+__/g);
  if (unresolved) {
    throw new Error(`mock-fastapi-origin: unresolved portal placeholders: ${[...new Set(unresolved)].join(", ")}`);
  }
  return rendered;
}

function getPortalHtml() {
  return renderPortalHtml();
}

const server = createServer((req, res) => {
  const url = new URL(req.url || "/", `http://${HOST}:${PORT}`);
  const pathname = url.pathname;

  res.setHeader("Cache-Control", "no-store");

  if (req.method === "GET" && (pathname === "/" || pathname === "/healthz" || pathname === "/ready")) {
    if (pathname === "/healthz" || pathname === "/ready") {
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("ok");
      return;
    }
    try {
      const body = getPortalHtml();
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(body);
    } catch (error) {
      res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(`mock-fastapi-origin: ${error?.message || String(error)}`);
    }
    return;
  }

  // The front-door's preflight-backend-auth.mjs probes
  // GET /v1/config-metadata?pipeline=lux-depth-v3 on dev-server start.
  // A 200 here lets the dev server come up; non-2xx fails closed.
  if (req.method === "GET" && pathname === "/v1/config-metadata") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ pipeline: url.searchParams.get("pipeline") || "" }));
    return;
  }

  if (req.method === "GET" && pathname === "/__mock-portal/portal.css") {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.writeHead(200, { "Content-Type": "text/css; charset=utf-8" });
    res.end(readPortalCss());
    return;
  }

  if (req.method === "GET" && FILE_ROUTES.has(pathname)) {
    const asset = FILE_ROUTES.get(pathname);
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.writeHead(200, { "Content-Type": asset.type });
    res.end(readFileSync(path.join(PORTAL_ASSETS_DIR, asset.file)));
    return;
  }

  if (req.method === "GET" && SCRIPT_STUB_ROUTES.has(pathname)) {
    const stub = SCRIPT_STUB_ROUTES.get(pathname);
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.writeHead(200, { "Content-Type": stub.type });
    res.end(stub.body);
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("not found");
});

server.listen(PORT, HOST, () => {
  process.stdout.write(`mock-fastapi-origin: listening on http://${HOST}:${PORT}\n`);
});

const shutdown = () => {
  server.close(() => process.exit(0));
};
process.once("SIGTERM", shutdown);
process.once("SIGINT", shutdown);
