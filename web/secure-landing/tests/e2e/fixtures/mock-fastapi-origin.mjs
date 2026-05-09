// Mock FastAPI origin for the @portal-browser Playwright suite.
//
// The front-door's /portal route fetches GET / from the configured
// TP_FASTAPI_ORIGIN and proxies the response body through to the
// browser (web/secure-landing/app/portal/route.js:174). That fetch is
// server-side from the Next.js process, so browser-level page.route()
// cannot intercept it. This tiny Node origin stands in for FastAPI
// and serves the real portal.html template with placeholder URLs
// substituted to inert mock-served stubs — keeping the assertions
// against live portal markup, not a hand-written fixture.
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

// Mock-served stub asset paths the placeholder substitutions point at.
// Each stub returns a minimal valid body so the browser does not log
// console errors that would trip the @portal-browser suite's console-error
// gate. The portal shell's interactive JS is intentionally NOT loaded —
// these tests assert markup contracts, not runtime behavior. Live JS/CSS
// behavior remains the responsibility of validate_portal_browser_smoke.py.
const STUB_ROUTES = new Map([
  ["/__mock-portal-asset.css", { type: "text/css; charset=utf-8", body: "/* mock portal css */\n" }],
  ["/__mock-portal-asset.js", { type: "text/javascript; charset=utf-8", body: "/* mock portal js */\n" }],
  ["/__mock-portal-asset.svg", {
    type: "image/svg+xml",
    body: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>'
  }]
]);

// Substitution map for the __PORTAL_*_URL__ placeholders defined in
// src/transformation_portal/portal/asset_bundle.py. CSS/JS placeholders
// route to the inert stubs above; the brand SVG and font placeholders
// route to a transparent SVG stub (the font is also stubbed to a tiny
// SVG since the preload is non-essential for shell-anchor coverage).
const PLACEHOLDER_SUBSTITUTIONS = Object.freeze({
  __PORTAL_CSS_URL__: "/__mock-portal-asset.css",
  __PORTAL_REVIEW_CSS_URL__: "/__mock-portal-asset.css",
  __PORTAL_JS_URL__: "/__mock-portal-asset.js",
  __PORTAL_BUILD_JS_URL__: "/__mock-portal-asset.js",
  __PORTAL_OPERATE_JS_URL__: "/__mock-portal-asset.js",
  __PORTAL_OVERVIEW_JS_URL__: "/__mock-portal-asset.js",
  __PORTAL_REVIEW_JS_URL__: "/__mock-portal-asset.js",
  __PORTAL_BRAND_LIGHT_URL__: "/__mock-portal-asset.svg",
  __PORTAL_BRAND_DARK_URL__: "/__mock-portal-asset.svg",
  __PORTAL_FONT_SANS_URL__: "/__mock-portal-asset.svg"
});

function renderPortalHtml() {
  const template = readFileSync(PORTAL_HTML_PATH, "utf-8");
  let rendered = template;
  for (const [placeholder, replacement] of Object.entries(PLACEHOLDER_SUBSTITUTIONS)) {
    rendered = rendered.split(placeholder).join(replacement);
  }
  // Drop the woff2 font preload — without a real font body the browser
  // emits a console error about the preload type/cors mismatch which would
  // trip @portal-browser console-error gates. The preload is a perf hint,
  // not part of the markup contract.
  rendered = rendered.replace(
    /<link\s+rel="preload"\s+as="font"[^>]*>\s*/g,
    ""
  );
  const unresolved = rendered.match(/__PORTAL_[A-Z0-9_]+__/g);
  if (unresolved) {
    throw new Error(`mock-fastapi-origin: unresolved portal placeholders: ${[...new Set(unresolved)].join(", ")}`);
  }
  return rendered;
}

let cachedPortalHtml = null;
function getPortalHtml() {
  if (cachedPortalHtml === null) {
    cachedPortalHtml = renderPortalHtml();
  }
  return cachedPortalHtml;
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

  if (req.method === "GET" && STUB_ROUTES.has(pathname)) {
    const stub = STUB_ROUTES.get(pathname);
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
