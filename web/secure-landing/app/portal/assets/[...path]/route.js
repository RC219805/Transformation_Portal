import { getConfig } from "../../../../lib/config.js";
import { applySecurityHeaders } from "../../../../lib/http.js";
import {
  buildUpstreamHeaders,
  buildUpstreamUrl,
  copyUpstreamResponseHeaders
} from "../../../../lib/proxy.js";

export const runtime = "nodejs";

const ALLOWED_PORTAL_ASSET_PATHS = new Set([
  "portal.css",
  "portal.js",
  "fonts/portal-sans.woff2",
  "fonts/portal-mono.woff2"
]);

function errorResponse(status, message) {
  return applySecurityHeaders(
    new Response(message, {
      status,
      headers: {
        "Cache-Control": "no-store"
      }
    })
  );
}

function normalizeAssetPath(pathParts) {
  if (!Array.isArray(pathParts) || pathParts.length === 0) {
    return null;
  }

  const normalized = [];
  for (const rawPart of pathParts) {
    const part = String(rawPart ?? "");
    if (
      !part ||
      part !== part.trim() ||
      part === "." ||
      part === ".." ||
      part.includes("/") ||
      part.includes("\\") ||
      part.includes("\0")
    ) {
      return null;
    }
    normalized.push(part);
  }

  const assetPath = normalized.join("/");
  return ALLOWED_PORTAL_ASSET_PATHS.has(assetPath) ? assetPath : null;
}

async function fetchUpstreamAsset(request, upstreamUrl, upstreamHeaders) {
  let upstream = await fetch(upstreamUrl, {
    method: request.method,
    headers: upstreamHeaders,
    cache: "no-store",
    redirect: "manual"
  });
  let usedGetFallback = false;

  if (request.method === "HEAD" && upstream.status === 405) {
    upstream = await fetch(upstreamUrl, {
      method: "GET",
      headers: upstreamHeaders,
      cache: "no-store",
      redirect: "manual"
    });
    usedGetFallback = true;
  }

  return {
    upstream,
    usedGetFallback
  };
}

async function proxyPortalAsset(request, { params }) {
  const resolvedParams = typeof params?.then === "function" ? await params : params;
  const assetPath = normalizeAssetPath(resolvedParams?.path);
  if (!assetPath) {
    return errorResponse(404, "Portal asset not found");
  }

  const config = getConfig();
  if (!config.backendApiKey) {
    return errorResponse(503, "TP_BACKEND_API_KEY is not configured");
  }

  const upstreamHeaders = buildUpstreamHeaders(request.headers, {
    backendApiKey: config.backendApiKey,
    actor: null,
    forwarding: {
      clientIp: String(request.headers.get("cf-connecting-ip") || "").trim() || null,
      host: request.headers.get("host") || request.nextUrl.host,
      proto: request.nextUrl.protocol.replace(":", "")
    }
  });

  let upstream;
  let usedGetFallback = false;
  try {
    ({ upstream, usedGetFallback } = await fetchUpstreamAsset(
      request,
      buildUpstreamUrl(`/portal/assets/${assetPath.split("/").map(encodeURIComponent).join("/")}`),
      upstreamHeaders
    ));
  } catch {
    return errorResponse(503, "Upstream service unavailable");
  }

  if (request.method === "HEAD" && usedGetFallback && upstream.body) {
    try {
      await upstream.body.cancel();
    } catch {
      // Ignore cancel failures and continue returning the HEAD response.
    }
  }

  const responseHeaders = copyUpstreamResponseHeaders(upstream.headers);
  responseHeaders.set("Cache-Control", upstream.status < 400 ? upstream.headers.get("cache-control") || "no-store" : "no-store");

  return applySecurityHeaders(
    new Response(request.method === "HEAD" ? null : upstream.body, {
      status: upstream.status,
      headers: responseHeaders
    })
  );
}

export const GET = proxyPortalAsset;
export const HEAD = proxyPortalAsset;
