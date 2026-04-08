import { getConfig } from "../../../../lib/config.js";
import { audit } from "../../../../lib/audit.js";
import { applySecurityHeaders } from "../../../../lib/http.js";
import { isAllowedPortalAssetPath } from "../../../../lib/portal-asset-manifest.js";
import {
  buildUpstreamHeaders,
  buildUpstreamUrl,
  copyUpstreamResponseHeaders
} from "../../../../lib/proxy.js";

export const runtime = "nodejs";

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
  return isAllowedPortalAssetPath(assetPath) ? assetPath : null;
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
    audit("portal_asset_proxy_not_found", {
      requestedPath: Array.isArray(resolvedParams?.path) ? resolvedParams.path.join("/") : ""
    });
    return errorResponse(404, "Portal asset not found");
  }

  const config = getConfig();
  if (!config.backendApiKey) {
    audit("portal_asset_proxy_config_error", {
      assetPath,
      reason: "missing_backend_api_key"
    });
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
    audit("portal_asset_proxy_upstream_unavailable", {
      assetPath
    });
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
  if (upstream.status >= 400) {
    audit("portal_asset_proxy_upstream_status", {
      assetPath,
      status: upstream.status
    });
  }

  return applySecurityHeaders(
    new Response(request.method === "HEAD" ? null : upstream.body, {
      status: upstream.status,
      headers: responseHeaders
    })
  );
}

export const GET = proxyPortalAsset;
export const HEAD = proxyPortalAsset;
