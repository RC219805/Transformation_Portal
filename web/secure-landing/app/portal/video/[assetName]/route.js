import { getConfig } from "../../../../lib/config.js";
import { applySecurityHeaders } from "../../../../lib/http.js";
import { buildUpstreamHeaders, buildUpstreamUrl, copyUpstreamResponseHeaders } from "../../../../lib/proxy.js";

export const runtime = "nodejs";

const PORTAL_VIDEO_CACHE_CONTROL = "public, max-age=86400";

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

async function proxyPortalVideo(request, { params }) {
  const resolvedParams = typeof params?.then === "function" ? await params : params;
  const assetName = String(resolvedParams?.assetName || "").trim();
  if (!assetName) {
    return errorResponse(400, "Invalid portal video asset");
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
  try {
    upstream = await fetch(buildUpstreamUrl(`/portal/video/${encodeURIComponent(assetName)}`), {
      method: request.method,
      headers: upstreamHeaders,
      cache: "no-store",
      redirect: "manual"
    });
  } catch {
    return errorResponse(503, "Upstream service unavailable");
  }

  const responseHeaders = copyUpstreamResponseHeaders(upstream.headers);
  responseHeaders.set("Cache-Control", upstream.ok ? PORTAL_VIDEO_CACHE_CONTROL : "no-store");

  return applySecurityHeaders(
    new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders
    })
  );
}

export const GET = proxyPortalVideo;
export const HEAD = proxyPortalVideo;
