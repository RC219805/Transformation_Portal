import { getConfig } from "../../../../lib/config.js";
import { audit } from "../../../../lib/audit.js";
import { applySecurityHeaders } from "../../../../lib/http.js";
import {
  auditManagedSurfaceFailure,
  classifyUpstreamFailureStatus,
  getManagedFailureMessage,
  MANAGED_FAILURE_REASON
} from "../../../../lib/managed-failure.js";
import { buildUpstreamHeaders, buildUpstreamUrl, copyUpstreamResponseHeaders } from "../../../../lib/proxy.js";

export const runtime = "nodejs";

const PORTAL_VIDEO_CACHE_CONTROL = "public, max-age=86400";
const ALLOWED_PORTAL_VIDEO_ASSETS = new Set(["dna-portal-video-2.mp4"]);

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
  const assetName = String(resolvedParams?.assetName ?? "");
  if (!ALLOWED_PORTAL_VIDEO_ASSETS.has(assetName)) {
    audit("portal_video_proxy_not_found", {
      assetName
    });
    return errorResponse(404, "Portal video asset not found");
  }

  const config = getConfig();
  if (!config.backendApiKey) {
    auditManagedSurfaceFailure("portal_video", {
      extra: { assetName },
      path: `/portal/video/${assetName}`,
      reason: MANAGED_FAILURE_REASON.CONFIG_FAILURE,
      status: 503
    });
    return errorResponse(503, getManagedFailureMessage("portal_video", MANAGED_FAILURE_REASON.CONFIG_FAILURE));
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
  } catch (error) {
    auditManagedSurfaceFailure("portal_video", {
      extra: { assetName },
      message: error instanceof Error ? error.message : String(error),
      path: `/portal/video/${assetName}`,
      reason: MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE,
      status: 503
    });
    return errorResponse(
      503,
      getManagedFailureMessage("portal_video", MANAGED_FAILURE_REASON.UPSTREAM_UNAVAILABLE)
    );
  }

  if (upstream.status >= 400) {
    const reason = classifyUpstreamFailureStatus(upstream.status, { clientErrorIsConfig: true });
    if (reason) {
      auditManagedSurfaceFailure("portal_video", {
        extra: { assetName },
        path: `/portal/video/${assetName}`,
        reason,
        status: 503,
        upstreamStatus: upstream.status
      });
      return errorResponse(503, getManagedFailureMessage("portal_video", reason));
    }
  }

  const responseHeaders = copyUpstreamResponseHeaders(upstream.headers);
  responseHeaders.set("Cache-Control", upstream.status < 400 ? PORTAL_VIDEO_CACHE_CONTROL : "no-store");
  if (upstream.status >= 400) {
    audit("portal_video_proxy_upstream_status", {
      assetName,
      status: upstream.status
    });
  }

  return applySecurityHeaders(
    new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders
    })
  );
}

export const GET = proxyPortalVideo;
export const HEAD = proxyPortalVideo;
