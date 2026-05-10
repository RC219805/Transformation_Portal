import { NextResponse } from "next/server.js";

import { revokeSessionOnAccessFailure, resolveAuthenticatedAccessSession } from "../../../lib/access.js";
import { audit } from "../../../lib/audit.js";
import { isPortalRumEnabled } from "../../../lib/config.js";
import { applySecurityHeaders } from "../../../lib/http.js";
import { stableRolloutBucket } from "../../../lib/rollout.js";
import {
  auditManagedSurfaceFailure,
  buildManagedBootstrapFailure,
  classifyManagedAccessFailure
} from "../../../lib/managed-failure.js";
import { clearSessionCookie } from "../../../lib/sessions.js";
import { resolveRequestTraceparent, traceIdFromTraceparent } from "../../../lib/trace.js";

export const runtime = "nodejs";

function parseRolloutPercent(rawValue) {
  const parsed = Number.parseInt(String(rawValue || "").trim(), 10);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, Math.min(100, parsed));
}

function resolveRolloutCohortKey(session) {
  return String(session?.username || session?.accessEmail || session?.role || "").trim().toLowerCase();
}

function resolvePortalRollout(session, envKey, env = process.env) {
  const rolloutPercent = parseRolloutPercent(env[envKey]);
  if (rolloutPercent <= 0) {
    return false;
  }
  const cohortKey = resolveRolloutCohortKey(session);
  if (!cohortKey) {
    return false;
  }
  return stableRolloutBucket(cohortKey) < rolloutPercent;
}

function resolveArtifactViewerModal(session, env = process.env) {
  return resolvePortalRollout(session, "TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT", env);
}

function resolveReviewSurfaceDeferred(session, env = process.env) {
  return resolvePortalRollout(session, "TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT", env);
}

function resolveRumTelemetry(session, env = process.env) {
  if (!isPortalRumEnabled(env)) {
    return false;
  }
  return resolvePortalRollout(session, "TP_PORTAL_RUM_ROLLOUT_PERCENT", env);
}

function resolveStagedUploads(session, env = process.env) {
  if (String(env.TP_PORTAL_UPLOAD_STAGING_ENABLED || "").trim().toLowerCase() !== "1") {
    return false;
  }
  return resolvePortalRollout(session, "TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", env);
}

function resolveFastVlmCaptioning(session, env = process.env) {
  if (String(env.TP_PORTAL_FASTVLM_CAPTIONING_ENABLED || "").trim().toLowerCase() !== "1") {
    return false;
  }
  return resolvePortalRollout(session, "TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", env);
}

function withTraceparent(response, traceparent) {
  response.headers.set("traceparent", traceparent);
  return response;
}

export async function GET(request) {
  const requestTraceparent = resolveRequestTraceparent(request);
  const traceId = traceIdFromTraceparent(requestTraceparent);
  const authState = await resolveAuthenticatedAccessSession(request, { touch: true });
  if (!authState.ok) {
    const reason = classifyManagedAccessFailure(authState.errorCode);
    auditManagedSurfaceFailure("portal_bootstrap", {
      actor: authState.session,
      errorCode: authState.errorCode,
      path: "/portal/bootstrap",
      reason,
      status: authState.status,
      extra: traceId ? { traceId } : {}
    });
    if (authState.revokeSession) {
      revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }

    const response = withTraceparent(applySecurityHeaders(
      NextResponse.json(
        buildManagedBootstrapFailure({
          reason,
          status: authState.status
        }),
        {
          status: authState.status,
          headers: {
            "Cache-Control": "no-store"
          }
        }
      )
    ), requestTraceparent);
    if (authState.revokeSession) {
      clearSessionCookie(response);
    }
    return response;
  }
  const { session } = authState;
  audit("portal_bootstrap", {
    username: session.username,
    accessEmail: session.accessEmail,
    path: "/portal/bootstrap",
    traceId
  });

  return withTraceparent(applySecurityHeaders(
    NextResponse.json(
      {
        authMode: "managed",
        csrfToken: session.csrfToken,
        actor: {
          username: session.username,
          role: session.role,
          accessEmail: session.accessEmail
        },
        features: {
          apiKeyInput: false,
          directDebug: false,
          artifactViewerModal: resolveArtifactViewerModal(session),
          reviewSurfaceDeferred: resolveReviewSurfaceDeferred(session),
          stagedUploads: resolveStagedUploads(session),
          rumTelemetry: resolveRumTelemetry(session),
          fastVlmCaptioning: resolveFastVlmCaptioning(session)
        }
      },
      {
        headers: {
          "Cache-Control": "no-store"
        }
      }
    )
  ), requestTraceparent);
}
