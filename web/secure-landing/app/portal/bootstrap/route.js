import { createHash } from "node:crypto";

import { NextResponse } from "next/server.js";

import { revokeSessionOnAccessFailure, resolveAuthenticatedAccessSession } from "../../../lib/access.js";
import { applySecurityHeaders } from "../../../lib/http.js";
import {
  auditManagedSurfaceFailure,
  buildManagedBootstrapFailure,
  classifyManagedAccessFailure
} from "../../../lib/managed-failure.js";
import { clearSessionCookie } from "../../../lib/sessions.js";

export const runtime = "nodejs";

function parseRolloutPercent(rawValue) {
  const parsed = Number.parseInt(String(rawValue || "").trim(), 10);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, Math.min(100, parsed));
}

function stableRolloutBucket(key) {
  const normalized = String(key || "").trim().toLowerCase();
  if (!normalized) {
    return 100;
  }
  const digest = createHash("sha256").update(normalized).digest("hex");
  return Number.parseInt(digest.slice(0, 8), 16) % 100;
}

function resolveArtifactViewerModal(session, env = process.env) {
  const rolloutPercent = parseRolloutPercent(env.TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT);
  if (rolloutPercent <= 0) {
    return false;
  }
  const cohortKey = String(session?.username || session?.accessEmail || session?.role || "").trim().toLowerCase();
  if (!cohortKey) {
    return false;
  }
  return stableRolloutBucket(cohortKey) < rolloutPercent;
}

export async function GET(request) {
  const authState = await resolveAuthenticatedAccessSession(request, { touch: true });
  if (!authState.ok) {
    const reason = classifyManagedAccessFailure(authState.errorCode);
    auditManagedSurfaceFailure("portal_bootstrap", {
      actor: authState.session,
      errorCode: authState.errorCode,
      path: "/portal/bootstrap",
      reason,
      status: authState.status
    });
    if (authState.revokeSession) {
      revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }

    const response = applySecurityHeaders(
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
    );
    if (authState.revokeSession) {
      clearSessionCookie(response);
    }
    return response;
  }
  const { session } = authState;

  return applySecurityHeaders(
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
          artifactViewerModal: resolveArtifactViewerModal(session)
        }
      },
      {
        headers: {
          "Cache-Control": "no-store"
        }
      }
    )
  );
}
