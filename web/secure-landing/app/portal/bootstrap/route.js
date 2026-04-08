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
          directDebug: false
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
