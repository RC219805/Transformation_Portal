import { NextResponse } from "next/server.js";

import { revokeSessionOnAccessFailure, resolveAuthenticatedAccessSession } from "../../../lib/access.js";
import { applySecurityHeaders } from "../../../lib/http.js";
import { clearSessionCookie } from "../../../lib/sessions.js";

export const runtime = "nodejs";

export async function GET(request) {
  const authState = await resolveAuthenticatedAccessSession(request, { touch: true });
  if (!authState.ok) {
    if (authState.revokeSession) {
      revokeSessionOnAccessFailure(authState.session, authState.errorCode);
    }

    const response = applySecurityHeaders(
      NextResponse.json(
        {
          error: authState.status === 503 ? "managed access unavailable" : "authentication required"
        },
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
