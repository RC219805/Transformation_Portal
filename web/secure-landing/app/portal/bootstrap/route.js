import { NextResponse } from "next/server.js";

import { applySecurityHeaders } from "../../../lib/http.js";
import { getSessionFromRequest } from "../../../lib/sessions.js";

export const runtime = "nodejs";

export async function GET(request) {
  const session = getSessionFromRequest(request, { touch: true });
  if (!session?.authenticated) {
    return applySecurityHeaders(
      NextResponse.json(
        {
          error: "authentication required"
        },
        {
          status: 401,
          headers: {
            "Cache-Control": "no-store"
          }
        }
      )
    );
  }

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
