import { NextResponse } from "next/server.js";

import { getConfig } from "../../lib/config.js";
import { applySecurityHeaders } from "../../lib/http.js";

export const runtime = "nodejs";

export async function GET() {
  let backendOk = false;
  let backendStatus = 0;

  try {
    const upstream = await fetch(new URL("/ready", getConfig().fastapiOrigin), {
      headers: {
        Accept: "application/json"
      },
      cache: "no-store",
      signal: AbortSignal.timeout(2000)
    });
    backendStatus = upstream.status;
    backendOk = upstream.ok;
  } catch {
    backendStatus = 0;
    backendOk = false;
  }

  return applySecurityHeaders(
    NextResponse.json(
      {
        ok: backendOk,
        frontend: "ready",
        backend: {
          ok: backendOk,
          status: backendStatus
        }
      },
      {
        status: backendOk ? 200 : 503,
        headers: {
          "Cache-Control": "no-store"
        }
      }
    )
  );
}
