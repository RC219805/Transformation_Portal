import { NextResponse } from "next/server.js";

import { getConfig } from "../../lib/config.js";
import { getDb } from "../../lib/db.js";
import { applySecurityHeaders } from "../../lib/http.js";
import { evaluateSessionScaling } from "../../lib/session-scaling.js";

export const runtime = "nodejs";

function evaluateAccessConfig(config) {
  const required = !config.allowLocalAccessBypass;
  const teamDomainConfigured = Boolean(config.cfAccessTeamDomain);
  const audienceConfigured = Boolean(config.cfAccessAud);
  const ok = !required || (teamDomainConfigured && audienceConfigured);

  return {
    ok,
    required,
    mode: config.allowLocalAccessBypass ? "development_bypass" : "cloudflare_access",
    teamDomainConfigured,
    audienceConfigured,
    reason: ok ? null : !teamDomainConfigured ? "missing_access_team_domain" : "missing_access_audience"
  };
}

function evaluateUserSource(config) {
  const source = config.usersFilePath ? "file" : "json_env";
  const userCount = Array.isArray(config.users) ? config.users.length : 0;
  const ok = userCount > 0;

  return {
    ok,
    required: true,
    source,
    userCount,
    reason: ok ? null : "no_configured_users"
  };
}

function evaluateSessionStore(config) {
  try {
    const db = getDb(config.sessionDbPath);
    db.prepare("SELECT 1 AS ok").get();
    return {
      ok: true,
      required: true,
      configured: Boolean(config.sessionDbPath),
      reason: null
    };
  } catch {
    return {
      ok: false,
      required: true,
      configured: Boolean(config.sessionDbPath),
      reason: "session_store_unavailable"
    };
  }
}

async function evaluateBackend(config) {
  if (!config.backendApiKey) {
    return {
      ok: false,
      required: true,
      configured: false,
      status: 503,
      auth_status: 0,
      reason: "missing_backend_api_key"
    };
  }

  let readyStatus;
  try {
    const upstream = await fetch(new URL("/ready", config.fastapiOrigin), {
      headers: {
        Accept: "application/json"
      },
      cache: "no-store",
      signal: AbortSignal.timeout(2000)
    });
    readyStatus = upstream.status;
    if (!upstream.ok) {
      return {
        ok: false,
        required: true,
        configured: true,
        status: readyStatus,
        auth_status: 0,
        reason: "backend_not_ready"
      };
    }
  } catch {
    return {
      ok: false,
      required: true,
      configured: true,
      status: 0,
      auth_status: 0,
      reason: "backend_unreachable"
    };
  }

  // Reachable + ready. Probe a protected endpoint to confirm the configured
  // TP_BACKEND_API_KEY actually matches what the backend expects. A 401/403
  // here means the keys have drifted — the visible 503 storm in the portal.
  let authStatus;
  try {
    const probe = await fetch(
      new URL("/v1/config-metadata?pipeline=lux-depth-v3", config.fastapiOrigin),
      {
        headers: {
          Accept: "application/json",
          "x-api-key": config.backendApiKey
        },
        cache: "no-store",
        signal: AbortSignal.timeout(2000)
      }
    );
    authStatus = probe.status;
  } catch {
    return {
      ok: false,
      required: true,
      configured: true,
      status: readyStatus,
      auth_status: 0,
      reason: "backend_unreachable"
    };
  }

  if (authStatus === 401 || authStatus === 403) {
    return {
      ok: false,
      required: true,
      configured: true,
      status: readyStatus,
      auth_status: authStatus,
      reason: "backend_auth_mismatch"
    };
  }
  if (authStatus >= 500) {
    return {
      ok: false,
      required: true,
      configured: true,
      status: readyStatus,
      auth_status: authStatus,
      reason: "backend_not_ready"
    };
  }
  return {
    ok: true,
    required: true,
    configured: true,
    status: readyStatus,
    auth_status: authStatus,
    reason: null
  };
}

export async function GET() {
  const config = getConfig();
  const backend = await evaluateBackend(config);
  const checks = {
    backend,
    access_config: evaluateAccessConfig(config),
    user_source: evaluateUserSource(config),
    session_store: evaluateSessionStore(config),
    session_scaling: evaluateSessionScaling(config)
  };
  const ok = Object.values(checks).every((check) => !check.required || check.ok);

  return applySecurityHeaders(
    NextResponse.json(
      {
        ok,
        frontend: ok ? "ready" : "degraded",
        backend: {
          ok: backend.ok,
          status: backend.status
        },
        checks
      },
      {
        status: ok ? 200 : 503,
        headers: {
          "Cache-Control": "no-store"
        }
      }
    )
  );
}
