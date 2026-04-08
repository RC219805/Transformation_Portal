import { NextResponse } from "next/server.js";

import { getConfig } from "../../lib/config.js";
import { getDb } from "../../lib/db.js";
import { applySecurityHeaders } from "../../lib/http.js";

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
      reason: "missing_backend_api_key"
    };
  }

  try {
    const upstream = await fetch(new URL("/ready", config.fastapiOrigin), {
      headers: {
        Accept: "application/json"
      },
      cache: "no-store",
      signal: AbortSignal.timeout(2000)
    });
    return {
      ok: upstream.ok,
      required: true,
      configured: true,
      status: upstream.status,
      reason: upstream.ok ? null : "backend_not_ready"
    };
  } catch {
    return {
      ok: false,
      required: true,
      configured: true,
      status: 0,
      reason: "backend_unreachable"
    };
  }
}

export async function GET() {
  const config = getConfig();
  const backend = await evaluateBackend(config);
  const checks = {
    backend,
    access_config: evaluateAccessConfig(config),
    user_source: evaluateUserSource(config),
    session_store: evaluateSessionStore(config)
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
