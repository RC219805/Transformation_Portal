// Frontdoor startup preflight: validate required env and probe the backend
// with the configured API key BEFORE Next.js begins serving traffic.
//
// Catches the most common operator failure: TP_BACKEND_API_KEY (frontdoor) and
// TP_API_KEY (backend) drifting apart, which manifests as a flood of
// 503 config_failure responses instead of a single actionable error.
//
// Behaviour:
//   - Always fails closed on missing env (TP_BACKEND_API_KEY, TP_FASTAPI_ORIGIN,
//     user source). In production, also requires Cloudflare Access config.
//   - Probes GET ${origin}/v1/config-metadata?pipeline=lux-depth-v3 with the
//     `x-api-key` header. 401/403 always fails closed (key mismatch).
//   - Connection failures (refused, DNS, timeout) fail closed in production;
//     in development they emit a warning and continue (the backend may still
//     be starting up).
//   - Hard escape hatch: TP_FRONTDOOR_PREFLIGHT_DISABLE=1 skips the script
//     entirely. Use only for emergencies.

import { readFileSync } from "node:fs";

const PROTECTED_PROBE_PATH = "/v1/config-metadata?pipeline=lux-depth-v3";
const PROBE_TIMEOUT_MS = 3000;

function isProduction() {
  return (process.env.NODE_ENV || "development") === "production";
}

function trimmed(value) {
  return String(value ?? "").trim();
}

function resolveBackendOrigin() {
  return trimmed(process.env.TP_FASTAPI_ORIGIN);
}

function resolveBackendKey() {
  return trimmed(process.env.TP_BACKEND_API_KEY);
}

function parseUsersJson(raw) {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => {
      if (!item || typeof item !== "object") return false;
      return Boolean(
        trimmed(item.username) &&
        trimmed(item.password_hash) &&
        trimmed(item.access_email)
      );
    });
  } catch {
    return [];
  }
}

function userSourceStatus() {
  const usersFile = trimmed(process.env.TP_FRONTDOOR_USERS_FILE);
  if (usersFile) {
    try {
      const users = parseUsersJson(readFileSync(usersFile, "utf-8"));
      return {
        ok: users.length > 0,
        source: "file",
        detail: users.length > 0
          ? `loaded ${users.length} configured frontdoor user(s) from ${usersFile}`
          : `TP_FRONTDOOR_USERS_FILE did not contain any valid users: ${usersFile}`
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        ok: false,
        source: "file",
        detail: `TP_FRONTDOOR_USERS_FILE could not be read: ${message}`
      };
    }
  }
  const json = trimmed(process.env.TP_FRONTDOOR_USERS_JSON);
  if (!json) {
    return {
      ok: false,
      source: "",
      detail: "Set TP_FRONTDOOR_USERS_FILE or TP_FRONTDOOR_USERS_JSON before starting."
    };
  }
  const users = parseUsersJson(json);
  return {
    ok: users.length > 0,
    source: "json_env",
    detail: users.length > 0
      ? `loaded ${users.length} configured frontdoor user(s) from TP_FRONTDOOR_USERS_JSON`
      : "TP_FRONTDOOR_USERS_JSON did not contain any valid users."
  };
}

function failClosed(reason, detail) {
  const lines = [
    "Frontdoor preflight failed: " + reason,
    detail,
    "",
    "Resolve the configuration before retrying. Generate a canonical local",
    "env file with: ./scripts/dev/write_local_env.sh"
  ].filter(Boolean);
  console.error(lines.join("\n"));
  process.exit(1);
}

function validateEnv() {
  const origin = resolveBackendOrigin();
  const apiKey = resolveBackendKey();

  if (!apiKey) {
    failClosed(
      "TP_BACKEND_API_KEY is not set",
      "The frontdoor cannot authenticate to the FastAPI origin without it."
    );
  }
  if (!origin) {
    failClosed(
      "TP_FASTAPI_ORIGIN is not set",
      "The frontdoor runtime proxies /v1/* through TP_FASTAPI_ORIGIN; TP_BACKEND_ORIGIN is not consumed."
    );
  }
  const users = userSourceStatus();
  if (!users.ok) {
    failClosed(
      "no frontdoor user source configured",
      users.detail
    );
  }
  if (isProduction()) {
    if (!trimmed(process.env.TP_CF_ACCESS_TEAM_DOMAIN)) {
      failClosed(
        "TP_CF_ACCESS_TEAM_DOMAIN is not set in production",
        "Cloudflare Access verification cannot operate without it."
      );
    }
    if (!trimmed(process.env.TP_CF_ACCESS_AUD)) {
      failClosed(
        "TP_CF_ACCESS_AUD is not set in production",
        "Cloudflare Access JWT audience is required for verification."
      );
    }
  }
}

async function probeBackendAuth() {
  const origin = resolveBackendOrigin();
  const apiKey = resolveBackendKey();
  const url = new URL(PROTECTED_PROBE_PATH, origin);
  let response;
  try {
    response = await fetch(url, {
      method: "GET",
      cache: "no-store",
      headers: {
        Accept: "application/json",
        "x-api-key": apiKey
      },
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS)
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (isProduction()) {
      failClosed(
        `backend protected probe could not connect to ${origin}`,
        `Underlying error: ${message}`
      );
    }
    console.warn(
      `[frontdoor preflight] backend protected probe could not connect to ${origin}: ${message}`
    );
    console.warn("[frontdoor preflight] continuing because NODE_ENV is not production.");
    return;
  }

  if (response.status === 401 || response.status === 403) {
    failClosed(
      `backend protected probe returned ${response.status}`,
      "TP_BACKEND_API_KEY does not match backend TP_API_KEY."
    );
  }
  if (!response.ok) {
    failClosed(
      `backend protected probe returned ${response.status}`,
      "The configured TP_FASTAPI_ORIGIN did not expose the expected protected config endpoint."
    );
  }
  console.log(`[frontdoor preflight] backend protected probe ok (${response.status}) at ${origin}`);
}

async function main() {
  if (trimmed(process.env.TP_FRONTDOOR_PREFLIGHT_DISABLE) === "1") {
    console.warn("[frontdoor preflight] disabled via TP_FRONTDOOR_PREFLIGHT_DISABLE=1");
    return;
  }
  validateEnv();
  await probeBackendAuth();
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[frontdoor preflight] unexpected error: ${message}`);
  process.exit(1);
}
