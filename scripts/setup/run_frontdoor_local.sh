#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTDOOR_ROOT="${REPO_ROOT}/web/secure-landing"
FRONTDOOR_HOST="${TP_FRONTDOOR_HOST:-127.0.0.1}"
FRONTDOOR_PORT="${TP_FRONTDOOR_PORT:-3000}"
FASTAPI_ORIGIN="${TP_FASTAPI_ORIGIN:-http://127.0.0.1:8000}"
SESSION_DB="${TP_FRONTDOOR_SESSION_DB:-/tmp/transformation-portal-frontdoor-sessions.db}"
BACKEND_API_KEY="${TP_BACKEND_API_KEY:-${TP_API_KEY:-}}"
FRONTDOOR_DIST_DIR="${TP_FRONTDOOR_DIST_DIR:-}"
DEFAULT_USERS_FILE="${TP_FRONTDOOR_USERS_FILE:-/tmp/tp-frontdoor-users.json}"
DEFAULT_FRONTDOOR_USERNAME="${TP_FRONTDOOR_USERNAME:-smoke-admin}"
DEFAULT_FRONTDOOR_PASSWORD="${TP_FRONTDOOR_PASSWORD:-correct horse battery staple}"
DEFAULT_FRONTDOOR_ACCESS_EMAIL="${TP_FRONTDOOR_ACCESS_EMAIL:-${DEFAULT_FRONTDOOR_USERNAME}@local.invalid}"

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "run_frontdoor_local.sh requires '$1' to be installed."
    exit 1
  fi
}

port_in_use() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${FRONTDOOR_PORT}" -sTCP:LISTEN >/dev/null 2>&1
    return
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "${FRONTDOOR_PORT}" >/dev/null 2>&1
    return
  fi
  return 1
}

ensure_command curl
ensure_command node
ensure_command npm

if [[ -z "${BACKEND_API_KEY}" ]]; then
  echo "Set TP_API_KEY or TP_BACKEND_API_KEY before starting the managed front door."
  echo "Generate a canonical local env file with: ./scripts/dev/write_local_env.sh"
  exit 1
fi

if [[ -n "${TP_API_KEY:-}" && -n "${TP_BACKEND_API_KEY:-}" \
      && "${TP_API_KEY}" != "${TP_BACKEND_API_KEY}" ]]; then
  echo "Refusing to start: TP_API_KEY (backend) and TP_BACKEND_API_KEY (frontdoor) differ."
  echo "These must be equal so the frontdoor can authenticate to the backend."
  echo "Source the canonical env file in both shells:"
  echo "  source /tmp/tp-local-http-all-on.env"
  echo "Or regenerate it: ./scripts/dev/write_local_env.sh --rotate"
  exit 1
fi

if [[ -z "${TP_FRONTDOOR_USERS_FILE:-}" && -z "${TP_FRONTDOOR_USERS_JSON:-}" ]]; then
  echo "No explicit front-door user source supplied. Seeding the canonical local smoke user fixture..."
  (
    cd "${FRONTDOOR_ROOT}"
    node ./scripts/seed-frontdoor-user.mjs \
      --output "${DEFAULT_USERS_FILE}" \
      --username "${DEFAULT_FRONTDOOR_USERNAME}" \
      --password "${DEFAULT_FRONTDOOR_PASSWORD}" \
      --access-email "${DEFAULT_FRONTDOOR_ACCESS_EMAIL}" \
      --role admin \
      --quiet
  )
  export TP_FRONTDOOR_USERS_FILE="${DEFAULT_USERS_FILE}"
  echo "Seeded local front-door user fixture at ${TP_FRONTDOOR_USERS_FILE}"
  echo "Local operator username: ${DEFAULT_FRONTDOOR_USERNAME}"
  if [[ "${TP_FRONTDOOR_PRINT_PASSWORD:-0}" == "1" ]]; then
    echo "Local operator password: ${DEFAULT_FRONTDOOR_PASSWORD}"
  else
    echo "Password not printed. Set TP_FRONTDOOR_PRINT_PASSWORD=1 to display it explicitly."
  fi
fi

if port_in_use; then
  echo "Refusing to start the managed front door because localhost:${FRONTDOOR_PORT} is already in use."
  echo "Stop the existing process instead of letting Next.js fall back to :3001."
  exit 1
fi

if ! curl -fsS "${FASTAPI_ORIGIN}/ready" >/dev/null; then
  echo "FastAPI origin is not ready at ${FASTAPI_ORIGIN}/ready."
  echo "Start the backend first, then retry."
  exit 1
fi

# When run alongside scripts/dev/run_cloudflared.sh, the tunnel hostname is
# written to a sentinel file. Append it to TP_TRUSTED_HOSTS so the FastAPI
# Trusted-Host middleware does not reject proxied requests.
CF_HOST_FILE="${TP_CLOUDFLARED_HOST_FILE:-/tmp/tp-cloudflared-host}"
if [[ -r "${CF_HOST_FILE}" ]]; then
  CF_HOSTNAME="$(head -n 1 "${CF_HOST_FILE}" | tr -d '[:space:]')"
  if [[ -n "${CF_HOSTNAME}" ]]; then
    if [[ -n "${TP_TRUSTED_HOSTS:-}" ]] \
        && ! [[ ",${TP_TRUSTED_HOSTS}," == *",${CF_HOSTNAME},"* ]]; then
      export TP_TRUSTED_HOSTS="${TP_TRUSTED_HOSTS},${CF_HOSTNAME}"
      echo "Appended ${CF_HOSTNAME} to TP_TRUSTED_HOSTS."
    elif [[ -z "${TP_TRUSTED_HOSTS:-}" ]]; then
      export TP_TRUSTED_HOSTS="localhost,127.0.0.1,::1,testserver,${CF_HOSTNAME}"
      echo "Set TP_TRUSTED_HOSTS to include ${CF_HOSTNAME}."
    fi
  fi
fi

export NODE_ENV=development
export TP_ALLOW_LOCAL_ACCESS_BYPASS=1
export TP_FASTAPI_ORIGIN="${FASTAPI_ORIGIN}"
export TP_BACKEND_API_KEY="${BACKEND_API_KEY}"
export TP_FRONTDOOR_SESSION_DB="${SESSION_DB}"
export TP_FRONTDOOR_SESSION_SCALING_MODE=single_instance
if [[ -n "${FRONTDOOR_DIST_DIR}" ]]; then
  export TP_NEXT_DIST_DIR="${FRONTDOOR_DIST_DIR}"
fi

echo "Starting managed front door on http://${FRONTDOOR_HOST}:${FRONTDOOR_PORT}"
echo "Using FastAPI origin ${TP_FASTAPI_ORIGIN}"
if [[ -n "${FRONTDOOR_DIST_DIR}" ]]; then
  echo "Using isolated Next distDir ${TP_FRONTDOOR_DIST_DIR}"
fi

cd "${FRONTDOOR_ROOT}"
npm run dev -- --hostname "${FRONTDOOR_HOST}" --port "${FRONTDOOR_PORT}"
