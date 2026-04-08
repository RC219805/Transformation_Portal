#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTDOOR_ROOT="${REPO_ROOT}/web/secure-landing"
FRONTDOOR_PORT="3000"
FASTAPI_ORIGIN="${TP_FASTAPI_ORIGIN:-http://127.0.0.1:8000}"
SESSION_DB="${TP_FRONTDOOR_SESSION_DB:-/tmp/transformation-portal-frontdoor-sessions.db}"
BACKEND_API_KEY="${TP_BACKEND_API_KEY:-${TP_API_KEY:-}}"

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
ensure_command npm

if [[ -z "${BACKEND_API_KEY}" ]]; then
  echo "Set TP_API_KEY or TP_BACKEND_API_KEY before starting the managed front door."
  exit 1
fi

if [[ -z "${TP_FRONTDOOR_USERS_FILE:-}" && -z "${TP_FRONTDOOR_USERS_JSON:-}" ]]; then
  echo "Set TP_FRONTDOOR_USERS_FILE or TP_FRONTDOOR_USERS_JSON before starting the managed front door."
  exit 1
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

export NODE_ENV=development
export TP_ALLOW_LOCAL_ACCESS_BYPASS=1
export TP_FASTAPI_ORIGIN="${FASTAPI_ORIGIN}"
export TP_BACKEND_API_KEY="${BACKEND_API_KEY}"
export TP_FRONTDOOR_SESSION_DB="${SESSION_DB}"
export TP_FRONTDOOR_SESSION_SCALING_MODE=single_instance

echo "Starting managed front door on http://localhost:${FRONTDOOR_PORT}"
echo "Using FastAPI origin ${TP_FASTAPI_ORIGIN}"

cd "${FRONTDOOR_ROOT}"
npm run dev -- --hostname 127.0.0.1 --port "${FRONTDOOR_PORT}"
