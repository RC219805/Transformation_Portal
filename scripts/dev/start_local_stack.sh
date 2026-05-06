#!/usr/bin/env bash
# Bring up the canonical local Transformation Portal stack:
#   1. Generate /tmp/tp-local-http-all-on.env (idempotent).
#   2. Tear down any leftover listeners on dev ports.
#   3. Launch the FastAPI backend with reload boundaries (make run-backend-local).
#   4. Wait for /ready, then launch the managed frontdoor.
#
# All processes are foregrounded into separate log files under /tmp so the
# operator can `tail -f` them. Send SIGINT / Ctrl+C to stop the launcher; the
# sub-processes propagate the signal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${TP_LOCAL_ENV_FILE:-/tmp/tp-local-http-all-on.env}"
BACKEND_LOG="${TP_DEV_BACKEND_LOG:-/tmp/tp-backend.log}"
FRONTDOOR_LOG="${TP_DEV_FRONTDOOR_LOG:-/tmp/tp-frontdoor.log}"
READY_TIMEOUT="${TP_DEV_READY_TIMEOUT:-30}"

cd "${REPO_ROOT}"

echo "[dev-start] Writing canonical env file (${ENV_FILE})..."
./scripts/dev/write_local_env.sh

# shellcheck disable=SC1090
source "${ENV_FILE}"

append_cloudflared_trusted_host() {
  local host_file="${TP_CLOUDFLARED_HOST_FILE:-/tmp/tp-cloudflared-host}"
  if [[ ! -r "${host_file}" ]]; then
    return
  fi
  local hostname
  hostname="$(head -n 1 "${host_file}" | tr -d '[:space:]')"
  if [[ -z "${hostname}" ]]; then
    return
  fi
  if [[ -n "${TP_TRUSTED_HOSTS:-}" ]] && [[ ",${TP_TRUSTED_HOSTS}," == *",${hostname},"* ]]; then
    return
  fi
  if [[ -n "${TP_TRUSTED_HOSTS:-}" ]]; then
    export TP_TRUSTED_HOSTS="${TP_TRUSTED_HOSTS},${hostname}"
  else
    export TP_TRUSTED_HOSTS="localhost,127.0.0.1,::1,testserver,${hostname}"
  fi
  echo "[dev-start] Added Cloudflare tunnel host to backend TP_TRUSTED_HOSTS before startup: ${hostname}"
}

append_cloudflared_trusted_host

echo "[dev-start] Stopping any existing local stack..."
./scripts/dev/stop_local_stack.sh || true

echo "[dev-start] Starting backend (logs: ${BACKEND_LOG})..."
( make run-backend-local >"${BACKEND_LOG}" 2>&1 ) &
BACKEND_PID=$!

cleanup() {
  echo
  echo "[dev-start] Stopping local stack..."
  kill "${BACKEND_PID}" 2>/dev/null || true
  if [[ -n "${FRONTDOOR_PID:-}" ]]; then
    kill "${FRONTDOOR_PID}" 2>/dev/null || true
  fi
  ./scripts/dev/stop_local_stack.sh || true
}
trap cleanup EXIT INT TERM

echo "[dev-start] Waiting up to ${READY_TIMEOUT}s for backend /ready..."
elapsed=0
until curl -fsS "${TP_FASTAPI_ORIGIN:-http://127.0.0.1:8000}/ready" >/dev/null 2>&1; do
  if [[ "${elapsed}" -ge "${READY_TIMEOUT}" ]]; then
    echo "[dev-start] Backend did not become ready within ${READY_TIMEOUT}s." >&2
    echo "[dev-start] Last 40 lines of backend log:" >&2
    tail -n 40 "${BACKEND_LOG}" >&2 || true
    exit 1
  fi
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[dev-start] Backend exited prematurely. See ${BACKEND_LOG}." >&2
    tail -n 40 "${BACKEND_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done
echo "[dev-start] Backend is ready."

echo "[dev-start] Starting frontdoor (logs: ${FRONTDOOR_LOG})..."
( make run-frontdoor-local >"${FRONTDOOR_LOG}" 2>&1 ) &
FRONTDOOR_PID=$!

echo "[dev-start] Local stack up:"
echo "  backend   pid=${BACKEND_PID}  log=${BACKEND_LOG}"
echo "  frontdoor pid=${FRONTDOOR_PID}  log=${FRONTDOOR_LOG}"
echo "Press Ctrl+C to stop."

while true; do
  if ! kill -0 "${FRONTDOOR_PID}" 2>/dev/null; then
    set +e
    wait "${FRONTDOOR_PID}"
    frontdoor_status=$?
    set -e
    echo "[dev-start] Frontdoor exited unexpectedly with status ${frontdoor_status}. See ${FRONTDOOR_LOG}." >&2
    tail -n 40 "${FRONTDOOR_LOG}" >&2 || true
    exit "${frontdoor_status}"
  fi
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    set +e
    wait "${BACKEND_PID}"
    backend_status=$?
    set -e
    echo "[dev-start] Backend exited unexpectedly with status ${backend_status}. See ${BACKEND_LOG}." >&2
    tail -n 40 "${BACKEND_LOG}" >&2 || true
    exit "${backend_status}"
  fi
  sleep 1
done
