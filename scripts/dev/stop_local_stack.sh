#!/usr/bin/env bash
# Tear down any local Transformation Portal processes left over from a prior
# `make dev-start` (or ad-hoc backend/frontdoor invocations).
#
# Kills listeners on the canonical local ports (8000 backend, 3000 frontdoor,
# 8001 archive-gate-c, 3002 alt-frontdoor) and any orphan Uvicorn parent +
# child processes that might be holding the port. Verifies that the ports are
# free at exit; lists any remaining listeners so the operator can investigate.

set -euo pipefail

PORTS=(8000 3000 8001 3002)
GRACE_SECONDS="${TP_DEV_STOP_GRACE_SECONDS:-5}"

kill_pids() {
  local signal="$1"
  shift
  local pid
  for pid in "$@"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${signal}" "${pid}" 2>/dev/null || true
    fi
  done
}

pids_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:"${port}" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${port}" 2>/dev/null \
      | awk -F'pid=' 'NF>1 {split($2, parts, ","); print parts[1]}'
  fi
}

clear_port() {
  local port="$1"
  local pids
  pids="$(pids_on_port "${port}" | xargs -r echo)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  echo "Stopping listeners on :${port}: ${pids}"
  # shellcheck disable=SC2086
  kill_pids -TERM ${pids}
  # Wait up to GRACE_SECONDS for graceful exit.
  local waited=0
  while [[ "${waited}" -lt "${GRACE_SECONDS}" ]]; do
    pids="$(pids_on_port "${port}" | xargs -r echo)"
    if [[ -z "${pids}" ]]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "Listeners on :${port} still up after ${GRACE_SECONDS}s; sending SIGKILL: ${pids}"
  # shellcheck disable=SC2086
  kill_pids -KILL ${pids}
}

# Kill any uvicorn reloader process trees that might not be bound to a port
# yet (e.g. starting up). pkill is best-effort; never fail the script on it.
if command -v pkill >/dev/null 2>&1; then
  pkill -f 'uvicorn[[:space:]]+app:app' 2>/dev/null || true
fi

for port in "${PORTS[@]}"; do
  clear_port "${port}"
done

# Final report.
echo
echo "Port status:"
remaining=0
for port in "${PORTS[@]}"; do
  pids="$(pids_on_port "${port}" | xargs -r echo)"
  if [[ -z "${pids}" ]]; then
    printf "  :%s  free\n" "${port}"
  else
    printf "  :%s  STILL IN USE by %s\n" "${port}" "${pids}"
    remaining=1
  fi
done

if [[ "${remaining}" -ne 0 ]]; then
  exit 1
fi
