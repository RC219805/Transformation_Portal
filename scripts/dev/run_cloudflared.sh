#!/usr/bin/env bash
# Launch a Cloudflare tunnel in front of the local FastAPI backend.
#
# Two modes:
#   - Named tunnel (preferred): set CLOUDFLARED_TUNNEL_NAME to a tunnel that
#     was created via `cloudflared tunnel create <name>` and DNS-routed via
#     `cloudflared tunnel route dns <name> <hostname>`. The hostname stays
#     stable across restarts, which is what production frontdoors need.
#   - Quick tunnel (dev only): no env set; falls back to `--url`. Forces the
#     HTTP/2 transport because QUIC has been observed to fail intermittently
#     during long-running sessions (per the executive diagnosis logs).
#
# After the tunnel is up, the assigned hostname is written to
# /tmp/tp-cloudflared-host so scripts/dev/start_local_stack.sh can add it to
# TP_TRUSTED_HOSTS before the FastAPI backend starts.

set -euo pipefail

SENTINEL="${TP_CLOUDFLARED_HOST_FILE:-/tmp/tp-cloudflared-host}"
LOCAL_PORT="${TP_BACKEND_LOCAL_PORT:-8000}"
LOCAL_URL="http://127.0.0.1:${LOCAL_PORT}"
PROTOCOL="${TP_CLOUDFLARED_PROTOCOL:-http2}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "run_cloudflared.sh: 'cloudflared' is not installed. See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/" >&2
  exit 1
fi

cleanup() {
  rm -f "${SENTINEL}" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -n "${CLOUDFLARED_TUNNEL_NAME:-}" ]]; then
  HOSTNAME="${CLOUDFLARED_TUNNEL_HOSTNAME:-}"
  if [[ -z "${HOSTNAME}" ]]; then
    echo "run_cloudflared.sh: CLOUDFLARED_TUNNEL_NAME is set but CLOUDFLARED_TUNNEL_HOSTNAME is not." >&2
    echo "Set CLOUDFLARED_TUNNEL_HOSTNAME to the hostname routed to the tunnel" >&2
    echo "(e.g. via 'cloudflared tunnel route dns <name> <hostname>')." >&2
    exit 1
  fi
  printf '%s\n' "${HOSTNAME}" > "${SENTINEL}"
  echo "Starting named Cloudflare tunnel '${CLOUDFLARED_TUNNEL_NAME}' (hostname=${HOSTNAME})"
  cloudflared tunnel --protocol "${PROTOCOL}" run "${CLOUDFLARED_TUNNEL_NAME}"
  exit $?
fi

echo "No CLOUDFLARED_TUNNEL_NAME set; starting an ephemeral quick tunnel against ${LOCAL_URL}."
echo "Quick tunnels are NOT stable for production frontdoor origins."

LOG_FILE="$(mktemp -t tp-cloudflared.XXXXXX.log)"
cloudflared tunnel --protocol "${PROTOCOL}" --url "${LOCAL_URL}" 2>&1 | tee "${LOG_FILE}" &
CF_PID=$!
trap 'kill ${CF_PID} 2>/dev/null || true; cleanup' EXIT INT TERM

# Wait briefly for cloudflared to advertise the trycloudflare hostname.
for _ in $(seq 1 60); do
  if grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | head -n 1 > /tmp/tp-cf-host.tmp; then
    HOST_URL="$(cat /tmp/tp-cf-host.tmp 2>/dev/null || true)"
    if [[ -n "${HOST_URL}" ]]; then
      printf '%s\n' "${HOST_URL#https://}" > "${SENTINEL}"
      echo "Tunnel hostname written to ${SENTINEL}: ${HOST_URL#https://}"
      break
    fi
  fi
  sleep 1
done

wait ${CF_PID}
