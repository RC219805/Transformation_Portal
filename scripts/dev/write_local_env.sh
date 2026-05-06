#!/usr/bin/env bash
# Write the canonical local-development environment file.
#
# Produces /tmp/tp-local-http-all-on.env with TP_API_KEY and TP_BACKEND_API_KEY
# bound to the same value, plus the other envs that the backend and managed
# frontdoor consume. This is the single authoritative source for local secrets;
# do not generate keys ad-hoc in shells without rewriting this file.
#
# Usage:
#   scripts/dev/write_local_env.sh             # idempotent: keep existing key
#   scripts/dev/write_local_env.sh --rotate    # generate a new key
#   TP_LOCAL_API_KEY=<value> scripts/dev/write_local_env.sh   # use a supplied key
#
# After writing, source the file in any shell that runs the backend or frontdoor:
#   source /tmp/tp-local-http-all-on.env

set -euo pipefail

ENV_FILE="${TP_LOCAL_ENV_FILE:-/tmp/tp-local-http-all-on.env}"
USERS_FILE="${TP_FRONTDOOR_USERS_FILE:-/tmp/tp-frontdoor-users.json}"
SESSION_DB="${TP_FRONTDOOR_SESSION_DB:-/tmp/transformation-portal-frontdoor-sessions.db}"
ROTATE=0

for arg in "$@"; do
  case "${arg}" in
    --rotate) ROTATE=1 ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "write_local_env.sh: unknown argument '${arg}'" >&2
      exit 2
      ;;
  esac
done

ensure_writable_dir() {
  local dir
  dir="$(dirname "$1")"
  if [[ ! -d "${dir}" ]]; then
    echo "write_local_env.sh: parent directory '${dir}' does not exist" >&2
    exit 1
  fi
  if ! [[ -w "${dir}" ]]; then
    echo "write_local_env.sh: parent directory '${dir}' is not writable" >&2
    exit 1
  fi
}

ensure_writable_dir "${ENV_FILE}"
ensure_writable_dir "${USERS_FILE}"
ensure_writable_dir "${SESSION_DB}"

decode_single_quoted_export_value() {
  local token="$1"
  local value=""
  local idx=0
  local len=${#token}
  local char
  local closed
  local next_idx

  while ((idx < len)); do
    char="${token:idx:1}"
    if [[ "${char}" == "'" ]]; then
      idx=$((idx + 1))
      closed=0
      while ((idx < len)); do
        char="${token:idx:1}"
        if [[ "${char}" == "'" ]]; then
          closed=1
          idx=$((idx + 1))
          break
        fi
        value+="${char}"
        idx=$((idx + 1))
      done
      if [[ "${closed}" -ne 1 ]]; then
        return 1
      fi
    elif [[ "${char}" == "\\" ]]; then
      next_idx=$((idx + 1))
      if ((next_idx >= len)) || [[ "${token:next_idx:1}" != "'" ]]; then
        return 1
      fi
      value+="'"
      idx=$((idx + 2))
    else
      return 1
    fi
  done

  printf '%s\n' "${value}"
}

decode_double_quoted_export_value() {
  local token="$1"
  local len=${#token}
  if ((len < 2)) || [[ "${token:0:1}" != '"' || "${token:len - 1:1}" != '"' ]]; then
    return 1
  fi

  local content_len=$((len - 2))
  local content="${token:1:content_len}"
  local value=""
  local idx=0
  local char
  local next
  local next_idx

  while ((idx < content_len)); do
    char="${content:idx:1}"
    if [[ "${char}" == '"' ]]; then
      return 1
    fi
    if [[ "${char}" == "\\" ]]; then
      next_idx=$((idx + 1))
      if ((next_idx >= content_len)); then
        return 1
      fi
      next="${content:next_idx:1}"
      case "${next}" in
        '$' | '"' | '`' | '\')
          value+="${next}"
          idx=$((idx + 2))
          ;;
        *)
          value+="\\"
          idx=$((idx + 1))
          ;;
      esac
    else
      value+="${char}"
      idx=$((idx + 1))
    fi
  done

  printf '%s\n' "${value}"
}

resolve_existing_key() {
  local line
  local token
  local parsed

  if [[ ! -f "${ENV_FILE}" ]]; then
    return 0
  fi

  while IFS= read -r line; do
    case "${line}" in
      export\ TP_API_KEY=*)
        token="${line#export TP_API_KEY=}"
        if parsed="$(decode_single_quoted_export_value "${token}")"; then
          printf '%s\n' "${parsed}"
          return 0
        fi
        if parsed="$(decode_double_quoted_export_value "${token}")"; then
          printf '%s\n' "${parsed}"
          return 0
        fi
        return 0
        ;;
    esac
  done < "${ENV_FILE}"
}

EXISTING_KEY="$(resolve_existing_key || true)"

if [[ -n "${TP_LOCAL_API_KEY:-}" ]]; then
  KEY="${TP_LOCAL_API_KEY}"
elif [[ "${ROTATE}" -eq 1 || -z "${EXISTING_KEY}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    KEY="$(openssl rand -base64 32 | tr -d '\n')"
  elif [[ -r /dev/urandom ]]; then
    KEY="$(head -c 32 /dev/urandom | base64 | tr -d '\n')"
  else
    echo "write_local_env.sh: cannot generate a key (no openssl, no /dev/urandom)" >&2
    exit 1
  fi
else
  KEY="${EXISTING_KEY}"
fi

if [[ -z "${KEY}" ]]; then
  echo "write_local_env.sh: refusing to write empty key" >&2
  exit 1
fi

quote_shell_value() {
  local value="$1"
  printf "'"
  printf '%s' "${value}" | sed "s/'/'\\\\''/g"
  printf "'"
}

write_export() {
  local name="$1"
  local value="$2"
  printf 'export %s=' "${name}"
  quote_shell_value "${value}"
  printf '\n'
}

umask 077

{
  printf '%s\n' "# Canonical local Transformation Portal environment."
  printf '%s\n' "# Generated by scripts/dev/write_local_env.sh — do not hand-edit."
  printf '%s\n' "# Both TP_API_KEY (backend) and TP_BACKEND_API_KEY (frontdoor) MUST agree."
  write_export TP_API_KEY "${KEY}"
  write_export TP_BACKEND_API_KEY "${KEY}"
  write_export TP_BACKEND_ORIGIN "http://127.0.0.1:8000"
  write_export TP_FASTAPI_ORIGIN "http://127.0.0.1:8000"
  write_export TP_FRONTDOOR_USERS_FILE "${USERS_FILE}"
  write_export TP_FRONTDOOR_SESSION_DB "${SESSION_DB}"
  write_export TP_FRONTDOOR_SESSION_SCALING_MODE "single_instance"
  write_export TP_TRUSTED_HOSTS "localhost,127.0.0.1,::1,testserver"
  write_export TP_ALLOW_LOCAL_ACCESS_BYPASS "1"
} > "${ENV_FILE}"

chmod 600 "${ENV_FILE}"

echo "Wrote ${ENV_FILE}"
if [[ "${ROTATE}" -eq 1 ]]; then
  echo "Rotated TP_API_KEY / TP_BACKEND_API_KEY (length=${#KEY})."
elif [[ -n "${EXISTING_KEY}" && "${EXISTING_KEY}" == "${KEY}" ]]; then
  echo "Reused existing key (length=${#KEY}). Use --rotate to generate a new one."
else
  echo "Wrote new key (length=${#KEY})."
fi
echo "Source it before starting backend or frontdoor:"
echo "  source ${ENV_FILE}"
