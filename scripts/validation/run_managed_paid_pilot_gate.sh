#!/usr/bin/env bash
#
# Run the managed-provider paid-pilot gate from a clean process.
#
# This intentionally does not create or populate the provider env file. Operators
# must provide real staging values and disposable TP_TEST_* resources outside the
# repository, usually at /tmp/tp-managed-staging.env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ENV_FILE="/tmp/tp-managed-staging.env"

ENV_FILE="${DEFAULT_ENV_FILE}"
PREFLIGHT_ONLY=0
CLEAN_PROCESS=0

usage() {
    cat <<'EOF'
Usage: scripts/validation/run_managed_paid_pilot_gate.sh [options]

Options:
  --env-file PATH      Provider env file to source (default: /tmp/tp-managed-staging.env)
  --preflight-only     Validate the clean env and stop before migrations/tests
  -h, --help           Show this help

The env file must be outside the repository, mode 0600 or stricter, and contain
real provider-managed Postgres, Redis, frontdoor Redis, and S3-compatible values.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --env-file)
            shift
            [[ $# -gt 0 ]] || die "--env-file requires a path"
            ENV_FILE="$1"
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=1
            ;;
        --_clean-process)
            CLEAN_PROCESS=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

if [[ "${CLEAN_PROCESS}" != "1" ]]; then
    NODE_BIN="$(command -v node 2>/dev/null || true)"
    PYTHON_BIN="$("${REPO_ROOT}/scripts/setup/resolve_python_311.sh" 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    CLEAN_PATH="/usr/bin:/bin:/usr/sbin:/sbin"
    if [[ -n "${NODE_BIN}" ]]; then
        NODE_DIR="$(cd "$(dirname "${NODE_BIN}")" && pwd)"
        CLEAN_PATH="${NODE_DIR}:${CLEAN_PATH}"
    fi
    if [[ -n "${PYTHON_BIN}" ]]; then
        PYTHON_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
        CLEAN_PATH="${PYTHON_DIR}:${CLEAN_PATH}"
    fi
    REEXEC_ARGS=(--env-file "${ENV_FILE}")
    if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
        REEXEC_ARGS+=(--preflight-only)
    fi

    exec env -i \
        HOME="${HOME:-}" \
        PATH="${CLEAN_PATH}" \
        USER="${USER:-}" \
        SHELL="/bin/zsh" \
        /bin/bash "$0" \
        --_clean-process \
        "${REEXEC_ARGS[@]}"
fi

cd "${REPO_ROOT}"

[[ -f "${ENV_FILE}" ]] || die "missing managed provider env file: ${ENV_FILE}"
[[ "${ENV_FILE}" != "${REPO_ROOT}"/* ]] || die "managed provider env file must live outside the repository"

PYTHON_BIN="$("${REPO_ROOT}/scripts/setup/resolve_python_311.sh")"

"${PYTHON_BIN}" - "${ENV_FILE}" <<'PY'
from __future__ import annotations

import stat
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
try:
    mode = stat.S_IMODE(env_file.stat().st_mode)
except OSError as exc:
    print(f"ERROR: cannot stat managed provider env file: {exc}", file=sys.stderr)
    raise SystemExit(1)

if mode & 0o077:
    print(
        f"ERROR: managed provider env file must be chmod 600 or stricter; observed {oct(mode)}",
        file=sys.stderr,
    )
    raise SystemExit(1)

repo_root = Path.cwd().resolve()
try:
    env_file.resolve().relative_to(repo_root)
except ValueError:
    pass
else:
    print("ERROR: managed provider env file must live outside the repository", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    PYTHON_BIN="$(command -v python)"
elif [[ "${PREFLIGHT_ONLY}" != "1" ]]; then
    die "missing .venv/bin/activate; run make install-core before this gate"
fi
./scripts/setup/ensure_node_version.sh

set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
import sys

required = {
    "TP_ORCHESTRATOR_STATE_BACKEND",
    "TP_DATABASE_URL",
    "TP_TEST_POSTGRES_URL",
    "TP_ORCHESTRATOR_QUEUE_BACKEND",
    "TP_REDIS_URL",
    "TP_TEST_REDIS_URL",
    "TP_FRONTDOOR_SESSION_STORE",
    "TP_FRONTDOOR_REDIS_URL",
    "TP_ARTIFACT_STORE",
    "TP_ARTIFACT_ENDPOINT_URL",
    "TP_TEST_S3_URL",
    "TP_ARTIFACT_BUCKET",
    "TP_TEST_S3_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
}

expected = {
    "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
    "TP_ORCHESTRATOR_QUEUE_BACKEND": "redis",
    "TP_FRONTDOOR_SESSION_STORE": "redis",
    "TP_ARTIFACT_STORE": "s3",
}

forbidden = {
    "TP_API_KEY",
    "TP_BACKEND_API_KEY",
    "TP_BACKEND_ORIGIN",
    "TP_FASTAPI_ORIGIN",
    "TP_ALLOWED_ORIGINS",
    "TP_ALLOW_LOCAL_ACCESS_BYPASS",
    "TP_FRONTDOOR_SESSION_DB",
    "TP_FRONTDOOR_SESSION_SCALING_MODE",
    "TP_PORTAL_DIRECT_DEBUG_COHORT_KEY",
    "TP_PORTAL_RUM_LOG_PATH",
    "TP_PORTAL_EVENT_LOG_PATH",
    "TP_LOG_HEALTHCHECKS",
}
forbidden_prefixes = ("TP_FASTVLM_", "TP_PORTAL_FASTVLM_", "TP_PORTAL_UPLOAD_")
placeholder_tokens = ("<", "redacted", "replace-", "...")


def placeholder_like(value: str) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in placeholder_tokens)


missing = sorted(name for name in required if not os.getenv(name, "").strip())
placeholder = sorted(name for name in required if placeholder_like(os.getenv(name, "")))
wrong = {name: os.getenv(name, "") for name, value in expected.items() if os.getenv(name) != value}
leaked = sorted(name for name in os.environ if name in forbidden or name.startswith(forbidden_prefixes))
unsafe_overlap: dict[str, str] = {}
if os.getenv("TP_DATABASE_URL") and os.getenv("TP_DATABASE_URL") == os.getenv("TP_TEST_POSTGRES_URL"):
    unsafe_overlap["TP_TEST_POSTGRES_URL"] = "must not equal TP_DATABASE_URL"
if os.getenv("TP_ARTIFACT_BUCKET") and os.getenv("TP_ARTIFACT_BUCKET") == os.getenv("TP_TEST_S3_BUCKET"):
    unsafe_overlap["TP_TEST_S3_BUCKET"] = "must not equal TP_ARTIFACT_BUCKET"

print("missing:", missing)
print("placeholder-like:", placeholder)
print("wrong selectors:", wrong)
print("leaked local-dev vars:", leaked)
print("unsafe managed/test overlap:", unsafe_overlap)

raise SystemExit(1 if missing or placeholder or wrong or leaked or unsafe_overlap else 0)
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    printf '%s\n' "Managed paid-pilot clean-env preflight passed."
    exit 0
fi

make db-upgrade
make test-paid-pilot-services-contract
