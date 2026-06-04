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
EVIDENCE_OUT=""
PREFLIGHT_ONLY=0
CLEAN_PROCESS=0
STEP_LOG=""

usage() {
    cat <<'EOF'
Usage: scripts/validation/run_managed_paid_pilot_gate.sh [options]

Options:
  --env-file PATH      Provider env file to source (default: /tmp/tp-managed-staging.env)
  --evidence-out PATH  Write a redacted managed-provider acceptance note outside the repository
  --preflight-only     Validate the clean env and stop before migrations/tests
  -h, --help           Show this help

The env file must be outside the repository, mode 0600 or stricter, and contain
real provider-managed Postgres, Redis, frontdoor Redis, and S3-compatible values.
The evidence output is optional and must also live outside the repository.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup_step_log() {
    [[ -z "${STEP_LOG}" ]] || rm -f "${STEP_LOG}"
}

init_step_log() {
    if [[ -z "${STEP_LOG}" ]]; then
        STEP_LOG="$(mktemp "${TMPDIR:-/tmp}/tp-managed-paid-pilot-steps.XXXXXX")"
        trap cleanup_step_log EXIT
    fi
}

record_step() {
    init_step_log
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >>"${STEP_LOG}"
}

while (($#)); do
    case "$1" in
        --env-file)
            shift
            [[ $# -gt 0 ]] || die "--env-file requires a path"
            ENV_FILE="$1"
            ;;
        --evidence-out)
            shift
            [[ $# -gt 0 ]] || die "--evidence-out requires a path"
            EVIDENCE_OUT="$1"
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
    if [[ -n "${EVIDENCE_OUT}" ]]; then
        REEXEC_ARGS+=(--evidence-out "${EVIDENCE_OUT}")
    fi
    if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
        REEXEC_ARGS+=(--preflight-only)
    fi

    exec env -i \
        HOME="${HOME:-}" \
        PATH="${CLEAN_PATH}" \
        USER="${USER:-}" \
        SHELL="/bin/bash" \
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

validate_evidence_out() {
    [[ -n "${EVIDENCE_OUT}" ]] || return 0
    "${PYTHON_BIN}" - "${EVIDENCE_OUT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
repo_root = Path.cwd().resolve()

resolved = evidence_path.expanduser().resolve()
try:
    resolved.relative_to(repo_root)
except ValueError:
    pass
else:
    print("ERROR: managed paid-pilot evidence output must live outside the repository", file=sys.stderr)
    raise SystemExit(1)

resolved.parent.mkdir(parents=True, exist_ok=True)
PY
}

write_evidence_note() {
    [[ -n "${EVIDENCE_OUT}" ]] || return 0
    "${PYTHON_BIN}" - "${EVIDENCE_OUT}" "$1" "${STEP_LOG}" <<'PY'
from __future__ import annotations

from collections import OrderedDict
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

evidence_path = Path(sys.argv[1])
gate_status = sys.argv[2]
step_log_path = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
repo_root = Path.cwd().resolve()

resolved = evidence_path.expanduser().resolve()
try:
    resolved.relative_to(repo_root)
except ValueError:
    pass
else:
    print("ERROR: managed paid-pilot evidence output must live outside the repository", file=sys.stderr)
    raise SystemExit(1)

resolved.parent.mkdir(parents=True, exist_ok=True)

selectors = {
    "TP_ORCHESTRATOR_STATE_BACKEND": os.getenv("TP_ORCHESTRATOR_STATE_BACKEND", ""),
    "TP_ORCHESTRATOR_QUEUE_BACKEND": os.getenv("TP_ORCHESTRATOR_QUEUE_BACKEND", ""),
    "TP_FRONTDOOR_SESSION_STORE": os.getenv("TP_FRONTDOOR_SESSION_STORE", ""),
    "TP_ARTIFACT_STORE": os.getenv("TP_ARTIFACT_STORE", ""),
}

def _url_summary(name: str) -> str:
    raw = os.getenv(name, "").strip()
    if not raw:
        return "unset"
    parsed = urlsplit(raw)
    scheme = parsed.scheme or "unknown"
    database_or_path = "set" if parsed.path.strip("/") else "unset"
    tls = "yes" if scheme.endswith("s") or scheme in {"https", "rediss"} else "no"
    return f"scheme={scheme}; tls={tls}; path={database_or_path}"

def _set_summary(name: str) -> str:
    return "set" if os.getenv(name, "").strip() else "unset"

def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

def _load_steps(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []

    steps: OrderedDict[str, dict[str, str]] = OrderedDict()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            label, command, status, exit_code = raw_line.split("\t", 3)
        except ValueError:
            continue
        if label not in steps:
            steps[label] = {
                "label": label,
                "command": command,
                "status": status,
                "exit_code": exit_code,
            }
        else:
            steps[label].update(
                {
                    "command": command,
                    "status": status,
                    "exit_code": exit_code,
                }
            )
    return list(steps.values())

try:
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"

generated_at = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
steps = _load_steps(step_log_path)
lines = [
    "# Managed Provider Paid-Pilot Acceptance Note",
    "",
    f"- generated_at_utc: `{generated_at}`",
    f"- git_commit: `{commit}`",
    f"- gate_status: `{gate_status}`",
    "",
    "## Redacted Selector Summary",
    "",
]
for name, value in selectors.items():
    lines.append(f"- `{name}`: `{value or 'unset'}`")

lines.extend(
    [
        "",
        "## Redacted Endpoint Summary",
        "",
        f"- `TP_DATABASE_URL`: `{_url_summary('TP_DATABASE_URL')}`",
        f"- `TP_TEST_POSTGRES_URL`: `{_url_summary('TP_TEST_POSTGRES_URL')}`",
        f"- `TP_REDIS_URL`: `{_url_summary('TP_REDIS_URL')}`",
        f"- `TP_TEST_REDIS_URL`: `{_url_summary('TP_TEST_REDIS_URL')}`",
        f"- `TP_FRONTDOOR_REDIS_URL`: `{_url_summary('TP_FRONTDOOR_REDIS_URL')}`",
        f"- `TP_ARTIFACT_ENDPOINT_URL`: `{_url_summary('TP_ARTIFACT_ENDPOINT_URL')}`",
        f"- `TP_TEST_S3_URL`: `{_url_summary('TP_TEST_S3_URL')}`",
        f"- `TP_ARTIFACT_BUCKET`: `{_set_summary('TP_ARTIFACT_BUCKET')}`",
        f"- `TP_TEST_S3_BUCKET`: `{_set_summary('TP_TEST_S3_BUCKET')}`",
        f"- `TP_ARTIFACT_REGION`: `{_set_summary('TP_ARTIFACT_REGION')}`",
        "",
        "## Gate Step Results",
        "",
    ]
)
if steps:
    lines.extend(
        [
            "| Step | Command | Result | Exit code |",
            "| --- | --- | --- | --- |",
        ]
    )
    for step in steps:
        exit_code = step["exit_code"] or "-"
        lines.append(
            "| "
            f"{_markdown_cell(step['label'])} | "
            f"`{_markdown_cell(step['command'])}` | "
            f"`{_markdown_cell(step['status'])}` | "
            f"`{_markdown_cell(exit_code)}` |"
        )
else:
    lines.append("- No gate steps recorded.")

lines.extend(
    [
        "",
        "## Evidence Checklist",
        "",
        "- Confirm every gate step above passed before opening paid-pilot admission.",
        "- Cleanup result: record validation Postgres/Redis/S3 cleanup confirmation.",
        "",
        "Do not add secrets, private hostnames, customer identifiers, real usernames, or bucket names to this note.",
        "",
    ]
)
resolved.write_text("\n".join(lines), encoding="utf-8")
print(f"Managed paid-pilot evidence note written: {resolved}")
PY
}

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

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


def postgres_identity(value: str) -> tuple[str, str, int, str]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.split("+", 1)[0]
    host = (parsed.hostname or "").lower()
    port = parsed.port or 5432
    database = parsed.path.lstrip("/")
    return (scheme, host, port, database)


provider_env_names = sorted(
    name for name in os.environ if name.startswith("TP_") or name.startswith("AWS_")
)
missing = sorted(name for name in required if not os.getenv(name, "").strip())
placeholder = sorted(name for name in provider_env_names if placeholder_like(os.getenv(name, "")))
wrong = {name: os.getenv(name, "") for name, value in expected.items() if os.getenv(name) != value}
leaked = sorted(name for name in os.environ if name in forbidden or name.startswith(forbidden_prefixes))
unsafe_overlap: dict[str, str] = {}
db_url = os.getenv("TP_DATABASE_URL", "")
test_db_url = os.getenv("TP_TEST_POSTGRES_URL", "")
if db_url and test_db_url and postgres_identity(db_url) == postgres_identity(test_db_url):
    unsafe_overlap["TP_TEST_POSTGRES_URL"] = (
        "must not target the same Postgres host/port/database as TP_DATABASE_URL"
    )
if os.getenv("TP_ARTIFACT_BUCKET") and os.getenv("TP_ARTIFACT_BUCKET") == os.getenv("TP_TEST_S3_BUCKET"):
    unsafe_overlap["TP_TEST_S3_BUCKET"] = "must not equal TP_ARTIFACT_BUCKET"

print("missing:", missing)
print("placeholder-like:", placeholder)
print("wrong selectors:", wrong)
print("leaked local-dev vars:", leaked)
print("unsafe managed/test overlap:", unsafe_overlap)

raise SystemExit(1 if missing or placeholder or wrong or leaked or unsafe_overlap else 0)
PY

validate_evidence_out
record_step "Clean env preflight" "managed provider env validation" "passed" "0"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    write_evidence_note "preflight_passed"
    printf '%s\n' "Managed paid-pilot clean-env preflight passed."
    exit 0
fi

initialize_gate_plan() {
    record_step "Database migrations" "make db-upgrade" "not_run" ""
    record_step "Orchestrator Postgres contract" "TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-orchestrator-postgres-contract" "not_run" ""
    record_step "Orchestrator Postgres app contract" "TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-orchestrator-postgres-app-contract" "not_run" ""
    record_step "Worker Redis contract" "TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-worker-redis-contract" "not_run" ""
    record_step "Artifact S3 contract" "make test-artifact-s3-contract" "not_run" ""
    record_step "Frontdoor Redis contract" "make test-frontdoor-redis-contract" "not_run" ""
    record_step "Integrated paid-pilot smoke" "TP_RUN_PAID_PILOT_SERVICES_CONTRACT=1 python -m pytest -q tests/orchestrator/test_paid_pilot_services_contract.py -m unit" "not_run" ""
}

run_gate_step() {
    local label="$1"
    local command_display="$2"
    shift 2

    printf 'Running managed paid-pilot gate step: %s\n' "${label}"
    set +e
    "$@"
    local exit_code=$?
    set -e

    if [[ "${exit_code}" == "0" ]]; then
        record_step "${label}" "${command_display}" "passed" "${exit_code}"
        return 0
    fi

    record_step "${label}" "${command_display}" "failed" "${exit_code}"
    write_evidence_note "gate_failed" || true
    exit "${exit_code}"
}

initialize_gate_plan
run_gate_step "Database migrations" "make db-upgrade" make db-upgrade
run_gate_step "Orchestrator Postgres contract" "TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-orchestrator-postgres-contract" env TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-orchestrator-postgres-contract
run_gate_step "Orchestrator Postgres app contract" "TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-orchestrator-postgres-app-contract" env TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-orchestrator-postgres-app-contract
run_gate_step "Worker Redis contract" "TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-worker-redis-contract" env TP_ORCHESTRATOR_STATE_BACKEND=memory TP_ORCHESTRATOR_QUEUE_BACKEND=memory TP_ARTIFACT_STORE=local make test-worker-redis-contract
run_gate_step "Artifact S3 contract" "make test-artifact-s3-contract" make test-artifact-s3-contract
run_gate_step "Frontdoor Redis contract" "make test-frontdoor-redis-contract" make test-frontdoor-redis-contract
run_gate_step "Integrated paid-pilot smoke" "TP_RUN_PAID_PILOT_SERVICES_CONTRACT=1 python -m pytest -q tests/orchestrator/test_paid_pilot_services_contract.py -m unit" env TP_RUN_PAID_PILOT_SERVICES_CONTRACT=1 "${PYTHON_BIN}" -m pytest -q tests/orchestrator/test_paid_pilot_services_contract.py -m unit
write_evidence_note "gate_passed"
