#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 <black|isort|parity> [args...]" >&2
    exit 64
fi

shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LINT_VENV="${TP_LINT_VENV:-$REPO_ROOT/.venv-lint}"
LINT_REQUIREMENTS="$REPO_ROOT/requirements-lint.txt"
LINT_STAMP="$LINT_VENV/.requirements-lint.sha256"

resolve_bootstrap_python() {
    if [[ -n "${TP_LINT_PYTHON:-}" ]]; then
        printf '%s\n' "$TP_LINT_PYTHON"
        return
    fi

    if command -v python3.12 >/dev/null 2>&1; then
        printf '%s\n' "python3.12"
        return
    fi

    if [[ -x "$LINT_VENV/bin/python" ]]; then
        printf '%s\n' "$LINT_VENV/bin/python"
        return
    fi

    echo "run_lint_tool: python3.12 is required for CI lint parity." >&2
    echo "Set TP_LINT_PYTHON=/path/to/python3.12 if it is installed outside PATH." >&2
    exit 1
}

BOOTSTRAP_PYTHON="$(resolve_bootstrap_python)"

compute_requirements_hash() {
    "$BOOTSTRAP_PYTHON" - <<'PY' "$LINT_REQUIREMENTS"
from pathlib import Path
import hashlib
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

matches_lint_requirements() {
    local python_bin="$1"
    "$python_bin" - <<'PY' "$LINT_REQUIREMENTS"
from importlib import metadata
from pathlib import Path
import sys

required = {}
for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "==" not in line:
        continue
    name, version = [part.strip() for part in line.split("==", 1)]
    key = name.lower()
    if key in {"black", "isort", "flake8", "pylint"}:
        required[key] = version

for name, version in required.items():
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(1)
    if installed != version:
        raise SystemExit(1)
PY
}

ensure_lint_env() {
    local requirements_hash

    if matches_lint_requirements "$BOOTSTRAP_PYTHON"; then
        LINT_PYTHON="$BOOTSTRAP_PYTHON"
        return
    fi

    if [[ ! -x "$LINT_VENV/bin/python" ]]; then
        "$BOOTSTRAP_PYTHON" -m venv "$LINT_VENV"
    fi

    if ! "$LINT_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        "$LINT_VENV/bin/python" -m ensurepip --upgrade >/dev/null
    fi

    requirements_hash="$(compute_requirements_hash)"

    if [[ ! -f "$LINT_STAMP" ]] || [[ "$(<"$LINT_STAMP")" != "$requirements_hash" ]] || ! matches_lint_requirements "$LINT_VENV/bin/python"; then
        "$LINT_VENV/bin/python" -m pip install --upgrade pip >/dev/null
        "$LINT_VENV/bin/python" -m pip install -r "$LINT_REQUIREMENTS" >/dev/null
        printf '%s\n' "$requirements_hash" > "$LINT_STAMP"
    fi

    LINT_PYTHON="$LINT_VENV/bin/python"
}

run_black() {
    ensure_lint_env
    exec "$LINT_PYTHON" -m black --line-length=127 "$@"
}

run_isort() {
    ensure_lint_env
    exec "$LINT_PYTHON" -m isort --profile=black --line-length=127 "$@"
}

run_parity() {
    ensure_lint_env

    "$LINT_PYTHON" -m black --check --diff --line-length=127 src/ tests/
    "$LINT_PYTHON" -m isort --check-only --diff --profile=black --line-length=127 src/ tests/
    PYTHON_BIN="$LINT_PYTHON" \
    LINT_RUNNER_GITHUB_EVENT_NAME=pull_request \
    "$REPO_ROOT/scripts/lint_runner.sh" pr
    "$LINT_PYTHON" "$REPO_ROOT/scripts/validation/check_raw_json_usage.py"
}

cd "$REPO_ROOT"

case "$MODE" in
    black)
        run_black "$@"
        ;;
    isort)
        run_isort "$@"
        ;;
    parity)
        run_parity
        ;;
    *)
        echo "Usage: $0 <black|isort|parity> [args...]" >&2
        exit 64
        ;;
esac
