#!/usr/bin/env bash
# Apply CI Dependency Resolution Patches (safer)
#
# Usage: ./apply-patches.sh [repo-root]
set -euo pipefail

REPO_ROOT="${1:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Applying CI Dependency Resolution Patches to ${REPO_ROOT}"

# Ensure repo path exists
if [ ! -d "${REPO_ROOT}" ]; then
  echo "ERROR: repo root '${REPO_ROOT}' does not exist."
  exit 2
fi

# Ensure requirements directory exists in script dir for copy source
if [ ! -f "${SCRIPT_DIR}/requirements/ml.in" ]; then
  echo "ERROR: patch source requirements/ml.in not found in ${SCRIPT_DIR}/requirements/ml.in"
  exit 2
fi

# Ensure workspace requirements dir exists
mkdir -p "${REPO_ROOT}/requirements"

# 1. Copy requirements/ml.in if it doesn't exist in repo
if [ ! -f "${REPO_ROOT}/requirements/ml.in" ]; then
    echo "✅ Creating ${REPO_ROOT}/requirements/ml.in..."
    cp "${SCRIPT_DIR}/requirements/ml.in" "${REPO_ROOT}/requirements/ml.in"
else
    echo "⚠️  ${REPO_ROOT}/requirements/ml.in already exists, skipping..."
fi

# 1b. Warn if constraints missing (CI expects this)
if [ ! -f "${REPO_ROOT}/requirements/constraints.txt" ]; then
    echo "⚠️  ${REPO_ROOT}/requirements/constraints.txt not found."
    echo "    CI workflows use -c requirements/constraints.txt; create or commit it before running CI."
fi

# 2. Backup and update performance-monitor.yml
PM_TARGET="${REPO_ROOT}/.github/workflows/performance-monitor.yml"
mkdir -p "$(dirname "${PM_TARGET}")"
if [ -f "${PM_TARGET}" ]; then
    BACKUP="${PM_TARGET}.bak.$(date +%s)"
    echo "🔁 Backing up existing ${PM_TARGET} -> ${BACKUP}"
    cp "${PM_TARGET}" "${BACKUP}"
fi
echo "✅ Updating ${PM_TARGET}..."
cp "${SCRIPT_DIR}/.github/workflows/performance-monitor.yml" "${PM_TARGET}"

# 3. Inform about ci-consolidated patch (manual)
CI_TARGET="${REPO_ROOT}/.github/workflows/ci-consolidated.yml"
PATCH_FILE="${SCRIPT_DIR}/ci-consolidated.yml.patch"
if [ -f "${CI_TARGET}" ]; then
    if [ -f "${PATCH_FILE}" ]; then
        echo "🔧 Attempting to apply patch to ${CI_TARGET}..."
        cd "${REPO_ROOT}"
        if git apply "${PATCH_FILE}"; then
            echo "✅ Patch applied cleanly to ${CI_TARGET}."
        else
            echo "⚠️  Patch failed to apply cleanly. Manual merge may be needed."
            echo "   Review: ${PATCH_FILE}"
        fi
    else
        echo "⚠️  Patch file ${PATCH_FILE} not found; skipping."
    fi
else
    echo "ℹ️  ${CI_TARGET} not found; skipping consolidated CI patch."
fi

echo "Done. Next steps: pip install pip-tools; pip-compile requirements/ml.in -o requirements/ml.txt; commit changes."
