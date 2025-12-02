#!/usr/bin/env bash
# Apply CI Dependency Resolution Patches (safer)
#
# Usage: ./apply-patches.sh [repo-root]
#
# This script is intended to apply CI dependency resolution patches to a target
# repository. When run from within the main repository with default arguments,
# it will skip self-referential copies.
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

# 2. Backup and update performance-monitor.yml (avoid self-referential copy)
PM_TARGET="${REPO_ROOT}/.github/workflows/performance-monitor.yml"
SRC_PM="${SCRIPT_DIR}/.github/workflows/performance-monitor.yml"

# Security: Resolve REPO_ROOT to canonical path (following symlinks) before any operations
ROOT_REAL="$(realpath "${REPO_ROOT}")"

# Create target directory within the resolved repo root
TARGET_DIR="${ROOT_REAL}/.github/workflows"
mkdir -p "${TARGET_DIR}"

# Construct the final target path using the resolved root
PM_TARGET_RESOLVED="${TARGET_DIR}/performance-monitor.yml"

# Resolve source path for self-reference check
SRC_PM_REAL="$(realpath "${SRC_PM}" 2>/dev/null || echo "${SRC_PM}")"

# Verify the resolved target is within the resolved repo root (protects against symlink traversal)
PM_REAL="$(realpath -m "${PM_TARGET_RESOLVED}" 2>/dev/null || echo)"

# Refuse to operate if target path escapes repo root or if final target is a symlink
if [ -z "${PM_REAL}" ] || [[ "${PM_REAL}" != "${ROOT_REAL}"/* ]]; then
    echo "ERROR: Refusing to operate on path outside repo: ${PM_TARGET_RESOLVED}" >&2
    exit 1
fi

# Additional check: refuse if the target itself is a symlink
if [ -L "${PM_TARGET_RESOLVED}" ]; then
    echo "ERROR: Refusing to operate on symlink target: ${PM_TARGET_RESOLVED}" >&2
    exit 1
fi

if [ "${SRC_PM_REAL}" = "${PM_REAL}" ]; then
    echo "⚠️  Source and destination for performance-monitor.yml are the same; skipping copy."
else
    if [ -f "${PM_TARGET_RESOLVED}" ]; then
        BACKUP="${PM_TARGET_RESOLVED}.bak.$(date +%s)"
        echo "🔁 Backing up existing ${PM_TARGET_RESOLVED} -> ${BACKUP}"
        cp -- "${PM_TARGET_RESOLVED}" "${BACKUP}"
    fi
    echo "✅ Updating ${PM_TARGET_RESOLVED}..."
    # Remove any existing file before copy to prevent symlink following
    rm -f -- "${PM_TARGET_RESOLVED}"
    cp -- "${SRC_PM}" "${PM_TARGET_RESOLVED}"
fi

echo "Done. Next steps: pip install pip-tools; pip-compile requirements/ml.in -o requirements/ml.txt; commit changes."
