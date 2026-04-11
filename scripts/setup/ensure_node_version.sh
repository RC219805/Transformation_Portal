#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# ensure_node_version.sh
#
# Node.js version enforcement wrapper for frontdoor validation.
# Auto-detects nvm/fnm/volta and provides actionable guidance.
#
# Exit codes:
#   0 - Node version is correct
#   1 - Node version is incorrect (with guidance)
# -----------------------------------------------------------------------------

set -euo pipefail

REQUIRED_MAJOR=22
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRONTDOOR_ROOT="${REPO_ROOT}/web/secure-landing"

# Colors for output (if terminal supports it)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

log_error() {
    echo -e "${RED}✗ $1${NC}" >&2
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_info() {
    echo "  $1"
}

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    log_error "Node.js is not installed or not in PATH"
    echo ""
    log_info "Install Node.js ${REQUIRED_MAJOR}.x using one of these methods:"
    echo ""

    # Check for version managers
    if command -v nvm &> /dev/null || [[ -d "$HOME/.nvm" ]]; then
        log_info "  nvm detected — run:"
        log_info "    nvm install ${REQUIRED_MAJOR}"
        log_info "    nvm use ${REQUIRED_MAJOR}"
    elif command -v fnm &> /dev/null; then
        log_info "  fnm detected — run:"
        log_info "    fnm install ${REQUIRED_MAJOR}"
        log_info "    fnm use ${REQUIRED_MAJOR}"
    elif command -v volta &> /dev/null; then
        log_info "  volta detected — run:"
        log_info "    volta install node@${REQUIRED_MAJOR}"
    else
        log_info "  Using a version manager (recommended):"
        log_info "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
        log_info "    nvm install ${REQUIRED_MAJOR}"
        echo ""
        log_info "  Or install directly from nodejs.org"
    fi
    exit 1
fi

# Get current Node version
NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
NODE_MAJOR=$(echo "$NODE_VERSION" | sed -E 's/^v?([0-9]+).*/\1/')

if [[ "$NODE_MAJOR" == "unknown" || -z "$NODE_MAJOR" ]]; then
    log_error "Could not determine Node.js version"
    log_info "  node --version returned: $NODE_VERSION"
    exit 1
fi

# Check .nvmrc if present
NVMRC_VERSION=""
if [[ -f "${FRONTDOOR_ROOT}/.nvmrc" ]]; then
    NVMRC_VERSION=$(cat "${FRONTDOOR_ROOT}/.nvmrc" | tr -d '[:space:]')
fi

# Validate version
if [[ "$NODE_MAJOR" -eq "$REQUIRED_MAJOR" ]]; then
    log_success "Node.js ${NODE_VERSION} meets requirement (${REQUIRED_MAJOR}.x)"

    # Show .nvmrc info if present
    if [[ -n "$NVMRC_VERSION" ]]; then
        log_info "  .nvmrc specifies: ${NVMRC_VERSION}"
    fi
    exit 0
else
    log_error "Node.js ${NODE_VERSION} does not match required ${REQUIRED_MAJOR}.x"
    echo ""

    # Provide specific guidance based on version manager
    if [[ -n "${NVM_DIR:-}" ]] || [[ -d "$HOME/.nvm" ]]; then
        log_info "nvm detected — switch version:"
        log_info "  nvm install ${REQUIRED_MAJOR}"
        log_info "  nvm use ${REQUIRED_MAJOR}"

        # Check if the version is already installed
        if [[ -d "$HOME/.nvm/versions/node" ]]; then
            INSTALLED=$(ls "$HOME/.nvm/versions/node" 2>/dev/null | grep "^v${REQUIRED_MAJOR}" | head -1 || true)
            if [[ -n "$INSTALLED" ]]; then
                log_info ""
                log_info "  Version ${INSTALLED} is already installed:"
                log_info "    nvm use ${REQUIRED_MAJOR}"
            fi
        fi
    elif command -v fnm &> /dev/null; then
        log_info "fnm detected — switch version:"
        log_info "  fnm install ${REQUIRED_MAJOR}"
        log_info "  fnm use ${REQUIRED_MAJOR}"
    elif command -v volta &> /dev/null; then
        log_info "volta detected — switch version:"
        log_info "  volta install node@${REQUIRED_MAJOR}"
    else
        log_info "Switch to Node.js ${REQUIRED_MAJOR}.x:"
        log_info "  Install a version manager like nvm, fnm, or volta"
        log_info "  Or download from https://nodejs.org/"
    fi

    echo ""
    log_warning "The secure-landing frontdoor requires Node ${REQUIRED_MAJOR}.x"
    log_info "  Native dependencies (better-sqlite3, argon2) are ABI-sensitive"
    log_info "  See: web/secure-landing/package.json engines constraint"

    exit 1
fi
