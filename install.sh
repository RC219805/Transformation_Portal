#!/usr/bin/env bash
#
# install.sh
# Transformation Portal Installation Script
#
# This script sets up the development environment for the Transformation Portal,
# including virtual environment creation, dependencies, and directory structure
# for the RAG system and processing pipelines.
#
# Usage:
#   ./install.sh [--help]
#
# Requirements:
#   - Python 3.8 or higher
#   - Git (for version control)
#
# See docs/guides/INSTALLATION.md for detailed documentation.

set -euo pipefail

# =============================================================================
# Constants and Colors
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

print_header() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║       Transformation Portal Installation Script           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_help() {
    echo "Usage: ./install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help      Show this help message and exit"
    echo ""
    echo "This script sets up the Transformation Portal development environment."
    echo "See docs/guides/INSTALLATION.md for detailed documentation."
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_python() {
    log_info "Checking Python installation..."

    # Try python3 first, then python
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD=python
    else
        log_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi

    # Check Python version (must be 3.8+)
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
        log_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python $PYTHON_VERSION found"
}

# =============================================================================
# Virtual Environment Setup
# =============================================================================

setup_venv() {
    log_info "Setting up virtual environment..."

    cd "$SCRIPT_DIR"

    if [ -d "venv" ]; then
        log_info "Virtual environment already exists at ./venv"
    else
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        log_success "Virtual environment created at ./venv"
    fi

    # Activate virtual environment
    # shellcheck source=/dev/null
    source venv/bin/activate
    log_success "Virtual environment activated"
}

# =============================================================================
# Dependencies Installation
# =============================================================================

install_dependencies() {
    log_info "Installing dependencies..."

    # Upgrade pip first
    log_info "Upgrading pip..."
    if ! pip install --upgrade pip --quiet; then
        log_warning "pip upgrade failed, continuing with current version"
    fi

    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install -r requirements.txt
        log_success "Core dependencies installed"
    else
        log_warning "requirements.txt not found, skipping core dependencies"
    fi

    # Install from requirements-dev.txt if it exists
    if [ -f "requirements-dev.txt" ]; then
        log_info "Installing development dependencies from requirements-dev.txt..."
        pip install -r requirements-dev.txt
        log_success "Development dependencies installed"
    else
        log_info "requirements-dev.txt not found, skipping development dependencies"
    fi
}

# =============================================================================
# Directory Structure Creation
# =============================================================================

create_directories() {
    log_info "Creating required directory structure..."

    # Define the required directories for RAG system and processing pipelines
    local directories=(
        "data/knowledge_base/memory_snapshots"
        "data/feedback_loops/audits"
        "assets/luts/imported"
        "assets/models"
        "src/transformation_portal/pipelines"
        "scripts/production"
        "scripts/utilities"
        "docs/guides"
        "archive/debug_artifacts"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "  Created: $dir"
        else
            log_info "  Exists:  $dir"
        fi
    done

    log_success "Directory structure ready"
}

# =============================================================================
# Set Permissions
# =============================================================================

set_permissions() {
    log_info "Setting executable permissions..."

    # Make install.sh executable
    chmod +x "$SCRIPT_DIR/install.sh" 2>/dev/null || true

    # Make all .sh files in scripts/ executable
    if [ -d "$SCRIPT_DIR/scripts" ]; then
        local script_count
        script_count=$(find "$SCRIPT_DIR/scripts" -name "*.sh" -type f | wc -l)
        if [ "$script_count" -gt 0 ]; then
            find "$SCRIPT_DIR/scripts" -name "*.sh" -type f -exec chmod +x {} \;
            log_success "Shell scripts in scripts/ are now executable ($script_count scripts)"
        else
            log_info "No shell scripts found in scripts/"
        fi
    else
        log_info "scripts/ directory not found, skipping"
    fi
}

# =============================================================================
# Environment File Setup
# =============================================================================

setup_env_file() {
    log_info "Checking environment configuration..."

    cd "$SCRIPT_DIR"

    # Create .env.example if it doesn't exist
    if [ ! -f ".env.example" ]; then
        log_info "Creating .env.example template..."
        cat > .env.example << 'EOF'
# Transformation Portal Environment Configuration
# ================================================
# Copy this file to .env and update with your values:
#   cp .env.example .env
#
# Note: .env is git-ignored and should not be committed.

# Environment mode: development, staging, production
TRANSFORMATION_ENV=development

# RAG System Configuration
RAG_MEMORY_PATH=data/knowledge_base

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
EOF
        log_success "Created .env.example"
    fi

    # Copy .env.example to .env if .env doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_success "Created .env from .env.example"
        log_warning "Please review and update .env with your configuration"
    else
        log_info ".env already exists, skipping"
    fi
}

# =============================================================================
# Main Installation Flow
# =============================================================================

main() {
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $arg"
                print_help
                exit 1
                ;;
        esac
    done

    print_header

    log_info "Starting installation..."
    echo ""

    # Step 1: Check prerequisites
    check_python

    # Step 2: Setup virtual environment
    setup_venv

    # Step 3: Install dependencies
    install_dependencies

    # Step 4: Create directory structure
    create_directories

    # Step 5: Set executable permissions
    set_permissions

    # Step 6: Setup environment file
    setup_env_file

    # Print success message
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Installation Complete!                          ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Activate the virtual environment:"
    echo "     ${YELLOW}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Review and update your environment configuration:"
    echo "     ${YELLOW}nano .env${NC}"
    echo ""
    echo "  3. Verify installation by running tests:"
    echo "     ${YELLOW}make test-fast${NC}"
    echo ""
    echo "For detailed documentation, see: docs/guides/INSTALLATION.md"
    echo ""
}

main "$@"
