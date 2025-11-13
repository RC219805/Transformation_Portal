#!/usr/bin/env bash
#
# pre-commit-check.sh
# Pre-commit hook to prevent committing misplaced files
#
# This hook checks for files in the repository root that should be
# organized into subdirectories according to REPO_ORGANIZATION.md
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Files that are allowed in root
ALLOWED_ROOT_FILES=(
    # Core documentation
    "README.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
    "REPO_ORGANIZATION.md"
    
    # Build configuration
    "Makefile"
    "pyproject.toml"
    "setup.py"
    "setup.cfg"
    
    # Dependency management
    "requirements.txt"
    "requirements-dev.txt"
    "requirements-ci.txt"
    "requirements-test.txt"
    "Pipfile"
    "Pipfile.lock"
    "poetry.lock"
    
    # Testing configuration
    "pytest.ini"
    "tox.ini"
    ".coveragerc"
    
    # Linting configuration
    ".pylintrc"
    ".flake8"
    "mypy.ini"
    
    # Docker
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.yaml"
    
    # Git
    ".gitignore"
    ".gitattributes"
    ".gitmodules"
    
    # Organization system
    ".auto-organize.sh"
    
    # Package metadata
    "PKG-INFO"
    "MANIFEST.in"
    
    # Python package
    "__init__.py"
)

# Patterns that are allowed in root (regex)
ALLOWED_ROOT_PATTERNS=(
    "^requirements.*\.txt$"
    "^\.git.*$"
    "^\..*rc$"
)

# Check if a file is allowed in root
is_allowed_in_root() {
    local file="$1"
    local basename=$(basename "$file")
    
    # Check exact matches
    for allowed in "${ALLOWED_ROOT_FILES[@]}"; do
        if [[ "$basename" == "$allowed" ]]; then
            return 0
        fi
    done
    
    # Check patterns
    for pattern in "${ALLOWED_ROOT_PATTERNS[@]}"; do
        if [[ "$basename" =~ $pattern ]]; then
            return 0
        fi
    done
    
    return 1
}

# Suggest destination for misplaced file
suggest_destination() {
    local file="$1"
    local basename=$(basename "$file")
    local ext="${basename##*.}"
    
    # Documentation
    if [[ "$basename" =~ \.md$ ]]; then
        if [[ "$basename" =~ (PLAN|STRATEGY|OPTIMIZATION|SUMMARY) ]]; then
            echo "docs/guides/"
        elif [[ "$basename" =~ (ARCHITECTURE|DESIGN) ]]; then
            echo "docs/architecture/"
        elif [[ "$basename" =~ (API|REFERENCE) ]]; then
            echo "docs/api/"
        elif [[ "$basename" =~ (DEPLOY|PRODUCTION) ]]; then
            echo "docs/deployment/"
        else
            echo "docs/guides/"
        fi
        return
    fi
    
    # Scripts
    if [[ "$basename" =~ \.(sh|py)$ ]] && [[ ! "$basename" =~ ^test_ ]]; then
        if [[ "$basename" =~ (install|setup|download) ]]; then
            echo "scripts/setup/"
        elif [[ "$basename" =~ (verify|navigate|util) ]]; then
            echo "scripts/utilities/"
        else
            echo "scripts/automation/"
        fi
        return
    fi
    
    # Data files
    if [[ "$ext" == "json" || "$ext" == "csv" || "$ext" == "txt" ]]; then
        echo "data/"
        return
    fi
    
    # Images
    if [[ "$ext" =~ ^(jpg|jpeg|png|gif|tiff|tif)$ ]]; then
        if [[ "$basename" =~ (debug|test) ]]; then
            echo "archive/"
        else
            echo "data/sample_images/"
        fi
        return
    fi
    
    # Code files
    if [[ "$ext" == "ts" || "$ext" == "js" ]]; then
        echo "archive/"
        return
    fi
    
    # Default
    echo "appropriate subdirectory"
}

# Main pre-commit check
main() {
    local exit_code=0
    local misplaced_files=()
    
    # Get list of staged files in root directory
    while IFS= read -r file; do
        # Skip if not in root directory
        if [[ "$file" == */* ]]; then
            continue
        fi
        
        # Skip if directory
        if [[ -d "$file" ]]; then
            continue
        fi
        
        # Check if file is allowed in root
        if ! is_allowed_in_root "$file"; then
            misplaced_files+=("$file")
        fi
    done < <(git diff --cached --name-only --diff-filter=ACM)
    
    # If there are misplaced files, show error and suggest fix
    if [[ ${#misplaced_files[@]} -gt 0 ]]; then
        echo -e "${RED}✗ Pre-commit check failed${NC}"
        echo ""
        echo "The following files should not be in the repository root:"
        echo ""
        
        for file in "${misplaced_files[@]}"; do
            local suggestion=$(suggest_destination "$file")
            echo -e "  ${YELLOW}$file${NC}"
            echo -e "    → Suggested: ${GREEN}$suggestion${NC}"
        done
        
        echo ""
        echo "Please move these files to appropriate directories and try again:"
        echo ""
        echo "  1. Run the organization script:"
        echo "     ./.auto-organize.sh --dry-run  # Preview changes"
        echo "     ./.auto-organize.sh            # Apply changes"
        echo ""
        echo "  2. Or move files manually:"
        for file in "${misplaced_files[@]}"; do
            local suggestion=$(suggest_destination "$file")
            if [[ "$suggestion" != "appropriate subdirectory" ]]; then
                echo "     mv $file $suggestion"
            fi
        done
        echo ""
        echo "  3. Then stage and commit again:"
        echo "     git add ."
        echo "     git commit"
        echo ""
        echo "To bypass this check (not recommended):"
        echo "  git commit --no-verify"
        echo ""
        echo "For more information, see: REPO_ORGANIZATION.md"
        echo ""
        
        exit_code=1
    fi
    
    exit $exit_code
}

main "$@"
