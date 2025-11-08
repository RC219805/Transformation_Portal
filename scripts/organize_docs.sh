#!/usr/bin/env bash
# Organize documentation files for Transformation Portal
# Moves excessive root markdown files to appropriate docs/ subdirectories
#
# Usage:
#   ./scripts/organize_docs.sh [--dry-run] [--auto]
#
# Options:
#   --dry-run    Show what would be moved without making changes
#   --auto       Move files automatically without prompts

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
DRY_RUN=0
AUTO_MODE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --auto)
            AUTO_MODE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--auto]"
            exit 1
            ;;
    esac
done

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Documentation Organization Tool          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"

if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[DRY-RUN MODE] No changes will be made${NC}\n"
fi

# Files to keep in root (always)
KEEP_IN_ROOT=(
    "README.md"
    "LICENSE"
    "LICENSE.md"
    "CONTRIBUTING.md"
    "CODE_OF_CONDUCT.md"
    "SECURITY.md"
    "CHANGELOG.md"
    "START_HERE.md"
)

# Create docs subdirectories if they don't exist
DOCS_DIRS=(
    "docs/migration"
    "docs/deprecation"
    "docs/guides"
    "docs/reference"
    "docs/archive"
)

for dir in "${DOCS_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        if [ $DRY_RUN -eq 1 ]; then
            echo -e "${YELLOW}[DRY-RUN] Would create: $dir${NC}"
        else
            mkdir -p "$dir"
            echo -e "${GREEN}✓ Created: $dir${NC}"
        fi
    fi
done

# Get all markdown files in root
MD_FILES=$(find . -maxdepth 1 -name "*.md" -type f)
MD_COUNT=$(echo "$MD_FILES" | wc -l | tr -d ' ')

echo -e "\n${BLUE}Found $MD_COUNT markdown files in root${NC}"

# Categorization function
categorize_file() {
    local file="$1"
    local filename=$(basename "$file")

    # Check if should stay in root
    for keep in "${KEEP_IN_ROOT[@]}"; do
        if [ "$filename" = "$keep" ]; then
            echo "root"
            return
        fi
    done

    # Categorize by content/name
    case "$filename" in
        *MIGRATION* | *MIGRAT*)
            echo "docs/migration"
            ;;
        *DEPRECAT* | *DEPRICATION*)
            echo "docs/deprecation"
            ;;
        *GUIDE* | *TUTORIAL* | *HOWTO* | *HOW_TO*)
            echo "docs/guides"
            ;;
        *WORKFLOW* | *BUGS* | *FIXED* | *REFACTOR*)
            echo "docs/reference"
            ;;
        *HISTORY* | *ARCHIVE* | *OLD* | *BACKUP*)
            echo "docs/archive"
            ;;
        *)
            echo "docs/reference"
            ;;
    esac
}


# Show categorization plan
echo -e "\n${YELLOW}Categorization Plan:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

KEEP_COUNT=0
MOVE_COUNT=0

# Process each file
for file in $MD_FILES; do
    category=$(categorize_file "$file")
    filename=$(basename "$file")

    if [ "$category" = "root" ]; then
        echo -e "${GREEN}  ✓ Keep in root: $filename${NC}"
        ((KEEP_COUNT++))
    else
        echo -e "${YELLOW}  → Move to $category: $filename${NC}"
        ((MOVE_COUNT++))
    fi
done

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Keep in root: $KEEP_COUNT${NC}"
echo -e "${YELLOW}Move to docs/: $MOVE_COUNT${NC}"

FINAL_COUNT=$((KEEP_COUNT))
MAX_MD=10

if [ $FINAL_COUNT -le $MAX_MD ]; then
    echo -e "${GREEN}✓ Final count ($FINAL_COUNT) will be within limit ($MAX_MD)${NC}"
else
    echo -e "${RED}✗ Final count ($FINAL_COUNT) still exceeds limit ($MAX_MD)${NC}"
    echo -e "${YELLOW}💡 Manual review recommended${NC}"
fi

# Confirm or execute
if [ $MOVE_COUNT -eq 0 ]; then
    echo -e "\n${GREEN}✓ No files need to be moved!${NC}"
    exit 0
fi

if [ $DRY_RUN -eq 0 ] && [ $AUTO_MODE -eq 0 ]; then
    echo -e "\n${YELLOW}Proceed with moving $MOVE_COUNT files? [y/N]${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cancelled${NC}"
        exit 0
    fi
fi

# Move files
echo -e "\n${BLUE}Moving files...${NC}"
MOVED_COUNT=0

for file in $MD_FILES; do
    category=$(categorize_file "$file")
    filename=$(basename "$file")

    if [ "$category" != "root" ]; then
        dest="$category/$filename"

        if [ $DRY_RUN -eq 1 ]; then
            echo -e "${YELLOW}[DRY-RUN] Would move: $filename → $dest${NC}"
            ((MOVED_COUNT++))
        else
            # Check if file already exists at destination
            if [ -f "$dest" ]; then
                echo -e "${YELLOW}⚠ Destination exists: $dest (skipping)${NC}"
                continue
            fi

            git mv "$file" "$dest" 2>/dev/null || mv "$file" "$dest"
            echo -e "${GREEN}✓ Moved: $filename → $dest${NC}"
            ((MOVED_COUNT++))
        fi
    fi
done

# Create index file in docs/
if [ $MOVED_COUNT -gt 0 ] && [ $DRY_RUN -eq 0 ]; then
    INDEX_FILE="docs/DOCUMENTATION_INDEX.md"

    echo -e "\n${BLUE}Creating documentation index...${NC}"

    cat > "$INDEX_FILE" << 'EOF'
# Documentation Index

This directory contains organized documentation for the Transformation Portal.

## Directory Structure

### `/migration`
Migration guides and version upgrade documentation.

### `/deprecation`
Deprecation notices and legacy system documentation.

### `/guides`
User guides, tutorials, and how-to documentation.

### `/reference`
Technical reference documentation, workflow guides, and bug fix logs.

### `/archive`
Historical documentation and archived materials.

## Root Documentation

Essential documentation remains in the repository root:
- `README.md` - Project overview and quick start
- `START_HERE.md` - Onboarding guide
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - License information

## Recently Moved Files

EOF

    # Add moved files to index
    for file in $MD_FILES; do
        category=$(categorize_file "$file")
        filename=$(basename "$file")

        if [ "$category" != "root" ]; then
            echo "- [$filename]($category/$filename)" >> "$INDEX_FILE"
        fi
    done

    echo -e "${GREEN}✓ Created: $INDEX_FILE${NC}"
fi

# Summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $DRY_RUN -eq 1 ]; then
    echo -e "${YELLOW}[DRY-RUN] Would move $MOVE_COUNT files${NC}"
else
    echo -e "${GREEN}✓ Moved $MOVED_COUNT files to docs/ subdirectories${NC}"
    echo -e "${BLUE}Final markdown count in root: $FINAL_COUNT${NC}"

    if [ $MOVED_COUNT -gt 0 ]; then
        echo -e "\n${YELLOW}💡 Next steps:${NC}"
        echo -e "  1. Review changes: git status"
        echo -e "  2. Update any broken links in documentation"
        echo -e "  3. Commit changes: git commit -m 'docs: organize markdown files'"
    fi
fi
