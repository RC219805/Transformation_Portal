#!/bin/bash
# ==========================================================
# Move Transformation Portal Outputs to Samsung T9 SSD
# Preserves disk space by moving large outputs to external drive
# ==========================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Move Outputs to Samsung T9 External SSD${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Configuration
REPO_ROOT="/Users/rc/Transformation_Portal"
EXTERNAL_SSD="/Volumes/T9"
EXTERNAL_BASE="${EXTERNAL_SSD}/Transformation_Portal_Sweeps"

# Directories to move
DIRS_TO_MOVE=(
    "sweep_runs"
    "sweep_runs_kitchen_only"
)

# Check if external SSD is mounted
if [ ! -d "$EXTERNAL_SSD" ]; then
    echo -e "${RED}✗ Samsung T9 not found at $EXTERNAL_SSD${NC}"
    echo -e "${YELLOW}Please mount the external SSD and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Samsung T9 found: $EXTERNAL_SSD${NC}"
echo ""

# Check disk space
echo "Disk Space Status:"
echo "─────────────────────────────────────────────────────────"
df -h "$REPO_ROOT" | grep -v Filesystem | awk '{print "  Main disk:   " $4 " available (" $5 " used)"}'
df -h "$EXTERNAL_SSD" | grep -v Filesystem | awk '{print "  Samsung T9:  " $4 " available (" $5 " used)"}'
echo ""

# Calculate total size to move
TOTAL_SIZE=0
echo "Directories to move:"
echo "─────────────────────────────────────────────────────────"
for dir in "${DIRS_TO_MOVE[@]}"; do
    SOURCE_DIR="$REPO_ROOT/$dir"
    if [ -d "$SOURCE_DIR" ]; then
        SIZE=$(du -sh "$SOURCE_DIR" 2>/dev/null | awk '{print $1}')
        echo -e "  ${GREEN}✓${NC} $dir ($SIZE)"
    else
        echo -e "  ${YELLOW}−${NC} $dir (not found, will skip)"
    fi
done
echo ""

# Confirmation
read -p "Continue with migration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Migration cancelled."
    exit 0
fi

# Create external base directory
mkdir -p "$EXTERNAL_BASE"
echo -e "${GREEN}✓${NC} Created external directory: $EXTERNAL_BASE"
echo ""

# Move each directory
for dir in "${DIRS_TO_MOVE[@]}"; do
    SOURCE_DIR="$REPO_ROOT/$dir"
    DEST_DIR="$EXTERNAL_BASE/$dir"
    SYMLINK="$SOURCE_DIR"
    
    if [ ! -d "$SOURCE_DIR" ]; then
        echo -e "${YELLOW}Skipping $dir (not found)${NC}"
        continue
    fi
    
    echo -e "${BLUE}Moving: $dir${NC}"
    
    # Check if it's already a symlink
    if [ -L "$SOURCE_DIR" ]; then
        echo -e "${YELLOW}  Already a symlink, skipping${NC}"
        continue
    fi
    
    # Move directory
    echo "  → Copying to external SSD..."
    rsync -ah --progress "$SOURCE_DIR/" "$DEST_DIR/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Copy complete${NC}"
        
        # Verify copy
        SOURCE_SIZE=$(du -sk "$SOURCE_DIR" | awk '{print $1}')
        DEST_SIZE=$(du -sk "$DEST_DIR" | awk '{print $1}')
        SIZE_DIFF=$((SOURCE_SIZE - DEST_SIZE))
        SIZE_DIFF_ABS=${SIZE_DIFF#-}  # absolute value
        
        if [ $SIZE_DIFF_ABS -lt 1024 ]; then  # Allow 1MB difference
            echo -e "${GREEN}  ✓ Verification passed${NC}"
            
            # Remove original
            echo "  → Removing original..."
            rm -rf "$SOURCE_DIR"
            
            # Create symlink
            echo "  → Creating symlink..."
            ln -s "$DEST_DIR" "$SOURCE_DIR"
            
            echo -e "${GREEN}  ✓ Migration complete: $dir${NC}"
        else
            echo -e "${RED}  ✗ Size mismatch! Source: ${SOURCE_SIZE}KB, Dest: ${DEST_SIZE}KB${NC}"
            echo -e "${YELLOW}  Keeping both copies for safety. Please verify manually.${NC}"
        fi
    else
        echo -e "${RED}  ✗ Copy failed for $dir${NC}"
    fi
    
    echo ""
done

# Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Migration Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "New locations:"
for dir in "${DIRS_TO_MOVE[@]}"; do
    if [ -L "$REPO_ROOT/$dir" ]; then
        TARGET=$(readlink "$REPO_ROOT/$dir")
        echo "  $dir → $TARGET"
    fi
done
echo ""
echo "Disk space freed:"
df -h "$REPO_ROOT" | grep -v Filesystem | awk '{print "  Main disk: " $4 " available (" $5 " used)"}'
echo ""
echo -e "${YELLOW}Note: Keep the Samsung T9 connected when running sweeps!${NC}"
echo ""
