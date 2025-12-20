#!/bin/bash
# External SSD Backup Script for Samsung T9
# Creates timestamped backups of the entire Transformation Portal repository
# Usage: ./backup_to_external_ssd.sh [destination_path]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default destination (can be overridden)
DEFAULT_DEST="/Volumes/Samsung_T9/Backups/Transformation_Portal"

# Use provided destination or default
DEST="${1:-$DEFAULT_DEST}"

# Timestamp for backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="transformation_portal_${TIMESTAMP}"
BACKUP_PATH="${DEST}/${BACKUP_NAME}"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Transformation Portal External Backup${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check if destination volume is mounted
VOLUME_NAME=$(echo "${DEST}" | cut -d'/' -f3)
if [ ! -d "/Volumes/${VOLUME_NAME}" ]; then
    echo -e "${RED}Error: Volume /Volumes/${VOLUME_NAME} not mounted${NC}"
    echo -e "${YELLOW}Please connect Samsung T9 SSD and try again${NC}"
    exit 1
fi

# Check available space
REPO_SIZE=$(du -sk . | cut -f1)
AVAILABLE_SPACE=$(df -k "${DEST}" | tail -1 | awk '{print $4}')

echo -e "${BLUE}Repository size:${NC} $(numfmt --to=iec-i --suffix=B $((REPO_SIZE * 1024)))"
echo -e "${BLUE}Available space:${NC} $(numfmt --to=iec-i --suffix=B $((AVAILABLE_SPACE * 1024)))"

if [ "$AVAILABLE_SPACE" -lt "$REPO_SIZE" ]; then
    echo -e "${RED}Error: Not enough space on destination${NC}"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "${DEST}"

echo ""
echo -e "${YELLOW}Creating backup: ${BACKUP_NAME}${NC}"
echo ""

# Create backup using rsync
rsync -avh \
    --progress \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.hypothesis' \
    --exclude='node_modules' \
    --exclude='.DS_Store' \
    --exclude='*.egg-info' \
    --exclude='outputs/' \
    --exclude='logs/*.log' \
    --exclude='.local_backup' \
    . "${BACKUP_PATH}"

# Create metadata file
cat > "${BACKUP_PATH}/BACKUP_METADATA.txt" <<EOF
Backup Created: $(date)
Source: $(pwd)
Hostname: $(hostname)
User: $(whoami)
Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Repository Size: $(numfmt --to=iec-i --suffix=B $((REPO_SIZE * 1024)))
Python Version: $(python3 --version)
EOF

echo ""
echo -e "${GREEN}✅ Backup completed successfully!${NC}"
echo ""
echo -e "${BLUE}Backup location:${NC} ${BACKUP_PATH}"
echo -e "${BLUE}Backup size:${NC} $(du -sh "${BACKUP_PATH}" | cut -f1)"

# List recent backups
echo ""
echo -e "${BLUE}Recent backups:${NC}"
ls -lt "${DEST}" | grep "transformation_portal_" | head -5

# Cleanup old backups (keep last 5)
BACKUP_COUNT=$(ls -1 "${DEST}" | grep "transformation_portal_" | wc -l)
if [ "$BACKUP_COUNT" -gt 5 ]; then
    echo ""
    echo -e "${YELLOW}Cleaning up old backups (keeping last 5)...${NC}"
    ls -1t "${DEST}" | grep "transformation_portal_" | tail -n +6 | while read OLD_BACKUP; do
        echo "  Removing: ${OLD_BACKUP}"
        rm -rf "${DEST}/${OLD_BACKUP}"
    done
fi

echo ""
echo -e "${GREEN}Done!${NC}"
