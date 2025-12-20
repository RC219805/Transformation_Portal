#!/bin/bash
# Create expanded validation dataset
# Copies selected images from input_images to validation_expanded directory

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VALIDATION_DIR="$REPO_ROOT/data/validation_expanded"
IMAGE_LIST="$VALIDATION_DIR/images.txt"

echo -e "${BLUE}=== Validation Dataset Creation ===${NC}"
echo "Repository: $REPO_ROOT"
echo "Validation dir: $VALIDATION_DIR"
echo ""

# Verify image list exists
if [[ ! -f "$IMAGE_LIST" ]]; then
    echo -e "${YELLOW}Error: Image list not found at $IMAGE_LIST${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$VALIDATION_DIR"
echo -e "${GREEN}✓${NC} Created validation directory"

# Count images (excluding comments and blank lines)
TOTAL_IMAGES=$(grep -v '^#' "$IMAGE_LIST" | grep -v '^$' | wc -l | tr -d ' ')
echo -e "${BLUE}Found $TOTAL_IMAGES images in selection list${NC}"
echo ""

# Copy images
COPIED=0
FAILED=0

while IFS= read -r image_path; do
    # Skip comments and blank lines
    if [[ "$image_path" =~ ^#.*$ ]] || [[ -z "$image_path" ]]; then
        continue
    fi
    
    # Check if source file exists
    if [[ ! -f "$image_path" ]]; then
        echo -e "${YELLOW}⚠ Missing: $(basename "$image_path")${NC}"
        ((FAILED++))
        continue
    fi
    
    # Copy with metadata preservation (-p flag)
    basename_file="$(basename "$image_path")"
    cp -p "$image_path" "$VALIDATION_DIR/"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓${NC} Copied: $basename_file"
        ((COPIED++))
    else
        echo -e "${YELLOW}✗ Failed: $basename_file${NC}"
        ((FAILED++))
    fi
done < "$IMAGE_LIST"

echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo "Total images: $TOTAL_IMAGES"
echo -e "Copied: ${GREEN}$COPIED${NC}"
if [[ $FAILED -gt 0 ]]; then
    echo -e "Failed: ${YELLOW}$FAILED${NC}"
fi

# Verify final count
ACTUAL_COUNT=$(find "$VALIDATION_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l | tr -d ' ')
echo -e "Files in validation dir: ${BLUE}$ACTUAL_COUNT${NC}"

if [[ $ACTUAL_COUNT -eq $COPIED ]]; then
    echo -e "${GREEN}✓ Dataset creation successful${NC}"
else
    echo -e "${YELLOW}⚠ Mismatch: Expected $COPIED, found $ACTUAL_COUNT${NC}"
fi

# Show size statistics
echo ""
echo -e "${BLUE}=== Size Statistics ===${NC}"
TOTAL_SIZE=$(du -sh "$VALIDATION_DIR" | cut -f1)
echo "Total size: $TOTAL_SIZE"

# Count by size category
echo ""
python3 << 'PYEOF'
import sys
from PIL import Image
from pathlib import Path

validation_dir = Path(sys.argv[1])
images = list(validation_dir.glob("*.jpg")) + list(validation_dir.glob("*.png"))

small, medium, large = 0, 0, 0
landscape, portrait, pano = 0, 0, 0

for img_path in images:
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            shortest = min(width, height)
            aspect = width / height
            
            # Size categories
            if shortest < 2000:
                small += 1
            elif shortest < 4000:
                medium += 1
            else:
                large += 1
            
            # Aspect categories
            if aspect > 2.0:
                pano += 1
            elif aspect > 1.3:
                landscape += 1
            elif aspect < 0.77:
                portrait += 1
    except Exception:
        pass

print(f"Size distribution:")
print(f"  Small (<2000px): {small}")
print(f"  Medium (2000-4000px): {medium}")
print(f"  Large (>4000px): {large}")
print(f"\nAspect ratio distribution:")
print(f"  Landscape: {landscape}")
print(f"  Portrait: {portrait}")
print(f"  Panorama: {pano}")
PYEOF

echo ""
echo -e "${GREEN}Dataset ready for validation!${NC}"
echo "Run: python scripts/automation/production_depth_validation_fixed.py --image-dir $VALIDATION_DIR"
