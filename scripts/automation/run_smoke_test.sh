#!/bin/bash
#
# Smoke Test: 2-image validation with V2 classifier integration
# This must pass before running the full 18-image validation
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Smoke Test: V2 Classifier Integration"
echo "========================================="

# Create smoke test directory
SMOKE_DIR="data/validation_smoke"
mkdir -p "$SMOKE_DIR"

# Copy 2 test images (if validation_expanded exists)
if [ -d "data/validation_expanded" ]; then
    echo "Copying test images from validation_expanded..."

    # Find first 2 images
    COUNT=0
    for img in data/validation_expanded/*.{jpg,jpeg,png,tif,tiff} 2>/dev/null; do
        if [ -f "$img" ]; then
            cp "$img" "$SMOKE_DIR/"
            echo "  Copied: $(basename $img)"
            COUNT=$((COUNT + 1))
            if [ $COUNT -ge 2 ]; then
                break
            fi
        fi
    done

    if [ $COUNT -lt 2 ]; then
        echo "⚠️  Warning: Only found $COUNT image(s) in validation_expanded"
        echo "Creating synthetic test images..."

        # Create synthetic images using Python
        python3 -c "
import numpy as np
from PIL import Image
from pathlib import Path

for i in range(2):
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(img).save(Path('$SMOKE_DIR') / f'synthetic_{i}.jpg')
    print(f'  Created: synthetic_{i}.jpg')
"
    fi
else
    echo "Creating synthetic test images..."
    python3 -c "
import numpy as np
from PIL import Image
from pathlib import Path

for i in range(2):
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(img).save(Path('$SMOKE_DIR') / f'synthetic_{i}.jpg')
    print(f'  Created: synthetic_{i}.jpg')
"
fi

# Count images
IMAGE_COUNT=$(find "$SMOKE_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tif" -o -name "*.tiff" \) | wc -l)
echo ""
echo "Test images ready: $IMAGE_COUNT"

if [ $IMAGE_COUNT -lt 1 ]; then
    echo "❌ FAILED: No test images available"
    exit 1
fi

# Create output directory
OUTPUT_DIR="outputs/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running validation..."
echo "  Input: $SMOKE_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run validator
python scripts/automation/production_depth_validation_fixed.py \
  --input-dir "$SMOKE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --tile-size 512 \
  --overlap 64

echo ""
echo "========================================="
echo "Smoke Test Results"
echo "========================================="

# Check metrics files
METRICS_COUNT=$(find "$OUTPUT_DIR" -name "*_metrics.json" | wc -l)
echo "Metrics files generated: $METRICS_COUNT"

if [ $METRICS_COUNT -eq 0 ]; then
    echo "❌ FAILED: No metrics files generated"
    exit 1
fi

# Validate metrics content
echo ""
echo "Validating metrics content..."

ALL_PASS=true

for metrics_file in "$OUTPUT_DIR"/*_metrics.json; do
    echo ""
    echo "File: $(basename $metrics_file)"

    # Extract key fields
    scene_type=$(python3 -c "import json; print(json.load(open('$metrics_file')).get('scene_type', 'NULL'))")
    edge_f1=$(python3 -c "import json; print(json.load(open('$metrics_file')).get('edge_f1', 'NULL'))")
    lenient_pass=$(python3 -c "import json; print(json.load(open('$metrics_file')).get('lenient_pass', 'NULL'))")

    echo "  scene_type: $scene_type"
    echo "  edge_f1: $edge_f1"
    echo "  lenient_pass: $lenient_pass"

    # Check for NULL values
    if [ "$scene_type" == "NULL" ] || [ "$scene_type" == "None" ]; then
        echo "  ❌ FAIL: scene_type is NULL"
        ALL_PASS=false
    fi

    if [ "$edge_f1" == "NULL" ] || [ "$edge_f1" == "None" ]; then
        echo "  ❌ FAIL: edge_f1 is NULL"
        ALL_PASS=false
    fi

    if [ "$lenient_pass" == "NULL" ] || [ "$lenient_pass" == "None" ]; then
        echo "  ❌ FAIL: lenient_pass is NULL"
        ALL_PASS=false
    fi

    # Check classification_factors exists
    has_factors=$(python3 -c "import json; print('classification_factors' in json.load(open('$metrics_file')))")
    echo "  classification_factors: $has_factors"

    if [ "$has_factors" != "True" ]; then
        echo "  ❌ FAIL: classification_factors missing"
        ALL_PASS=false
    fi

    if [ "$ALL_PASS" = true ]; then
        echo "  ✅ PASS: All required fields populated"
    fi
done

echo ""
echo "========================================="
if [ "$ALL_PASS" = true ]; then
    echo "✅ SMOKE TEST PASSED"
    echo ""
    echo "Next steps:"
    echo "  1. Review metrics in: $OUTPUT_DIR"
    echo "  2. Proceed to full 18-image validation"
    echo "  3. Run: python scripts/automation/production_depth_validation_fixed.py --input-dir data/validation_expanded --output-dir outputs/validation_v2_\$(date +%Y%m%d_%H%M%S) --tile-size 1024 --overlap 192"
    exit 0
else
    echo "❌ SMOKE TEST FAILED"
    echo ""
    echo "DO NOT PROCEED to full validation."
    echo "Fix integration issues first."
    exit 1
fi
