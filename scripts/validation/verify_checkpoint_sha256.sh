#!/bin/bash
#
# Checkpoint SHA256 Verification Script
# Validates model checkpoint integrity
#
# Usage:
#   ./scripts/validation/verify_checkpoint_sha256.sh
#

set -euo pipefail

CHECKPOINT_DIR="checkpoints"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checkpoint SHA256 Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

CHECKPOINTS_FOUND=0
VERIFIED=0
MISSING=0

# Known checksums from presets
declare -A EXPECTED_CHECKSUMS=(
    ["depth_pro.pt"]="3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"
    # Add more as verified
)

echo "Scanning checkpoints..."
echo ""

for checkpoint in "$CHECKPOINT_DIR"/*.pt "$CHECKPOINT_DIR"/*.pth; do
    if [ -f "$checkpoint" ]; then
        CHECKPOINTS_FOUND=$((CHECKPOINTS_FOUND + 1))
        BASENAME=$(basename "$checkpoint")

        echo "📦 $BASENAME"

        # Compute actual SHA256
        ACTUAL_SHA=$(shasum -a 256 "$checkpoint" | awk '{print $1}')
        echo "   Actual:   $ACTUAL_SHA"

        # Check if expected checksum exists
        if [ -n "${EXPECTED_CHECKSUMS[$BASENAME]:-}" ]; then
            EXPECTED_SHA="${EXPECTED_CHECKSUMS[$BASENAME]}"
            echo "   Expected: $EXPECTED_SHA"

            if [ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]; then
                echo "   Status:   ✅ VERIFIED"
                VERIFIED=$((VERIFIED + 1))
            else
                echo "   Status:   ❌ MISMATCH"
            fi
        else
            echo "   Expected: (not in registry)"
            echo "   Status:   ⚠️  UNKNOWN"
            MISSING=$((MISSING + 1))
        fi
        echo ""
    fi
done

if [ $CHECKPOINTS_FOUND -eq 0 ]; then
    echo "⚠️  No checkpoints found in $CHECKPOINT_DIR"
    echo ""
    echo "Download checkpoints with:"
    echo "  python scripts/download_checkpoints.py"
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary:"
echo "  Total checkpoints: $CHECKPOINTS_FOUND"
echo "  Verified:          $VERIFIED"
echo "  Unknown:           $MISSING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "⚠️  Action Required:"
    echo "  Update EXPECTED_CHECKSUMS in this script with verified hashes"
fi
