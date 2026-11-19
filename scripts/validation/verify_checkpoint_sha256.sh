#!/bin/bash
#
# Checkpoint SHA256 Verification Script
# Validates model checkpoint integrity
#
# Usage:
#   ./scripts/validation/verify_checkpoint_sha256.sh [--strict]
#
# Exit codes:
#   0  every checkpoint with a registered SHA256 matches it
#   1  a checkpoint does not match its registered SHA256, or the checkpoint
#      directory is missing; with --strict also when a checkpoint has no
#      registered SHA256 or no checkpoint was found (nothing was verified)
#   2  usage error
#
# Compatible with macOS system bash 3.2 (no associative arrays).
#

set -euo pipefail

CHECKPOINT_DIR="checkpoints"
STRICT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --strict)
            STRICT=1
            ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--strict]" >&2
            exit 2
            ;;
    esac
    shift
done

# Known checksums from presets. Prints nothing for a checkpoint that is not registered.
expected_checksum_for() {
    case "$1" in
        depth_pro.pt) echo "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce" ;;
        # Add more as verified
        *) ;;
    esac
}

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
MISMATCHED=0
MISSING=0

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
        EXPECTED_SHA=$(expected_checksum_for "$BASENAME")
        if [ -n "$EXPECTED_SHA" ]; then
            echo "   Expected: $EXPECTED_SHA"

            if [ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]; then
                echo "   Status:   ✅ VERIFIED"
                VERIFIED=$((VERIFIED + 1))
            else
                echo "   Status:   ❌ MISMATCH"
                MISMATCHED=$((MISMATCHED + 1))
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
    if [ $STRICT -eq 1 ]; then
        echo ""
        echo "❌ --strict: nothing was verified"
        exit 1
    fi
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary:"
echo "  Total checkpoints: $CHECKPOINTS_FOUND"
echo "  Verified:          $VERIFIED"
echo "  Mismatched:        $MISMATCHED"
echo "  Unknown:           $MISSING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "⚠️  Action Required:"
    echo "  Register verified hashes in expected_checksum_for() in this script"
fi

if [ $MISMATCHED -gt 0 ]; then
    echo ""
    echo "❌ $MISMATCHED checkpoint(s) do not match their registered SHA256"
    exit 1
fi

if [ $STRICT -eq 1 ] && [ $MISSING -gt 0 ]; then
    echo ""
    echo "❌ --strict: $MISSING checkpoint(s) have no registered SHA256"
    exit 1
fi
