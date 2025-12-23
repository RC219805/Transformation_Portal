#!/bin/bash
#
# Kitchen-Only Phase 1 Sweep
# Uses existing sweep infrastructure with modified source directory
#

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Kitchen-Only Phase 1 Parameter Sweep                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Temporarily modify SOURCE_DIR in sweep_runner.py
SWEEP_RUNNER="exploration/sweep_runner.py"
BACKUP_RUNNER="exploration/sweep_runner.py.kitchen_backup"

# Backup original
cp "$SWEEP_RUNNER" "$BACKUP_RUNNER"

# Modify SOURCE_DIR to point to kitchen-only directory
sed -i.tmp 's|SOURCE_DIR = Path("projects/750_picacho_lane/Final_Production_UltraQuality")|SOURCE_DIR = Path("projects/750_picacho_lane/Kitchen_Only_Test")|g' "$SWEEP_RUNNER"

# Update SWEEP_ROOT to avoid conflicts
sed -i.tmp2 's|SWEEP_ROOT = Path("sweep_runs")|SWEEP_ROOT = Path("sweep_runs_kitchen_only")|g' "$SWEEP_RUNNER"

# Update BASELINE_DIR as well
sed -i.tmp3 's|BASELINE_DIR = Path("sweep_runs/baseline")|BASELINE_DIR = Path("sweep_runs_kitchen_only/baseline")|g' "$SWEEP_RUNNER"

echo "✓ Modified sweep_runner.py for kitchen-only testing"
echo "✓ Input: projects/750_picacho_lane/Kitchen_Only_Test/"
echo "✓ Output: sweep_runs_kitchen_only/"
echo ""

# Cleanup function to restore original
cleanup() {
    echo ""
    echo "Restoring original sweep_runner.py..."
    mv "$BACKUP_RUNNER" "$SWEEP_RUNNER"
    rm -f "${SWEEP_RUNNER}.tmp" "${SWEEP_RUNNER}.tmp2"
    echo "✓ Original configuration restored"
}

trap cleanup EXIT

# Run Phase 1 with the modified configuration
echo "🚀 Starting Kitchen-Only Phase 1 Sweep..."
echo ""
echo "This will run all 9 parameter sweeps on just the kitchen image."
echo "Estimated duration: ~1 hour"
echo "Estimated disk usage: ~30-40GB"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Execute Phase 1
bash exploration/execute_phase1.sh --all

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Kitchen-Only Phase 1 Sweep Complete!                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: sweep_runs_kitchen_only/"
echo ""
echo "Review results:"
echo "  ls -la sweep_runs_kitchen_only/"
echo "  cat sweep_runs_kitchen_only/*/notes.md"
echo ""
