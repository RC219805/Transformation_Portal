#!/bin/bash
# Quick Launch Script - Phase 1 Parameter Sweep
# Usage: bash LAUNCH_PHASE1.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Phase 1 Parameter Sweep - Quick Launcher          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Select execution mode:"
echo ""
echo "  [1] Full Phase 1 with Live Monitoring (2-4 hours)"
echo "  [2] Depth Parameters Only (30-45 min)"
echo "  [3] Materials V3 Only (20-30 min)"
echo "  [4] Color/Tone Only (15-20 min)"
echo "  [5] Standard Full Phase 1 - No Monitoring (2-4 hours)"
echo "  [6] Verify Environment Only"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Full Phase 1 with Live Monitoring..."
        echo ""
        bash exploration/phase1_live_monitor.sh --all
        ;;
    2)
        echo ""
        echo "🚀 Starting Depth Parameter Sweeps..."
        echo ""
        bash exploration/phase1_live_monitor.sh --depth
        ;;
    3)
        echo ""
        echo "🚀 Starting Materials V3 Parameter Sweeps..."
        echo ""
        bash exploration/phase1_live_monitor.sh --materials
        ;;
    4)
        echo ""
        echo "🚀 Starting Color/Tone Parameter Sweeps..."
        echo ""
        bash exploration/phase1_live_monitor.sh --color
        ;;
    5)
        echo ""
        echo "🚀 Starting Standard Full Phase 1..."
        echo ""
        bash exploration/execute_phase1.sh --all
        ;;
    6)
        echo ""
        echo "🔍 Verifying environment..."
        echo ""
        bash exploration/execute_phase1.sh --verify-only
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac
