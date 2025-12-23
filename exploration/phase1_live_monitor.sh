#!/bin/bash
# ==============================================
# Phase 1 Live Progress Monitor
# Wraps execute_phase1.sh with real-time metrics
# ==============================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MODE="${1:---all}"
LOG_FILE="sweep_runs/phase1_execution_$(date +%Y%m%d_%H%M%S).log"
SWEEP_ROOT="sweep_runs"

# Progress tracking
START_TIME=$(date +%s)
SWEEPS_COMPLETED=0
SWEEPS_FAILED=0

# =============================================================================
# Helper Functions
# =============================================================================

log_metric() {
    echo -e "${CYAN}[METRIC]${NC} $1"
}

calculate_duration() {
    local current=$(date +%s)
    local elapsed=$((current - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

show_progress() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Phase 1 Progress Summary${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    log_metric "Elapsed time: $(calculate_duration)"
    log_metric "Sweeps completed: ${GREEN}${SWEEPS_COMPLETED}${NC}"
    log_metric "Sweeps failed: ${RED}${SWEEPS_FAILED}${NC}"
    
    # Count output directories
    if [ -d "$SWEEP_ROOT" ]; then
        local run_count=$(find "$SWEEP_ROOT" -maxdepth 1 -type d -name "*delta*" 2>/dev/null | wc -l | tr -d ' ')
        log_metric "Run directories created: $run_count"
        
        # Show latest outputs
        local latest=$(find "$SWEEP_ROOT" -maxdepth 2 -type f -name "*.tif" -o -name "*_report.json" 2>/dev/null | tail -5)
        if [ ! -z "$latest" ]; then
            echo -e "${CYAN}Latest outputs:${NC}"
            echo "$latest" | while read -r file; do
                local size=$(du -h "$file" 2>/dev/null | cut -f1)
                echo "  - $(basename $file) ($size)"
            done
        fi
    fi
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

monitor_sweep_runner() {
    # Monitor the sweep_runner.py process
    while true; do
        # Check if sweep_runner is running
        if pgrep -f "sweep_runner.py" > /dev/null; then
            # Show progress every 30 seconds
            sleep 30
            show_progress
        else
            # No sweep_runner process found - execution might be done or between sweeps
            sleep 5
        fi
    done
}

# =============================================================================
# Main Execution
# =============================================================================

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Phase 1 Live Monitor - Lux Depth V2 Parameter Sweep${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Mode: $MODE"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo ""

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Start background monitor
monitor_sweep_runner &
MONITOR_PID=$!

# Trap to cleanup background process
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Execute Phase 1 with tee to capture output
echo -e "${BLUE}Starting Phase 1 execution...${NC}"
echo ""

if bash exploration/execute_phase1.sh "$MODE" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   Phase 1 Execution Complete - SUCCESS${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
else
    echo ""
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}   Phase 1 Execution Failed${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
fi

# Kill background monitor
kill $MONITOR_PID 2>/dev/null || true

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
show_progress
echo ""
echo -e "${CYAN}Final Summary:${NC}"
echo "  Total duration: $(calculate_duration)"
echo "  Log file: $LOG_FILE"
echo ""

# Show sweep results summary
if [ -f "$LOG_FILE" ]; then
    echo -e "${CYAN}Processing summary from log:${NC}"
    grep -E "✓|✗|Completed|failed" "$LOG_FILE" | tail -20 || true
fi

echo ""
echo -e "${GREEN}View detailed results:${NC}"
echo "  ls -la $SWEEP_ROOT/"
echo "  cat $SWEEP_ROOT/*/notes.md"
echo ""
