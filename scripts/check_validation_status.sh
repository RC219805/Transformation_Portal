#!/bin/bash
# Quick status check for 50-image validation run
# Usage: ./scripts/check_validation_status.sh

set -e

echo "=================================================="
echo "50-Image Validation Status Check"
echo "=================================================="
echo ""

# Check if process is still running
RUNNING=$(ps aux | grep production_depth_validation_fixed.py | grep -v grep | wc -l)

if [ "$RUNNING" -gt 0 ]; then
    echo "⏳ Validation is STILL RUNNING"
    echo ""
    ps aux | grep production_depth_validation_fixed.py | grep -v grep
    echo ""
    
    # Check progress from latest output directory
    OUTPUT_DIR=$(ls -td outputs/validation_full_50img_* 2>/dev/null | head -1)
    if [ -n "$OUTPUT_DIR" ]; then
        COMPLETED=$(ls -1 "$OUTPUT_DIR"/*_metrics.json 2>/dev/null | wc -l | tr -d ' ')
        echo "Progress: $COMPLETED/50 images completed ($(echo "scale=1; $COMPLETED * 100 / 50" | bc)%)"
        echo ""
        
        # Show latest image processed
        LATEST=$(ls -t "$OUTPUT_DIR"/*_metrics.json 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "Latest: $(basename "$LATEST" _metrics.json)"
        fi
    fi
    
    echo ""
    echo "Tail log:"
    tail -20 validation_full_50img_run.log 2>/dev/null || echo "  (log not found)"
    
    exit 0
fi

echo "✅ Validation COMPLETE (no process running)"
echo ""

# Find latest output directory
OUTPUT_DIR=$(ls -td outputs/validation_full_50img_* 2>/dev/null | head -1)

if [ -z "$OUTPUT_DIR" ]; then
    echo "❌ No output directory found matching: outputs/validation_full_50img_*"
    exit 1
fi

echo "Output: $OUTPUT_DIR"
echo ""

# Count metrics files
METRICS_COUNT=$(ls -1 "$OUTPUT_DIR"/*_metrics.json 2>/dev/null | wc -l | tr -d ' ')
echo "Metrics files: $METRICS_COUNT/50"

if [ "$METRICS_COUNT" -ne 50 ]; then
    echo "⚠️  WARNING: Expected 50 metrics files, found $METRICS_COUNT"
    echo ""
    echo "Missing or failed images - check log for errors:"
    echo "  grep -i error validation_full_50img_run.log"
    exit 1
fi

echo ""

# Check for validation_report.json
if [ -f "$OUTPUT_DIR/validation_report.json" ]; then
    echo "✅ validation_report.json exists"
    echo ""
    echo "Quality Summary:"
    jq -r '.quality | "  Lenient: \(.lenient_pass)/\(.total) (\(.lenient_pass_rate * 100 | floor)%)\n  Strict:  \(.strict_pass)/\(.total) (\(.strict_pass_rate * 100 | floor)%)"' "$OUTPUT_DIR/validation_report.json" 2>/dev/null || echo "  (unable to parse JSON)"
    echo ""
    
    # Scene type breakdown
    echo "Scene Type Breakdown:"
    jq -r '.images[] | .scene_type' "$OUTPUT_DIR/validation_report.json" 2>/dev/null | sort | uniq -c | while read count type; do
        echo "  $type: $count"
    done
    echo ""
else
    echo "⚠️  validation_report.json NOT FOUND"
    echo ""
fi

echo "=================================================="
echo "Next Steps:"
echo "=================================================="
echo ""
echo "1. Analyze results:"
echo "   python scripts/analyze_validation_v2.py $OUTPUT_DIR"
echo ""
echo "2. Generate confusion matrix and per-class metrics"
echo ""
echo "3. Review top failures:"
echo "   jq '.images[] | select(.lenient_pass == false) | {filename, scene_type, edge_f1, chamfer_distance, gate_reason}' $OUTPUT_DIR/validation_report.json | head -20"
echo ""
echo "4. If classifier healthy (≥90%), proceed to structure input-size sweep"
echo "   See: SESSION_COMPLETE_50IMAGE_VALIDATION_IN_PROGRESS.md"
echo ""
