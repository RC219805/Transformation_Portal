#!/usr/bin/env bash
#
# Phase 2 Benchmark Matrix Runner
#
# Runs a comprehensive test matrix of Phase 2 features across
# representative interior and exterior scenes with different quality tiers.
#
# Usage:
#   ./scripts/run_phase2_bench_matrix.sh [--quick] [--output-dir DIR]
#
# Options:
#   --quick        Run only APEX tier tests (faster)
#   --output-dir   Output directory (default: outputs/phase2_bench_matrix)
#   --help         Show this help message

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="$REPO_ROOT/assets/phase2_bench"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/phase2_bench_matrix}"
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --output-dir)
            OUT_ROOT="$2"
            shift 2
            ;;
        --help)
            head -n 15 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Phase 2 Benchmark Matrix ===${NC}"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUT_ROOT"
echo "Quick mode: $QUICK_MODE"
echo

# Create output directory
mkdir -p "$OUT_ROOT"

# Helper function to run a test case
run_case() {
    local name="$1"
    local preset="$2"
    local extra_flags="${3:-}"
    local input_file="$INPUT_DIR/${name}.tif"
    
    if [[ ! -f "$input_file" ]]; then
        echo -e "${YELLOW}⚠ Skipping $name (file not found)${NC}"
        return 0
    fi
    
    local output_dir="$OUT_ROOT/${name}_${preset}"
    local log_file="$output_dir/pipeline.log"
    
    mkdir -p "$output_dir"
    
    echo -e "${BLUE}▶ Running:${NC} $name | preset=$preset $extra_flags"
    
    # Run pipeline
    if python -m lux_depth_v2.cli \
        --input "$input_file" \
        --output-dir "$output_dir" \
        --preset "$preset" \
        $extra_flags \
        > "$log_file" 2>&1; then
        
        echo -e "${GREEN}✓${NC} Completed: $name | $preset"
        
        # Check for expected outputs
        local output_count=$(find "$output_dir" -type f \( -name "*.png" -o -name "*.tiff" -o -name "*.tif" \) | wc -l)
        echo "  Generated $output_count output files"
    else
        echo -e "${RED}✗${NC} Failed: $name | $preset (see $log_file)"
        return 1
    fi
}

# Helper for auto-preset tests
run_auto_preset() {
    local name="$1"
    local tier="$2"
    local input_file="$INPUT_DIR/${name}.tif"
    
    if [[ ! -f "$input_file" ]]; then
        echo -e "${YELLOW}⚠ Skipping $name (file not found)${NC}"
        return 0
    fi
    
    local output_dir="$OUT_ROOT/${name}_auto_${tier}"
    local log_file="$output_dir/pipeline.log"
    
    mkdir -p "$output_dir"
    
    echo -e "${BLUE}▶ Running:${NC} $name | auto-preset (tier=$tier)"
    
    if python -m lux_depth_v2.cli \
        --input "$input_file" \
        --output-dir "$output_dir" \
        --auto-preset \
        --quality-tier "$tier" \
        > "$log_file" 2>&1; then
        
        echo -e "${GREEN}✓${NC} Completed: $name | auto-preset ($tier)"
        
        # Extract selected preset from logs
        if grep -q "Auto-selected preset" "$log_file"; then
            local selected_preset=$(grep "Auto-selected preset" "$log_file" | head -1)
            echo "  $selected_preset"
        fi
    else
        echo -e "${RED}✗${NC} Failed: $name | auto-preset (see $log_file)"
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Test matrix
echo -e "\n${BLUE}=== Interior Scenes ===${NC}\n"

# Kitchen - Full tier matrix
if [[ "$QUICK_MODE" = false ]]; then
    run_case "750Picacho_Kitchen_Ultimate" "interior_luxury" && ((PASSED++)) || ((FAILED++))
    run_case "750Picacho_Kitchen_Ultimate" "interior_luxury_max_quality" && ((PASSED++)) || ((FAILED++))
fi
run_case "750Picacho_Kitchen_Ultimate" "interior_luxury_apex_quality" && ((PASSED++)) || ((FAILED++))

# Bedroom - APEX only
run_case "750Picacho_PrimaryBedroom_Ultimate" "interior_luxury_apex_quality" && ((PASSED++)) || ((FAILED++))

# Bathroom - APEX only  
run_case "750Picacho_PrimaryBathroom_Ultimate" "interior_luxury_apex_quality" && ((PASSED++)) || ((FAILED++))

echo -e "\n${BLUE}=== Exterior Scenes ===${NC}\n"

# Pool - APEX
run_case "750Picacho_Pool_Ultimate" "exterior_pool_apex_quality" && ((PASSED++)) || ((FAILED++))

# Pool - Wrong preset (control test - should still work but suboptimal)
if [[ "$QUICK_MODE" = false ]]; then
    run_case "750Picacho_Pool_Ultimate" "interior_luxury" && ((PASSED++)) || ((FAILED++))
fi

# Aerial - Standard
if [[ "$QUICK_MODE" = false ]]; then
    run_case "750Picacho_Aerial_Ultimate" "exterior_showcase" && ((PASSED++)) || ((FAILED++))
fi

echo -e "\n${BLUE}=== Auto-Preset Tests ===${NC}\n"

# Auto-preset on various scenes
run_auto_preset "750Picacho_Kitchen_Ultimate" "apex" && ((PASSED++)) || ((FAILED++))
run_auto_preset "750Picacho_PrimaryBedroom_Ultimate" "max" && ((PASSED++)) || ((FAILED++))
run_auto_preset "750Picacho_Pool_Ultimate" "apex" && ((PASSED++)) || ((FAILED++))

if [[ "$QUICK_MODE" = false ]]; then
    run_auto_preset "750Picacho_Aerial_Ultimate" "standard" && ((PASSED++)) || ((FAILED++))
fi

# Summary
echo
echo -e "${BLUE}=== Benchmark Summary ===${NC}"
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo
echo "Results saved to: $OUT_ROOT"

# Create summary JSON
cat > "$OUT_ROOT/benchmark_summary.json" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "passed": $PASSED,
  "failed": $FAILED,
  "skipped": $SKIPPED,
  "quick_mode": $QUICK_MODE,
  "output_directory": "$OUT_ROOT"
}
EOF

# Exit with appropriate code
if [[ $FAILED -gt 0 ]]; then
    exit 1
else
    echo -e "${GREEN}✓ All tests passed${NC}"
    exit 0
fi
