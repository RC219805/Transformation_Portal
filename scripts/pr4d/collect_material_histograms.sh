#!/usr/bin/env bash
# PR-4D Data Collection: Run canary preset on 5-6 scenes to collect material histograms
# Focus: Stone (primary), Wood (secondary)
# Excludes: Pool (water detector not ready), Bathroom (MPS OOM risk)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SCENES=(
    "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Kitchen_UltraQuality.tif"
    "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_GreatRoom_UltraQuality.tif"
    "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_PrimaryBedroom_UltraQuality.tif"
    "projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Aerial_UltraQuality.tif"
)

OUTPUT_BASE="outputs/pr4d_data"
PRESET="interior_luxury_apex_quality_materials_v3_glass"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PR-4D Material Histogram Collection"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Preset: $PRESET"
echo "Scenes: ${#SCENES[@]}"
echo "Output: $OUTPUT_BASE"
echo ""

mkdir -p "$OUTPUT_BASE"

for scene_path in "${SCENES[@]}"; do
    if [[ ! -f "$scene_path" ]]; then
        echo "⚠️  Skipping missing: $scene_path"
        continue
    fi
    
    scene_name="$(basename "$scene_path" .tif)"
    scene_output="$OUTPUT_BASE/$scene_name"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing: $scene_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python -m lux_depth_v2.cli \
        --input "$scene_path" \
        --output-dir "$scene_output" \
        --preset "$PRESET" \
        --device auto \
        --upscaler-backend torch
    
    # Verify report was generated
    report_json="$scene_output/${scene_name}_report.json"
    if [[ -f "$report_json" ]]; then
        echo "✅ Report generated: $report_json"
        
        # Quick preview of materials detected
        echo ""
        echo "Materials detected:"
        python3 -c "
import json, sys
with open('$report_json') as f:
    r = json.load(f)
    plan = r.get('materials_v3_response_plan', {})
    present = plan.get('summary', {}).get('present_classes', [])
    print('  ' + ', '.join(present) if present else '  (none)')
"
    else
        echo "❌ Report not found: $report_json"
    fi
    
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Data collection complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next step: Run aggregation script"
echo "  python scripts/pr4d/aggregate_histograms.py"
