#!/bin/bash
################################################################################
# PRODUCTION DEPTH VALIDATION - COMPREHENSIVE RUN
################################################################################
#
# ALL 5 CRITICAL FIXES IMPLEMENTED - READY FOR EXECUTION
#
# This script runs the comprehensive validation with all fixes:
# - PRIORITY 1: Separated reporting (execution/seam/quality)
# - PRIORITY 2: Spatial calibration smoothing (seam stabilization)
# - PRIORITY 3: Overshoot heatmap generation
# - PRIORITY 4: Readable edge overlay (RED/BLUE/GREEN)
# - PRIORITY 5: Full dataset + category reporting
#
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "PRODUCTION DEPTH VALIDATION - COMPREHENSIVE RUN"
echo "================================================================================"
echo ""
echo "Expected runtime: 30-45 minutes (6 large TIFF images)"
echo "Expected outcome: 6/6 execution, 5/6 seam pass, 2-3/6 strict quality pass"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to begin..."
sleep 5

python production_depth_validation.py \
  --input-dir input_images/750_Picacho/Source_TIFFs_Base \
  --output-dir outputs/production_validation_comprehensive_20251218 \
  --tile-size 1024 \
  --overlap 192

echo ""
echo "================================================================================"
echo "VALIDATION COMPLETE"
echo "================================================================================"
echo ""
echo "Review outputs at: outputs/production_validation_comprehensive_20251218/"
echo "Check validation_report.json for aggregate results"
echo "Review production_validation.log for detailed logs"
echo ""
