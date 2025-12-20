#!/bin/bash
# Quick Reference: Classifier Improvement Session (2025-12-18)
# Use these commands to review results and proceed with next steps

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                CLASSIFIER IMPROVEMENT - QUICK REFERENCE                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# REVIEW CURRENT RESULTS
# ============================================================================
echo "📊 1. VIEW VALIDATION RESULTS"
echo "────────────────────────────────────────────────────────────────────────────"
echo "# Generate confusion matrix and pass rates"
echo "python3 scripts/analyze_validation_results.py outputs/validation_v2_20251218_170022_8197588/"
echo ""
echo "# Simulate filename hint improvements"
echo "python3 scripts/reanalyze_with_filenames.py outputs/validation_v2_20251218_170022_8197588/"
echo ""

# ============================================================================
# OPTION A: PROCEED WITH 77.8% BASELINE
# ============================================================================
echo "🅰  2a. OPTION A: PROCEED WITH 77.8% BASELINE (Pragmatic)"
echo "────────────────────────────────────────────────────────────────────────────"
echo "# Accept current classifier, calibrate gates by scene type"
echo ""
echo "# Step 1: Run validation with current classifier"
echo "python3 scripts/automation/production_depth_validation_fixed.py \\"
echo "    --input-dir input_images/750_Picacho \\"
echo "    --output-dir outputs/validation_gate_calibration_\$(date +%Y%m%d)"
echo ""
echo "# Step 2: Analyze results and identify gate threshold adjustments"
echo "python3 scripts/analyze_validation_results.py outputs/validation_gate_calibration_*/"
echo ""
echo "# Step 3: Document gate thresholds and acceptance criteria"
echo "# (Manual: update quality_metrics.py or create gate_config.yaml)"
echo ""

# ============================================================================
# OPTION B: INTEGRATE MATERIALSV3
# ============================================================================
echo "🅱  2b. OPTION B: INTEGRATE MATERIALSV3 (Ambitious)"
echo "────────────────────────────────────────────────────────────────────────────"
echo "# Implement ML-based scene classifier"
echo ""
echo "# Step 1: Add --scene-classifier flag to production_depth_validation_fixed.py"
echo "# (Code change required)"
echo ""
echo "# Step 2: Integrate MaterialsV3 in shadow mode"
echo "python3 scripts/automation/production_depth_validation_fixed.py \\"
echo "    --input-dir data/validation_expanded_18 \\"
echo "    --output-dir outputs/validation_materials_v3_\$(date +%Y%m%d) \\"
echo "    --scene-classifier materials_v3"
echo ""
echo "# Step 3: Compare classifications"
echo "python3 scripts/compare_classifiers.py \\"
echo "    outputs/validation_v2_20251218_170022_8197588/ \\"
echo "    outputs/validation_materials_v3_*/"
echo ""
echo "# Step 4: Decision gate - promote to active only if accuracy ≥85%"
echo ""

# ============================================================================
# OPTION C: MANUAL GROUND TRUTH REVIEW
# ============================================================================
echo "🅲  2c. OPTION C: MANUAL GROUND TRUTH REVIEW (Conservative)"
echo "────────────────────────────────────────────────────────────────────────────"
echo "# Validate inferred labels with human inspection"
echo ""
echo "# Step 1: Create ground truth CSV"
echo "cat > data/validation_expanded_18/ground_truth.csv << 'CSV'"
echo "filename,expected_scene_type,notes"
echo "750Picacho_Aerial.jpg,texture_dominated,Pool/courtyard aerial view"
echo "750Picacho_Pool.jpg,texture_dominated,Pool with water surface"
echo "# ... (add all 18 images)"
echo "CSV"
echo ""
echo "# Step 2: Re-run analysis with true labels"
echo "python3 scripts/analyze_validation_results.py \\"
echo "    outputs/validation_v2_20251218_170022_8197588/ \\"
echo "    --ground-truth data/validation_expanded_18/ground_truth.csv"
echo ""
echo "# Step 3: Adjust classifier if inferred labels were wrong"
echo ""

# ============================================================================
# DOCUMENTATION
# ============================================================================
echo "📖 3. READ FULL DOCUMENTATION"
echo "────────────────────────────────────────────────────────────────────────────"
echo "# Comprehensive handoff document"
echo "cat docs/guides/CLASSIFIER_IMPROVEMENT_HANDOFF_20251218.md"
echo ""
echo "# Full session summary"
echo "cat docs/CLASSIFIER_IMPROVEMENT_20251218.md"
echo ""

# ============================================================================
# TESTING
# ============================================================================
echo "🧪 4. RUN TESTS"
echo "────────────────────────────────────────────────────────────────────────────"
echo "# All scene classifier V2 tests (15 tests)"
echo "python3 -m pytest high_fidelity_depth/test_scene_classifier_v2.py -v"
echo ""
echo "# Smoke test (2 images)"
echo "python3 scripts/automation/production_depth_validation_fixed.py \\"
echo "    --input-dir data/validation_smoke \\"
echo "    --output-dir outputs/smoke_test_\$(date +%Y%m%d)"
echo ""

# ============================================================================
# GIT STATUS
# ============================================================================
echo "📦 5. GIT STATUS"
echo "────────────────────────────────────────────────────────────────────────────"
echo "git log --oneline -5"
echo "git status"
echo ""

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                            RESULTS AT A GLANCE                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Baseline:          55.6% accuracy (10/18)"
echo "  Depth Gradient:    61.1% accuracy (11/18) [+5.5%]"
echo "  Filename Hints:    77.8% accuracy (14/18) [+22.2% total]"
echo ""
echo "  Target:            85-90% accuracy"
echo "  Status:            ⚠️  Marginal (7% below target)"
echo ""
echo "  Override Precision: 100% (4/4 correct)"
echo "  Remaining Errors:   4/18 (generic filenames: 800-picacho-*.jpg)"
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         DECISION REQUIRED                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Choose ONE of the following:"
echo ""
echo "  🅰  Proceed with 77.8% (pragmatic, 1 session)"
echo "  🅱  Integrate MaterialsV3 (ambitious, 1-2 sessions)"
echo "  🅲  Manual review first (conservative, 30 min)"
echo ""
echo "  💡 RECOMMENDED: Option C → Option A"
echo ""
