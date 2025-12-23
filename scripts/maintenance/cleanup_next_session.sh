#!/bin/bash
# Cleanup script for depth quality session artifacts
# Run at start of next session to organize documentation

set -e

echo "=== Transformation Portal: Depth Quality Session Cleanup ==="
echo ""

# Create archive directory
ARCHIVE_DIR="docs/sessions/2025_12_18_depth_quality"
mkdir -p "$ARCHIVE_DIR"

echo "Archiving session documentation to $ARCHIVE_DIR..."

# Move session-specific documentation
mv DEPTH_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv PRODUCTION_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv VALIDATION_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv TILING_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv TERMINAL_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv RESPONSE_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv QUALITY_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv HIGH_FIDELITY_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv CRITICAL_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv IMPLEMENTATION_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv INTEGRATED_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv EXECUTIVE_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv FIXES_*.txt "$ARCHIVE_DIR/" 2>/dev/null || true
mv PRIORITY_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv VALIDATED_*.md "$ARCHIVE_DIR/" 2>/dev/null || true
mv COMPREHENSIVE_*.md "$ARCHIVE_DIR/" 2>/dev/null || true

# Keep critical quick-refs at root
cp "$ARCHIVE_DIR/VALIDATION_QUICK_START.md" . 2>/dev/null || true
cp "$ARCHIVE_DIR/PRODUCTION_VALIDATION_QUICK_START.md" . 2>/dev/null || true

echo "Organizing validation scripts..."

# Move standalone validation scripts to scripts/validation/
mkdir -p scripts/validation/depth_quality/
mv quick_validation.py scripts/validation/depth_quality/ 2>/dev/null || true
mv run_isolation_tests.py scripts/validation/depth_quality/ 2>/dev/null || true
mv production_validation_*.py scripts/validation/depth_quality/ 2>/dev/null || true
mv run_ab_*.py scripts/validation/depth_quality/ 2>/dev/null || true
mv full_dataset_validation.py scripts/validation/depth_quality/ 2>/dev/null || true
mv ab_validation.py scripts/validation/depth_quality/ 2>/dev/null || true
mv depth_map_generator_standalone.py scripts/validation/depth_quality/ 2>/dev/null || true
mv research_grade_depth_pipeline.py scripts/validation/depth_quality/ 2>/dev/null || true

# Move old validation logs
mkdir -p logs/depth_validation_2025_12_18/
mv production_validation*.log logs/depth_validation_2025_12_18/ 2>/dev/null || true
mv validation_*.log logs/depth_validation_2025_12_18/ 2>/dev/null || true
mv quick_test_fix.log logs/depth_validation_2025_12_18/ 2>/dev/null || true

echo ""
echo "=== Cleanup Summary ==="
echo "✓ Documentation archived to: $ARCHIVE_DIR"
echo "✓ Validation scripts moved to: scripts/validation/depth_quality/"
echo "✓ Logs archived to: logs/depth_validation_2025_12_18/"
echo ""
echo "Critical files kept at root:"
echo "  - VALIDATION_QUICK_START.md"
echo "  - PRODUCTION_VALIDATION_QUICK_START.md"
echo "  - SESSION_END_SUMMARY_2025-12-18_DEPTH_QUALITY.md"
echo ""
echo "Next steps:"
echo "  1. Review git status"
echo "  2. Commit working state: high_fidelity_depth/, lux_depth_v2/, scripts/"
echo "  3. Fix sliver tiles (border padding)"
echo "  4. Run full 10+ image validation"
echo ""
