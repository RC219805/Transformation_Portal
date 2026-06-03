#!/bin/bash
set -e

echo "Organizing output directories..."

# Move all 750_picacho outputs
echo "→ Moving 750_picacho outputs..."
mv -v output_750_picacho* outputs/750_picacho/ 2>/dev/null || true

# Move test outputs
echo "→ Moving test outputs..."
mv -v output_*test* outputs/tests/ 2>/dev/null || true
mv -v test_artifacts outputs/tests/ 2>/dev/null || true
mv -v test_view_configs outputs/tests/ 2>/dev/null || true

# Move other outputs to archive
echo "→ Moving other outputs to archive..."
mv -v output_* outputs/archive/ 2>/dev/null || true
mv -v processed_images outputs/archive/ 2>/dev/null || true

# Move single output directory if exists
if [ -d "output" ]; then
    mv -v output outputs/archive/output_general 2>/dev/null || true
fi

echo "✓ Output organization complete"

# Show summary
echo ""
echo "Summary of outputs directory:"
ls -la outputs/
echo ""
echo "750_picacho outputs:"
ls -1 outputs/750_picacho/ 2>/dev/null | head -10
echo ""
echo "Test outputs:"
ls -1 outputs/tests/ 2>/dev/null | head -10
