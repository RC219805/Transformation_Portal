#!/bin/bash
# Context-Aware Rendering - Quick Start
# Transformation Portal

echo "=================================="
echo "CONTEXT-AWARE RENDERING QUICK START"
echo "=================================="
echo

# Check dependencies
echo "Checking dependencies..."
python3 -c "import fitz; print('✓ PyMuPDF installed')" 2>/dev/null || {
    echo "Installing PyMuPDF..."
    pip install pymupdf
}

# Example 1: Extract context from architectural PDF
echo
echo "Example 1: Extract Architectural Context"
echo "========================================="
echo
echo "python scripts/architectural_context_extractor.py \\"
echo "    'path/to/architectural_plans.pdf' \\"
echo "    --output extracted_context \\"
echo "    --verbose"
echo
echo "This extracts:"
echo "  • Room types and dimensions"
echo "  • Material palette"
echo "  • Design style"
echo "  • Floor plans (images)"
echo "  • Project metadata"
echo

# Example 2: Generate context-aware strategy
echo "Example 2: Generate Rendering Strategy"
echo "======================================="
echo
echo "python scripts/context_aware_rendering.py \\"
echo "    'input_images/Kitchen_Rendering.tiff' \\"
echo "    --context 'extracted_context/project_context.json' \\"
echo "    --output output_strategies"
echo
echo "This creates:"
echo "  • Room-specific processing strategy"
echo "  • Depth pipeline configuration"
echo "  • Material Response settings"
echo "  • Color grading parameters"
echo

# Example 3: Full premium processing
echo "Example 3: Full Premium Pipeline"
echo "================================="
echo
echo "python scripts/premium_context_pipeline.py \\"
echo "    'input_images/Kitchen_Rendering.tiff' \\"
echo "    --context 'extracted_context/project_context.json' \\"
echo "    --quality premium \\"
echo "    --output output_premium"
echo
echo "Quality levels:"
echo "  • standard - Fast processing (30-45 sec)"
echo "  • premium  - Balanced quality/speed (60-90 sec) [RECOMMENDED]"
echo "  • ultimate - Maximum quality with 4K upscale (3-5 min)"
echo

# Example 4: 750 Picacho Lane workflow
echo
echo "Example 4: Complete 750 Picacho Lane Workflow"
echo "=============================================="
echo
echo "# Step 1: Extract context from architectural PDF"
echo "python scripts/architectural_context_extractor.py \\"
echo "    '~/24098.00_750 PICACHO LANE.pdf' \\"
echo "    --output extracted_context \\"
echo "    --verbose"
echo
echo "# Step 2: Process kitchen rendering"
echo "python scripts/premium_context_pipeline.py \\"
echo "    'input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.tiff' \\"
echo "    --context 'extracted_context/24098.00_750 PICACHO LANE_context.json' \\"
echo "    --quality premium \\"
echo "    --output output_750Picacho"
echo
echo "# Step 3: Review outputs"
echo "ls -lh output_750Picacho/"
echo

# Batch processing example
echo "Example 5: Batch Process All Rooms"
echo "==================================="
echo
echo "# Process all renderings for a project"
echo "for render in input_images/750Picacho_*.tiff; do"
echo "    python scripts/premium_context_pipeline.py \\"
echo "        \"\$render\" \\"
echo "        --context 'extracted_context/750Picacho_context.json' \\"
echo "        --quality premium \\"
echo "        --output 'deliverables/750_Picacho'"
echo "done"
echo

# Quality comparison
echo "Example 6: Compare Generic vs Context-Aware"
echo "============================================"
echo
echo "# Generic processing (old way)"
echo "python luxury_tiff_batch_processor.py \\"
echo "    'input_images/Kitchen.tiff' \\"
echo "    --preset signature_estate \\"
echo "    --output-dir output_generic"
echo
echo "# Context-aware processing (new way)"
echo "python scripts/premium_context_pipeline.py \\"
echo "    'input_images/Kitchen.tiff' \\"
echo "    --context 'extracted_context/project_context.json' \\"
echo "    --quality premium \\"
echo "    --output output_context_aware"
echo
echo "# Compare results visually"
echo

# Troubleshooting
echo
echo "Troubleshooting"
echo "==============="
echo
echo "Issue: 'Could not identify room type'"
echo "Solution: Ensure filename includes room name (Kitchen, Bedroom, etc.)"
echo
echo "Issue: 'Depth processing failed'"
echo "Solution: Verify depth_pipeline/ directory exists and is configured"
echo
echo "Issue: 'No materials detected'"
echo "Solution: PDF may need OCR - run: ocrmypdf input.pdf output.pdf"
echo

# Documentation
echo
echo "Documentation"
echo "============="
echo "Full guide: docs/guides/CONTEXT_AWARE_RENDERING.md"
echo "Examples: Browse output_context_aware/ for sample strategies"
echo

echo
echo "✓ Quick start complete!"
echo
echo "Next steps:"
echo "  1. Extract context from your architectural PDF"
echo "  2. Process a test rendering with context intelligence"
echo "  3. Compare quality against generic processing"
echo "  4. Batch process entire project"
echo
