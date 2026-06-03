#!/bin/bash
set -e

echo "Organizing remaining files and directories..."

# Move specialized enhancers/processors to archive/experiments
echo "→ Moving experimental/specialized files..."
mv -v ai_enhance*.py archive/experiments/ 2>/dev/null || true
mv -v board_material_aerial_enhancer.py archive/experiments/ 2>/dev/null || true
mv -v enhance_pool_aerial.py archive/experiments/ 2>/dev/null || true
mv -v agx_batch_processor.py archive/experiments/ 2>/dev/null || true

# Move legacy pipeline files
echo "→ Moving legacy pipeline files..."
mv -v pipeline.py archive/legacy/ 2>/dev/null || true
mv -v final_quality_boost.py archive/legacy/ 2>/dev/null || true

# Move deprecated files
echo "→ Moving deprecated files..."
mv -v evolutionary_checkpoint.py archive/deprecated/ 2>/dev/null || true
mv -v filter_nodes.py archive/deprecated/ 2>/dev/null || true
mv -v holographic_node.py archive/deprecated/ 2>/dev/null || true
mv -v prophetic_orchestrator.py archive/deprecated/ 2>/dev/null || true
mv -v temporal_evolution.py archive/deprecated/ 2>/dev/null || true

# Move utility modules that should be in src
echo "→ Moving modules to scripts/utilities..."
mv -v color_science.py scripts/utilities/ 2>/dev/null || true
mv -v format_utils*.py scripts/utilities/ 2>/dev/null || true
mv -v image_utils.py scripts/utilities/ 2>/dev/null || true
mv -v helpers.py scripts/utilities/ 2>/dev/null || true

# Move depth-related to scripts/utilities
mv -v depth_*.py scripts/utilities/ 2>/dev/null || true
mv -v coreml_wrapper.py scripts/utilities/ 2>/dev/null || true

# Move BIM/context extractors
mv -v bim_metadata_extractor.py scripts/utilities/ 2>/dev/null || true
mv -v architectural_context*.py scripts/utilities/ 2>/dev/null || true
mv -v context_aware_renderer.py scripts/utilities/ 2>/dev/null || true

# Move material/rendering utilities
mv -v material_response*.py scripts/utilities/ 2>/dev/null || true
mv -v create_board_textures.py scripts/utilities/ 2>/dev/null || true
mv -v tiff_quality_optimizer.py scripts/utilities/ 2>/dev/null || true
mv -v tonemapper_agx_filmic.py scripts/utilities/ 2>/dev/null || true

# Move misc utilities
mv -v realize_*.py scripts/utilities/ 2>/dev/null || true
mv -v pdf_spec_parser.py scripts/utilities/ 2>/dev/null || true
mv -v resolve_*.py scripts/utilities/ 2>/dev/null || true
mv -v check_source_quality.py scripts/utilities/ 2>/dev/null || true
mv -v compare_images.py scripts/utilities/ 2>/dev/null || true
mv -v multiformat_converter.py scripts/utilities/ 2>/dev/null || true

# Move batch processors
mv -v batch_*.py scripts/utilities/ 2>/dev/null || true
mv -v world_class_batch_processor.py scripts/utilities/ 2>/dev/null || true
mv -v luxury_tiff_batch_processor*.py scripts/utilities/ 2>/dev/null || true
mv -v luxury_video_master_grader.py scripts/utilities/ 2>/dev/null || true

# Move phase/golden hour scripts to pipelines
mv -v phase*.py scripts/pipelines/ 2>/dev/null || true
mv -v golden_hour*.py scripts/pipelines/ 2>/dev/null || true

# Move visualization tools
mv -v visualize_*.py scripts/utilities/ 2>/dev/null || true
mv -v synthetic_viewer.py scripts/utilities/ 2>/dev/null || true
mv -v presence_*.py archive/experiments/ 2>/dev/null || true

echo "✓ File organization complete"
