#!/bin/bash
# Organize documentation files to fix test failure

# Create docs structure
mkdir -p docs/750_picacho
mkdir -p docs/pipeline
mkdir -p docs/quality_analysis
mkdir -p docs/visual_review
mkdir -p docs/depth_model

# Move 750 Picacho related docs
mv 750_PICACHO_*.md docs/750_picacho/ 2>/dev/null
mv QUALITY_ANALYSIS_SUMMARY.txt docs/750_picacho/ 2>/dev/null

# Move pipeline documentation
mv PIPELINE_*.md docs/pipeline/ 2>/dev/null
mv LUXURY_ESTATE_*.md docs/pipeline/ 2>/dev/null
mv ELITE_PIPELINE_*.md docs/pipeline/ 2>/dev/null
mv README_PIPELINE_FIXES.md docs/pipeline/ 2>/dev/null

# Move quality analysis docs
mv QA_Improvements.md docs/quality_analysis/ 2>/dev/null
mv Corrective_Action_Plan.md docs/quality_analysis/ 2>/dev/null
mv Immediate_Recommendations.md docs/quality_analysis/ 2>/dev/null

# Move visual review docs
mv Visual_Feedback_Analysis_Summary.md docs/visual_review/ 2>/dev/null
mv Root_Cause_Analysis.md docs/visual_review/ 2>/dev/null
mv CRITICAL_VISUAL_REVIEW_FINDINGS.md docs/visual_review/ 2>/dev/null
mv Analysis_Complete.md docs/visual_review/ 2>/dev/null

# Move depth model docs
mv DEPTH_MODEL_*.md docs/depth_model/ 2>/dev/null
mv PHASE*.md docs/depth_model/ 2>/dev/null

# Move git/push related docs
mv COMMIT_SUMMARY.md docs/ 2>/dev/null
mv PUSH_*.md docs/ 2>/dev/null
mv FILES_*.md docs/ 2>/dev/null

echo "Documentation reorganization complete!"
