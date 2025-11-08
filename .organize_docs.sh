#!/bin/bash

# Session summaries and temporary docs -> docs/session_summaries/
mv -f TIFF_CONVERSION_FIX.md docs/session_summaries/ 2>/dev/null || true
mv -f CODE_QUALITY_IMPROVEMENTS.md docs/session_summaries/ 2>/dev/null || true
mv -f SESSION_SUMMARY_NOV8.md docs/session_summaries/ 2>/dev/null || true
mv -f TIFF_FIX_SUMMARY_NOV8.md docs/session_summaries/ 2>/dev/null || true
mv -f UNIFIED_PIPELINE_SUMMARY.md docs/session_summaries/ 2>/dev/null || true
mv -f PHASE_2_COMPLETE.md docs/session_summaries/ 2>/dev/null || true

# Implementation and task reports -> docs/
mv -f IMPLEMENTATION_TEST_REPORT.md docs/ 2>/dev/null || true
mv -f TASK_COMPLETION_SUMMARY.md docs/ 2>/dev/null || true
mv -f WORKFLOW_OPTIMIZATION_SUMMARY.md docs/ 2>/dev/null || true
mv -f QUALITY_SYSTEM_SUMMARY.md docs/ 2>/dev/null || true

# Archive old summaries
mv -f AUDIT_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f CHANGES_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f EXEC_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f GREATROOM_ANALYSIS_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f GREATROOM_QUICK_REFERENCE.txt docs/archive/ 2>/dev/null || true
mv -f HEALTH_CHECK_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f KITCHEN_ANALYSIS_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f MERGE_SUMMARY.txt docs/archive/ 2>/dev/null || true
mv -f PR232_VERIFICATION.txt docs/archive/ 2>/dev/null || true
mv -f README_FIX.txt docs/archive/ 2>/dev/null || true
mv -f WORKFLOW_VISUAL_GUIDE.txt docs/archive/ 2>/dev/null || true
mv -f FILES_CHANGED.md docs/archive/ 2>/dev/null || true

echo "Documentation organized successfully"
