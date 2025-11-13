#!/bin/bash
set -e

echo "Organizing shell scripts..."

# Move build/enhancement scripts
echo "→ Moving shell scripts to scripts/..."
mv -v *.sh scripts/ 2>/dev/null || true

# Move specific workflow scripts to tools
echo "→ Organizing in tools directory..."
[ -f "scripts/hdr_production_pipeline.sh" ] && mv -v scripts/hdr_production_pipeline.sh scripts/pipelines/ || true
[ -f "scripts/health_report.sh" ] && mv -v scripts/health_report.sh scripts/analysis/ || true

echo "✓ Shell script organization complete"

# Show current Python files still in root
echo ""
echo "Remaining Python files in root:"
ls -1 *.py 2>/dev/null | wc -l
ls -1 *.py 2>/dev/null | head -10
