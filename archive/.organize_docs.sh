#!/bin/bash
# Organize markdown files to comply with codebase structure test

# Create documentation subdirectories if they don't exist
mkdir -p docs/sessions
mkdir -p docs/projects/750_picacho_lane
mkdir -p docs/guides

# Move session summaries
for file in SESSION_SUMMARY_*.md TASK_COMPLETION_REPORT.md; do
    if [ -f "$file" ]; then
        git mv "$file" docs/sessions/ 2>/dev/null || mv "$file" docs/sessions/
        echo "Moved $file to docs/sessions/"
    fi
done

# Move 750 Picacho project documentation
for file in 750_PICACHO_*.md AERIAL_DUPLICATE_RESOLUTION.md; do
    if [ -f "$file" ]; then
        git mv "$file" docs/projects/750_picacho_lane/ 2>/dev/null || mv "$file" docs/projects/750_picacho_lane/
        echo "Moved $file to docs/projects/750_picacho_lane/"
    fi
done

# Move integration guides
for file in BIM_PDF_*.md FINAL_QUALITY_BOOST_README.md; do
    if [ -f "$file" ]; then
        git mv "$file" docs/guides/ 2>/dev/null || mv "$file" docs/guides/
        echo "Moved $file to docs/guides/"
    fi
done

# Keep in root: README.md, START_HERE.md, MIGRATION_GUIDE.md, DEPRECATION_POLICY.md
echo ""
echo "Root markdown files remaining:"
find . -maxdepth 1 -name "*.md" -type f
echo ""
echo "Total: $(find . -maxdepth 1 -name "*.md" -type f | wc -l)"
