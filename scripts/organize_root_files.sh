#!/bin/bash
set -e

echo "Organizing root Python files into categorized directories..."

# Categorize and move pipeline scripts
echo "→ Moving pipeline scripts..."
for file in process_*.py run_*.py *_pipeline*.py coastal_*.py; do
    [ -f "$file" ] && mv -v "$file" scripts/pipelines/ || true
done

# Move utility scripts
echo "→ Moving utility scripts..."
for file in convert_*.py fix_*.py verify_*.py save_*.py update_*.py; do
    [ -f "$file" ] && mv -v "$file" scripts/utilities/ || true
done

# Move analysis scripts
echo "→ Moving analysis scripts..."
for file in analyze_*.py diagnose_*.py audit_*.py; do
    [ -f "$file" ] && mv -v "$file" scripts/analysis/ || true
done

# Move setup scripts
echo "→ Moving setup scripts..."
for file in install_*.py download_*.py; do
    [ -f "$file" ] && mv -v "$file" scripts/setup/ || true
done

# Move example files
echo "→ Moving example files..."
for file in example_*.py *_example*.py; do
    [ -f "$file" ] && mv -v "$file" examples/workflows/ || true
done

# Move RAG-related files to examples/rag
echo "→ Moving RAG examples..."
for file in rag_*.py *rag*.py; do
    [ -f "$file" ] && [ "$file" != "install_and_run_rag.py" ] && mv -v "$file" examples/rag/ || true
done

# Move test files to tests/
echo "→ Moving test files..."
for file in test_*.py; do
    [ -f "$file" ] && mv -v "$file" tests/ || true
done

echo "✓ File organization complete"
echo ""
echo "Remaining root Python files (for manual review):"
ls -1 *.py 2>/dev/null | head -20 || echo "None"
