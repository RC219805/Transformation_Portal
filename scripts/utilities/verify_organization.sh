#!/bin/bash

echo "============================================"
echo "Directory Structure Verification"
echo "============================================"
echo ""

echo "📊 Root Directory Status:"
echo "-------------------------"
echo "Python files in root: $(ls -1 *.py 2>/dev/null | wc -l | tr -d ' ')"
echo "Shell scripts in root: $(ls -1 *.sh 2>/dev/null | wc -l | tr -d ' ')"
echo ""

echo "📁 Scripts Organization:"
echo "-------------------------"
echo "Pipelines:  $(ls -1 scripts/pipelines/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "Utilities:  $(ls -1 scripts/utilities/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "Analysis:   $(ls -1 scripts/analysis/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "Setup:      $(ls -1 scripts/setup/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

echo "📚 Examples Organization:"
echo "-------------------------"
echo "Pipelines:  $(ls -1 examples/pipelines/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "RAG:        $(ls -1 examples/rag/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "Workflows:  $(ls -1 examples/workflows/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

echo "📦 Archive Organization:"
echo "-------------------------"
echo "Experiments: $(ls -1 archive/experiments/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "Deprecated:  $(ls -1 archive/deprecated/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo "Legacy:      $(ls -1 archive/legacy/*.py 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

echo "🗂️  Output Directories:"
echo "-------------------------"
echo "750_picacho: $(ls -1d outputs/750_picacho/* 2>/dev/null | wc -l | tr -d ' ') directories"
echo "Tests:       $(ls -1d outputs/tests/* 2>/dev/null | wc -l | tr -d ' ') directories"
echo "Archive:     $(ls -1d outputs/archive/* 2>/dev/null | wc -l | tr -d ' ') directories"
echo ""

echo "✅ Verification Summary:"
echo "-------------------------"
total_scripts=$(($(ls -1 scripts/pipelines/*.py 2>/dev/null | wc -l) + $(ls -1 scripts/utilities/*.py 2>/dev/null | wc -l) + $(ls -1 scripts/analysis/*.py 2>/dev/null | wc -l) + $(ls -1 scripts/setup/*.py 2>/dev/null | wc -l)))
total_examples=$(($(ls -1 examples/pipelines/*.py 2>/dev/null | wc -l) + $(ls -1 examples/rag/*.py 2>/dev/null | wc -l) + $(ls -1 examples/workflows/*.py 2>/dev/null | wc -l)))
total_archive=$(($(ls -1 archive/experiments/*.py 2>/dev/null | wc -l) + $(ls -1 archive/deprecated/*.py 2>/dev/null | wc -l) + $(ls -1 archive/legacy/*.py 2>/dev/null | wc -l)))

echo "Total scripts organized: $total_scripts"
echo "Total examples organized: $total_examples"
echo "Total archived: $total_archive"
echo "Total Python files: $(($total_scripts + $total_examples + $total_archive))"
echo ""

if [ "$(ls -1 *.py 2>/dev/null | wc -l | tr -d ' ')" -le 1 ]; then
    echo "✅ Root directory is clean!"
else
    echo "⚠️  Root directory still has Python files"
fi
