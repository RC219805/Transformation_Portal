#!/bin/bash
# Quick navigation helper for the optimized directory structure

echo "Transformation Portal - Directory Navigator"
echo "==========================================="
echo ""

case "${1}" in
    scripts|s)
        echo "📁 Scripts Directory:"
        echo ""
        echo "scripts/"
        echo "├── pipelines/     (42 files) - Pipeline execution scripts"
        echo "├── utilities/     (40 files) - Conversion, fixing, verification"
        echo "├── analysis/      (3 files)  - Quality analysis tools"
        echo "└── setup/         (4 files)  - Installation scripts"
        echo ""
        echo "Examples:"
        echo "  python scripts/pipelines/process_750_picacho.py"
        echo "  python scripts/utilities/verify_tiff_quality.py"
        echo "  python scripts/analysis/analyze_750_picacho_quality.py"
        ;;
    examples|e)
        echo "📚 Examples Directory:"
        echo ""
        echo "examples/"
        echo "├── rag/           (2 files) - RAG system examples"
        echo "├── workflows/     (2 files) - Workflow demonstrations"
        echo "└── pipelines/               - Pipeline examples"
        echo ""
        echo "Examples:"
        echo "  python examples/rag/rag_query.py 'depth pipeline'"
        echo "  python examples/rag/rag_workflow_demo.py"
        ;;
    archive|a)
        echo "🗄️  Archive Directory:"
        echo ""
        echo "archive/"
        echo "├── experiments/   (6 files) - Experimental features"
        echo "├── deprecated/    (5 files) - Superseded code"
        echo "└── legacy/        (2 files) - Historical implementations"
        ;;
    outputs|o)
        echo "📤 Outputs Directory:"
        echo ""
        echo "outputs/ (gitignored)"
        echo "├── 750_picacho/   (4 dirs)  - Project outputs"
        echo "├── tests/         (6 dirs)  - Test outputs"
        echo "└── archive/       (6 dirs)  - Archived outputs"
        ;;
    help|h|*)
        echo "Usage: ./navigate.sh [section]"
        echo ""
        echo "Sections:"
        echo "  scripts, s   - View scripts organization (89 files)"
        echo "  examples, e  - View examples organization (4 files)"
        echo "  archive, a   - View archived code (13 files)"
        echo "  outputs, o   - View outputs structure (16 dirs)"
        echo "  help, h      - Show this help"
        echo ""
        echo "Quick Reference:"
        echo "  scripts/pipelines/  - Process and run scripts (42)"
        echo "  scripts/utilities/  - Convert, fix, verify (40)"
        echo "  scripts/analysis/   - Quality analysis (3)"
        echo "  scripts/setup/      - Installation (4)"
        echo "  examples/rag/       - RAG examples (2)"
        echo "  examples/workflows/ - Workflow demos (2)"
        ;;
esac

