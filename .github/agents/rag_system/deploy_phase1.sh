#!/bin/bash
# ============================================================================
# Phase 1 RAG System Deployment Script
# Transformation Portal - Persistent Cache & Vector Search Activation
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_SYSTEM_DIR=".github/agents/rag_system"
OUTPUTS_DIR="/mnt/user-data/outputs"

echo "=============================================="
echo "Phase 1 RAG System Deployment"
echo "=============================================="
echo ""

# Check if we're in a repository root (look for .github)
check_repo_root() {
    if [ ! -d ".github" ]; then
        echo "⚠️  Warning: .github directory not found"
        echo "   Please run from repository root"
        echo "   Creating structure for demonstration..."
        mkdir -p "$RAG_SYSTEM_DIR"
    fi
}

# Create RAG system directory structure
create_directory_structure() {
    echo "📁 Creating directory structure..."
    mkdir -p "$RAG_SYSTEM_DIR"
    mkdir -p ".rag_cache"
    mkdir -p ".rag_cache/backups"
    echo "   ✓ Directory structure ready"
}

# Deploy configuration
deploy_config() {
    echo ""
    echo "⚙️  Deploying configuration..."

    if [ -f "$OUTPUTS_DIR/phase1_rag_config.yaml" ]; then
        cp "$OUTPUTS_DIR/phase1_rag_config.yaml" "$RAG_SYSTEM_DIR/config.yaml"
        echo "   ✓ Configuration deployed to $RAG_SYSTEM_DIR/config.yaml"
    else
        echo "   ⚠️  Config file not found in outputs, using local copy"
    fi
}

# Deploy Python modules
deploy_modules() {
    echo ""
    echo "🐍 Deploying Python modules..."

    modules=("cache_manager.py" "enhanced_retriever.py" "phase1_integration.py")

    for module in "${modules[@]}"; do
        if [ -f "$OUTPUTS_DIR/$module" ]; then
            cp "$OUTPUTS_DIR/$module" "$RAG_SYSTEM_DIR/"
            echo "   ✓ $module deployed"
        else
            echo "   ⚠️  $module not found"
        fi
    done
}

# Update __init__.py with new exports
update_init() {
    echo ""
    echo "📦 Updating module exports..."

    INIT_FILE="$RAG_SYSTEM_DIR/__init__.py"

    # Check if init file exists
    if [ ! -f "$INIT_FILE" ]; then
        touch "$INIT_FILE"
    fi

    # Check if Phase 1 exports already exist
    if grep -q "Phase 1 Enhancements" "$INIT_FILE" 2>/dev/null; then
        echo "   ✓ Phase 1 exports already present"
        return
    fi

    # Append Phase 1 exports
    cat >> "$INIT_FILE" << 'INITEOF'

# =============================================================================
# Phase 1 Enhancements - Persistence & Vector Search
# =============================================================================

# Cache Manager - Persistent storage with content-hash invalidation
try:
    from .cache_manager import (
        CacheManager,
        CacheConfig,
        CacheMetadata,
        ContentHasher,
        create_cache_manager,
        get_cache_status,
    )
except ImportError:
    pass  # Optional dependency

# Enhanced Retriever - Hybrid BM25 + Semantic Vector Search
try:
    from .enhanced_retriever import (
        EnhancedHybridRetriever,
        RetrieverConfig,
        RetrievalResult,
        RetrievalStats,
        BM25Retriever,
        VectorRetriever,
        create_retriever,
    )
except ImportError:
    pass  # Optional dependency (requires sentence-transformers)

# Unified RAG System Interface
try:
    from .phase1_integration import (
        RAGSystem,
        RAGConfig,
        Chunk,
    )
except ImportError:
    pass  # Optional dependency

# Phase 1 version marker
__phase1_version__ = "2.0.0"
INITEOF

    echo "   ✓ Module exports updated"
}

# Verify deployment
verify_deployment() {
    echo ""
    echo "🔍 Verifying deployment..."

    # Check files exist
    files=("config.yaml" "cache_manager.py" "enhanced_retriever.py" "phase1_integration.py")
    all_present=true

    for file in "${files[@]}"; do
        if [ -f "$RAG_SYSTEM_DIR/$file" ]; then
            echo "   ✓ $file present"
        else
            echo "   ✗ $file MISSING"
            all_present=false
        fi
    done

    # Check Python syntax
    echo ""
    echo "🐍 Validating Python syntax..."

    for pyfile in "$RAG_SYSTEM_DIR"/*.py; do
        if [ -f "$pyfile" ]; then
            if python3 -m py_compile "$pyfile" 2>/dev/null; then
                echo "   ✓ $(basename $pyfile) syntax valid"
            else
                echo "   ✗ $(basename $pyfile) syntax ERROR"
                all_present=false
            fi
        fi
    done

    if [ "$all_present" = true ]; then
        echo ""
        echo "   ✅ All verifications passed"
    else
        echo ""
        echo "   ⚠️  Some verifications failed"
    fi
}

# Check dependencies
check_dependencies() {
    echo ""
    echo "📚 Checking dependencies..."

    # Check numpy
    if python3 -c "import numpy" 2>/dev/null; then
        echo "   ✓ numpy available"
    else
        echo "   ⚠️  numpy not installed (required)"
    fi

    # Check PyYAML
    if python3 -c "import yaml" 2>/dev/null; then
        echo "   ✓ PyYAML available"
    else
        echo "   ⚠️  PyYAML not installed (required)"
    fi

    # Check sentence-transformers (optional but recommended)
    if python3 -c "from sentence_transformers import SentenceTransformer" 2>/dev/null; then
        echo "   ✓ sentence-transformers available (vector search enabled)"
    else
        echo "   ⚠️  sentence-transformers not installed (vector search disabled)"
        echo "      Install with: pip install sentence-transformers"
    fi

    # Check torch
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_DEVICE="cpu"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            TORCH_DEVICE="cuda"
        elif python3 -c "import torch; exit(0 if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
            TORCH_DEVICE="mps"
        fi
        echo "   ✓ PyTorch available (device: $TORCH_DEVICE)"
    else
        echo "   ⚠️  PyTorch not installed (required for vector search)"
    fi
}

# Initialize cache directory
init_cache() {
    echo ""
    echo "💾 Initializing cache directory..."

    mkdir -p .rag_cache/backups

    # Create initial metadata
    if [ ! -f ".rag_cache/metadata.json" ]; then
        cat > ".rag_cache/metadata.json" << 'METAEOF'
{
  "created_at": "",
  "updated_at": "",
  "chunk_count": 0,
  "embedding_count": 0,
  "indexed_files": 0,
  "cache_version": "2.0.0",
  "statistics": {
    "cache_hits": 0,
    "cache_misses": 0,
    "invalidations": 0
  }
}
METAEOF
        echo "   ✓ Cache metadata initialized"
    else
        echo "   ✓ Cache metadata exists"
    fi

    # Add cache to gitignore if not present
    if [ -f ".gitignore" ]; then
        if ! grep -q ".rag_cache" ".gitignore" 2>/dev/null; then
            echo "" >> .gitignore
            echo "# RAG System Cache (Phase 1)" >> .gitignore
            echo ".rag_cache/" >> .gitignore
            echo "   ✓ Added .rag_cache to .gitignore"
        fi
    fi
}

# Generate deployment report
generate_report() {
    echo ""
    echo "=============================================="
    echo "Deployment Summary"
    echo "=============================================="
    echo ""
    echo "📁 RAG System Location: $RAG_SYSTEM_DIR"
    echo "💾 Cache Location: .rag_cache/"
    echo ""
    echo "Files Deployed:"
    ls -la "$RAG_SYSTEM_DIR"/*.py "$RAG_SYSTEM_DIR"/*.yaml 2>/dev/null | awk '{print "   " $NF " (" $5 " bytes)"}'
    echo ""
    echo "Next Steps:"
    echo "   1. Install dependencies: pip install sentence-transformers torch"
    echo "   2. Run initial indexing: python -c \"from rag_system import RAGSystem; RAGSystem().index()\""
    echo "   3. Verify cache: ls -la .rag_cache/"
    echo ""
    echo "=============================================="
    echo "Phase 1 Deployment Complete"
    echo "=============================================="
}

# Main execution
main() {
    check_repo_root
    create_directory_structure
    deploy_config
    deploy_modules
    update_init
    verify_deployment
    check_dependencies
    init_cache
    generate_report
}

main "$@"
