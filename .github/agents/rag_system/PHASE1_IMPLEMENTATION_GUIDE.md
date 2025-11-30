# Phase 1 RAG System Implementation Guide

## Transformation Portal - Persistent Cache & Vector Search Activation

**Version:** 2.0.0  
**Status:** Active Development  
**Last Updated:** 2025-11-30

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Performance](#performance)
8. [Troubleshooting](#troubleshooting)
9. [Migration Guide](#migration-guide)

---

## Overview

Phase 1 introduces two critical enhancements to the Transformation Portal RAG system:

### 1. Persistent Cache System
- **Content-hash-based invalidation**: Only re-index files that have actually changed
- **Embedding persistence**: Avoid recomputing expensive vector embeddings
- **Automatic backup management**: Versioned backups prevent data loss
- **Thread-safe operations**: Safe for concurrent access

### 2. Semantic Vector Search
- **Hybrid retrieval**: Combines BM25 (keyword) with semantic embeddings
- **Sentence Transformers integration**: State-of-the-art embedding models
- **Configurable weights**: Tune the balance between keyword and semantic matching
- **GPU/MPS acceleration**: Automatic device selection for optimal performance

### Benefits

| Feature | Before Phase 1 | After Phase 1 |
|---------|---------------|---------------|
| Index startup | 30-60s full reindex | ~200ms cache load |
| Query accuracy | BM25 only | Hybrid BM25 + Semantic |
| Session persistence | None | Full state preservation |
| Embedding compute | Every session | Cached across sessions |
| File change detection | None | Content-hash based |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAGSystem                                 │
│                   (Unified Interface)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │ CacheManager  │  │ SimpleIndexer │  │ EnhancedRetriever │   │
│  │               │  │               │  │                   │   │
│  │ • Persistence │  │ • Chunking    │  │ ┌─────────────┐   │   │
│  │ • Backups     │  │ • File scan   │  │ │ BM25        │   │   │
│  │ • Validation  │  │ • Metadata    │  │ │ (Sparse)    │   │   │
│  │               │  │               │  │ └─────────────┘   │   │
│  └───────────────┘  └───────────────┘  │ ┌─────────────┐   │   │
│         │                   │          │ │ Vector      │   │   │
│         │                   │          │ │ (Dense)     │   │   │
│         v                   v          │ └─────────────┘   │   │
│  ┌─────────────────────────────────┐  │         │         │   │
│  │           .rag_cache/           │  │         v         │   │
│  │  • chunks.pkl                   │  │ ┌─────────────┐   │   │
│  │  • embeddings.npy               │  │ │ Hybrid      │   │   │
│  │  • metadata.json                │  │ │ Scorer      │   │   │
│  │  • file_hashes.json             │  │ └─────────────┘   │   │
│  │  • backups/                     │  └───────────────────┘   │
│  └─────────────────────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **CacheManager** | Persistent storage, content-hash invalidation | `cache_manager.py` |
| **EnhancedRetriever** | Hybrid BM25 + semantic search | `enhanced_retriever.py` |
| **RAGSystem** | Unified interface, orchestration | `phase1_integration.py` |
| **Config** | YAML-based configuration | `phase1_rag_config.yaml` |

---

## Installation

### Prerequisites

```bash
# Core dependencies
pip install numpy pyyaml

# For vector search (recommended)
pip install sentence-transformers torch
```

### Quick Start

1. **Deploy Phase 1 components:**

```bash
cd /path/to/Transformation_Portal
chmod +x .github/agents/rag_system/deploy_phase1.sh
./.github/agents/rag_system/deploy_phase1.sh
```

2. **Verify installation:**

```python
from rag_system.phase1_integration import RAGSystem

# Initialize with defaults
rag = RAGSystem()

# Index repository
count = rag.index()
print(f"Indexed {count} chunks")

# Search
results = rag.search("depth pipeline processing")
for r in results[:3]:
    print(f"{r.file_path}: {r.score:.3f}")
```

### Manual Installation

If the deployment script doesn't work, copy files manually:

```bash
# Copy to RAG system directory
cp cache_manager.py .github/agents/rag_system/
cp enhanced_retriever.py .github/agents/rag_system/
cp phase1_integration.py .github/agents/rag_system/
cp phase1_rag_config.yaml .github/agents/rag_system/

# Create cache directory
mkdir -p .rag_cache/backups
```

---

## Configuration

### Configuration File: `phase1_rag_config.yaml`

```yaml
# Core Phase 1 settings
indexer:
  cache_enabled: true
  cache_dir: .rag_cache
  chunk_size_tokens: 750
  overlap_tokens: 75

retriever:
  enable_vector_search: true
  bm25_weight: 0.6      # Keyword matching weight
  vector_weight: 0.4    # Semantic matching weight
  vector_model: all-MiniLM-L6-v2

feature_flags:
  vector_search: true
  persistent_cache: true
  embedding_cache: true
  query_cache: true
```

### Configuration Priority

1. **Constructor arguments** (highest priority)
2. **YAML configuration file**
3. **Default values** (lowest priority)

### Key Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_vector_search` | bool | `true` | Enable semantic embedding search |
| `bm25_weight` | float | `0.6` | Weight for keyword matching (0-1) |
| `vector_weight` | float | `0.4` | Weight for semantic matching (0-1) |
| `cache_enabled` | bool | `true` | Enable persistent caching |
| `vector_model` | str | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `top_k_default` | int | `10` | Default number of results |

---

## Usage Guide

### Basic Usage

```python
from rag_system.phase1_integration import RAGSystem, RAGConfig

# Initialize with defaults
rag = RAGSystem()

# Index repository (uses cache if valid)
count = rag.index()

# Search
results = rag.search("atmospheric depth effects", top_k=5)

# Process results
for result in results:
    print(f"File: {result.file_path}")
    print(f"Score: {result.score:.4f} ({result.retrieval_method})")
    print(f"Content: {result.chunk.content[:100]}...")
    print()
```

### Custom Configuration

```python
config = RAGConfig(
    repo_root="/path/to/repo",
    enable_vector_search=True,
    bm25_weight=0.5,
    vector_weight=0.5,
    cache_dir=".my_cache",
)
rag = RAGSystem(config)
```

### Force Re-indexing

```python
# Ignore cache, rebuild from scratch
rag.index(force_reindex=True)
```

### Filter Results

```python
# Filter by chunk type
results = rag.search(
    "depth processing",
    chunk_types={"code", "doc"},  # Only code and documentation
)

# Filter by file path pattern
results = rag.search(
    "tone mapping",
    file_pattern=r"depth_pipeline/.*\.py",  # Only depth pipeline Python files
)
```

### Generate Citations

```python
results = rag.search("material response")
citations = rag.cite(results, format_type="markdown", max_citations=3)
print(citations)
```

### Cache Management

```python
# Save current state
rag.save()

# Clear cache
rag.clear_cache()

# Get statistics
stats = rag.get_statistics()
print(f"Indexed chunks: {stats['indexed_chunks']}")
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
```

### CLI Usage

```bash
# Index repository
python -m rag_system.phase1_integration index --repo-root .

# Search
python -m rag_system.phase1_integration search "depth pipeline"

# Show statistics
python -m rag_system.phase1_integration stats

# Clear cache
python -m rag_system.phase1_integration clear
```

---

## API Reference

### RAGSystem

```python
class RAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        """Initialize the RAG system."""

    def index(self, force_reindex: bool = False) -> int:
        """Index repository. Returns chunk count."""

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        chunk_types: Optional[Set[str]] = None,
        file_pattern: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Search for relevant chunks."""

    def cite(
        self,
        results: List[RetrievalResult],
        format_type: str = "markdown",
        max_citations: int = 5,
    ) -> str:
        """Generate citations from results."""

    def save(self) -> None:
        """Save current state to cache."""

    def clear_cache(self) -> None:
        """Clear all cached data."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
```

### CacheManager

```python
class CacheManager:
    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize cache manager."""

    def save_chunks(
        self,
        chunks: List[Any],
        source_files: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """Save chunks to persistent storage."""

    def load_chunks(self) -> Optional[List[Any]]:
        """Load chunks from cache."""

    def load_chunks_with_validation(
        self,
        current_files: Dict[str, Path],
    ) -> Tuple[Optional[List[Any]], Set[str]]:
        """Load with content-hash validation."""

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> bool:
        """Save embeddings to cache."""

    def load_embeddings(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Load embeddings from cache."""

    def clear(self) -> None:
        """Clear all cached data."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
```

### EnhancedHybridRetriever

```python
class EnhancedHybridRetriever:
    def __init__(self, config: Optional[RetrieverConfig] = None) -> None:
        """Initialize hybrid retriever."""

    def index(self, chunks: List[Any]) -> None:
        """Index chunks for retrieval."""

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        chunk_type_filter: Optional[Set[str]] = None,
        file_path_filter: Optional[str] = None,
        method: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks."""

    def load_cached_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> None:
        """Load pre-computed embeddings."""

    def save_embeddings(self, cache_manager: Any) -> bool:
        """Save embeddings to cache manager."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
```

---

## Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Cache load (1000 chunks) | ~50-200ms | From pickle |
| Full indexing (1000 files) | ~30-60s | First time only |
| Embedding compute (1000 chunks) | ~3-5s | GPU accelerated |
| BM25 query | <10ms | Keyword matching |
| Vector query | ~15-25ms | Semantic similarity |
| Hybrid query | ~20-30ms | Combined scoring |
| Cached query | <1ms | LRU cache hit |

### Memory Usage

- **Base**: ~50MB (model not loaded)
- **With model**: ~200-300MB (all-MiniLM-L6-v2)
- **With embeddings**: +~150MB per 100k chunks

### Optimization Tips

1. **Use cache**: Always enable caching for repeated sessions
2. **Batch queries**: Group similar queries to maximize cache hits
3. **Filter early**: Use `chunk_types` and `file_pattern` to reduce candidates
4. **Tune weights**: Adjust `bm25_weight`/`vector_weight` for your corpus

---

## Troubleshooting

### Common Issues

#### 1. "sentence-transformers not installed"

```bash
pip install sentence-transformers torch
```

Vector search will be disabled without these dependencies.

#### 2. "No cached chunks found"

The cache is empty. Run `rag.index()` first.

#### 3. Slow first query

The embedding model loads on first use. Enable `model_warmup: true` in config.

#### 4. Out of memory

Reduce `embedding_batch_size` in config, or use a smaller model like `all-MiniLM-L6-v2`.

#### 5. Cache invalidation not working

Ensure `content_hash_algorithm: sha256` is set in config.

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run RAG operations
rag = RAGSystem()
rag.index()
```

### Cache Inspection

```bash
# View cache contents
ls -la .rag_cache/

# Check metadata
cat .rag_cache/metadata.json

# Clear cache
rm -rf .rag_cache/*
```

---

## Migration Guide

### From Base RAG System

If you're upgrading from the base RAG system:

1. **Backup existing code**: The new components are additive
2. **Install dependencies**: `pip install sentence-transformers torch`
3. **Run deployment**: `.github/agents/rag_system/deploy_phase1.sh`
4. **Update imports**:

```python
# Old
from rag_system.retriever import HybridRetriever

# New
from rag_system.enhanced_retriever import EnhancedHybridRetriever
```

5. **Use unified interface**:

```python
# New unified interface
from rag_system.phase1_integration import RAGSystem

rag = RAGSystem()
rag.index()
results = rag.search("query")
```

### Backwards Compatibility

The Phase 1 components are designed to work alongside existing RAG components:

- Existing `HybridRetriever` remains available
- `RepositoryIndexer` can be used with `EnhancedHybridRetriever`
- Cache is optional and can be disabled

---

## Future Phases

### Phase 2: Incremental Indexing
- Git hook integration for automatic re-indexing
- Differential updates instead of full reindex
- Real-time file watching

### Phase 3: Knowledge Evolution
- Query pattern tracking
- Automatic relevance feedback
- Multi-repository search

---

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review logs in `.rag_cache/rag_system.log`
3. Open an issue with debug output

---

**Document Version:** 1.0  
**Phase:** 1 - Persistence & Vector Search  
**Author:** Transformation Portal Team
