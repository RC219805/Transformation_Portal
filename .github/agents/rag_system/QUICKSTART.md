# RAG System - Quick Start Guide

**Retrieval-Augmented Generation system for intelligent code and documentation search**

[![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen)](tests/test_rag_pipeline.py)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## What is the RAG System?

The RAG system provides intelligent retrieval of code and documentation from the Transformation Portal repository. It helps developers:

- 🔍 **Find relevant code patterns** quickly and accurately
- 📚 **Discover documentation** with confidence scores
- 🎯 **Get structured citations** with file paths and line numbers
- 📝 **Generate workflow templates** for feature implementation and bug fixes
- 🧠 **Reduce hallucinations** by grounding responses in actual repository content

## Quick Start (30 seconds)

```bash
# 1. Index the repository
cd .github/agents/rag_system
python3 cli.py index --repo-root /path/to/repo

# 2. Search for code
python3 cli.py search "depth pipeline processing" --top-k 5

# 3. Generate citations
python3 cli.py cite "material response" --max-citations 3 --format markdown

# 4. Run full demo
python3 demo_full_pipeline.py --repo-root /path/to/repo
```

## System Architecture

```
┌──────────────────┐
│   Repository     │
│   (docs, src,    │
│   tests, agents) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  1. INDEXER      │  ← Chunks content (1,938 chunks)
│  (indexer.py)    │  ← Extracts metadata (functions, classes)
└────────┬─────────┘  ← Generates unique IDs
         │
         ▼
┌──────────────────┐
│  2. RETRIEVER    │  ← BM25 sparse retrieval
│  (retriever.py)  │  ← Filter by type/path
└────────┬─────────┘  ← 10-50ms query time
         │
         ▼
┌──────────────────┐
│  3. RERANKER     │  ← Multi-signal reranking
│  (reranker.py)   │  ← Code quality, docs, tests
└────────┬─────────┘  ← +1.21 avg boost
         │
         ▼
┌──────────────────┐
│  4. CITATIONS    │  ← Confidence scores (0-100%)
│  (citation.py)   │  ← Snippets & relevance notes
└────────┬─────────┘  ← Multiple formats
         │
         ▼
┌──────────────────┐
│  5. TEMPLATES    │  ← Feature implementation
│  (templates.py)  │  ← Bug triage
└──────────────────┘  ← CI workflows
```

## Key Features

### 1. Repository Indexing
- **1,938+ chunks** indexed from repository
- **4 chunk types**: agent (8%), code (33%), doc (23%), test (36%)
- **3 languages**: Python (72%), Markdown (28%), JSON (0.3%)
- **Metadata extraction**: Functions, classes, docstrings

### 2. Hybrid Retrieval
- **BM25 ranking** for precise text matching
- **Filtering**: By chunk type (code/doc/test/agent) and file path (regex)
- **Context windows**: Get surrounding chunks for better understanding
- **Performance**: 10-50ms query time

### 3. Multi-Signal Reranking
- **Exact match bonus** (2.0x): Query phrase in content
- **Code quality signals** (0.3x): Docstrings, type hints, naming
- **Documentation signals** (0.2x): Titles, examples, links
- **Test relevance** (0.1x): Function names matching query
- **Average boost**: +1.21 for high-quality matches

### 4. Structured Citations
- **Confidence scores**: 0-100% based on rank and score
- **File paths & line numbers**: Exact location in codebase
- **Snippets**: 10 lines, 500 chars max
- **Relevance notes**: Entity types, document types
- **Multiple formats**: Markdown, Text, JSON

### 5. Workflow Templates
- **Feature implementation**: Requirements → Files → Tests → PR
- **Bug triage**: Error → Root cause → Fix → Testing
- **CI changes**: Workflow → Changes → Testing → Impact
- **Structured responses**: JSON schema for machine parsing

## Usage Examples

### Example 1: Find Code Patterns

```python
from indexer import RepositoryIndexer
from retriever import HybridRetriever

# Index repository
indexer = RepositoryIndexer(repo_root='.')
chunks = indexer.index_repository()

# Search for depth processing code
retriever = HybridRetriever()
retriever.index(chunks)
results = retriever.retrieve(
    query="depth estimation processing",
    chunk_type_filter=['code'],
    top_k=5
)

for result in results:
    print(f"{result.file_path}:{result.start_line}")
    print(f"Score: {result.score:.2f}")
    if 'function_name' in result.metadata:
        print(f"Function: {result.metadata['function_name']}")
```

### Example 2: Generate Citations

```python
from reranker import ResultReranker
from citation import CitationGenerator

# Rerank results
reranker = ResultReranker()
reranked = reranker.rerank(results, query, top_k=3)

# Generate citations
citation_gen = CitationGenerator()
citations = citation_gen.generate_citations(reranked, max_citations=3)

# Format as markdown
formatted = citation_gen.format_citations(citations, format_type='markdown')
print(formatted)
```

Output:
```markdown
## Citations

### [1] src/transformation_portal/depth/pipeline.py:35-53
**Confidence**: 99%
**Relevance**: Class: ArchitecturalDepthPipeline | Has documentation

```python
class ArchitecturalDepthPipeline:
    """Production depth-aware enhancement pipeline..."""
    ...
```
```

### Example 3: Use Prompt Templates

```python
from templates import PromptTemplates

# Generate feature implementation template
template = PromptTemplates.feature_implementation(
    feature_description="Add atmospheric haze effect based on depth",
    context="Existing depth_pipeline infrastructure"
)

print(template)
```

## CLI Reference

### Index Command
```bash
python3 cli.py index [OPTIONS]

Options:
  --repo-root PATH        Repository root (default: .)
  --chunk-size TOKENS     Chunk size in tokens (default: 750)
  --chunk-overlap TOKENS  Overlap in tokens (default: 75)
  --output FILE          Save stats to JSON
```

### Search Command
```bash
python3 cli.py search QUERY [OPTIONS]

Options:
  --repo-root PATH    Repository root (default: .)
  --top-k N          Number of results (default: 10)
  --types TYPES      Filter by chunk types (code,doc,test,agent)
  --no-rerank        Skip reranking
```

### Citation Command
```bash
python3 cli.py cite QUERY [OPTIONS]

Options:
  --repo-root PATH       Repository root (default: .)
  --max-citations N      Max citations (default: 5)
  --format FORMAT        Output: markdown, text, json (default: markdown)
```

## Performance

| Metric | Value |
|--------|-------|
| **Indexing** | 1,938 chunks, 2.1M chars |
| **Query Time** | 10-50ms (BM25) |
| **Reranking** | 5-10ms (top-k) |
| **End-to-End** | 20-60ms |
| **Memory** | ~50-100MB |

## Testing

```bash
# Run all tests (10 tests, all passing)
pytest tests/test_rag_pipeline.py -v

# Run demo
python3 demo_full_pipeline.py --repo-root .
```

**Expected output:**
```
================================================== 10 passed in 0.06s ==================================================
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| **Indexer** | `indexer.py` | Index repository into chunks |
| **Retriever** | `retriever.py` | BM25 retrieval with filtering |
| **Reranker** | `reranker.py` | Multi-signal reranking |
| **Citations** | `citation.py` | Generate structured citations |
| **Templates** | `templates.py` | Workflow prompt templates |
| **CLI** | `cli.py` | Command-line interface |
| **Demo** | `demo_full_pipeline.py` | Complete pipeline demo |
| **Tests** | `tests/test_rag_pipeline.py` | Integration tests |

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete API reference with examples
- **[README.md](README.md)** - This quick start guide
- **[tests/test_rag_pipeline.py](tests/test_rag_pipeline.py)** - Test examples
- **[demo_full_pipeline.py](demo_full_pipeline.py)** - Live demonstration

## Troubleshooting

### No results found?
```python
# Try broader query
results = retriever.retrieve("depth", top_k=10)
```

### Low confidence scores?
```python
# Use more specific query
results = retriever.retrieve(
    query="ArchitecturalDepthPipeline process_image",
    top_k=10
)
```

### Import errors?
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.github/agents/rag_system')))
```

## Best Practices

1. ✅ **Index regularly** - Re-index when repository changes
2. ✅ **Use specific queries** - More specific = better results
3. ✅ **Filter by type** - Narrow search with chunk_type_filter
4. ✅ **Always rerank** - Reranking improves precision significantly
5. ✅ **Check confidence** - Only trust citations with >50% confidence
6. ✅ **Use context** - Get surrounding chunks for better understanding

## What's Next?

- Read the **[USAGE_GUIDE.md](USAGE_GUIDE.md)** for detailed examples
- Run the **demo**: `python3 demo_full_pipeline.py --repo-root .`
- Explore **tests**: Check `tests/test_rag_pipeline.py` for usage patterns
- Try the **CLI**: `python3 cli.py --help`

## Support

For issues or questions:
1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md) first
2. Review test examples in `tests/test_rag_pipeline.py`
3. Run demo script for complete example
4. Check repository documentation in `docs/`

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Tests**: 10/10 Passing  
**Last Updated**: November 2025
