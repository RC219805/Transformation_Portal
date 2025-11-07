# RAG Pipeline Usage Guide

Complete guide to using the RAG (Retrieval-Augmented Generation) system for the Transformation Portal.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Components](#components)
- [CLI Usage](#cli-usage)
- [Programmatic Usage](#programmatic-usage)
- [Examples](#examples)
- [Testing](#testing)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Overview

The RAG system provides intelligent code and documentation retrieval with confidence scoring and structured citations. It helps developers find relevant code patterns, understand implementation details, and generate contextual responses.

### Key Features

- **Repository Indexing**: Indexes 1900+ chunks from docs/, src/, tests/, and agent files
- **Hybrid Retrieval**: BM25 sparse retrieval for precise text matching
- **Multi-Signal Reranking**: Improves precision using code quality, documentation, and relevance signals
- **Citation Generation**: Structured citations with confidence scores (0-100%)
- **Prompt Templates**: Canonical templates for feature implementation, bug triage, and CI changes
- **Knowledge Integration**: Pattern analysis and recommendation engine

## Quick Start

### 1. Index Your Repository

```bash
cd .github/agents/rag_system
python3 cli.py index --repo-root /path/to/repo --output stats.json
```

**Output:**
```
Indexing complete:
  Total chunks: 1938
  Chunk types:
    agent: 155
    code: 638
    doc: 447
    test: 698
```

### 2. Search for Relevant Code

```bash
python3 cli.py search "depth pipeline processing" --top-k 5
```

**Output:**
```
[1] src/transformation_portal/depth/pipeline.py:35-53
Score: 9.293 | Method: bm25
Metadata: class_name=ArchitecturalDepthPipeline, entity_type=class
Preview: Production depth-aware enhancement pipeline...
```

### 3. Generate Citations

```bash
python3 cli.py cite "material response enhancement" --max-citations 3 --format markdown
```

**Output:**
```markdown
## Citations

### [1] src/transformation_portal/processors/material_response/__init__.py:1-2
**Confidence**: 84%
**Relevance**: Text match
...
```

### 4. Run Complete Demo

```bash
python3 demo_full_pipeline.py --repo-root /path/to/repo
```

This demonstrates the entire pipeline from indexing to citation generation.

## Components

### 1. Repository Indexer (`indexer.py`)

Indexes repository content into searchable chunks with metadata.

**Features:**
- Chunks with 500-1000 token windows and 50-100 token overlap
- Extracts metadata from Python code (functions, classes, docstrings)
- Detects programming languages (Python, Markdown, YAML, JSON)
- Generates unique chunk IDs with SHA-256 hashing

**Supported File Types:**
- Python (`.py`)
- Markdown (`.md`)
- YAML (`.yaml`, `.yml`)
- JSON (`.json`)
- Shell scripts (`.sh`, `.bash`)
- Config files (`.toml`, `.cfg`)

**Example:**
```python
from indexer import RepositoryIndexer

indexer = RepositoryIndexer(repo_root='.')
chunks = indexer.index_repository()
stats = indexer.get_statistics()

print(f"Indexed {stats['total_chunks']} chunks")
```

### 2. Hybrid Retriever (`retriever.py`)

Retrieves relevant chunks using BM25 ranking.

**Features:**
- BM25 sparse retrieval (k1=1.5, b=0.75)
- Filter by chunk type (code, doc, test, agent)
- Filter by file path (regex patterns)
- Context window retrieval (surrounding chunks)

**Example:**
```python
from retriever import HybridRetriever

retriever = HybridRetriever()
retriever.index(chunks)

results = retriever.retrieve(
    query="depth processing",
    top_k=10,
    chunk_type_filter=['code', 'doc'],
    file_path_filter=r'depth.*\.py'
)
```

### 3. Result Reranker (`reranker.py`)

Improves precision using multiple quality signals.

**Signals:**
- **Exact match bonus** (2.0): Exact query phrase in content
- **Code quality bonus** (0.3): Docstrings, type hints, good naming
- **Documentation bonus** (0.2): Title, code examples, links
- **Test relevance bonus** (0.1): Test function names matching query

**Example:**
```python
from reranker import ResultReranker

reranker = ResultReranker()
reranked = reranker.rerank(results, query, top_k=5)

for result in reranked:
    boost = result.metadata['rerank_boost']
    print(f"Score: {result.score:.3f} (boost: {boost:+.3f})")
```

### 4. Citation Generator (`citation.py`)

Generates structured citations with confidence scoring.

**Features:**
- Confidence based on rank and retrieval score
- Snippet extraction (10 lines, 500 chars max)
- Relevance notes (function/class names, documentation type)
- Multiple formats (markdown, text, JSON)

**Example:**
```python
from citation import CitationGenerator

citation_gen = CitationGenerator()
citations = citation_gen.generate_citations(results, max_citations=5)

# Format as markdown
formatted = citation_gen.format_citations(citations, format_type='markdown')
print(formatted)
```

### 5. Prompt Templates (`templates.py`)

Canonical templates for common workflows.

**Templates:**
- **Feature Implementation**: Requirements → Files → Tests → PR description
- **Bug Triage**: Error log → Root cause → Fix strategy → Testing
- **CI Change**: Workflow → Changes → Testing → Impact assessment

**Example:**
```python
from templates import PromptTemplates

template = PromptTemplates.feature_implementation(
    feature_description="Add atmospheric haze effect",
    context="Existing depth_pipeline infrastructure"
)

print(template)
```

## CLI Usage

### Index Command

```bash
python3 cli.py index [OPTIONS]

Options:
  --repo-root PATH        Repository root directory (default: .)
  --chunk-size TOKENS     Chunk size in tokens (default: 750)
  --chunk-overlap TOKENS  Overlap in tokens (default: 75)
  --output FILE          Save statistics to JSON file
  --verbose              Verbose output
```

### Search Command

```bash
python3 cli.py search QUERY [OPTIONS]

Options:
  --repo-root PATH    Repository root directory (default: .)
  --top-k N          Number of results (default: 10)
  --types TYPES      Comma-separated chunk types (code,doc,test,agent)
  --no-rerank        Skip reranking
```

### Citation Command

```bash
python3 cli.py cite QUERY [OPTIONS]

Options:
  --repo-root PATH       Repository root directory (default: .)
  --max-citations N      Maximum citations (default: 5)
  --format FORMAT        Output format: markdown, text, json (default: markdown)
```

### Template Command

```bash
python3 cli.py template TYPE [OPTIONS]

Types:
  feature       Feature implementation template
  bug           Bug triage template
  ci            CI workflow change template
```

## Programmatic Usage

### Complete Pipeline Example

```python
from indexer import RepositoryIndexer
from retriever import HybridRetriever
from reranker import ResultReranker
from citation import CitationGenerator

# 1. Index repository
indexer = RepositoryIndexer(repo_root='.')
chunks = indexer.index_repository()
print(f"Indexed {len(chunks)} chunks")

# 2. Retrieve relevant content
retriever = HybridRetriever()
retriever.index(chunks)
results = retriever.retrieve(
    query="How to process depth maps?",
    top_k=20
)

# 3. Rerank for precision
reranker = ResultReranker()
reranked = reranker.rerank(results, query="depth maps", top_k=5)

# 4. Generate citations
citation_gen = CitationGenerator()
citations = citation_gen.generate_citations(reranked, max_citations=3)

# 5. Format output
formatted = citation_gen.format_citations(citations, format_type='markdown')
print(formatted)
```

### Context Window Retrieval

```python
# Get surrounding chunks for context
context_chunks = retriever.get_context_window(
    chunk_id="path/to/file.py:50:abc123",
    window_size=2  # 2 chunks before and after
)

for chunk in context_chunks:
    print(f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
    print(f"Method: {chunk.retrieval_method}")  # 'target' or 'context'
```

### Custom Reranking Signals

```python
from reranker import ResultReranker, RerankingSignal

# Custom signal weights
signals = RerankingSignal(
    exact_match_bonus=3.0,      # Boost exact matches more
    code_quality_bonus=0.5,     # Prioritize quality code
    documentation_bonus=0.3,
    test_bonus=0.1
)

reranker = ResultReranker(signals=signals)
reranked = reranker.rerank(results, query, top_k=5)
```

## Examples

### Example 1: Find Implementation Patterns

```python
# Search for how depth processing is implemented
results = retriever.retrieve(
    query="depth estimation and processing",
    chunk_type_filter=['code'],
    top_k=10
)

for result in results[:3]:
    print(f"\n{result.file_path}:{result.start_line}")
    if 'function_name' in result.metadata:
        print(f"Function: {result.metadata['function_name']}")
    print(f"Preview: {result.content[:200]}...")
```

### Example 2: Documentation Search

```python
# Find documentation about Material Response
results = retriever.retrieve(
    query="Material Response surface enhancement",
    chunk_type_filter=['doc'],
    top_k=5
)

reranked = reranker.rerank(results, query, top_k=3)

for result in reranked:
    if 'title' in result.metadata:
        print(f"Document: {result.metadata['title']}")
    print(f"File: {result.file_path}")
    print(f"Confidence: {result.score:.2f}\n")
```

### Example 3: Test Discovery

```python
# Find tests related to a feature
results = retriever.retrieve(
    query="test depth pipeline",
    chunk_type_filter=['test'],
    top_k=10
)

for result in results:
    func_name = result.metadata.get('function_name', 'unknown')
    print(f"Test: {func_name}")
    print(f"File: {result.file_path}:{result.start_line}")
```

### Example 4: Structured Response

```python
from templates import CodeModificationResponse, FileModification

# Create structured response
response = CodeModificationResponse(
    summary="Add depth-based haze effect",
    files=[
        FileModification(
            path="depth_pipeline/processors/atmospheric.py",
            patch="+ def apply_haze(image, depth, intensity=0.3): pass",
            description="New atmospheric haze processor"
        )
    ],
    tests=["tests/test_atmospheric.py"],
    explanation="Implements depth-based atmospheric haze using fog color blending",
    confidence=0.85,
    citations=[
        {
            "file_path": "depth_pipeline/processors/denoising.py",
            "snippet": "Similar depth-aware processing pattern",
            "relevance": "shows depth map usage"
        }
    ]
)

# Convert to JSON
json_output = response.to_json()
print(json_output)
```

## Testing

### Run All Tests

```bash
pytest .github/agents/rag_system/tests/test_rag_pipeline.py -v
```

### Run Specific Tests

```bash
# Test indexing
pytest tests/test_rag_pipeline.py::TestRAGPipeline::test_indexer_creates_chunks -v

# Test retrieval
pytest tests/test_rag_pipeline.py::TestRAGPipeline::test_retriever_finds_relevant_chunks -v

# Test end-to-end workflow
pytest tests/test_rag_pipeline.py::TestRAGPipeline::test_end_to_end_workflow -v
```

### Expected Test Results

```
10 passed in 0.08s
```

All tests should pass, demonstrating:
- ✅ Indexing creates valid chunks
- ✅ Retrieval finds relevant results
- ✅ Reranking improves result quality
- ✅ Citations have proper structure
- ✅ Templates generate correct format
- ✅ End-to-end pipeline works

## Performance

### Indexing Performance

- **Throughput**: ~200-300 files/second
- **Chunks created**: 1,938 chunks from repository
- **Total characters**: ~2.1M characters indexed
- **Memory usage**: ~50-100MB for index

### Retrieval Performance

- **Query time**: ~10-50ms for BM25 search
- **Reranking time**: ~5-10ms for top-k results
- **Total latency**: ~20-60ms end-to-end

### Optimization Tips

1. **Cache index**: Save indexed chunks to avoid reindexing
   ```python
   import pickle
   
   # Save
   with open('index.pkl', 'wb') as f:
       pickle.dump(chunks, f)
   
   # Load
   with open('index.pkl', 'rb') as f:
       chunks = pickle.load(f)
   ```

2. **Filter early**: Use chunk_type_filter and file_path_filter to reduce search space
   ```python
   results = retriever.retrieve(
       query="test",
       chunk_type_filter=['test'],  # Only search tests
       file_path_filter=r'test_.*\.py'  # Only test files
   )
   ```

3. **Adjust chunk size**: Larger chunks = fewer chunks but less precision
   ```python
   indexer = RepositoryIndexer(
       repo_root='.',
       chunk_size_tokens=1000,  # Larger chunks
       overlap_tokens=100
   )
   ```

## Troubleshooting

### Issue: No results found

**Solution**: Check if query matches content
```python
# Try broader query
results = retriever.retrieve("depth", top_k=10)

# Check chunk types
for chunk in chunks[:10]:
    print(f"{chunk.chunk_type}: {chunk.file_path}")
```

### Issue: Low confidence scores

**Solution**: Increase reranking signals or adjust query
```python
# More specific query
results = retriever.retrieve(
    query="ArchitecturalDepthPipeline process_image",
    top_k=10
)

# Custom reranking
signals = RerankingSignal(exact_match_bonus=3.0)
reranker = ResultReranker(signals=signals)
```

### Issue: Import errors

**Solution**: Ensure correct Python path
```python
import sys
from pathlib import Path

# Add RAG system to path
sys.path.insert(0, str(Path('.github/agents/rag_system')))

from indexer import RepositoryIndexer
```

### Issue: Slow indexing

**Solution**: Exclude large directories
```python
# The indexer already skips:
# - __pycache__
# - .git
# - node_modules
# - build/dist artifacts

# To add more exclusions, modify _should_index() method
```

## Advanced Features

### Custom Document Chunk Processing

```python
from indexer import DocumentChunk

# Create custom chunk
chunk = DocumentChunk(
    content="Custom content",
    file_path="custom.py",
    start_line=1,
    end_line=10,
    chunk_type='code',
    language='python',
    metadata={'custom_field': 'value'}
)

# Chunk ID is auto-generated
print(chunk.chunk_id)  # custom.py:1:abc12345
```

### Pattern Analysis

```python
from knowledge_engine import KnowledgeIntegrationEngine

engine = KnowledgeIntegrationEngine()

# Analyze patterns in results
patterns = engine.analyze_patterns(chunks)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence}")
    print(f"Examples: {pattern.examples}")
```

## Best Practices

1. **Index regularly**: Re-index when repository changes significantly
2. **Use specific queries**: More specific queries = better results
3. **Filter by type**: Use chunk_type_filter to narrow search
4. **Rerank always**: Reranking improves precision significantly
5. **Check confidence**: Only trust citations with >50% confidence
6. **Provide context**: Use context windows for better understanding
7. **Validate results**: Review citations before using in production

## Resources

- **Main README**: `.github/agents/rag_system/README.md`
- **Integration Tests**: `.github/agents/rag_system/tests/test_rag_pipeline.py`
- **Demo Script**: `.github/agents/rag_system/demo_full_pipeline.py`
- **CLI Reference**: Run `python3 cli.py --help`

## Support

For issues or questions:
1. Check this guide first
2. Review test examples in `test_rag_pipeline.py`
3. Run demo script for complete example
4. Check repository documentation in `docs/`

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Maintainer**: Transformation Portal Team
