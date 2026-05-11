# RAG System for Transformation Portal Specialist Agent

Retrieval-Augmented Generation (RAG) system that enhances the Transformation Portal Specialist custom agent with:

Current baseline: `main` through PR #1721. This package supports retrieval and
citation workflows; live agent behavior is governed by `.github/agents/README.md`,
`.github/copilot-instructions.md`, and `docs/architecture/agent_governance.md`.

1. **Repository content indexing** with intelligent chunking
2. **Hybrid retrieval** combining BM25 sparse and dense vector search
3. **Result reranking** for improved precision
4. **Citation generation** with confidence scores
5. **Canonical prompt templates** for common workflows
6. **Structured JSON response schemas** for machine parsing

## ✨ New in v2.0 (Enhanced Features)

- 🚀 **Persistent Caching**: 10-100x faster indexing with pickle-based cache
- ⚙️ **Configuration System**: YAML-based config with environment variable overrides
- 📊 **Structured Logging**: Python logging module with configurable levels
- 🧠 **Semantic Vector Search**: Dense embeddings with Sentence Transformers
- ⚡ **Query Caching**: LRU cache for 80-120x faster repeated queries
- 🛡️ **Custom Exceptions**: Specific error types for better error handling

See [RAG_SYSTEM_ENHANCEMENTS.md](../RAG_SYSTEM_ENHANCEMENTS.md) for detailed documentation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG System Architecture                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Indexer    │─────▶│  Retriever   │─────▶│  Reranker    │
│              │      │              │      │              │
│ • Chunks     │      │ • BM25       │      │ • Exact      │
│ • Metadata   │      │ • Vector     │      │   matches    │
│ • Overlap    │      │ • Hybrid     │      │ • Quality    │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Citation Generator                            │
│  • File paths + line numbers                                    │
│  • Code/doc snippets                                            │
│  • Confidence scores (0.0-1.0)                                  │
│  • Relevance notes                                              │
└─────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Prompt Templates                               │
│  • Feature implementation                                        │
│  • Bug triage                                                   │
│  • CI changes                                                   │
│  • Few-shot examples                                            │
│  • JSON response schema                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Indexer (`indexer.py`)

Indexes repository content into searchable chunks:

- **Chunk size**: 500-1000 tokens (~2000-4000 characters)
- **Overlap**: 50-100 tokens (~200-400 characters) between chunks
- **Indexed content**:
  - `docs/` - Documentation
  - `src/` - Source code
  - `tests/` - Test files
  - `.github/agents/` - Agent definitions
  - Top-level markdown files (READMEs, CHANGELOGs, guides)
  - `examples/` - Example code

**Features**:
- Python-aware chunking (preserves functions/classes)
- Metadata extraction (function names, docstrings, titles)
- File path and line number tracking

**Usage**:
```bash
PYTHONPATH=.github/agents python -m rag_system.indexer \
    --repo-root /path/to/repo \
    --output index_stats.json \
    --verbose
```

### 2. Retriever (`retriever.py`)

Hybrid retrieval using BM25 for sparse keyword matching:

- **BM25 parameters**: k1=1.5, b=0.75 (tuned for code/docs)
- **Filtering**: By chunk type (code/doc/test) or file path pattern
- **Context window**: Retrieve surrounding chunks for additional context

**Features**:
- Efficient sparse retrieval with TF-IDF
- Query tokenization and normalization
- Configurable top-k results
- File path and type filtering

**Usage**:
```bash
PYTHONPATH=.github/agents python -m rag_system.retriever \
    --repo-root /path/to/repo \
    --query "depth pipeline atmospheric effects" \
    --top-k 5 \
    --type doc code
```

### 3. Reranker (`reranker.py`)

Reranks retrieval results using additional signals:

- **Exact match bonus**: +2.0 for exact query phrases
- **Code quality bonus**: +0.3 for docstrings, type hints
- **Documentation bonus**: +0.2 for titles, examples, links
- **Test relevance bonus**: +0.1 for matching test names

**Features**:
- Multi-signal scoring
- Configurable signal weights
- Metadata-aware reranking

**Usage**:
```bash
PYTHONPATH=.github/agents python -m rag_system.reranker \
    --repo-root /path/to/repo \
    --query "ffmpeg filter graph" \
    --top-k 5
```

### 4. Citation Generator (`citation.py`)

Generates structured citations with confidence scores:

- **Confidence**: Computed from retrieval rank and score (0.0-1.0)
- **Snippets**: Trimmed to 10 lines / 500 characters max
- **Formats**: Markdown, plain text, JSON

**Features**:
- Rank-based confidence scoring
- Relevance notes (function names, doc types)
- Multiple output formats

**Usage**:
```bash
PYTHONPATH=.github/agents python -m rag_system.citation \
    --repo-root /path/to/repo \
    --query "material response enhancement" \
    --max-citations 5 \
    --format markdown
```

### 5. Prompt Templates (`templates.py`)

Canonical templates for common workflows:

#### Feature Implementation Template
- Requirements → Files to modify → Tests to add → PR body
- Includes few-shot examples from repository
- Structured JSON response schema

#### Bug Triage Template
- Error log → Probable cause → Minimal repro → Fix steps
- Root cause analysis workflow
- Similar issues injection

#### CI Change Template
- Workflow name → Job steps → Test coverage → Required secrets
- Impact assessment
- Existing workflow examples

**Features**:
- Structured JSON response schema (`CodeModificationResponse`)
- Few-shot examples from repository history
- Schema validation

**Usage**:
```bash
# Generate feature template
PYTHONPATH=.github/agents python -m rag_system.templates \
    --type feature \
    --description "Add depth-based fog effect" \
    --with-examples

# Generate bug triage template
PYTHONPATH=.github/agents python -m rag_system.templates \
    --type bug \
    --description "ImportError: No module named 'torch'" \
    --context "Python 3.11, Ubuntu 22.04"

# Validate response schema
PYTHONPATH=.github/agents python -m rag_system.templates \
    --validate response.json
```

### 6. Artifact Classifier (`classifier.py`)

Automatically classifies and organizes image processing artifacts:

- **Auto-classification**: Analyses, depth maps, color grades, HDR outputs, metrics, logs, profiles
- **Metadata extraction**: Pipeline, parameters, timestamps, hardware, success/failure, timing
- **Hierarchical organization**: By pipeline type with relational links (parent/child/related)
- **Tag generation**: Pipeline name, AI model, error patterns, hardware, color space, resolution, processing time
- **Version control**: Track pipeline evolution and parameter tuning
- **Lineage tracking**: Full transformation chains with audit trails

**Artifact Types**:
- Analysis, Depth Map, Color Grade, HDR Output, Metric, Log, Profile, Render, Material Response, LUT Application

**Pipeline Types**:
- Depth Pipeline, Lux Render, Material Response, Video Grader, TIFF Processor, HDR Production, AGX Filmic

**Features**:
- Pattern-based classification from filenames
- Content-based classification from file contents
- Metadata extraction (resolution, color space, bit depth, AI models, timestamps)
- Hierarchical artifact relationships (parent → child → grandchild)
- Tag-based search and filtering
- Statistics and export to JSON

**Usage**:
```bash
# Classify artifacts in a directory
python .github/agents/rag_system/classifier.py \
    --input-dir output/ \
    --output artifacts.json \
    --verbose

# Search by tags
python .github/agents/rag_system/classifier.py \
    --input-dir output/ \
    --tags depth_map 4k_plus success \
    --require-all-tags
```

### 7. Knowledge Integration Engine (`knowledge_engine.py`)

Provides pattern analysis, feedback loops, and recommendations for continuous improvement:

- **Pattern analysis**: Success rates, failure modes, performance trends, quality evolution
- **Feedback loops**: Historical outcomes inform current decisions
- **Recommendations**: Gap analysis (missing tests, undocumented features, regressions, optimizations)
- **Natural language query**: Ask questions about pipeline performance, errors, and quality
- **KPI tracking**: Track accuracy, latency, quality, and success rates over time

**Analysis Features**:
- Success rate tracking per pipeline
- Processing time statistics (avg, median, P95)
- Failure mode detection and categorization
- Trend analysis (improving, degrading, stable)
- Common and optimal parameter identification

**Recommendation Types**:
- **Regression**: Low success rate, quality degradation
- **Optimization**: Performance degradation
- **Missing Test**: Recurring errors without test coverage
- **Undocumented Feature**: Features used but not documented

**Natural Language Queries**:
- "What is the success rate for depth_pipeline?"
- "How is the performance of material_response?"
- "What errors occurred in lux_render?"
- "What improvements do you recommend?"

**Usage**:
```bash
# Analyze a pipeline
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --analyze-pipeline depth_pipeline \
    --days 30

# Generate recommendations
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --recommendations

# Natural language query
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --query "What is the average processing time?"

# Export knowledge base
python .github/agents/rag_system/knowledge_engine.py \
    --feedback-file feedback.json \
    --export knowledge_base.json
```

## JSON Response Schema

All code modification responses must follow this schema:

```json
{
  "summary": "Brief summary of changes (1-2 sentences)",
  "files": [
    {
      "path": "relative/path/to/file.py",
      "patch": "unified diff or change description",
      "description": "Why this change is needed"
    }
  ],
  "tests": [
    "tests/test_module.py",
    "tests/integration/test_feature.py"
  ],
  "explanation": "Detailed explanation of approach, trade-offs, alternatives",
  "confidence": 0.85,
  "citations": [
    {
      "file_path": "existing_code.py",
      "snippet": "relevant code snippet",
      "relevance": "shows similar pattern"
    }
  ]
}
```

**Benefits**:
- Machine-parseable for CI validation
- Structured patches for automated application
- Confidence scoring for human review
- Citations for verification

## Integration with Agent

The RAG system enhances the Transformation Portal Specialist agent by:

1. **Reducing hallucinations**: Grounding responses in actual repository content
2. **Improving relevance**: Finding repo-specific patterns and examples
3. **Providing evidence**: Citations with file paths and snippets
4. **Structuring responses**: JSON schema for automation

### Workflow Integration

```python
# Run from repository root with: export PYTHONPATH=.github/agents
from rag_system import (
    RepositoryIndexer,
    HybridRetriever,
    ResultReranker,
    CitationGenerator,
    PromptTemplates,
)

# 1. Index repository (one-time or periodic)
indexer = RepositoryIndexer('/path/to/repo')
chunks = indexer.index_repository()

# 2. Setup retrieval pipeline
retriever = HybridRetriever()
retriever.index(chunks)
reranker = ResultReranker()

# 3. Process query
query = "How to add a new LUT preset?"
results = retriever.retrieve(query, top_k=10)
reranked = reranker.rerank(results, query, top_k=5)

# 4. Generate citations
citation_gen = CitationGenerator()
citations = citation_gen.generate_citations(reranked)

# 5. Use template with citations
template = PromptTemplates.feature_implementation(
    "Add warm sunset LUT preset",
    context=citation_gen.format_citations(citations)
)
```

## Performance Characteristics

### Indexing
- **Time**: ~2-5 seconds for typical repo size (100+ files)
- **Memory**: ~50-100 MB for index in memory
- **Chunks**: ~500-1000 chunks for Transformation Portal

### Retrieval
- **BM25 search**: <10ms for typical queries
- **Reranking**: <5ms for top-10 results
- **Citation generation**: <1ms

### Scalability
- **Small repos** (<100 files): In-memory BM25 (current implementation)
- **Medium repos** (100-1000 files): FAISS for vector embeddings
- **Large repos** (>1000 files): Weaviate or Pinecone for distributed search

## Vector Database Options

Current implementation uses **in-memory BM25** for simplicity. For enhanced semantic search, consider:

### Self-Hosted Options
- **FAISS** (Facebook AI Similarity Search)
  - Best for: Single-machine, fast vector search
  - Pros: No dependencies, very fast, free
  - Cons: In-memory only, no distributed search

- **Weaviate**
  - Best for: Self-hosted semantic search with GraphQL API
  - Pros: Full-featured, persistent storage, hybrid search
  - Cons: Requires Docker/Kubernetes, more complex setup

### Cloud Options
- **Pinecone**
  - Best for: Managed vector database with low maintenance
  - Pros: Fully managed, scalable, easy to use
  - Cons: Costs money, vendor lock-in

- **Redis Vector Search**
  - Best for: Existing Redis users
  - Pros: Integrated with Redis, fast, flexible
  - Cons: Requires Redis Stack

## Future Enhancements

1. **Dense vector embeddings**: Add sentence-transformers for semantic search
2. **Persistent index**: Save/load indexed chunks to avoid reindexing
3. **Incremental updates**: Update index when files change (git hooks)
4. **Query expansion**: Automatic synonym expansion for better recall
5. **Code understanding**: AST-based code analysis for deeper understanding
6. **Embedding caching**: Cache embeddings for faster retrieval

## Testing

Run tests for RAG components:

```bash
# End-to-end RAG pipeline test
pytest .github/agents/rag_system/tests/test_rag_pipeline.py -v

# Target specific component behavior within the same suite
pytest .github/agents/rag_system/tests/test_rag_pipeline.py -k indexer -v
pytest .github/agents/rag_system/tests/test_rag_pipeline.py -k retriever -v
pytest .github/agents/rag_system/tests/test_rag_pipeline.py -k templates -v
```

## Contributing

When adding new features to the RAG system:

1. **Maintain compatibility**: Don't break existing indexer/retriever APIs
2. **Add tests**: Test new chunking strategies, retrieval methods
3. **Document performance**: Benchmark and document timing/memory
4. **Update examples**: Add new few-shot examples to `templates.py`
5. **Validate schemas**: Ensure JSON responses validate

## References

- **BM25**: Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
- **Hybrid Search**: Combining sparse (BM25) and dense (vector) retrieval for optimal results
- **RAG**: Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

## License

Same as parent repository (see root LICENSE file).
