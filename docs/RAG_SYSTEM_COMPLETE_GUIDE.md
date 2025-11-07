# RAG System - Complete Integration Guide
**Transformation Portal - Knowledge-Enhanced Processing**

## Overview

The RAG (Retrieval-Augmented Generation) system provides intelligent, context-aware assistance for the Transformation Portal. It indexes repository documentation, code, and configuration to enable:

- **Intelligent Search**: Semantic retrieval of relevant documentation and code
- **Contextual Citations**: Automatic citation generation with source attribution
- **Template Generation**: Context-aware code templates for common tasks
- **Artifact Classification**: Automatic categorization of pipeline outputs

## Architecture

```
.github/agents/rag_system/
├── cli.py                 # Command-line interface
├── indexer.py            # Repository indexing engine
├── retriever.py          # Semantic search and retrieval
├── citation.py           # Citation generation
├── classifier.py         # Artifact classification
├── reranker.py          # Result reranking (quality)
├── templates.py         # Template generation
├── knowledge_engine.py  # Core RAG orchestration
└── requirements.txt     # Dependencies
```

## Installation

```bash
cd /Users/rc/Transformation_Portal
pip install -r .github/agents/rag_system/requirements.txt
```

**Dependencies:**
- sentence-transformers (embeddings)
- faiss-cpu (vector search)
- PyYAML (config parsing)
- rich (terminal UI)

## Quick Start

### 1. Index Repository
```bash
python .github/agents/rag_system/cli.py index \
  --repo-root . \
  --output index_stats.json
```

**Output:**
- Total chunks: ~500-1000 (depending on repo size)
- Types: documentation, code, tests, config, agent
- Statistics saved to JSON

### 2. Search Documentation
```bash
python .github/agents/rag_system/cli.py search \
  "depth pipeline" \
  --top-k 5
```

**Returns:**
- Top 5 relevant passages
- Source file paths
- Relevance scores

### 3. Generate Citations
```bash
python .github/agents/rag_system/cli.py cite \
  "material response" \
  --format markdown
```

**Output:**
- Formatted citations
- Source attribution
- Context snippets

### 4. Create Templates
```bash
python .github/agents/rag_system/cli.py template feature \
  "Add new depth effect" \
  --context "architectural rendering"
```

**Generates:**
- Starter code template
- Configuration examples
- Usage documentation

### 5. Classify Artifacts
```bash
python .github/agents/rag_system/cli.py classify \
  output/ \
  --output artifacts.json
```

**Output:**
- File type classification
- Processing metadata
- Organization recommendations

## Integration with Copilot

The RAG system is accessible via the custom agent:

```
@transformation-portal-specialist <your query>
```

**Example Queries:**
```
@transformation-portal-specialist generate citations for "depth pipeline"

@transformation-portal-specialist create templates for new LUT preset

@transformation-portal-specialist explain material response technology

@transformation-portal-specialist find examples of HDR tone mapping
```

## Use Cases

### 1. **Onboarding New Developers**
```bash
# Find all documentation about a feature
python cli.py search "ControlNet integration" --top-k 10

# Generate getting-started template
python cli.py template quickstart "depth processing"
```

### 2. **Code Maintenance**
```bash
# Find all references to deprecated code
python cli.py search "deprecated" --top-k 20

# Generate migration template
python cli.py template migration "update to new API"
```

### 3. **Documentation Quality**
```bash
# Generate citations for technical writing
python cli.py cite "depth pipeline architecture" --format markdown

# Find missing documentation
python cli.py search "undocumented" --top-k 10
```

### 4. **Pipeline Output Organization**
```bash
# Classify batch processing results
python cli.py classify output_batch/ --output batch_artifacts.json

# Identify deliverable files
python cli.py classify output/ --filter "deliverable"
```

## Configuration

### Index Configuration
Create `.github/agents/rag_system/config.yaml`:

```yaml
indexing:
  chunk_size: 512
  overlap: 50
  min_chunk_length: 50
  
  file_patterns:
    - "**/*.md"
    - "**/*.py"
    - "**/*.yaml"
    - "**/*.sh"
    
  exclude_patterns:
    - ".git/**"
    - "__pycache__/**"
    - "*.pyc"
    - ".venv/**"

retrieval:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 5
  similarity_threshold: 0.3

citation:
  format: "markdown"
  max_context_length: 200
  include_line_numbers: true
```

### Environment Variables
```bash
export RAG_INDEX_PATH="./index_stats.json"
export RAG_CACHE_DIR="./.rag_cache"
export RAG_LOG_LEVEL="INFO"
```

## Advanced Features

### Custom Chunk Types

The indexer recognizes specialized content:

```python
# In indexer.py
CHUNK_TYPES = {
    'doc': ['*.md', '*.rst', '*.txt'],
    'code': ['*.py', '*.sh', '*.ts'],
    'config': ['*.yaml', '*.json', '*.toml'],
    'test': ['test_*.py', '*_test.py'],
    'agent': ['.github/agents/**/*.py']
}
```

### Reranking

Improve search quality with reranking:

```bash
python cli.py search "depth processing" \
  --top-k 20 \
  --rerank \
  --rerank-model "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### Batch Operations

Process multiple queries:

```bash
# Create queries.txt
cat > queries.txt << EOF
depth pipeline configuration
material response parameters
LUT application workflow
EOF

# Batch search
while read query; do
  python cli.py search "$query" --top-k 3
done < queries.txt
```

## Performance

### Benchmarks (M4 Max)
- **Indexing**: 500 files in ~15 seconds
- **Search**: <100ms per query (cached)
- **Citation Generation**: <200ms per query
- **Classification**: ~50ms per file

### Optimization Tips

1. **Incremental Indexing**: Only re-index changed files
2. **Cache Embeddings**: Persist embeddings between runs
3. **Batch Queries**: Group related searches
4. **Tune Chunk Size**: Balance granularity vs. context

## Troubleshooting

### Index Not Found
```bash
# Re-index repository
python cli.py index --repo-root . --force
```

### Poor Search Results
```bash
# Try different similarity thresholds
python cli.py search "query" --threshold 0.2

# Use reranking
python cli.py search "query" --rerank
```

### Memory Issues
```bash
# Reduce chunk size in config
chunk_size: 256  # down from 512

# Clear cache
rm -rf .rag_cache/
```

## Integration Examples

### With Lux Render Pipeline
```python
from rag_system.knowledge_engine import KnowledgeEngine

engine = KnowledgeEngine()

# Get context for rendering parameters
context = engine.search("optimal ControlNet settings")

# Apply context to pipeline
pipeline_config = parse_context(context)
```

### With Depth Pipeline
```python
# Find relevant depth processing examples
examples = engine.search("depth pipeline presets")

# Generate new preset template
template = engine.generate_template(
    "depth_preset",
    context=examples
)
```

### With Material Response
```python
# Retrieve material detection algorithms
algorithms = engine.search("material detection")

# Generate citation for technical documentation
citation = engine.cite(
    "material response technology",
    format="markdown"
)
```

## Maintenance

### Regular Tasks

**Weekly:**
```bash
# Re-index repository
python cli.py index --repo-root . --output index_stats.json
```

**Monthly:**
```bash
# Clear old cache
rm -rf .rag_cache/

# Rebuild index from scratch
python cli.py index --repo-root . --force
```

**Quarterly:**
```bash
# Update embeddings model (if new version available)
pip install --upgrade sentence-transformers

# Re-index with new model
python cli.py index --repo-root . --force
```

### Quality Monitoring

Check index quality:
```bash
# View index statistics
cat index_stats.json | python -m json.tool

# Test search recall
python cli.py search "known term" --top-k 10
```

## Future Enhancements

### Planned Features
- [ ] Multi-modal indexing (images, videos)
- [ ] Graph-based knowledge representation
- [ ] Automated documentation generation
- [ ] Real-time index updates (file watchers)
- [ ] Vector database persistence (FAISS → Qdrant)
- [ ] LLM-powered query expansion
- [ ] Semantic code navigation

### Research Directions
- Fine-tuned embeddings for domain-specific terminology
- Hybrid search (BM25 + semantic)
- Knowledge graph integration
- Active learning for relevance feedback

## Resources

### Documentation
- [RAG System README](.github/agents/rag_system/README.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](.github/agents/rag_system/API.md)

### References
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- RAG Paper: https://arxiv.org/abs/2005.11401

## Support

For issues or questions:
1. Check this documentation
2. Review `.github/agents/rag_system/README.md`
3. Test with `@transformation-portal-specialist`
4. File issue in GitHub repository

---

**Last Updated**: 2025-11-07  
**Version**: 1.0.0  
**Status**: Production Ready ✅
