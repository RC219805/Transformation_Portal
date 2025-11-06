# RAG System Integration Guide

This guide demonstrates how to use the newly integrated RAG (Retrieval-Augmented Generation) system for the Transformation Portal repository.

## Quick Start

### 1. Install Dependencies

```bash
cd .github/agents/rag_system
pip install -r requirements.txt
```

The RAG system requires:
- `numpy>=1.24.0` - Numerical operations
- `scikit-learn>=1.3.0` - BM25 implementation (TF-IDF)

### 2. Index Your Repository

```bash
# Index the current repository
python .github/agents/rag_system/cli.py index --repo-root . --output index_stats.json

# Output shows:
# - Total chunks created
# - Chunk type distribution (doc, code, test, config, agent)
# - Statistics saved to JSON
```

### 3. Search for Relevant Content

```bash
# Search for depth pipeline information
python .github/agents/rag_system/cli.py search "depth pipeline atmospheric effects" --top-k 5

# Filter by chunk type (code only)
python .github/agents/rag_system/cli.py search "process image" --types code --top-k 3

# Skip reranking for faster results
python .github/agents/rag_system/cli.py search "LUT preset" --no-rerank
```

### 4. Generate Citations

```bash
# Generate markdown citations
python .github/agents/rag_system/cli.py cite "material response enhancement" --max-citations 3

# Generate text format
python .github/agents/rag_system/cli.py cite "FFmpeg filter graph" --format text --output citations.txt

# Generate JSON format
python .github/agents/rag_system/cli.py cite "depth map processing" --format json --output citations.json
```

### 5. Use Prompt Templates

```bash
# Feature implementation template
python .github/agents/rag_system/cli.py template feature \
  "Add depth-based atmospheric haze effect" \
  --context "Existing atmospheric processor in depth_pipeline/processors/" \
  --output feature_request.md

# Bug triage template
python .github/agents/rag_system/cli.py template bug \
  "ImportError: No module named 'torch'" \
  --environment "Python 3.10, Ubuntu 20.04" \
  --context "Run python pipeline.py"

# CI workflow change template
python .github/agents/rag_system/cli.py template ci \
  "Add Python 3.12 to test matrix" \
  --workflow build.yml \
  --context "Ensure compatibility with latest Python"
```

## Python API Usage

### Basic Workflow

```python
from pathlib import Path
import sys

# Add RAG system to path
sys.path.insert(0, '.github/agents/rag_system')

from indexer import RepositoryIndexer
from retriever import HybridRetriever
from reranker import ResultReranker
from citation import CitationGenerator

# 1. Index repository
indexer = RepositoryIndexer(str(Path.cwd()))
chunks = indexer.index_repository()
print(f"Indexed {len(chunks)} chunks")

# 2. Setup retrieval
retriever = HybridRetriever()
retriever.index(chunks)

# 3. Search for content
results = retriever.retrieve("How to add a new LUT preset?", top_k=10)
print(f"Found {len(results)} results")

# 4. Rerank for better precision
reranker = ResultReranker()
reranked = reranker.rerank(results, "How to add a new LUT preset?", top_k=5)

# 5. Generate citations
citation_gen = CitationGenerator()
citations = citation_gen.generate_citations(reranked, max_citations=3)

# 6. Format citations
formatted = citation_gen.format_citations(citations, format_type='markdown')
print(formatted)
```

### Using Prompt Templates

```python
from templates import PromptTemplates, CodeModificationResponse, FileModification

# Generate feature implementation template
template = PromptTemplates.feature_implementation(
    feature_description="Add HDR tone mapping with custom transfer function",
    context="Existing tone mapping in tonemapper_agx_filmic.py"
)
print(template)

# Create structured code modification response
response = CodeModificationResponse(
    summary="Add atmospheric haze effect to depth pipeline",
    files=[
        FileModification(
            path="depth_pipeline/processors/atmospheric.py",
            patch="+ def apply_haze(image, depth, intensity=0.3): ...",
            description="Add haze effect implementation"
        )
    ],
    tests=["tests/test_atmospheric_processor.py"],
    explanation="Atmospheric haze blends fog color proportional to depth distance",
    confidence=0.85,
    citations=[{"file_path": "depth_pipeline/processors/clarity.py", "relevance": "Similar pattern"}]
)

# Convert to JSON for CI validation
json_output = response.to_json()
```

### Artifact Classification

```python
from classifier import ArtifactClassifier

# Classify artifacts in output directory
classifier = ArtifactClassifier()
artifacts = classifier.classify_directory('output/', recursive=True)

# Get statistics
stats = classifier.get_statistics()
print(f"Artifacts by type: {stats['by_type']}")
print(f"Artifacts by pipeline: {stats['by_pipeline']}")

# Export to JSON
classifier.export_to_json('artifacts_catalog.json')
```

### Knowledge Engine

```python
from knowledge_engine import KnowledgeIntegrationEngine

# Create engine and add feedback
engine = KnowledgeIntegrationEngine()

feedback = {
    "pipeline": "depth_pipeline",
    "success": True,
    "processing_time": 0.045,
    "parameters": {"model": "depth_anything_v2", "tone_mapping": "agx"}
}
engine.add_feedback(feedback)

# Analyze pipeline performance
analysis = engine.analyze_pipeline("depth_pipeline")
print(f"Success rate: {analysis.success_rate:.1%}")
print(f"Avg processing time: {analysis.avg_processing_time:.3f}s")

# Generate recommendations
recommendations = engine.generate_recommendations()
for rec in recommendations:
    print(f"{rec.type}: {rec.title}")

# Natural language queries
answer = engine.query("What is the success rate for depth_pipeline?")
print(answer)
```

## Running Tests

```bash
# Run all RAG system tests
pytest .github/agents/rag_system/tests/ -v

# Run specific test
pytest .github/agents/rag_system/tests/test_rag_pipeline.py::TestRAGPipeline::test_indexer_creates_chunks -v

# Run with coverage
pytest .github/agents/rag_system/tests/ --cov=.github/agents/rag_system --cov-report=html
```

## Example Workflows

### Scenario 1: Finding Similar Code Patterns

```bash
# Search for existing LUT processing code
python .github/agents/rag_system/cli.py search "LUT application video processing" \
  --types code \
  --top-k 5 \
  --output lut_examples.txt
```

### Scenario 2: Documentation Lookup

```bash
# Find depth pipeline documentation
python .github/agents/rag_system/cli.py search "depth estimation CoreML" \
  --types doc \
  --top-k 3

# Generate citations for documentation
python .github/agents/rag_system/cli.py cite "depth pipeline usage examples" \
  --format markdown \
  --max-citations 5
```

### Scenario 3: Feature Implementation with Context

```bash
# 1. Search for related code
python .github/agents/rag_system/cli.py cite "atmospheric effects depth map" \
  --format text \
  --max-citations 3 \
  --output context.txt

# 2. Generate feature template with context
python .github/agents/rag_system/cli.py template feature \
  "Add fog density parameter to atmospheric effects" \
  --context "$(cat context.txt)" \
  --output feature_plan.md
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

## Advanced Configuration

### Custom Chunk Sizes

```bash
# Larger chunks for documentation-heavy repos
python .github/agents/rag_system/cli.py index \
  --chunk-size 1500 \
  --chunk-overlap 150 \
  --repo-root .
```

### Filtering by File Path

```python
# Only search in depth_pipeline directory
results = retriever.retrieve(
    query="depth estimation",
    top_k=10,
    file_path_filter=r"depth_pipeline/.*"
)
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the repository root
cd /path/to/Transformation_Portal

# Add RAG system to Python path
export PYTHONPATH="${PYTHONPATH}:.github/agents/rag_system"
```

### No Results Found

- Check that the repository has been indexed
- Try broader search terms
- Disable reranking with `--no-rerank`
- Increase `--top-k` value

### Performance Issues

- Reduce chunk size for faster indexing
- Use file path filters to limit search scope
- Skip reranking for faster (but less precise) results

## Integration with Custom Agents

The RAG system is designed to enhance the Transformation Portal Specialist agent by:

1. **Grounding responses** in actual repository content
2. **Reducing hallucinations** by citing real code/docs
3. **Providing evidence** with file paths and snippets
4. **Structuring responses** with JSON schemas

See `.github/agents/transformation-portal-specialist.md` for the full agent configuration that uses this RAG system.

## Next Steps

1. **Index your repository** regularly (e.g., via git hooks)
2. **Use templates** for consistent feature requests and bug reports
3. **Generate citations** to validate agent responses
4. **Classify artifacts** to organize pipeline outputs
5. **Analyze performance** with the knowledge engine

For more details, see:
- [RAG System README](.github/agents/rag_system/README.md)
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Example Usage Script](example_rag_usage.py)
