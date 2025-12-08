# RAG System Quick Start Guide 🚀

**Status**: ✅ Operational | **Chunks**: 3,229 | **Last Updated**: Dec 7, 2025

## 🎯 Quick Commands

### Search
```bash
# Basic search
python .github/agents/rag_system/cli.py search "your query" --top-k 5

# Search code only
python .github/agents/rag_system/cli.py search "depth pipeline" --types code

# Search docs only
python .github/agents/rag_system/cli.py search "architecture" --types doc
```

### Index
```bash
# Re-index repository
python .github/agents/rag_system/cli.py index --repo-root .

# Force full re-index (no cache)
python .github/agents/rag_system/cli.py index --repo-root . --force
```

### Classify
```bash
# Classify a file
python .github/agents/rag_system/cli.py classify path/to/file.py
```

### Templates
```bash
# Generate prompt template
python .github/agents/rag_system/cli.py template --type code_generation
python .github/agents/rag_system/cli.py template --type debugging
python .github/agents/rag_system/cli.py template --type refactoring
```

## 📊 System Stats

Run full initialization:
```bash
python init_rag_system.py
```

Check knowledge base:
```bash
ls -lh .github/agents/rag_system/knowledge_base/
```

## 🔧 Python API

### Basic Search
```python
import sys
sys.path.insert(0, '.github/agents/rag_system')

from indexer import RepositoryIndexer
from retriever import HybridRetriever

# Initialize
indexer = RepositoryIndexer(repo_root='.')
chunks = indexer.index_repository()

retriever = HybridRetriever(repo_root='.')
retriever.index(chunks)

# Search
results = retriever.search('depth pipeline', top_k=10)
for result in results:
    print(f"{result.chunk.file_path}: {result.score:.3f}")
```

### Knowledge Engine
```python
from knowledge_engine import KnowledgeIntegrationEngine

engine = KnowledgeIntegrationEngine()

# Add feedback
engine.add_feedback(
    pipeline="depth_v2",
    artifact_id="pool_test",
    success=True,
    processing_time=24.5,
    parameters={"preset": "interior_luxury"},
    quality_score=0.95
)

# Analyze patterns
patterns = engine.analyze_patterns("depth_v2", days=30)
print(f"Success rate: {patterns.success_rate:.1%}")
```

## 📁 Files

- **Main Script**: `init_rag_system.py`
- **Status Report**: `RAG_SYSTEM_STATUS.md`
- **CLI Tool**: `.github/agents/rag_system/cli.py`
- **Knowledge Base**: `.github/agents/rag_system/knowledge_base/`
- **Cache**: `.rag_cache/` (auto-generated)

## 🎓 Common Use Cases

### 1. Find Implementation Examples
```bash
python .github/agents/rag_system/cli.py search "how to add LUT preset" --top-k 3
```

### 2. Search Test Files
```bash
python .github/agents/rag_system/cli.py search "depth pipeline test" --types test
```

### 3. Find Documentation
```bash
python .github/agents/rag_system/cli.py search "architecture design" --types doc
```

### 4. Code Navigation
```bash
python .github/agents/rag_system/cli.py search "class DepthPipeline" --types code
```

## 🔄 Maintenance

### Re-initialize System
```bash
python init_rag_system.py
```

### Clear Cache
```bash
rm -rf .rag_cache/
python init_rag_system.py
```

### Update Knowledge Base
Knowledge base auto-updates on:
- Repository indexing
- Test execution (CI/CD)
- Manual analysis

## 📖 Documentation

- **Full Status**: `RAG_SYSTEM_STATUS.md`
- **Architecture**: `.github/agents/rag_system/README.md`
- **Phase 1 Guide**: `.github/agents/rag_system/PHASE1_IMPLEMENTATION_GUIDE.md`
- **Phase 2 Status**: `.github/agents/rag_system/PHASE2_IMPLEMENTATION_STATUS.md`

## ⚡ Performance

- **Index Load**: < 5 seconds (cached)
- **Search**: < 2 seconds per query
- **Memory**: ~100-200 MB
- **Throughput**: 100+ queries/min

## 🎯 Next Steps

1. Try searching for your current task
2. Generate a code template
3. Classify your working files
4. Review the full status report

**Need help?** See `RAG_SYSTEM_STATUS.md` for detailed documentation.
