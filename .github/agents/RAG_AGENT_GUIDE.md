# RAG Integration Agent - Complete Guide

This guide provides comprehensive documentation for using the RAG Integration Agent, an advanced autonomous system for optimizing RAG-powered workflows in the Transformation Portal.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Usage Patterns](#usage-patterns)
5. [Advanced Features](#advanced-features)
6. [Cross-Agent Coordination](#cross-agent-coordination)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Overview

The **RAG Integration Agent** is a specialized AI agent that:
- Orchestrates intelligent RAG queries with multiple strategies
- Fuses knowledge from multiple sources (code, docs, tests)
- Validates retrieval quality and assesses confidence
- Coordinates with other agents (Specialist, Architect)
- Learns adaptively from feedback

### When to Use

Use the RAG Integration Agent when you need:
- **Context-aware code assistance**: Retrieve relevant examples and patterns
- **Multi-source knowledge fusion**: Combine insights from code, docs, and tests
- **Confidence assessment**: Know when retrieved knowledge is reliable
- **Gap analysis**: Identify missing documentation or tests
- **Agent coordination**: Prepare context for Specialist/Architect agents

## Quick Start

### Command Line Usage

```bash
# Invoke via GitHub Copilot Chat
@rag-integration-agent How do I add atmospheric effects to the depth pipeline?

# With specific intent
@rag-integration-agent [IMPLEMENTATION] Add sunset LUT preset to video grader

# Request multi-source retrieval
@rag-integration-agent [MULTI_SOURCE] Find all depth processing patterns
```

### Python API Usage

```python
from .github.agents.rag_system.rag_agent import RAGAgent, QueryContext, UserIntent

# Initialize agent
agent = RAGAgent(repo_root='.')
agent.initialize()

# Simple query
response = agent.query("How to add a new LUT preset?")
print(response.answer)
print(f"Confidence: {response.confidence.value}")

# Print citations
for citation in response.citations:
    print(f"  {citation['file_path']}:{citation['line_range']}")
```

### With Context

```python
# Provide context for better results
context = QueryContext(
    user_intent=UserIntent.IMPLEMENTATION,
    priority="high",
    conversation_history=["Asked about video processing earlier"]
)

response = agent.query(
    "Add HDR tone mapping to video pipeline",
    context=context,
    strategy=RetrievalStrategy.MULTI_SOURCE,
    top_k=15
)
```

## Core Concepts

### Retrieval Strategies

The agent supports multiple retrieval strategies:

#### 1. Single Query (Default for Simple Queries)
```python
response = agent.query(
    "What is BM25?",
    strategy=RetrievalStrategy.SINGLE_QUERY
)
```
- **Use case**: Simple lookups, definitions, straightforward questions
- **Performance**: Fastest (~20-30ms)
- **Best for**: Exploratory queries

#### 2. Multi-Source (Best for Implementation)
```python
response = agent.query(
    "Add vignetting effect",
    strategy=RetrievalStrategy.MULTI_SOURCE
)
```
- **Use case**: Feature implementation requiring code, docs, and tests
- **Performance**: Medium (~50-100ms)
- **Best for**: Implementation tasks, comprehensive understanding

#### 3. Chain Reasoning (For Complex Workflows)
```python
response = agent.query(
    "First process depth map, then apply atmospheric effects, then tone map",
    strategy=RetrievalStrategy.CHAIN_REASONING
)
```
- **Use case**: Multi-step workflows, complex dependencies
- **Performance**: Slower (~100-200ms)
- **Best for**: Workflow understanding, end-to-end processes

#### 4. Adaptive (Automatic Selection)
```python
response = agent.query(
    "Your query here",
    strategy=RetrievalStrategy.ADAPTIVE
)
```
- **Use case**: Agent automatically selects optimal strategy
- **Performance**: Varies based on query complexity
- **Best for**: General use, when unsure which strategy to use

### User Intent Classification

The agent classifies user intent to optimize retrieval:

```python
# Automatically classified from query
response = agent.query("add new feature")  # → IMPLEMENTATION
response = agent.query("fix bug in pipeline")  # → BUG_FIX
response = agent.query("how does depth processing work?")  # → EXPLORATION

# Explicitly specify intent
context = QueryContext(user_intent=UserIntent.IMPLEMENTATION)
response = agent.query("atmospheric effects", context=context)
```

**Intent Types**:
- `IMPLEMENTATION`: Adding/creating new features
- `BUG_FIX`: Debugging and fixing issues
- `EXPLORATION`: Understanding code/documentation
- `REFACTORING`: Improving existing code
- `DOCUMENTATION`: Documenting features
- `OPTIMIZATION`: Performance improvements

### Confidence Levels

All responses include a confidence assessment:

- **HIGH** (>0.8): Strong evidence, recent, consistent sources
- **MEDIUM** (0.5-0.8): Good evidence, may have gaps
- **LOW** (<0.5): Weak evidence, outdated, or conflicting

```python
response = agent.query("depth processing")

if response.confidence == ConfidenceLevel.HIGH:
    print("High confidence - safe to proceed")
elif response.confidence == ConfidenceLevel.MEDIUM:
    print("Medium confidence - validate assumptions")
else:
    print("Low confidence - consult documentation or experts")
```

## Usage Patterns

### Pattern 1: Feature Implementation

**Goal**: Implement a new feature with RAG-retrieved patterns

```python
# Step 1: Query for implementation patterns
context = QueryContext(user_intent=UserIntent.IMPLEMENTATION)
response = agent.query(
    "Add depth-based vignetting effect",
    context=context,
    strategy=RetrievalStrategy.MULTI_SOURCE,
    top_k=10
)

# Step 2: Review retrieved sources
print(f"Found {len(response.sources)} relevant sources")
for source in response.sources[:3]:
    print(f"  - {source.file_path} ({source.chunk_type}): {source.score:.2f}")

# Step 3: Check confidence
if response.confidence != ConfidenceLevel.LOW:
    print("Sufficient context found - proceeding with implementation")
    
    # Step 4: Generate implementation plan
    for citation in response.citations:
        print(f"\nReference: {citation['file_path']}")
        print(f"Snippet: {citation['snippet'][:100]}...")
else:
    print("Low confidence - reviewing gaps:")
    for gap in response.gaps_identified:
        print(f"  - {gap}")
```

### Pattern 2: Bug Investigation

**Goal**: Investigate and fix a bug using RAG context

```python
error_message = "FFmpeg failing with 'Cannot determine format'"

# Step 1: Query for error context
response = agent.query(
    f"FFmpeg error: {error_message}",
    context=QueryContext(user_intent=UserIntent.BUG_FIX),
    strategy=RetrievalStrategy.MULTI_SOURCE
)

# Step 2: Review error handling patterns
print("Error Handling Patterns Found:")
for source in response.sources:
    if 'error' in source.content.lower() or 'exception' in source.content.lower():
        print(f"  {source.file_path}:{source.start_line}-{source.end_line}")

# Step 3: Check for conflicts (outdated solutions)
if response.conflicts:
    print("\nWarning - Conflicting Information:")
    for conflict in response.conflicts:
        print(f"  - {conflict}")

# Step 4: Get recommendations
print("\nRecommendations:")
for rec in response.recommendations:
    print(f"  - {rec}")
```

### Pattern 3: Code Exploration

**Goal**: Understand how a pipeline works

```python
# Step 1: Get architectural overview
response = agent.query(
    "depth processing pipeline architecture",
    strategy=RetrievalStrategy.MULTI_SOURCE
)

# Step 2: Build knowledge map
print("Pipeline Components:")
code_sources = [s for s in response.sources if s.chunk_type == 'code']
doc_sources = [s for s in response.sources if s.chunk_type == 'doc']

print(f"\nCode Files ({len(code_sources)}):")
for source in code_sources:
    print(f"  - {source.file_path}")

print(f"\nDocumentation ({len(doc_sources)}):")
for source in doc_sources:
    print(f"  - {source.file_path}")

# Step 3: Identify entry points
entry_point_query = "depth pipeline main function entry point"
entry_response = agent.query(entry_point_query, top_k=5)
print("\nEntry Points:")
for citation in entry_response.citations:
    print(f"  - {citation['file_path']}:{citation['line_range']}")
```

### Pattern 4: Cross-Agent Coordination

**Goal**: Prepare context for Specialist/Architect agents

```python
# Prepare context for Specialist agent
specialist_context = agent.prepare_context_for_agent(
    "transformation-portal-specialist",
    task="Implement depth-based atmospheric haze",
    include_history=True
)

print("Context prepared for Specialist:")
print(f"  Task: {specialist_context['task']}")
print(f"  Sources: {specialist_context['retrieved_sources']}")
print(f"  Confidence: {specialist_context['confidence']}")
print(f"  Citations: {len(specialist_context['citations'])}")

# Now invoke Specialist with prepared context
# @transformation-portal-specialist [with context above]
```

## Advanced Features

### Knowledge Fusion

The agent automatically fuses knowledge from multiple sources:

```python
# Query returns fused results
response = agent.query("LUT processing")

# Fused sources combine:
# - Adjacent code chunks from same file
# - Related documentation sections
# - Relevant test cases

print("Knowledge Fusion Results:")
for source in response.sources:
    print(f"\nSource: {source.file_path}")
    print(f"  Lines: {source.start_line}-{source.end_line}")
    print(f"  Type: {source.chunk_type}")
    print(f"  Score: {source.score:.3f} (retrieval)")
    print(f"  Recency: {source.recency_score:.3f}")
    print(f"  Quality: {source.quality_score:.3f}")
```

### Gap Analysis

Identify documentation and test gaps:

```python
response = agent.query("material response enhancement")

if response.gaps_identified:
    print("Knowledge Gaps Detected:")
    for gap in response.gaps_identified:
        print(f"  ⚠️  {gap}")
    
    print("\nRecommendations:")
    for rec in response.recommendations:
        print(f"  ✅  {rec}")
```

### Conflict Detection

Detect outdated or conflicting information:

```python
response = agent.query("FFmpeg filter configuration")

if response.conflicts:
    print("⚠️  Conflicts Detected:")
    for conflict in response.conflicts:
        print(f"  - {conflict}")
    
    # Review sources manually
    print("\nReview these sources:")
    for source in response.sources:
        mtime = source.metadata.get('modification_time', 'unknown')
        print(f"  {source.file_path} (modified: {mtime})")
```

### Adaptive Learning

The agent learns from feedback:

```python
# Execute query
response = agent.query("depth pipeline optimization")

# Provide feedback
agent.add_feedback(
    query="depth pipeline optimization",
    helpful=True,
    comment="Found exactly what I needed!"
)

# Agent learns successful strategy for similar queries
# Future optimization queries will prefer this strategy

# View learning statistics
stats = agent.get_statistics()
print(f"Successful strategies: {stats['successful_strategies']}")
```

### Query Caching

Automatic caching improves performance:

```python
# First query (retrieves from RAG system)
response1 = agent.query("depth processing")
print(f"Query time: {response1.metrics.query_time_ms:.1f}ms")

# Second query (from cache)
response2 = agent.query("depth processing")
print(f"Query time: {response2.metrics.query_time_ms:.1f}ms")  # Much faster!

# View cache statistics
stats = agent.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

## Cross-Agent Coordination

### With Specialist Agent

```python
# Step 1: RAG Agent retrieves implementation context
response = agent.query(
    "Add sunset LUT preset to video grader",
    context=QueryContext(user_intent=UserIntent.IMPLEMENTATION),
    strategy=RetrievalStrategy.MULTI_SOURCE
)

# Step 2: Prepare context for Specialist
specialist_context = agent.prepare_context_for_agent(
    "transformation-portal-specialist",
    "Implement sunset LUT preset"
)

# Step 3: Pass to Specialist (via Copilot Chat)
print("Context for Specialist:")
print(f"Citations: {len(specialist_context['citations'])}")
for citation in specialist_context['citations']:
    print(f"  - {citation['file_path']}")

# @transformation-portal-specialist
# Task: Implement sunset LUT preset
# Context: [specialist_context above]
```

### With Architect Agent

```python
# Step 1: RAG Agent analyzes system design
response = agent.query(
    "Cross-pipeline dependencies for depth and video processing",
    strategy=RetrievalStrategy.CHAIN_REASONING
)

# Step 2: Prepare architectural context
architect_context = agent.prepare_context_for_agent(
    "transformation-portal-architect",
    "Design integration between depth and video pipelines"
)

# Step 3: Pass to Architect
print("Context for Architect:")
print(f"System-wide sources: {architect_context['retrieved_sources']}")
print(f"Confidence: {architect_context['confidence']}")

# @transformation-portal-architect
# Task: Design pipeline integration
# Context: [architect_context above]
```

## Best Practices

### 1. Choose the Right Strategy

- **Single Query**: Simple lookups, definitions
- **Multi-Source**: Feature implementation, comprehensive understanding
- **Chain Reasoning**: Complex workflows, multi-step processes
- **Adaptive**: When unsure, let the agent decide

### 2. Provide Context

```python
# ❌ Bad - no context
response = agent.query("add effect")

# ✅ Good - with context
context = QueryContext(
    user_intent=UserIntent.IMPLEMENTATION,
    priority="high",
    conversation_history=["Discussed depth processing earlier"]
)
response = agent.query("add atmospheric haze effect", context=context)
```

### 3. Check Confidence Levels

```python
response = agent.query("your query")

if response.confidence == ConfidenceLevel.LOW:
    print("⚠️  Low confidence - validate carefully")
    print("Gaps:", response.gaps_identified)
    print("Conflicts:", response.conflicts)
```

### 4. Use Citations

```python
# Always review citations for critical decisions
for citation in response.citations:
    print(f"\n📄 {citation['file_path']}:{citation['line_range']}")
    print(f"Confidence: {citation['confidence']:.2f}")
    print(f"Snippet:\n{citation['snippet']}")
```

### 5. Provide Feedback

```python
# Help the agent learn
agent.add_feedback(
    query="your query",
    helpful=True/False,
    comment="Optional feedback comment"
)
```

### 6. Monitor Statistics

```python
# Periodically check performance
stats = agent.get_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Avg query time: {stats['avg_query_time_ms']:.1f}ms")
```

## Troubleshooting

### Issue: Low Confidence Responses

**Symptoms**: `response.confidence == ConfidenceLevel.LOW`

**Solutions**:
1. Check gaps: `response.gaps_identified`
2. Review sources: Are they relevant?
3. Try different strategy (e.g., MULTI_SOURCE)
4. Refine query with more specific terms

```python
# Check what's missing
if response.confidence == ConfidenceLevel.LOW:
    print("Gaps found:")
    for gap in response.gaps_identified:
        print(f"  - {gap}")
    
    # Try broader query
    response2 = agent.query(
        "broader query terms",
        strategy=RetrievalStrategy.MULTI_SOURCE
    )
```

### Issue: No Relevant Sources

**Symptoms**: `len(response.sources) == 0`

**Solutions**:
1. Repository not indexed: `agent.initialize(force_reindex=True)`
2. Query too specific: Broaden search terms
3. Content not in repository: Check if feature exists

```python
# Force reindex
agent.initialize(force_reindex=True)

# Try broader query
response = agent.query("broader terms")
```

### Issue: Conflicting Information

**Symptoms**: `len(response.conflicts) > 0`

**Solutions**:
1. Check source recency: Prefer newer sources
2. Review file modification times
3. Manually validate against current code

```python
if response.conflicts:
    # Sort sources by recency
    sources_by_recency = sorted(
        response.sources,
        key=lambda s: s.recency_score,
        reverse=True
    )
    print("Most recent source:")
    print(sources_by_recency[0].file_path)
```

### Issue: Slow Query Performance

**Symptoms**: `response.metrics.query_time_ms > 500`

**Solutions**:
1. Enable caching: `RAGAgent(..., enable_cache=True)`
2. Reduce `top_k` parameter
3. Use SINGLE_QUERY strategy for simple lookups

```python
# Faster query
response = agent.query(
    "quick lookup",
    strategy=RetrievalStrategy.SINGLE_QUERY,
    top_k=5  # Fewer results
)
```

## API Reference

### RAGAgent

```python
class RAGAgent:
    def __init__(
        self,
        repo_root: str,
        config_path: Optional[str] = None,
        enable_cache: bool = True,
        enable_learning: bool = True
    )
    
    def initialize(self, force_reindex: bool = False) -> int
    
    def query(
        self,
        query_text: str,
        context: Optional[QueryContext] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE,
        top_k: int = 10,
        include_citations: bool = True,
        max_citations: int = 5
    ) -> RAGResponse
    
    def prepare_context_for_agent(
        self,
        agent_name: str,
        task: str,
        include_history: bool = False
    ) -> Dict[str, Any]
    
    def add_feedback(
        self,
        query: str,
        helpful: bool,
        comment: Optional[str] = None
    )
    
    def get_statistics(self) -> Dict[str, Any]
```

### QueryContext

```python
@dataclass
class QueryContext:
    conversation_history: List[str] = []
    user_intent: Optional[UserIntent] = None
    priority: str = "medium"
    constraints: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
```

### RAGResponse

```python
@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[KnowledgeSource]
    confidence: ConfidenceLevel
    citations: List[Dict[str, Any]]
    metrics: RetrievalMetrics
    recommendations: List[str]
    gaps_identified: List[str]
    conflicts: List[str]
    
    def to_dict(self) -> Dict[str, Any]
    def to_json(self, indent: int = 2) -> str
```

### Enums

```python
class RetrievalStrategy(Enum):
    SINGLE_QUERY = "single"
    MULTI_SOURCE = "multi_source"
    CHAIN_REASONING = "chain"
    ADAPTIVE = "adaptive"
    CACHED_ONLY = "cached"

class UserIntent(Enum):
    IMPLEMENTATION = "implementation"
    BUG_FIX = "bug_fix"
    EXPLORATION = "exploration"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
```

## Examples

See `examples/rag/` directory for complete examples:
- `feature_implementation_example.py` - Implementing a new feature
- `bug_investigation_example.py` - Debugging with RAG
- `code_exploration_example.py` - Understanding a pipeline
- `cross_agent_coordination_example.py` - Coordinating with other agents

## Next Steps

1. **Try the Quick Start**: Get familiar with basic queries
2. **Experiment with Strategies**: Test different retrieval strategies
3. **Provide Feedback**: Help the agent learn from your usage
4. **Integrate with Workflow**: Use for feature development and debugging
5. **Monitor Performance**: Track statistics to optimize usage

For more information:
- **RAG System**: `.github/agents/rag_system/README.md`
- **Agent Definition**: `.github/agents/rag-integration-agent.md`
- **Integration Guide**: `.github/agents/RAG_INTEGRATION_GUIDE.md`
- **System Enhancements**: `.github/agents/RAG_SYSTEM_ENHANCEMENTS.md`

---

**Note**: The RAG Integration Agent is designed to work seamlessly with the Transformation Portal Specialist and Architect agents, providing them with accurate, relevant, and well-cited repository knowledge to eliminate hallucinations and maximize response quality.
