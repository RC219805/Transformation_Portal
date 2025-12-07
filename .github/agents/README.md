# Transformation Portal Custom Agents

This directory contains specialized GitHub Copilot agents tailored for the Transformation Portal repository.

## Available Agents

### 🤖 RAG Integration Agent (NEW)

**File**: `rag-integration-agent.md`

**Purpose**: Advanced autonomous RAG orchestration agent for optimizing knowledge retrieval, fusion, and context-aware code assistance. Coordinates with other agents to provide accurate, well-cited responses.

**Core Capabilities**:
- **Intelligent Query Orchestration**: Multi-strategy retrieval (single, multi-source, chain reasoning, adaptive)
- **Knowledge Fusion**: Combines code, docs, and tests into coherent responses
- **Confidence Assessment**: High/medium/low confidence with gap analysis
- **Cross-Agent Coordination**: Prepares context for Specialist/Architect agents
- **Adaptive Learning**: Improves from feedback over time
- **Quality Assurance**: Validates retrieval results, detects conflicts

**Best Used For**:
- Preparing context for implementation tasks
- Investigating bugs with multi-source knowledge
- Understanding complex pipelines and workflows
- Identifying documentation/test gaps
- Coordinating between Specialist and Architect agents
- Validating information accuracy with citations

**Key Features**:
- Four retrieval strategies: SINGLE_QUERY, MULTI_SOURCE, CHAIN_REASONING, ADAPTIVE
- Intent classification: IMPLEMENTATION, BUG_FIX, EXPLORATION, OPTIMIZATION, etc.
- Confidence scoring with recency and quality metrics
- Gap and conflict detection
- Query caching for performance
- Feedback-driven learning

**Usage**:
```
@rag-integration-agent How do I add atmospheric effects to the depth pipeline?

@rag-integration-agent [IMPLEMENTATION] Add sunset LUT preset to video grader

@rag-integration-agent [MULTI_SOURCE] Find all depth processing patterns
```

**Documentation**: See `RAG_AGENT_GUIDE.md` for complete usage guide

---

### 🎨 Transformation Portal Specialist (RAG-Enhanced)

**File**: `transformation-portal-specialist.md`

**Purpose**: Expert agent for luxury real estate rendering, architectural visualization, and professional image/video processing pipelines. **Now enhanced with Retrieval-Augmented Generation (RAG)** for grounded, evidence-based responses.

**RAG Enhancements**:
- **Repository indexing**: Searches docs/, src/, tests/, agents/, changelogs, READMEs
- **Hybrid retrieval**: BM25 sparse + dense vector embeddings for optimal results
- **Citation system**: File paths, line numbers, code snippets with confidence scores
- **Structured responses**: JSON schema for code modifications (machine-parseable)
- **Canonical templates**: Feature implementation, bug triage, CI changes with few-shot examples

**Best Used For**:
- Implementing or modifying depth-aware processing pipelines
- Working with AI/ML models (Stable Diffusion, ControlNet, Depth Anything V2)
- Creating or optimizing Material Response enhancements
- Developing FFmpeg-based video processing workflows
- Optimizing batch processing performance
- Writing tests for complex processing pipelines
- Troubleshooting hardware acceleration (CoreML, CUDA, MPS)
- Adding new LUT presets and color grading workflows
- Ensuring metadata preservation (IPTC/XMP/GPS)

**Key Capabilities**:
- **RAG-powered context retrieval**: Grounds responses in actual repository code and documentation
- **Structured code modifications**: JSON schema for machine-parseable patches and test recommendations
- **Citation-backed responses**: Every code suggestion includes file paths, snippets, and confidence scores
- **Canonical workflow templates**: Predefined templates for feature implementation, bug triage, CI changes
- Deep understanding of all core pipelines (Depth, Lux Render, Material Response, Video Grader)
- Expertise in PyTorch, FFmpeg, NumPy, Pillow, and color science
- Knowledge of Apple Silicon optimization (CoreML, MPS)
- Performance profiling and optimization strategies
- Professional color grading and HDR workflows
- Comprehensive testing strategies (pytest, hypothesis, mocking)

---

### 🏗️ Transformation Portal Architect

**File**: `transformation-portal-architect.md`

**Purpose**: Senior technical authority for system design, security, and long-term health of the Transformation Portal. Focuses on architecture, cross-module integration, and technical debt management.

**Core Responsibilities**:
- **System Architecture & Integration**: Design interactions between pipelines
- **Security & Compliance**: Audit vulnerabilities, manage dependencies
- **Technical Debt Management**: Identify patterns, propose refactoring
- **Infrastructure & DevOps**: Manage CI/CD, containerization, deployment
- **API Governance**: Define contracts between components

**Best Used For**:
- Designing new modules or major refactoring
- Security audits and vulnerability reviews
- Cross-module integration design
- CI/CD workflow creation and optimization
- Architectural decision records (ADRs)
- Dependency governance and conflict resolution

**Key Capabilities**:
- Cross-module dependency analysis
- Security vulnerability scanning
- Architectural pattern enforcement
- Infrastructure as code (Docker, CI/CD)
- API contract design
- Technical debt assessment

**Usage**:
```
@transformation-portal-architect Design a new module for batch video processing

@transformation-portal-architect Review the security of this input handler

@transformation-portal-architect How should we refactor the legacy batch processor?
```

## How to Use Custom Agents

### In GitHub Copilot Chat

1. **Reference the agent in your prompt**:
   ```
   @transformation-portal-specialist help me optimize the depth pipeline for batch processing
   ```

2. **Ask specific questions**:
   ```
   @transformation-portal-specialist how do I add a new tone mapping operator to the depth pipeline?
   ```

3. **Get implementation help**:
   ```
   @transformation-portal-specialist implement a new atmospheric haze effect based on depth information
   ```

### Agent Selection Guidelines

Use the **Transformation Portal Specialist** when:
- Working on any image or video processing code
- Implementing ML/AI features
- Optimizing performance for batch operations
- Dealing with FFmpeg filter graphs
- Adding new pipeline features
- Writing tests for processing code
- Troubleshooting hardware acceleration issues

Use **general Copilot** for:
- Generic Python questions unrelated to image/video processing
- Infrastructure and CI/CD changes (though the specialist can help with these too)
- Basic file operations or utilities

## RAG System Overview

The **Retrieval-Augmented Generation (RAG) system** enhances the Transformation Portal Specialist agent by:

### 1. Repository Content Indexing
- **Indexed content**: docs/, src/, tests/, .github/agents/, CHANGELOGs, READMEs, examples/
- **Chunking**: 500-1000 tokens with 50-100 token overlap
- **Metadata**: File paths, line numbers, function/class names, docstrings
- **Total chunks**: ~1200 chunks for Transformation Portal (as of Nov 2025)

### 2. Hybrid Retrieval System
- **BM25 sparse retrieval**: Keyword-based matching for precise term lookup
- **Vector embeddings**: (Extensible) Dense semantic search for similar concepts
- **Filtering**: By file type (code/doc/test) or path pattern
- **Context windows**: Retrieve surrounding chunks for additional context

### 3. Result Reranking
- **Exact match bonus**: +2.0 for exact query phrases in content
- **Code quality signals**: +0.3 for docstrings and type hints
- **Documentation signals**: +0.2 for titles, examples, links
- **Test relevance**: +0.1 for matching test function names

### 4. Citation Generation
- **File references**: `path/to/file.py:10-25`
- **Confidence scores**: 0.0-1.0 based on retrieval rank and score
- **Code snippets**: Trimmed to 10 lines / 500 chars
- **Relevance notes**: Function names, doc types, match quality

### 5. Canonical Templates
- **Feature implementation**: Requirements → Files → Tests → PR body
- **Bug triage**: Error log → Root cause → Minimal repro → Fix
- **CI changes**: Workflow → Job steps → Testing → Secrets
- **Few-shot examples**: Real examples from repository history
- **JSON schema**: Structured responses for CI validation

### Benefits
- ✅ **Reduces hallucinations**: Responses grounded in actual repo content
- ✅ **Increases relevance**: Finds repo-specific patterns and examples
- ✅ **Provides evidence**: Citations with file paths and snippets
- ✅ **Enables automation**: JSON schema for machine parsing
- ✅ **Improves consistency**: Canonical templates for common workflows

### Using the RAG System

```bash
# Index repository
python .github/agents/rag_system/indexer.py --repo-root .

# Test retrieval
python .github/agents/rag_system/retriever.py \
    --query "depth pipeline atmospheric effects" \
    --top-k 5

# Generate citations
python .github/agents/rag_system/citation.py \
    --query "material response enhancement" \
    --format markdown

# Create workflow template
python .github/agents/rag_system/templates.py \
    --type feature \
    --description "Add sunset LUT preset" \
    --with-examples
```

See `.github/agents/rag_system/README.md` for full documentation.

## Agent Design Philosophy

The Transformation Portal Specialist agent is designed with:

1. **RAG-Enhanced Context**: Retrieval-augmented responses grounded in repository content
2. **Domain Expertise**: Deep knowledge of image/video processing, color science, and ML/AI
3. **Repository Context**: Understanding of the specific architecture, pipelines, and coding standards
4. **Practical Focus**: Emphasis on working code, performance, and testing
5. **Professional Standards**: Knowledge of industry best practices (HDR, color spaces, metadata)
6. **Hardware Awareness**: Optimization for specific accelerators (Apple Silicon, CUDA)
7. **Structured Output**: JSON schemas and templates for automation

## Creating Additional Agents

To create a new custom agent for this repository:

1. **Create a new `.md` file** in this directory (`.github/agents/`)
2. **Use the frontmatter format**:
   ```markdown
   ---
   name: Your Agent Name
   description: Brief description of what your agent does
   ---
   
   # Your Agent Name
   
   [Agent instructions and expertise...]
   ```

3. **Define clear expertise areas** and provide examples
4. **Document use cases** and best practices
5. **Test the agent** by asking it questions through Copilot

## Agent Maintenance

### When to Update Agents

Update agents when:
- New pipelines or major features are added
- Coding standards or best practices change
- New dependencies or tools are introduced
- Performance characteristics change significantly
- Common issues or FAQs emerge

### How to Update

1. Edit the relevant `.md` file in `.github/agents/`
2. Update expertise areas, examples, or troubleshooting info
3. Test the changes by asking the agent questions
4. Commit and push the changes

## Examples of Agent Usage

### Example 1: Adding a New Feature

**Prompt**:
```
@transformation-portal-specialist I need to add depth-based vignetting to the 
ArchitecturalDepthPipeline. It should darken the background more than the foreground.
```

**Expected Response**:
- Context about where this fits in the pipeline
- Code implementation for the vignetting effect
- Configuration additions to YAML presets
- Test cases to verify the effect
- Performance considerations

### Example 2: Optimizing Performance

**Prompt**:
```
@transformation-portal-specialist the batch processing is using too much memory 
when processing 4K images. How can I optimize this?
```

**Expected Response**:
- Analysis of current memory usage patterns
- Specific optimization strategies (lazy loading, batch size reduction)
- Code examples with memory profiling
- Trade-offs between memory and speed
- Testing approach to verify improvements

### Example 3: Troubleshooting

**Prompt**:
```
@transformation-portal-specialist FFmpeg is failing with "Cannot determine format 
of input stream" when processing HDR video
```

**Expected Response**:
- Explanation of the error
- Common causes (codec support, format detection)
- Diagnostic commands (ffprobe)
- Solution with corrected filter graph
- Prevention strategies for future

## Agent Effectiveness

To get the best results from custom agents:

1. **Be specific**: Mention which pipeline or component you're working with
2. **Provide context**: Share error messages, code snippets, or test output
3. **Ask for examples**: Request concrete code rather than just explanations
4. **Request testing**: Ask for test cases along with implementations
5. **Clarify constraints**: Mention performance requirements or hardware limitations

## Feedback and Improvements

If you find the custom agents could be improved:

1. Note what worked well and what didn't
2. Identify missing expertise or incorrect information
3. Suggest new examples or troubleshooting scenarios
4. Update the agent file with improvements
5. Share feedback with the team

## Resources

- **Repository Documentation**: `/docs/`
- **Copilot Instructions**: `../.github/copilot-instructions.md`
- **Architecture Guide**: `/docs/ARCHITECTURE.md`
- **Pipeline Operations**: `/docs/PIPELINE_OPERATIONS_GUIDE.md`
- **Performance Guide**: `/docs/PERFORMANCE_OPTIMIZATION.md`

---

**Note**: Custom agents are a powerful way to provide specialized assistance for domain-specific tasks. The Transformation Portal Specialist agent encapsulates years of knowledge about professional image/video processing workflows, making it easier to maintain and extend this complex codebase.
